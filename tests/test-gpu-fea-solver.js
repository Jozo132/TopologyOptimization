#!/usr/bin/env node
/**
 * Tests for the GPUFEASolver class (js/gpu-fea-solver.js).
 *
 * Validates the module API, class instantiation, and solver behavior.
 * Adapts test expectations based on whether WebGPU (Dawn) is available.
 */

import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toURL = (p) => pathToFileURL(p).href;

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        passed++;
        console.log(`  ✓ ${message}`);
    } else {
        failed++;
        console.error(`  ✗ ${message}`);
    }
}

async function runTests() {
    // ─── Test 1: Module exports GPUFEASolver ───
    console.log('Test 1: Module exports GPUFEASolver');
    const mod = await import(toURL(join(__dirname, '..', 'js', 'gpu-fea-solver.js')));
    assert(typeof mod.GPUFEASolver === 'function', 'GPUFEASolver should be an exported class');
    assert(typeof mod._getGPU === 'function', '_getGPU should be an exported function');

    // ─── Test 2: Constructor ───
    console.log('Test 2: Constructor creates instance');
    const solver = new mod.GPUFEASolver();
    assert(solver !== null && solver !== undefined, 'Instance should be created');
    assert(solver.device === null, 'device should be null initially');
    assert(solver.available === false, 'available should be false initially');

    // ─── Test 3: isAvailable() before init ───
    console.log('Test 3: isAvailable() before init');
    assert(solver.isAvailable() === false, 'isAvailable() should be false before init');

    // ─── Test 4: init() ───
    console.log('Test 4: init()');
    const result = await solver.init();
    const gpuAvailable = result === true;
    assert(typeof result === 'boolean', `init() should return a boolean (got ${result})`);
    if (gpuAvailable) {
        console.log('  → WebGPU (Dawn) is available — GPU tests will run');
    } else {
        console.log('  → WebGPU not available — testing graceful degradation');
    }

    // ─── Test 5: isAvailable() after init ───
    console.log('Test 5: isAvailable() after init');
    assert(solver.isAvailable() === gpuAvailable, `isAvailable() should be ${gpuAvailable} after init`);
    assert(solver.available === gpuAvailable, `available property should be ${gpuAvailable}`);

    // ─── Test 6: Repeated init() returns same result ───
    console.log('Test 6: Repeated init() returns cached result');
    const result2 = await solver.init();
    assert(result2 === result, 'Second init() should return same cached result');

    // ─── Test 7: setup() when GPU not available ───
    console.log('Test 7: setup() behavior');
    if (!gpuAvailable) {
        let setupThrew = false;
        try {
            solver.setup({
                KEflat: new Float32Array(576),
                edofArray: new Int32Array(24),
                densities: new Float32Array(1),
                F: new Float32Array(24),
                fixedMask: new Uint8Array(24),
                nel: 1, ndof: 24, E0: 1.0, Emin: 1e-9, penal: 3,
            });
        } catch (e) {
            setupThrew = true;
            assert(e.message.includes('GPU not available'), `Error message: "${e.message}"`);
        }
        assert(setupThrew, 'setup() should throw when GPU is not available');
    } else {
        // GPU is available — setup with minimal valid data should not throw
        let setupOk = true;
        try {
            solver.setup({
                KEflat: new Float32Array(576),
                edofArray: new Int32Array(24),
                densities: new Float32Array([1.0]),
                F: new Float32Array(24),
                fixedMask: new Uint8Array(24),
                nel: 1, ndof: 24, E0: 1.0, Emin: 1e-9, penal: 3,
            });
        } catch (e) {
            setupOk = false;
            console.error(`  setup() threw unexpectedly: ${e.message}`);
        }
        assert(setupOk, 'setup() should succeed when GPU is available');
    }

    // ─── Test 8: solve() when not properly set up ───
    console.log('Test 8: solve() behavior');
    if (!gpuAvailable) {
        let solveThrew = false;
        try {
            await solver.solve();
        } catch (e) {
            solveThrew = true;
            assert(e.message.includes('not initialized'), `Error message: "${e.message}"`);
        }
        assert(solveThrew, 'solve() should throw when not initialized');
    } else {
        // GPU is available and setup was called — try a tiny solve
        let solveOk = true;
        try {
            const solveResult = await solver.solve({ maxIterations: 5, tolerance: 1e-4 });
            assert(solveResult.U instanceof Float32Array, 'solve() should return Float32Array U');
            assert(typeof solveResult.iterations === 'number', 'solve() should return iterations count');
        } catch (e) {
            solveOk = false;
            console.error(`  solve() threw unexpectedly: ${e.message}`);
        }
        assert(solveOk, 'solve() should succeed with valid setup');
    }

    // ─── Test 9: destroy() is safe on uninitialized instance ───
    console.log('Test 9: destroy() is safe on uninitialized instance');
    let destroyOk = true;
    try {
        solver.destroy();
    } catch (e) {
        destroyOk = false;
    }
    assert(destroyOk, 'destroy() should not throw');
    assert(solver.device === null, 'device should be null after destroy');
    assert(solver.available === false, 'available should be false after destroy');
    assert(solver._initPromise === null, '_initPromise should be null after destroy');

    // ─── Test 10: Fresh instance after destroy can re-init ───
    console.log('Test 10: Re-init after destroy');
    const solver2 = new mod.GPUFEASolver();
    const result3 = await solver2.init();
    assert(result3 === gpuAvailable, `Re-init should return ${gpuAvailable}`);
    solver2.destroy();

    // ─── Summary ───
    console.log(`\nResults: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
    console.error('Test runner error:', err);
    process.exit(1);
});
