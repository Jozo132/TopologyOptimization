#!/usr/bin/env node
/**
 * Tests for the WASM-accelerated applyAEbe3D function.
 * Verifies that the WASM element-by-element matvec produces the same results
 * as the JavaScript reference implementation.
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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

function assertClose(a, b, tol, message) {
    const diff = Math.abs(a - b);
    if (diff <= tol) {
        passed++;
        console.log(`  ✓ ${message}`);
    } else {
        failed++;
        console.error(`  ✗ ${message} (got ${a}, expected ${b}, diff ${diff})`);
    }
}

// ─── Load WASM module ───
async function loadWasm() {
    const wasmPath = join(__dirname, '..', 'wasm', 'matrix-ops.wasm');
    const buffer = await readFile(wasmPath);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module, {
        env: {
            abort: () => { throw new Error('WASM abort'); },
            seed: () => Date.now()
        }
    });
    return instance;
}

// ─── Reference JS implementation of applyA (element-by-element) ───
function applyA_JS(KEflat, edofArray, E_vals, active, activeCount, p, Ap, ndof) {
    Ap.fill(0);
    const loc = new Float64Array(24);
    for (let ai = 0; ai < activeCount; ai++) {
        const e = active[ai];
        const E = E_vals[e];
        const eOff = e * 24;
        for (let j = 0; j < 24; j++) {
            loc[j] = p[edofArray[eOff + j]];
        }
        for (let i = 0; i < 24; i++) {
            const gi = edofArray[eOff + i];
            let sum = 0.0;
            const row = i * 24;
            for (let j = 0; j < 24; j++) {
                sum += KEflat[row + j] * loc[j];
            }
            Ap[gi] += E * sum;
        }
    }
}

// ─── Generate a simple test element stiffness matrix (positive definite diagonal) ───
function makeSimpleKE() {
    const KE = new Float64Array(24 * 24);
    // Use a simple matrix: diagonal + some off-diagonal coupling
    for (let i = 0; i < 24; i++) {
        KE[i * 24 + i] = 10.0 + i; // diagonal
        if (i + 1 < 24) {
            KE[i * 24 + (i + 1)] = -1.0;
            KE[(i + 1) * 24 + i] = -1.0;
        }
    }
    return KE;
}

// ─── Precompute edof array for a small mesh ───
function precomputeEdofs3D(nelx, nely, nelz) {
    const nel = nelx * nely * nelz;
    const nny = nely + 1;
    const nnz = nelz + 1;
    const edofArray = new Int32Array(nel * 24);
    for (let elz = 0; elz < nelz; elz++) {
        for (let ely = 0; ely < nely; ely++) {
            for (let elx = 0; elx < nelx; elx++) {
                const idx = elx + ely * nelx + elz * nelx * nely;
                const offset = idx * 24;
                const n0 = elx * nny * nnz + ely * nnz + elz;
                const n1 = (elx + 1) * nny * nnz + ely * nnz + elz;
                const n2 = (elx + 1) * nny * nnz + (ely + 1) * nnz + elz;
                const n3 = elx * nny * nnz + (ely + 1) * nnz + elz;
                const n4 = elx * nny * nnz + ely * nnz + (elz + 1);
                const n5 = (elx + 1) * nny * nnz + ely * nnz + (elz + 1);
                const n6 = (elx + 1) * nny * nnz + (ely + 1) * nnz + (elz + 1);
                const n7 = elx * nny * nnz + (ely + 1) * nnz + (elz + 1);
                const nodes = [n0, n1, n2, n3, n4, n5, n6, n7];
                for (let ni = 0; ni < 8; ni++) {
                    edofArray[offset + ni * 3] = 3 * nodes[ni];
                    edofArray[offset + ni * 3 + 1] = 3 * nodes[ni] + 1;
                    edofArray[offset + ni * 3 + 2] = 3 * nodes[ni] + 2;
                }
            }
        }
    }
    return edofArray;
}

// ─── WASM applyA wrapper ───
function applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, activeCount, p, Ap, ndof) {
    const mem = wasmMod.exports.memory;
    const align8 = (v) => (v + 7) & ~7; // Ensure 8-byte alignment

    // Calculate offsets with proper alignment
    const keSize = 576 * 8;
    const scratchSize = 24 * 8;
    const edofsSize = edofArray.length * 4;
    const evalsSize = E_vals.length * 8;
    const activeSize = activeCount * 4;
    const pSize = ndof * 8;
    const apSize = ndof * 8;
    const totalBytes = keSize + scratchSize + edofsSize + evalsSize + activeSize + pSize + apSize + 64;

    const neededPages = Math.ceil(totalBytes / 65536) + 1;
    const dataStart = mem.buffer.byteLength;
    mem.grow(neededPages);

    let offset = dataStart;
    const keOff = offset; offset += keSize;
    const scratchOff = offset; offset += scratchSize;
    const edofsOff = offset; offset += edofsSize;
    offset = align8(offset);
    const evalsOff = offset; offset += evalsSize;
    const activeOff = offset; offset += activeSize;
    offset = align8(offset);
    const pOff = offset; offset += pSize;
    const apOff = offset; offset += apSize;

    // Copy data to WASM memory
    new Float64Array(mem.buffer, keOff, 576).set(KEflat);
    new Int32Array(mem.buffer, edofsOff, edofArray.length).set(edofArray);
    new Float64Array(mem.buffer, evalsOff, E_vals.length).set(E_vals);
    new Int32Array(mem.buffer, activeOff, activeCount).set(active.subarray(0, activeCount));
    new Float64Array(mem.buffer, pOff, ndof).set(p);

    // Call WASM function
    wasmMod.exports.applyAEbe3D(keOff, edofsOff, evalsOff, activeOff, activeCount, pOff, apOff, ndof, scratchOff);

    // Read result
    Ap.set(new Float64Array(mem.buffer, apOff, ndof));
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════
async function runTests() {
    const wasmMod = await loadWasm();

    // ─── Test 1: WASM module loads and has applyAEbe3D ───
    console.log('Test 1: WASM module exports applyAEbe3D');
    assert(typeof wasmMod.exports.applyAEbe3D === 'function', 'applyAEbe3D should be exported');
    assert(typeof wasmMod.exports.denseMatVecRaw === 'function', 'denseMatVecRaw should be exported');

    // ─── Test 2: Single element, all active ───
    console.log('Test 2: Single element (1x1x1 mesh)');
    {
        const nelx = 1, nely = 1, nelz = 1;
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const KEflat = makeSimpleKE();
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const E_vals = new Float64Array([1.0]);
        const active = new Int32Array([0]);
        const activeCount = 1;
        const p = new Float64Array(ndof);
        // Set some non-zero values
        for (let i = 0; i < ndof; i++) p[i] = (i + 1) * 0.1;

        const Ap_js = new Float64Array(ndof);
        const Ap_wasm = new Float64Array(ndof);

        applyA_JS(KEflat, edofArray, E_vals, active, activeCount, p, Ap_js, ndof);
        applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, activeCount, p, Ap_wasm, ndof);

        let maxDiff = 0;
        for (let i = 0; i < ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(Ap_js[i] - Ap_wasm[i]));
        }
        assert(maxDiff < 1e-10, `WASM matches JS, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 3: Multiple elements, all active ───
    console.log('Test 3: Multiple elements (3x3x3 mesh, all active)');
    {
        const nelx = 3, nely = 3, nelz = 3;
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const KEflat = makeSimpleKE();
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const E_vals = new Float64Array(nel);
        const active = new Int32Array(nel);
        for (let e = 0; e < nel; e++) {
            E_vals[e] = 0.5 + 0.5 * Math.random();
            active[e] = e;
        }
        const activeCount = nel;
        const p = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) p[i] = Math.sin(i * 0.7);

        const Ap_js = new Float64Array(ndof);
        const Ap_wasm = new Float64Array(ndof);

        applyA_JS(KEflat, edofArray, E_vals, active, activeCount, p, Ap_js, ndof);
        applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, activeCount, p, Ap_wasm, ndof);

        let maxDiff = 0;
        for (let i = 0; i < ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(Ap_js[i] - Ap_wasm[i]));
        }
        assert(maxDiff < 1e-10, `WASM matches JS, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 4: Partially active elements ───
    console.log('Test 4: Partially active elements (4x4x4 mesh, 50% active)');
    {
        const nelx = 4, nely = 4, nelz = 4;
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const KEflat = makeSimpleKE();
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const E_vals = new Float64Array(nel);
        const active = new Int32Array(nel);
        let activeCount = 0;
        for (let e = 0; e < nel; e++) {
            if (e % 2 === 0) {
                E_vals[e] = 1.0 + e * 0.01;
                active[activeCount++] = e;
            } else {
                E_vals[e] = 0.0; // inactive
            }
        }
        const p = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) p[i] = Math.cos(i * 0.3);

        const Ap_js = new Float64Array(ndof);
        const Ap_wasm = new Float64Array(ndof);

        applyA_JS(KEflat, edofArray, E_vals, active, activeCount, p, Ap_js, ndof);
        applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, activeCount, p, Ap_wasm, ndof);

        let maxDiff = 0;
        for (let i = 0; i < ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(Ap_js[i] - Ap_wasm[i]));
        }
        assert(maxDiff < 1e-10, `WASM matches JS, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 5: Zero input vector ───
    console.log('Test 5: Zero input vector');
    {
        const nelx = 2, nely = 2, nelz = 2;
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const KEflat = makeSimpleKE();
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const E_vals = new Float64Array(nel).fill(1.0);
        const active = new Int32Array(nel);
        for (let e = 0; e < nel; e++) active[e] = e;

        const p = new Float64Array(ndof); // all zeros
        const Ap_wasm = new Float64Array(ndof);
        applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, nel, p, Ap_wasm, ndof);

        let maxVal = 0;
        for (let i = 0; i < ndof; i++) maxVal = Math.max(maxVal, Math.abs(Ap_wasm[i]));
        assert(maxVal === 0, `Zero input produces zero output, max val = ${maxVal}`);
    }

    // ─── Test 6: No active elements ───
    console.log('Test 6: No active elements');
    {
        const nelx = 2, nely = 2, nelz = 2;
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const KEflat = makeSimpleKE();
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const E_vals = new Float64Array(nel).fill(1.0);
        const active = new Int32Array(nel);
        const activeCount = 0; // no active elements

        const p = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) p[i] = i + 1;
        const Ap_wasm = new Float64Array(ndof);
        applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, activeCount, p, Ap_wasm, ndof);

        let maxVal = 0;
        for (let i = 0; i < ndof; i++) maxVal = Math.max(maxVal, Math.abs(Ap_wasm[i]));
        assert(maxVal === 0, `No active elements produces zero output, max val = ${maxVal}`);
    }

    // ─── Test 7: Larger mesh stress test ───
    console.log('Test 7: Larger mesh (6x6x6, all active)');
    {
        const nelx = 6, nely = 6, nelz = 6;
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const KEflat = makeSimpleKE();
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const E_vals = new Float64Array(nel);
        const active = new Int32Array(nel);
        for (let e = 0; e < nel; e++) {
            E_vals[e] = 0.1 + 0.9 * Math.random();
            active[e] = e;
        }
        const p = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) p[i] = Math.sin(i) * 2.0;

        const Ap_js = new Float64Array(ndof);
        const Ap_wasm = new Float64Array(ndof);

        applyA_JS(KEflat, edofArray, E_vals, active, nel, p, Ap_js, ndof);
        applyA_WASM(wasmMod, KEflat, edofArray, E_vals, active, nel, p, Ap_wasm, ndof);

        let maxDiff = 0;
        for (let i = 0; i < ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(Ap_js[i] - Ap_wasm[i]));
        }
        assert(maxDiff < 1e-8, `WASM matches JS on large mesh, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 8: denseMatVecRaw correctness ───
    console.log('Test 8: denseMatVecRaw correctness');
    {
        const n = 50;
        const K = new Float64Array(n * n);
        const p = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            p[i] = (i + 1) * 0.5;
            for (let j = 0; j < n; j++) {
                K[i * n + j] = (i === j) ? 5.0 + i : -0.1 * Math.abs(i - j);
            }
        }

        // JS reference
        const Ap_js = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += K[i * n + j] * p[j];
            Ap_js[i] = s;
        }

        // WASM
        const mem = wasmMod.exports.memory;
        const totalBytes = n * n * 8 + n * 8 + n * 8;
        const neededPages = Math.ceil(totalBytes / 65536) + 1;
        const dataStart = mem.buffer.byteLength;
        mem.grow(neededPages);
        let offset = dataStart;
        const kOff = offset; offset += n * n * 8;
        const pOff = offset; offset += n * 8;
        const apOff = offset; offset += n * 8;

        new Float64Array(mem.buffer, kOff, n * n).set(K);
        new Float64Array(mem.buffer, pOff, n).set(p);

        wasmMod.exports.denseMatVecRaw(kOff, pOff, apOff, n);

        const Ap_wasm = new Float64Array(n);
        Ap_wasm.set(new Float64Array(mem.buffer, apOff, n));

        let maxDiff = 0;
        for (let i = 0; i < n; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(Ap_js[i] - Ap_wasm[i]));
        }
        assert(maxDiff < 1e-10, `denseMatVecRaw matches JS, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Summary ───
    console.log(`\nResults: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
    console.error('Test runner error:', err);
    process.exit(1);
});
