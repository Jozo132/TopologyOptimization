#!/usr/bin/env node
/**
 * Cross-compatibility test: GPUCompute class should work identically
 * in Node.js (via dawn.node) and browser (via navigator.gpu).
 *
 * This test exercises the same GPUCompute class from js/gpu-compute.js
 * that the browser benchmark-gpu.html uses.
 */

// Let GPUCompute._getGPU() handle Dawn loading, polyfilling, and caching.
// Do NOT import/create a second Dawn instance — multiple Dawn instances
// on the same D3D12 GPU can cause native crashes.
import { GPUCompute } from '../js/gpu-compute.js';

// ── Helpers ─────────────────────────────────────────────────────────
function randVec(n) {
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.random();
    return v;
}

function generateSPD(n) {
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A[i * n + j] = Math.random();
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            let s = 0;
            for (let k = 0; k < n; k++) s += A[k * n + i] * A[k * n + j];
            K[i * n + j] = s + (i === j ? n : 0);
        }
    return K;
}

function jsMatVecMul(A, x, n) {
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += A[i * n + j] * x[j];
        r[i] = s;
    }
    return r;
}

function jsDot(a, b, n) {
    let s = 0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

function jsAxpy(x, y, alpha, n) {
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = alpha * x[i] + y[i];
    return r;
}

function maxDiff(a, b) {
    let max = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
        const d = Math.abs(a[i] - b[i]);
        if (d > max) max = d;
    }
    return max;
}

// ── Main test ───────────────────────────────────────────────────────
const N = 128;
let passed = 0;
let failed = 0;

function assert(condition, label, detail) {
    if (condition) {
        console.log(`  ✓ ${label}`);
        passed++;
    } else {
        console.log(`  ✗ ${label} — ${detail}`);
        failed++;
    }
}

async function main() {
    console.log('═══════════════════════════════════════════════════════');
    console.log('  GPUCompute Cross-Compatibility Test (Node.js)');
    console.log('═══════════════════════════════════════════════════════\n');

    // 1. Init
    console.log('[Init] Creating GPUCompute instance...');
    const gpu = new GPUCompute();
    const ok = await gpu.init();
    assert(ok === true, 'GPUCompute.init() returns true');
    assert(gpu.isAvailable() === true, 'GPUCompute.isAvailable() returns true');

    if (!ok) {
        console.log('\nWebGPU not available — cannot run further tests.');
        process.exit(1);
    }

    // 2. Generate data
    console.log('\n[Data] Generating test data (n=' + N + ')...');
    const K = generateSPD(N);
    const x = randVec(N);
    const a = randVec(N);
    const b = randVec(N);
    const alpha = 2.5;

    // 3. Mat-Vec Mul
    console.log('\n[MatVecMul]');
    const jsR = jsMatVecMul(K, x, N);
    const gpuR = await gpu.matVecMul(K, x, N);
    const mvDiff = maxDiff(jsR, gpuR);
    assert(gpuR.length === N, `Result length = ${N}`, `got ${gpuR.length}`);
    assert(mvDiff < 1.0, `max|diff| = ${mvDiff.toExponential(4)} < 1.0 (f32 ok)`, `diff = ${mvDiff}`);

    // 4. Dot Product
    console.log('\n[DotProduct]');
    const jsDR = jsDot(a, b, N);
    const gpuDR = await gpu.dotProduct(a, b, N);
    const dpDiff = Math.abs(jsDR - gpuDR);
    assert(typeof gpuDR === 'number', `Returns a number`, `got ${typeof gpuDR}`);
    assert(dpDiff < 1.0, `|diff| = ${dpDiff.toExponential(4)} < 1.0 (f32 ok)`, `diff = ${dpDiff}`);

    // 5. AXPY
    console.log('\n[AXPY]');
    const jsAR = jsAxpy(a, b, alpha, N);
    const gpuAR = await gpu.axpy(a, b, alpha, N);
    const axDiff = maxDiff(jsAR, gpuAR);
    assert(gpuAR.length === N, `Result length = ${N}`, `got ${gpuAR.length}`);
    assert(axDiff < 0.01, `max|diff| = ${axDiff.toExponential(4)} < 0.01`, `diff = ${axDiff}`);

    // 6. Destroy
    console.log('\n[Destroy]');
    gpu.destroy();
    assert(gpu.isAvailable() === false, 'After destroy, isAvailable() = false');

    // Summary
    console.log('\n═══════════════════════════════════════════════════════');
    console.log(`  Results: ${passed} passed, ${failed} failed`);
    console.log('═══════════════════════════════════════════════════════');

    if (failed > 0) process.exit(1);
}

main().catch(err => {
    console.error('FATAL:', err);
    process.exit(1);
});
