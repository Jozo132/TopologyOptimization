#!/usr/bin/env node
/**
 * Three-Way Performance Benchmark: JS vs WASM vs WebGPU
 *
 * Solves K·U = F via Conjugate Gradient on a dense SPD system.
 * The JS path takes about 1 second, so you can feel the difference.
 *
 * Each backend uses the SAME algorithm (CG) but a different mat-vec multiply:
 *   JS    → pure JavaScript nested loops
 *   WASM  → denseMatVecRaw() in matrix-ops.wasm
 *   WebGPU→ GPUCompute.matVecMul() via Dawn (node) / navigator.gpu (browser)
 *
 * After solving, the three solution vectors are compared element-by-element
 * to verify they produce the same result.
 *
 * Usage:
 *   node benchmark-three-way.js                  # auto-size (~1s JS)
 *   node benchmark-three-way.js --size=600       # explicit matrix size
 *   node benchmark-three-way.js --maxiter=500    # cap CG iterations
 */

import fs from 'fs';
import { performance } from 'perf_hooks';
import { GPUCompute } from './js/gpu-compute.js';

// ── CLI args ────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const getArg = (name, def) => {
    const m = args.find(a => a.startsWith(`--${name}=`));
    return m ? m.split('=')[1] : def;
};
const USER_SIZE = getArg('size', '');
const USER_MAXITER = parseInt(getArg('maxiter', '0'));

// ── Constants ───────────────────────────────────────────────────────
const CG_TOL = 1e-10;
const EPSILON = 1e-30;

// ═══════════════════════════════════════════════════════════════════
//  WASM loader
// ═══════════════════════════════════════════════════════════════════
let wasmExports = null;

async function loadWasm() {
    try {
        const buf = fs.readFileSync('./wasm/matrix-ops.wasm');
        const module = await WebAssembly.compile(buf);
        const instance = await WebAssembly.instantiate(module, {
            env: { abort: () => {}, seed: () => Date.now() }
        });
        wasmExports = instance.exports;
        return true;
    } catch (e) {
        console.log('  WASM load failed:', e.message);
        return false;
    }
}

function wasmDenseMatVec(K, p, Ap, n) {
    const mem = wasmExports.memory;
    const needed = (n * n + n + n) * 8 + 4096;
    const pages = Math.ceil(needed / 65536);
    while (mem.buffer.byteLength < pages * 65536) wasmExports.memory.grow(1);

    const base = 1024;
    const kPtr = base;
    const pPtr = kPtr + n * n * 8;
    const apPtr = pPtr + n * 8;

    new Float64Array(mem.buffer, kPtr, n * n).set(K);
    new Float64Array(mem.buffer, pPtr, n).set(p);

    wasmExports.denseMatVecRaw(kPtr, pPtr, apPtr, n);

    Ap.set(new Float64Array(mem.buffer, apPtr, n));
}

// ═══════════════════════════════════════════════════════════════════
//  Test data generation
// ═══════════════════════════════════════════════════════════════════

/** Generate a random Symmetric Positive-Definite matrix K = Aᵀ·A + εI
 *  The small diagonal shift (ε = 0.1) makes K SPD but poorly conditioned,
 *  requiring many CG iterations.  */
function generateSPD(n) {
    let seed = 42;
    const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

    const A = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) A[i] = rand() - 0.5;

    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            let s = 0;
            for (let k = 0; k < n; k++) s += A[k * n + i] * A[k * n + j];
            if (i === j) s += 0.1; // tiny shift → still SPD, but ill-conditioned
            K[i * n + j] = s;
            K[j * n + i] = s;
        }
    }
    return K;
}

function generateRHS(n) {
    let seed = 7;
    const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const F = new Float64Array(n);
    for (let i = 0; i < n; i++) F[i] = rand();
    return F;
}

// ═══════════════════════════════════════════════════════════════════
//  CG Solvers — all three share identical algorithm, differ only in
//               how A*p is computed
// ═══════════════════════════════════════════════════════════════════

/** Pure JS mat-vec: Ap = K * p */
function jsMatVec(K, p, Ap, n) {
    for (let i = 0; i < n; i++) {
        let s = 0;
        const row = i * n;
        for (let j = 0; j < n; j++) s += K[row + j] * p[j];
        Ap[i] = s;
    }
}

/**
 * CG solve K·U = F using a supplied mat-vec function.
 * @param {Float64Array} K  - n×n SPD matrix (row-major)
 * @param {Float64Array} F  - RHS vector
 * @param {number} n
 * @param {number} maxIter
 * @param {(K,p,Ap,n)=>void|Promise<void>} matvecFn
 * @returns {Promise<{U: Float64Array, iters: number, residual: number, time: number}>}
 */
async function solveCG(K, F, n, maxIter, matvecFn) {
    const t0 = performance.now();

    const U = new Float64Array(n);
    const r = new Float64Array(n);
    const p = new Float64Array(n);
    const Ap = new Float64Array(n);

    // r = F  (since U starts at 0, r = F - K*0 = F)
    r.set(F);
    p.set(F);

    let rho = 0;
    for (let i = 0; i < n; i++) rho += r[i] * r[i];

    let iters = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        if (rho < CG_TOL * CG_TOL) break;

        // Ap = K * p
        await matvecFn(K, p, Ap, n);

        // alpha = rho / (pᵀ·Ap)
        let pAp = 0;
        for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
        const alpha = rho / (pAp + EPSILON);

        // U += alpha·p,  r -= alpha·Ap
        let rho_new = 0;
        for (let i = 0; i < n; i++) {
            U[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rho_new += r[i] * r[i];
        }

        // beta = rho_new / rho
        const beta = rho_new / (rho + EPSILON);
        for (let i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rho = rho_new;
        iters = iter + 1;
    }

    const time = performance.now() - t0;
    return { U, iters, residual: Math.sqrt(rho), time };
}

// ═══════════════════════════════════════════════════════════════════
//  GPU mat-vec wrapper (uses GPUCompute class from gpu-compute.js)
// ═══════════════════════════════════════════════════════════════════

let gpuCompute = null;

/**
 * GPU mat-vec: Ap = K * p  (via WebGPU compute shader)
 * Note: WebGPU uses f32 internally, so results will differ slightly.
 */
async function gpuMatVec(K, p, Ap, n) {
    const result = await gpuCompute.matVecMul(K, p, n);
    Ap.set(result);
}

// ═══════════════════════════════════════════════════════════════════
//  Comparison utilities
// ═══════════════════════════════════════════════════════════════════

function compareVectors(a, b, n) {
    let maxDiff = 0, sumSqDiff = 0, maxIdx = 0;
    for (let i = 0; i < n; i++) {
        const d = Math.abs(a[i] - b[i]);
        sumSqDiff += d * d;
        if (d > maxDiff) { maxDiff = d; maxIdx = i; }
    }
    const rmsDiff = Math.sqrt(sumSqDiff / n);
    // Relative error (against norm of a)
    let normA = 0;
    for (let i = 0; i < n; i++) normA += a[i] * a[i];
    normA = Math.sqrt(normA);
    const relErr = normA > 0 ? Math.sqrt(sumSqDiff) / normA : 0;
    return { maxDiff, rmsDiff, relErr, maxIdx };
}

// ═══════════════════════════════════════════════════════════════════
//  Auto-sizing: find n so that JS CG takes ≈1 second
// ═══════════════════════════════════════════════════════════════════

async function autoSize() {
    // Calibrate: time a small solve with the ill-conditioned matrix
    const probe = 200;
    const Kp = generateSPD(probe);
    const Fp = generateRHS(probe);
    const maxIterProbe = 500;
    const res = await solveCG(Kp, Fp, probe, maxIterProbe, jsMatVec);
    // Total cost ≈ iters × n² (mat-vec dominates)
    // At probe size: totalProbe = res.time, iters_probe × probe²
    // At target n:   total_n    ≈ iters_n × n²
    // iters scales roughly linearly with n for ill-conditioned CG
    // So total ∝ n³  →  n = probe * (targetMs / res.time)^(1/3)
    const targetMs = 1000;
    const n = Math.round(probe * Math.pow(targetMs / res.time, 1 / 3));
    // Clamp and round
    return Math.max(200, Math.min(Math.round(n / 25) * 25, 1500));
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

async function main() {
    const sep = '═'.repeat(72);
    const line = '─'.repeat(72);

    console.log(sep);
    console.log('  Three-Way CG Solver Benchmark:  JS  vs  WASM  vs  WebGPU');
    console.log(sep);

    // ── 1. Load backends ────────────────────────────────────────────
    console.log('\n[Backends]');

    const haveWasm = await loadWasm();
    console.log(`  WASM:   ${haveWasm ? '✓ loaded' : '✗ not available'}`);

    gpuCompute = new GPUCompute();
    const haveGPU = await gpuCompute.init();
    console.log(`  WebGPU: ${haveGPU ? '✓ loaded' : '✗ not available'}`);

    // ── 2. Determine matrix size ────────────────────────────────────
    let N;
    if (USER_SIZE) {
        N = parseInt(USER_SIZE);
        console.log(`\n[Size] User-specified: ${N}`);
    } else {
        console.log('\n[Auto-size] Calibrating for ~1s JS solve time...');
        N = await autoSize();
        console.log(`  Selected n = ${N}`);
    }

    const maxIter = USER_MAXITER > 0 ? USER_MAXITER : N;

    console.log(`  System:   ${N} × ${N} dense SPD  (${(N * N * 8 / 1024 / 1024).toFixed(1)} MB)`);
    console.log(`  Max iter: ${maxIter}`);

    // ── 3. Generate problem ─────────────────────────────────────────
    console.log('\n[Generate] Building SPD matrix and RHS...');
    const genT0 = performance.now();
    const K = generateSPD(N);
    const F = generateRHS(N);
    console.log(`  Done in ${((performance.now() - genT0) / 1000).toFixed(2)}s`);

    // ── 4. Solve with each backend ──────────────────────────────────
    console.log('\n' + line);
    console.log('  Solving K·U = F via Conjugate Gradient');
    console.log(line);

    // JS
    console.log('\n  [JS]     Solving...');
    const jsRes = await solveCG(K, F, N, maxIter, jsMatVec);
    console.log(`           ${jsRes.iters} iterations, residual ${jsRes.residual.toExponential(4)}, time ${(jsRes.time / 1000).toFixed(3)}s`);

    // WASM
    let wasmRes = null;
    if (haveWasm) {
        console.log('\n  [WASM]   Solving...');
        wasmRes = await solveCG(K, F, N, maxIter, (K, p, Ap, n) => wasmDenseMatVec(K, p, Ap, n));
        console.log(`           ${wasmRes.iters} iterations, residual ${wasmRes.residual.toExponential(4)}, time ${(wasmRes.time / 1000).toFixed(3)}s`);
    } else {
        console.log('\n  [WASM]   Skipped (not available)');
    }

    // WebGPU
    let gpuRes = null;
    if (haveGPU) {
        console.log('\n  [WebGPU] Solving...');
        gpuRes = await solveCG(K, F, N, maxIter, gpuMatVec);
        console.log(`           ${gpuRes.iters} iterations, residual ${gpuRes.residual.toExponential(4)}, time ${(gpuRes.time / 1000).toFixed(3)}s`);
    } else {
        console.log('\n  [WebGPU] Skipped (not available)');
    }

    // ── 5. Performance comparison ───────────────────────────────────
    console.log('\n' + line);
    console.log('  Performance');
    console.log(line);

    const pad = (s, w) => String(s).padStart(w);
    const padL = (s, w) => String(s).padEnd(w);
    const fmtTime = (ms) => ms < 1000 ? `${ms.toFixed(1)} ms` : `${(ms / 1000).toFixed(3)} s `;
    const fmtSpeedup = (base, cmp) => {
        if (!cmp) return '  —  ';
        const ratio = base / cmp;
        return ratio >= 1 ? `${ratio.toFixed(2)}x faster` : `${(1 / ratio).toFixed(2)}x slower`;
    };

    console.log();
    console.log(`  ${padL('Backend', 10)} ${pad('Time', 12)} ${pad('Iterations', 12)} ${pad('Residual', 14)} ${pad('vs JS', 16)}`);
    console.log(`  ${'-'.repeat(10)} ${'-'.repeat(12)} ${'-'.repeat(12)} ${'-'.repeat(14)} ${'-'.repeat(16)}`);
    console.log(`  ${padL('JS', 10)} ${pad(fmtTime(jsRes.time), 12)} ${pad(jsRes.iters, 12)} ${pad(jsRes.residual.toExponential(4), 14)} ${pad('baseline', 16)}`);
    if (wasmRes) {
        console.log(`  ${padL('WASM', 10)} ${pad(fmtTime(wasmRes.time), 12)} ${pad(wasmRes.iters, 12)} ${pad(wasmRes.residual.toExponential(4), 14)} ${pad(fmtSpeedup(jsRes.time, wasmRes.time), 16)}`);
    }
    if (gpuRes) {
        console.log(`  ${padL('WebGPU', 10)} ${pad(fmtTime(gpuRes.time), 12)} ${pad(gpuRes.iters, 12)} ${pad(gpuRes.residual.toExponential(4), 14)} ${pad(fmtSpeedup(jsRes.time, gpuRes.time), 16)}`);
    }

    // ── 6. Correctness comparison ───────────────────────────────────
    console.log('\n' + line);
    console.log('  Solution Accuracy  (should be near-identical)');
    console.log(line);

    console.log();
    console.log(`  ${padL('Comparison', 20)} ${pad('Max |diff|', 14)} ${pad('RMS diff', 14)} ${pad('Rel error', 14)} ${pad('Verdict', 10)}`);
    console.log(`  ${'-'.repeat(20)} ${'-'.repeat(14)} ${'-'.repeat(14)} ${'-'.repeat(14)} ${'-'.repeat(10)}`);

    if (wasmRes) {
        const cmp = compareVectors(jsRes.U, wasmRes.U, N);
        const verdict = cmp.relErr < 1e-10 ? '✓ MATCH' : cmp.relErr < 1e-6 ? '≈ CLOSE' : '✗ DIFFERS';
        console.log(`  ${padL('JS vs WASM', 20)} ${pad(cmp.maxDiff.toExponential(4), 14)} ${pad(cmp.rmsDiff.toExponential(4), 14)} ${pad(cmp.relErr.toExponential(4), 14)} ${pad(verdict, 10)}`);
    }
    if (gpuRes) {
        const cmp = compareVectors(jsRes.U, gpuRes.U, N);
        const verdict = cmp.relErr < 1e-4 ? '✓ MATCH' : cmp.relErr < 1e-2 ? '≈ CLOSE' : '✗ DIFFERS';
        console.log(`  ${padL('JS vs WebGPU', 20)} ${pad(cmp.maxDiff.toExponential(4), 14)} ${pad(cmp.rmsDiff.toExponential(4), 14)} ${pad(cmp.relErr.toExponential(4), 14)} ${pad(verdict, 10)}`);
    }
    if (wasmRes && gpuRes) {
        const cmp = compareVectors(wasmRes.U, gpuRes.U, N);
        const verdict = cmp.relErr < 1e-4 ? '✓ MATCH' : cmp.relErr < 1e-2 ? '≈ CLOSE' : '✗ DIFFERS';
        console.log(`  ${padL('WASM vs WebGPU', 20)} ${pad(cmp.maxDiff.toExponential(4), 14)} ${pad(cmp.rmsDiff.toExponential(4), 14)} ${pad(cmp.relErr.toExponential(4), 14)} ${pad(verdict, 10)}`);
    }

    // ── 7. Show a few solution values ───────────────────────────────
    const show = Math.min(5, N);
    console.log(`\n  First ${show} solution values:`);
    console.log(`  ${'idx'.padStart(5)}  ${'JS'.padStart(14)}  ${wasmRes ? 'WASM'.padStart(14) : ''}  ${gpuRes ? 'WebGPU'.padStart(14) : ''}`);
    for (let i = 0; i < show; i++) {
        let line_str = `  ${String(i).padStart(5)}  ${jsRes.U[i].toFixed(10).padStart(14)}`;
        if (wasmRes) line_str += `  ${wasmRes.U[i].toFixed(10).padStart(14)}`;
        if (gpuRes) line_str += `  ${gpuRes.U[i].toFixed(10).padStart(14)}`;
        console.log(line_str);
    }

    // ── 8. Notes ────────────────────────────────────────────────────
    console.log('\n' + sep);
    console.log('  Notes:');
    console.log('  • JS and WASM both use f64 — results should be bit-identical');
    console.log('  • WebGPU computes mat-vec in f32 — small precision loss expected');
    console.log('  • WebGPU overhead (buffer copies) dominates at small n; gains at n>1000');
    console.log('  • WASM typically 1.5–3x faster than JS for dense mat-vec');
    console.log(sep);

    // ── Cleanup ─────────────────────────────────────────────────────
    if (gpuCompute) gpuCompute.destroy();
}

main().catch(err => {
    console.error('FATAL:', err);
    process.exit(1);
});
