#!/usr/bin/env node
/**
 * Scaling Benchmark: JS vs WASM vs WebGPU
 *
 * Doubles the matrix size each iteration. When a backend exceeds 15s
 * it is retired. The loop continues until all backends have timed out.
 * A summary table is printed at the end.
 *
 * Usage:
 *   node benchmark-scaling.js                  # default start=50
 *   node benchmark-scaling.js --start=100      # custom start size
 *   node benchmark-scaling.js --timeout=10     # custom timeout in seconds
 *   node benchmark-scaling.js --maxiter=0      # 0 = use n as maxIter (default)
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
const START_SIZE = parseInt(getArg('start', '50'));
const TIMEOUT_S = parseFloat(getArg('timeout', '15'));
const USER_MAXITER = parseInt(getArg('maxiter', '0'));

// ── Constants ───────────────────────────────────────────────────────
const CG_TOL = 1e-10;
const EPSILON = 1e-30;
const TIMEOUT_MS = TIMEOUT_S * 1000;

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

/** Pre-load K into WASM memory once, return a fast mat-vec that only copies p each call */
function wasmPreloadMatVec(K, n) {
    const mem = wasmExports.memory;
    const needed = (n * n + n + n) * 8 + 4096;
    const pages = Math.ceil(needed / 65536);
    while (mem.buffer.byteLength < pages * 65536) wasmExports.memory.grow(1);

    const base = 1024;
    const kPtr = base;
    const pPtr = kPtr + n * n * 8;
    const apPtr = pPtr + n * 8;

    // Copy K once
    new Float64Array(mem.buffer, kPtr, n * n).set(K);

    return (_, p, Ap, nn) => {
        new Float64Array(mem.buffer, pPtr, nn).set(p);
        wasmExports.denseMatVecRaw(kPtr, pPtr, apPtr, nn);
        Ap.set(new Float64Array(mem.buffer, apPtr, nn));
    };
}

// ═══════════════════════════════════════════════════════════════════
//  Test data generation
// ═══════════════════════════════════════════════════════════════════

/** Generate a symmetric positive-definite matrix — O(n²).
 *  Uses diagonal dominance with a tiny margin (+0.1) to guarantee SPD
 *  while keeping the matrix ill-conditioned for many CG iterations. */
function generateSPD(n) {
    let seed = 42;
    const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

    const K = new Float64Array(n * n);
    // Fill upper triangle with random values, mirror to lower
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const v = (rand() - 0.5) * 2;
            K[i * n + j] = v;
            K[j * n + i] = v;
        }
    }
    // Diagonal = row abs-sum + tiny margin → barely diagonally dominant → very ill-conditioned SPD
    // Small ε means smallest eigenvalue ≈ ε, condition number ≈ O(n/ε) → many CG iterations
    const eps = 1e-6;
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        const row = i * n;
        for (let j = 0; j < n; j++) {
            if (j !== i) rowSum += Math.abs(K[row + j]);
        }
        K[row + i] = rowSum + eps;
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
//  CG Solver
// ═══════════════════════════════════════════════════════════════════

function jsMatVec(K, p, Ap, n) {
    for (let i = 0; i < n; i++) {
        let s = 0;
        const row = i * n;
        for (let j = 0; j < n; j++) s += K[row + j] * p[j];
        Ap[i] = s;
    }
}

async function solveCG(K, F, n, maxIter, matvecFn) {
    const t0 = performance.now();

    const U = new Float64Array(n);
    const r = new Float64Array(n);
    const p = new Float64Array(n);
    const Ap = new Float64Array(n);

    r.set(F);
    p.set(F);

    let rho = 0;
    for (let i = 0; i < n; i++) rho += r[i] * r[i];

    let iters = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        if (rho < CG_TOL * CG_TOL) break;

        await matvecFn(K, p, Ap, n);

        let pAp = 0;
        for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
        const alpha = rho / (pAp + EPSILON);

        let rho_new = 0;
        for (let i = 0; i < n; i++) {
            U[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rho_new += r[i] * r[i];
        }

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
//  GPU mat-vec wrapper
// ═══════════════════════════════════════════════════════════════════

let gpuCompute = null;

async function gpuMatVec(K, p, Ap, n) {
    const result = await gpuCompute.matVecMul(K, p, n);
    Ap.set(result);
}

// ═══════════════════════════════════════════════════════════════════
//  Formatting helpers
// ═══════════════════════════════════════════════════════════════════

const pad = (s, w) => String(s).padStart(w);
const padL = (s, w) => String(s).padEnd(w);

function fmtTime(ms) {
    if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
    if (ms < 1000) return `${ms.toFixed(1)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
}

function fmtSize(n) {
    const mb = n * n * 8 / 1024 / 1024;
    if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
    return `${mb.toFixed(1)} MB`;
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

async function main() {
    const sep = '═'.repeat(80);
    const line = '─'.repeat(80);

    console.log(sep);
    console.log('  Scaling Benchmark:  JS  vs  WASM  vs  WebGPU');
    console.log('  Double size each round — retire backend after >' + TIMEOUT_S + 's');
    console.log(sep);

    // ── 1. Load backends ────────────────────────────────────────────
    console.log('\n[Backends]');
    const haveWasm = await loadWasm();
    console.log(`  WASM:   ${haveWasm ? '✓ loaded' : '✗ not available'}`);

    gpuCompute = new GPUCompute();
    const haveGPU = await gpuCompute.init();
    console.log(`  WebGPU: ${haveGPU ? '✓ loaded' : '✗ not available'}`);

    // Backend tracking
    const backends = {
        JS:     { active: true,     fn: jsMatVec,     label: 'JS',     prepareFn: null },
        WASM:   { active: haveWasm, fn: null,          label: 'WASM',   prepareFn: null },
        WebGPU: { active: haveGPU,  fn: gpuMatVec,    label: 'WebGPU', prepareFn: null },
    };

    // Results table: array of { n, js: {time,iters}, wasm: {time,iters}, gpu: {time,iters} }
    const results = [];

    // ── 2. Scaling loop ─────────────────────────────────────────────
    let n = START_SIZE;
    let round = 0;

    console.log(`\n[Starting] n=${n}, doubling each round, timeout=${TIMEOUT_S}s\n`);
    console.log(line);

    while (backends.JS.active || backends.WASM.active || backends.WebGPU.active) {
        round++;
        const maxIter = USER_MAXITER > 0 ? USER_MAXITER : n;

        console.log(`\n  Round ${round}:  n = ${n}  (${fmtSize(n)} matrix,  maxIter = ${maxIter})`);
        console.log('  ' + '-'.repeat(76));

        // Generate problem for this size
        let K, F;
        try {
            const genT0 = performance.now();
            K = generateSPD(n);
            F = generateRHS(n);
            const genTime = performance.now() - genT0;
            console.log(`  Generated in ${fmtTime(genTime)}`);
        } catch (err) {
            console.log(`  ⚠ Failed to generate n=${n}: ${err.message} — stopping`);
            break;
        }

        // Pre-load K into WASM memory once per round (avoid re-copying 312MB each iteration)
        if (backends.WASM.active) {
            try {
                backends.WASM.fn = wasmPreloadMatVec(K, n);
            } catch (err) {
                console.log(`  [WASM  ]  preload failed: ${err.message} — retiring`);
                backends.WASM.active = false;
            }
        }

        const row = { n, matrixSize: fmtSize(n) };

        // Run each active backend
        for (const [key, backend] of Object.entries(backends)) {
            if (!backend.active) {
                console.log(`  [${padL(backend.label, 6)}]  retired`);
                row[key] = null;
                continue;
            }

            process.stdout.write(`  [${padL(backend.label, 6)}]  solving...`);

            try {
                const res = await solveCG(K, F, n, maxIter, backend.fn);
                row[key] = { time: res.time, iters: res.iters, residual: res.residual };

                const timeStr = fmtTime(res.time);
                console.log(`  ${res.iters} iters,  ${timeStr}`);

                // Check timeout
                if (res.time > TIMEOUT_MS) {
                    console.log(`  [${padL(backend.label, 6)}]  ⏱ exceeded ${TIMEOUT_S}s — retiring`);
                    backend.active = false;
                }
            } catch (err) {
                console.log(`  ERROR: ${err.message}`);
                row[key] = { time: -1, iters: 0, residual: -1, error: err.message };
                backend.active = false;
            }
        }

        results.push(row);

        // Double the size
        n *= 2;

        // Safety: if n > 16000 (matrix would be ~2 GB), stop regardless
        if (n > 16000) {
            console.log(`\n  ⚠ n=${n} would exceed safe memory limits — stopping`);
            break;
        }
    }

    // ── 3. Summary table ────────────────────────────────────────────
    console.log('\n' + sep);
    console.log('  RESULTS — Scaling Benchmark');
    console.log(sep);

    // Header
    const colW = { n: 7, size: 10, time: 12, iters: 7, vs: 14 };
    console.log();
    console.log(
        '  ' +
        pad('n', colW.n) + '  ' +
        pad('Matrix', colW.size) + '  │  ' +
        pad('JS', colW.time) + '  ' +
        pad('Iters', colW.iters) + '  │  ' +
        pad('WASM', colW.time) + '  ' +
        pad('Iters', colW.iters) + '  ' +
        pad('vs JS', colW.vs) + '  │  ' +
        pad('WebGPU', colW.time) + '  ' +
        pad('Iters', colW.iters) + '  ' +
        pad('vs JS', colW.vs)
    );
    console.log(
        '  ' +
        '─'.repeat(colW.n) + '──' +
        '─'.repeat(colW.size) + '──┼──' +
        '─'.repeat(colW.time) + '──' +
        '─'.repeat(colW.iters) + '──┼──' +
        '─'.repeat(colW.time) + '──' +
        '─'.repeat(colW.iters) + '──' +
        '─'.repeat(colW.vs) + '──┼──' +
        '─'.repeat(colW.time) + '──' +
        '─'.repeat(colW.iters) + '──' +
        '─'.repeat(colW.vs)
    );

    for (const row of results) {
        const jsTime = row.JS ? fmtTime(row.JS.time) : '—';
        const jsIters = row.JS ? String(row.JS.iters) : '—';

        const wasmTime = row.WASM ? fmtTime(row.WASM.time) : '—';
        const wasmIters = row.WASM ? String(row.WASM.iters) : '—';
        let wasmVs = '—';
        if (row.JS && row.WASM && row.JS.time > 0 && row.WASM.time > 0) {
            const ratio = row.JS.time / row.WASM.time;
            wasmVs = ratio >= 1
                ? `${ratio.toFixed(2)}x faster`
                : `${(1 / ratio).toFixed(2)}x slower`;
        }

        const gpuTime = row.WebGPU ? fmtTime(row.WebGPU.time) : '—';
        const gpuIters = row.WebGPU ? String(row.WebGPU.iters) : '—';
        let gpuVs = '—';
        if (row.JS && row.WebGPU && row.JS.time > 0 && row.WebGPU.time > 0) {
            const ratio = row.JS.time / row.WebGPU.time;
            gpuVs = ratio >= 1
                ? `${ratio.toFixed(2)}x faster`
                : `${(1 / ratio).toFixed(2)}x slower`;
        }

        console.log(
            '  ' +
            pad(row.n, colW.n) + '  ' +
            pad(row.matrixSize, colW.size) + '  │  ' +
            pad(jsTime, colW.time) + '  ' +
            pad(jsIters, colW.iters) + '  │  ' +
            pad(wasmTime, colW.time) + '  ' +
            pad(wasmIters, colW.iters) + '  ' +
            pad(wasmVs, colW.vs) + '  │  ' +
            pad(gpuTime, colW.time) + '  ' +
            pad(gpuIters, colW.iters) + '  ' +
            pad(gpuVs, colW.vs)
        );
    }

    // ── 4. Notes ────────────────────────────────────────────────────
    console.log('\n' + sep);
    console.log('  Notes:');
    console.log('  • CG cost ≈ iters × n²  (dense mat-vec dominates)');
    console.log('  • JS and WASM use f64 — WASM typically 1.5–3x faster');
    console.log('  • WebGPU uses f32 mat-vec — precision loss expected');
    console.log('  • WebGPU buffer copy overhead dominates at small n');
    console.log('  • Backends retired after exceeding ' + TIMEOUT_S + 's');
    console.log(sep);

    // ── Cleanup ─────────────────────────────────────────────────────
    if (gpuCompute) gpuCompute.destroy();
}

main().catch(err => {
    console.error('FATAL:', err);
    process.exit(1);
});
