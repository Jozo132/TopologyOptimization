#!/usr/bin/env node
/**
 * Benchmark: Full Self-Contained WASM FEA vs Default JS FEA
 *
 * Compares the performance of:
 *   1. Default JS FEA — Jacobi-preconditioned CG with element-by-element
 *      matrix-vector multiplication, entirely in JavaScript
 *   2. WASM FEA (ebePCG) — The entire PCG solve runs inside a single WASM
 *      call, eliminating per-iteration JS↔WASM boundary crossings
 *
 * Both solvers produce identical results (verified by correctness checks).
 *
 * Usage:
 *   node benchmark-fea.js
 */

import fs from 'fs';
import { performance } from 'perf_hooks';

const CG_TOLERANCE = 1e-8;
const MAX_CG_ITERATIONS = 2000;
const EPSILON = 1e-12;

// ═══════════════════════════════════════════════════════════════════════
// WASM loading
// ═══════════════════════════════════════════════════════════════════════
let wasmModule = null;

async function loadWasmModule() {
    try {
        const buffer = fs.readFileSync('./wasm/matrix-ops.wasm');
        const module = await WebAssembly.compile(buffer);
        wasmModule = await WebAssembly.instantiate(module, {
            env: {
                abort: () => console.error('WASM abort called'),
                seed: () => Date.now()
            }
        });
        return true;
    } catch (error) {
        console.error('Failed to load WASM:', error.message);
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2D element stiffness matrix (from optimizer-worker.js)
// ═══════════════════════════════════════════════════════════════════════
function lk2D(nu) {
    const k = [
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8
    ];
    const KE = Array(8).fill(0).map(() => Array(8).fill(0));
    KE[0][0] = k[0]; KE[0][1] = k[1]; KE[0][2] = k[2]; KE[0][3] = k[3];
    KE[0][4] = k[4]; KE[0][5] = k[5]; KE[0][6] = k[6]; KE[0][7] = k[7];
    KE[1][0] = k[1]; KE[1][1] = k[0]; KE[1][2] = k[7]; KE[1][3] = k[6];
    KE[1][4] = k[5]; KE[1][5] = k[4]; KE[1][6] = k[3]; KE[1][7] = k[2];
    KE[2][0] = k[2]; KE[2][1] = k[7]; KE[2][2] = k[0]; KE[2][3] = k[5];
    KE[2][4] = k[6]; KE[2][5] = k[3]; KE[2][6] = k[4]; KE[2][7] = k[1];
    KE[3][0] = k[3]; KE[3][1] = k[6]; KE[3][2] = k[5]; KE[3][3] = k[0];
    KE[3][4] = k[7]; KE[3][5] = k[2]; KE[3][6] = k[1]; KE[3][7] = k[4];
    KE[4][0] = k[4]; KE[4][1] = k[5]; KE[4][2] = k[6]; KE[4][3] = k[7];
    KE[4][4] = k[0]; KE[4][5] = k[1]; KE[4][6] = k[2]; KE[4][7] = k[3];
    KE[5][0] = k[5]; KE[5][1] = k[4]; KE[5][2] = k[3]; KE[5][3] = k[2];
    KE[5][4] = k[1]; KE[5][5] = k[0]; KE[5][6] = k[7]; KE[5][7] = k[6];
    KE[6][0] = k[6]; KE[6][1] = k[3]; KE[6][2] = k[4]; KE[6][3] = k[1];
    KE[6][4] = k[2]; KE[6][5] = k[7]; KE[6][6] = k[0]; KE[6][7] = k[5];
    KE[7][0] = k[7]; KE[7][1] = k[2]; KE[7][2] = k[1]; KE[7][3] = k[4];
    KE[7][4] = k[3]; KE[7][5] = k[6]; KE[7][6] = k[5]; KE[7][7] = k[0];

    const scale = 1 / (1 - nu * nu);
    return KE.map(row => row.map(val => val * scale));
}

function flattenKE(KE, size) {
    const flat = new Float64Array(size * size);
    for (let i = 0; i < size; i++)
        for (let j = 0; j < size; j++)
            flat[i * size + j] = KE[i][j];
    return flat;
}

// ═══════════════════════════════════════════════════════════════════════
// Mesh setup
// ═══════════════════════════════════════════════════════════════════════
function setupMesh(nelx, nely) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofSize = 8;

    // Element DOF connectivity
    const edofArray = new Int32Array(nel * edofSize);
    for (let ely = 0; ely < nely; ely++) {
        for (let elx = 0; elx < nelx; elx++) {
            const idx = ely + elx * nely;
            const offset = idx * edofSize;
            const n1 = (nely + 1) * elx + ely;
            const n2 = (nely + 1) * (elx + 1) + ely;
            edofArray[offset] = 2 * n1;
            edofArray[offset + 1] = 2 * n1 + 1;
            edofArray[offset + 2] = 2 * n2;
            edofArray[offset + 3] = 2 * n2 + 1;
            edofArray[offset + 4] = 2 * n2 + 2;
            edofArray[offset + 5] = 2 * n2 + 3;
            edofArray[offset + 6] = 2 * n1 + 2;
            edofArray[offset + 7] = 2 * n1 + 3;
        }
    }

    // Fixed DOFs: left edge (cantilever beam)
    const fixeddofs = [];
    for (let j = 0; j <= nely; j++) {
        fixeddofs.push(2 * j);     // x DOF
        fixeddofs.push(2 * j + 1); // y DOF
    }

    // Free DOFs
    const fixedSet = new Set(fixeddofs);
    const freedofs = new Int32Array(ndof - fixedSet.size);
    let fi = 0;
    for (let i = 0; i < ndof; i++) {
        if (!fixedSet.has(i)) freedofs[fi++] = i;
    }

    // Force vector: downward force at bottom-right corner
    const F = new Float64Array(ndof);
    const loadNode = (nely + 1) * nelx + nely;
    F[2 * loadNode + 1] = -1.0;

    // Densities: all solid
    const densities = new Float64Array(nel).fill(1.0);

    return { nel, ndof, edofSize, edofArray, freedofs, F, densities };
}

// ═══════════════════════════════════════════════════════════════════════
// JS FEA Solver (matches optimizer-worker.js logic)
// ═══════════════════════════════════════════════════════════════════════
function solveFEA_JS(nelx, nely, densities, KEflat, edofArray, F, freedofs, E0, Emin, penal) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofSize = 8;
    const nfree = freedofs.length;
    const dE = E0 - Emin;
    const skipThreshold = Emin * 1000;

    // Precompute element stiffnesses
    const E_vals = new Float64Array(nel);
    const activeElements = [];
    for (let e = 0; e < nel; e++) {
        const E = Emin + Math.pow(densities[e], penal) * dE;
        E_vals[e] = E;
        if (E > skipThreshold) activeElements.push(e);
    }

    // Compute Jacobi preconditioner
    const diag = new Float64Array(ndof);
    for (let ae = 0; ae < activeElements.length; ae++) {
        const e = activeElements[ae];
        const E = E_vals[e];
        const eOff = e * edofSize;
        for (let i = 0; i < edofSize; i++) {
            diag[edofArray[eOff + i]] += E * KEflat[i * edofSize + i];
        }
    }
    const invDiag = new Float64Array(nfree);
    for (let i = 0; i < nfree; i++) {
        const d = diag[freedofs[i]];
        invDiag[i] = d > 1e-30 ? 1.0 / d : 0.0;
    }

    // Full-space work buffers
    const p_full = new Float64Array(ndof);
    const Ap_full = new Float64Array(ndof);

    // EbE matvec function
    function ebeMatVec(p_reduced, Ap_reduced) {
        p_full.fill(0);
        for (let i = 0; i < nfree; i++) p_full[freedofs[i]] = p_reduced[i];
        Ap_full.fill(0);
        for (let ae = 0; ae < activeElements.length; ae++) {
            const e = activeElements[ae];
            const E = E_vals[e];
            const eOff = e * edofSize;
            for (let i = 0; i < edofSize; i++) {
                const gi = edofArray[eOff + i];
                let sum = 0;
                const keRow = i * edofSize;
                for (let j = 0; j < edofSize; j++) {
                    sum += KEflat[keRow + j] * p_full[edofArray[eOff + j]];
                }
                Ap_full[gi] += E * sum;
            }
        }
        for (let i = 0; i < nfree; i++) Ap_reduced[i] = Ap_full[freedofs[i]];
    }

    // CG solve
    const Uf = new Float64Array(nfree);
    const r = new Float64Array(nfree);
    const z = new Float64Array(nfree);
    const p = new Float64Array(nfree);
    const Ap = new Float64Array(nfree);

    for (let i = 0; i < nfree; i++) r[i] = F[freedofs[i]];

    let rz = 0;
    for (let i = 0; i < nfree; i++) {
        z[i] = invDiag[i] * r[i];
        p[i] = z[i];
        rz += r[i] * z[i];
    }

    const maxIter = Math.min(nfree, MAX_CG_ITERATIONS);
    const tolSq = CG_TOLERANCE * CG_TOLERANCE;

    for (let iter = 0; iter < maxIter; iter++) {
        let rnorm2 = 0;
        for (let i = 0; i < nfree; i++) rnorm2 += r[i] * r[i];
        if (rnorm2 < tolSq) break;

        ebeMatVec(p, Ap);

        let pAp = 0;
        for (let i = 0; i < nfree; i++) pAp += p[i] * Ap[i];
        const alpha = rz / (pAp + EPSILON);

        let rz_new = 0;
        for (let i = 0; i < nfree; i++) {
            Uf[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            z[i] = invDiag[i] * r[i];
            rz_new += r[i] * z[i];
        }

        const beta = rz_new / (rz + EPSILON);
        for (let i = 0; i < nfree; i++) {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }

    // Map back to full DOF vector
    const U = new Float64Array(ndof);
    for (let i = 0; i < nfree; i++) U[freedofs[i]] = Uf[i];

    let compliance = 0;
    for (let i = 0; i < ndof; i++) compliance += F[i] * U[i];

    return { U, compliance };
}

// ═══════════════════════════════════════════════════════════════════════
// WASM FEA Solver (ebePCG — full self-contained)
// ═══════════════════════════════════════════════════════════════════════
function solveFEA_WASM(nelx, nely, densities, KEflat, edofArray, F, freedofs, E0, Emin, penal) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofSize = 8;
    const nfree = freedofs.length;

    const mem = wasmModule.exports.memory;
    const align8 = (v) => (v + 7) & ~7;

    // Calculate memory layout
    const densSize = nel * 8;
    const keSize = edofSize * edofSize * 8;
    const edofsSize = nel * edofSize * 4;
    const fSize = ndof * 8;
    const uSize = ndof * 8;
    const freedofsSize = nfree * 4;
    // Workspace: E_vals[nel] + active[nel(i32)] + diag[ndof] +
    //            Uf[nfree] + r[nfree] + z[nfree] + p[nfree] + Ap[nfree] +
    //            p_full[ndof] + Ap_full[ndof] + scratch[edofSize] + invDiag[nfree]
    const workSize = nel * 8 + nel * 4 + ndof * 8 +
                     5 * nfree * 8 + 2 * ndof * 8 + edofSize * 8 + nfree * 8;

    const totalBytes = densSize + keSize + edofsSize + fSize + uSize + freedofsSize + workSize + 128;

    // Ensure enough WASM memory
    const currentBytes = mem.buffer.byteLength;
    if (currentBytes < totalBytes + 65536) {
        const additionalPages = Math.ceil((totalBytes + 65536 - currentBytes) / 65536);
        mem.grow(additionalPages);
    }
    const dataStart = mem.buffer.byteLength - totalBytes - 64;

    let offset = align8(dataStart);
    const densOff = offset; offset += densSize;
    const keOff = offset; offset += keSize;
    const edofsOff = offset; offset += edofsSize;
    offset = align8(offset);
    const fOff = offset; offset += fSize;
    const uOff = offset; offset += uSize;
    const freedofsOff = offset; offset += freedofsSize;
    offset = align8(offset);
    const workOff = offset;

    // Copy input data
    new Float64Array(mem.buffer, densOff, nel).set(densities);
    new Float64Array(mem.buffer, keOff, edofSize * edofSize).set(KEflat);
    new Int32Array(mem.buffer, edofsOff, nel * edofSize).set(edofArray);
    new Float64Array(mem.buffer, fOff, ndof).set(F);
    new Int32Array(mem.buffer, freedofsOff, nfree).set(freedofs);

    // Call WASM ebePCG
    const iterations = wasmModule.exports.ebePCG(
        densOff, keOff, edofsOff, fOff, uOff, freedofsOff,
        nel, edofSize, ndof, nfree,
        Emin, E0, penal, MAX_CG_ITERATIONS, CG_TOLERANCE, workOff
    );

    // Read results
    const U = new Float64Array(ndof);
    U.set(new Float64Array(mem.buffer, uOff, ndof));

    let compliance = 0;
    for (let i = 0; i < ndof; i++) compliance += F[i] * U[i];

    return { U, compliance, iterations };
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark runner
// ═══════════════════════════════════════════════════════════════════════
async function runBenchmark(nelx, nely, iterations = 5) {
    const nu = 0.3;
    const E0 = 1.0;
    const Emin = 1e-9;
    const penal = 3;

    const KE = lk2D(nu);
    const KEflat = flattenKE(KE, 8);
    const mesh = setupMesh(nelx, nely);

    console.log(`\n${'='.repeat(65)}`);
    console.log(`Mesh: ${nelx}×${nely} (${mesh.nel} elements, ${mesh.ndof} DOFs, ${mesh.freedofs.length} free DOFs)`);
    console.log('='.repeat(65));

    // Warm-up
    console.log('Warming up...');
    solveFEA_JS(nelx, nely, mesh.densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
    solveFEA_WASM(nelx, nely, mesh.densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

    // Correctness check
    const jsResult = solveFEA_JS(nelx, nely, mesh.densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
    const wasmResult = solveFEA_WASM(nelx, nely, mesh.densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

    let maxDiff = 0;
    for (let i = 0; i < mesh.ndof; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
    }
    const complianceDiff = Math.abs(jsResult.compliance - wasmResult.compliance);

    console.log(`Correctness: max |U_js - U_wasm| = ${maxDiff.toExponential(2)}`);
    console.log(`             |c_js - c_wasm| = ${complianceDiff.toExponential(2)}`);
    if (maxDiff < 1e-6) {
        console.log('✓ Results match within tolerance');
    } else {
        console.log('✗ Warning: Results differ significantly!');
    }

    // Benchmark JS
    const jsTimes = [];
    console.log('\nRunning JS FEA benchmarks...');
    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        solveFEA_JS(nelx, nely, mesh.densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const end = performance.now();
        jsTimes.push(end - start);
        process.stdout.write(`  Run ${i + 1}/${iterations}: ${(end - start).toFixed(2)}ms\r`);
    }
    console.log('');

    // Benchmark WASM
    const wasmTimes = [];
    console.log('Running WASM FEA benchmarks...');
    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        solveFEA_WASM(nelx, nely, mesh.densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const end = performance.now();
        wasmTimes.push(end - start);
        process.stdout.write(`  Run ${i + 1}/${iterations}: ${(end - start).toFixed(2)}ms\r`);
    }
    console.log('');

    const jsAvg = jsTimes.reduce((a, b) => a + b) / jsTimes.length;
    const wasmAvg = wasmTimes.reduce((a, b) => a + b) / wasmTimes.length;
    const jsMin = Math.min(...jsTimes);
    const wasmMin = Math.min(...wasmTimes);
    const speedup = jsAvg / wasmAvg;
    const improvement = ((jsAvg - wasmAvg) / jsAvg * 100);

    console.log('─'.repeat(65));
    console.log(`JS FEA:   avg ${jsAvg.toFixed(2)}ms  min ${jsMin.toFixed(2)}ms`);
    console.log(`WASM FEA: avg ${wasmAvg.toFixed(2)}ms  min ${wasmMin.toFixed(2)}ms`);
    if (improvement > 0) {
        console.log(`🚀 WASM is ${improvement.toFixed(1)}% FASTER (${speedup.toFixed(2)}x speedup)`);
    } else {
        console.log(`⚠️  WASM is ${Math.abs(improvement).toFixed(1)}% slower (${speedup.toFixed(2)}x)`);
    }

    return { nelx, nely, nel: mesh.nel, ndof: mesh.ndof, nfree: mesh.freedofs.length, jsAvg, wasmAvg, speedup, improvement };
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════
async function main() {
    console.log('╔═════════════════════════════════════════════════════════════════╗');
    console.log('║  Full Self-Contained WASM FEA vs Default JS FEA Benchmark      ║');
    console.log('║  (Jacobi-preconditioned CG with EbE matvec)                    ║');
    console.log('╚═════════════════════════════════════════════════════════════════╝');

    console.log('\nLoading WASM module...');
    const loaded = await loadWasmModule();
    if (!loaded) {
        console.error('Cannot run benchmark: WASM module failed to load');
        process.exit(1);
    }
    console.log('✓ WASM module loaded successfully');

    // Test different mesh sizes
    const testCases = [
        { nelx: 10, nely: 10, iterations: 20 },
        { nelx: 20, nely: 20, iterations: 10 },
        { nelx: 40, nely: 20, iterations: 5 },
        { nelx: 60, nely: 30, iterations: 5 },
        { nelx: 80, nely: 40, iterations: 3 },
        { nelx: 100, nely: 50, iterations: 3 },
    ];

    const results = [];
    for (const tc of testCases) {
        try {
            const result = await runBenchmark(tc.nelx, tc.nely, tc.iterations);
            results.push(result);
        } catch (err) {
            console.error(`  Failed for ${tc.nelx}×${tc.nely}: ${err.message}`);
        }
    }

    // Summary table
    console.log('\n\n' + '═'.repeat(75));
    console.log('SUMMARY: Full Self-Contained WASM FEA vs Default JS FEA');
    console.log('═'.repeat(75));
    console.log('Mesh       | Elements | Free DOFs | JS (ms) | WASM (ms) | Speedup');
    console.log('─'.repeat(75));

    for (const r of results) {
        const meshStr = `${r.nelx}×${r.nely}`.padEnd(10);
        const elStr = String(r.nel).padStart(8);
        const freeStr = String(r.nfree).padStart(9);
        const jsStr = r.jsAvg.toFixed(2).padStart(7);
        const wasmStr = r.wasmAvg.toFixed(2).padStart(9);
        const speedStr = `${r.speedup.toFixed(2)}x`;
        console.log(`${meshStr} | ${elStr} | ${freeStr} | ${jsStr} | ${wasmStr} | ${speedStr}`);
    }

    const avgSpeedup = results.reduce((s, r) => s + r.speedup, 0) / results.length;
    const avgImprovement = results.reduce((s, r) => s + r.improvement, 0) / results.length;
    console.log('─'.repeat(75));
    console.log(`Average speedup: ${avgSpeedup.toFixed(2)}x (${avgImprovement > 0 ? '+' : ''}${avgImprovement.toFixed(1)}%)`);
    console.log('═'.repeat(75));

    if (avgSpeedup > 1) {
        console.log('✓ Self-contained WASM FEA eliminates per-iteration JS↔WASM overhead!');
    } else {
        console.log('Note: JS engine optimizations may outperform WASM for small problems.');
        console.log('WASM FEA benefits increase with larger meshes due to reduced boundary crossings.');
    }
}

main().catch(err => {
    console.error('Benchmark failed:', err);
    process.exit(1);
});
