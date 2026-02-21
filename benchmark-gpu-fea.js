#!/usr/bin/env node
/**
 * Benchmark: GPUFEASolver Scaling Analysis
 *
 * Since WebGPU is not available in Node.js, this benchmark:
 *   1. Runs the JS Jacobi-PCG solver (same algorithm as GPUFEASolver) to
 *      measure actual per-solve timings across scaling cube sizes
 *   2. Uses measured data to project GPU solver performance via conservative
 *      speedup estimates derived from GPU compute characteristics
 *   3. Provides estimated comparisons with MGPCG and WASM solvers
 *
 * Starts at 50×50×50, scales voxel count by +20% each step until the JS
 * Jacobi-PCG solver exceeds 30 s per solve.
 *
 * Usage:
 *   node benchmark-gpu-fea.js
 */

import { performance } from 'perf_hooks';

const EPSILON = 1e-12;
const CG_TOLERANCE = 1e-8;
const MAX_CG_ITERATIONS = 2000;
const E0 = 1;
const Emin = 1e-9;
const NU = 0.3;
const PENAL = 3;
const TIMEOUT_SEC = 30;

// ═══════════════════════════════════════════════════════════════════════
// 3D element stiffness (8-node hex, 2×2×2 Gauss)
// ═══════════════════════════════════════════════════════════════════════
function lk3D(nu) {
    const fact = 1.0 / ((1 + nu) * (1 - 2 * nu));
    const C = Array.from({ length: 6 }, () => new Float64Array(6));
    C[0][0] = fact * (1 - nu); C[0][1] = fact * nu;       C[0][2] = fact * nu;
    C[1][0] = fact * nu;       C[1][1] = fact * (1 - nu); C[1][2] = fact * nu;
    C[2][0] = fact * nu;       C[2][1] = fact * nu;       C[2][2] = fact * (1 - nu);
    C[3][3] = fact * (1 - 2 * nu) / 2;
    C[4][4] = fact * (1 - 2 * nu) / 2;
    C[5][5] = fact * (1 - 2 * nu) / 2;

    const KE = Array.from({ length: 24 }, () => new Float64Array(24));
    const gp = [-1 / Math.sqrt(3), 1 / Math.sqrt(3)];

    for (let gi = 0; gi < 2; gi++) {
        for (let gj = 0; gj < 2; gj++) {
            for (let gk = 0; gk < 2; gk++) {
                const xi = gp[gi], eta = gp[gj], zeta = gp[gk];
                const dN = [
                    [-(1 - eta) * (1 - zeta), -(1 - xi) * (1 - zeta), -(1 - xi) * (1 - eta)],
                    [(1 - eta) * (1 - zeta),  -(1 + xi) * (1 - zeta), -(1 + xi) * (1 - eta)],
                    [(1 + eta) * (1 - zeta),   (1 + xi) * (1 - zeta), -(1 + xi) * (1 + eta)],
                    [-(1 + eta) * (1 - zeta),  (1 - xi) * (1 - zeta), -(1 - xi) * (1 + eta)],
                    [-(1 - eta) * (1 + zeta), -(1 - xi) * (1 + zeta),  (1 - xi) * (1 - eta)],
                    [(1 - eta) * (1 + zeta),  -(1 + xi) * (1 + zeta),  (1 + xi) * (1 - eta)],
                    [(1 + eta) * (1 + zeta),   (1 + xi) * (1 + zeta),  (1 + xi) * (1 + eta)],
                    [-(1 + eta) * (1 + zeta),  (1 - xi) * (1 + zeta),  (1 - xi) * (1 + eta)]
                ];
                for (let n = 0; n < 8; n++) { dN[n][0] *= 0.125; dN[n][1] *= 0.125; dN[n][2] *= 0.125; }

                const B = Array.from({ length: 6 }, () => new Float64Array(24));
                for (let n = 0; n < 8; n++) {
                    const col = n * 3;
                    B[0][col] = dN[n][0];
                    B[1][col + 1] = dN[n][1];
                    B[2][col + 2] = dN[n][2];
                    B[3][col] = dN[n][1]; B[3][col + 1] = dN[n][0];
                    B[4][col + 1] = dN[n][2]; B[4][col + 2] = dN[n][1];
                    B[5][col] = dN[n][2]; B[5][col + 2] = dN[n][0];
                }

                for (let i = 0; i < 24; i++) {
                    for (let j = 0; j < 24; j++) {
                        let sum = 0;
                        for (let s = 0; s < 6; s++) {
                            let cb = 0;
                            for (let t = 0; t < 6; t++) cb += C[s][t] * B[t][j];
                            sum += B[s][i] * cb;
                        }
                        KE[i][j] += sum;
                    }
                }
            }
        }
    }

    return KE;
}

function flattenKE(KE) {
    const n = KE.length;
    const flat = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            flat[i * n + j] = KE[i][j];
    return flat;
}

// ═══════════════════════════════════════════════════════════════════════
// Mesh setup
// ═══════════════════════════════════════════════════════════════════════
function precomputeEdofs3D(nelx, nely, nelz) {
    const nel = nelx * nely * nelz;
    const nny = nely + 1, nnz = nelz + 1;
    const edofArray = new Int32Array(nel * 24);
    for (let elz = 0; elz < nelz; elz++) {
        for (let ely = 0; ely < nely; ely++) {
            for (let elx = 0; elx < nelx; elx++) {
                const idx = elx + ely * nelx + elz * nelx * nely;
                const off = idx * 24;
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
                    edofArray[off + ni * 3] = 3 * nodes[ni];
                    edofArray[off + ni * 3 + 1] = 3 * nodes[ni] + 1;
                    edofArray[off + ni * 3 + 2] = 3 * nodes[ni] + 2;
                }
            }
        }
    }
    return edofArray;
}

function setupProblem3D(nelx, nely, nelz) {
    const nel = nelx * nely * nelz;
    const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
    const nny = nely + 1, nnz = nelz + 1;

    // Fix bottom corners (4 corner nodes, all 3 DOFs each)
    const fixeddofs = [];
    const corners = [
        0 * nny * nnz + 0 * nnz + 0,
        nelx * nny * nnz + 0 * nnz + 0,
        0 * nny * nnz + nely * nnz + 0,
        nelx * nny * nnz + nely * nnz + 0
    ];
    for (const n of corners) fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);

    const fixedMask = new Uint8Array(ndof);
    for (const d of fixeddofs) fixedMask[d] = 1;

    let nFree = 0;
    for (let i = 0; i < ndof; i++) if (!fixedMask[i]) nFree++;
    const freedofs = new Int32Array(nFree);
    let fp = 0;
    for (let i = 0; i < ndof; i++) if (!fixedMask[i]) freedofs[fp++] = i;

    // Downward force at top center
    const F = new Float64Array(ndof);
    const n_tc = Math.floor(nelx / 2) * nny * nnz + Math.floor(nely / 2) * nnz + nelz;
    F[3 * n_tc + 1] = -1.0;

    return { nel, ndof, fixedMask, freedofs, F };
}

// ═══════════════════════════════════════════════════════════════════════
// Jacobi-PCG solver (same algorithm as GPUFEASolver, but in JS/f64)
// ═══════════════════════════════════════════════════════════════════════
function solveJacobiPCG(KEflat, edofArray, nel, ndof, F, fixedMask, penal) {
    const dE = E0 - Emin;
    const skipThreshold = Emin * 1000;

    // Precompute element stiffnesses
    const E_vals = new Float64Array(nel);
    const activeElements = [];
    for (let e = 0; e < nel; e++) {
        const E = Emin + Math.pow(1.0, penal) * dE; // density = 1.0 (all solid)
        E_vals[e] = E;
        if (E > skipThreshold) activeElements.push(e);
    }

    // Diagonal preconditioner
    const diag = new Float64Array(ndof);
    for (let ae = 0; ae < activeElements.length; ae++) {
        const e = activeElements[ae], E = E_vals[e], eOff = e * 24;
        for (let i = 0; i < 24; i++) diag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
    }
    const invDiag = new Float64Array(ndof);
    for (let i = 0; i < ndof; i++) {
        if (!fixedMask[i] && diag[i] > 1e-30) invDiag[i] = 1.0 / diag[i];
    }

    // EbE matvec
    const loc = new Float64Array(24);
    function fullSpaceMatVec(p, Ap) {
        Ap.fill(0);
        for (let ae = 0; ae < activeElements.length; ae++) {
            const e = activeElements[ae], E = E_vals[e], eOff = e * 24;
            for (let j = 0; j < 24; j++) loc[j] = p[edofArray[eOff + j]];
            for (let i = 0; i < 24; i++) {
                const gi = edofArray[eOff + i];
                let sum = 0;
                const keRow = i * 24;
                for (let j = 0; j < 24; j++) sum += KEflat[keRow + j] * loc[j];
                Ap[gi] += E * sum;
            }
        }
    }

    const U = new Float64Array(ndof);
    const r = new Float64Array(ndof);
    const z = new Float64Array(ndof);
    const p = new Float64Array(ndof);
    const Ap = new Float64Array(ndof);

    // r = F (with fixed DOFs zeroed)
    for (let i = 0; i < ndof; i++) r[i] = fixedMask[i] ? 0 : F[i];

    // z = M^{-1} r, p = z
    let rz = 0;
    for (let i = 0; i < ndof; i++) {
        z[i] = invDiag[i] * r[i];
        p[i] = z[i];
        rz += r[i] * z[i];
    }

    let r0n2 = 0;
    for (let i = 0; i < ndof; i++) r0n2 += r[i] * r[i];
    const tolSq = CG_TOLERANCE * CG_TOLERANCE * Math.max(r0n2, 1e-30);

    let iters = 0;
    for (let iter = 0; iter < MAX_CG_ITERATIONS; iter++) {
        let rn2 = 0;
        for (let i = 0; i < ndof; i++) rn2 += r[i] * r[i];
        if (rn2 < tolSq) break;
        iters++;

        fullSpaceMatVec(p, Ap);
        for (let i = 0; i < ndof; i++) if (fixedMask[i]) Ap[i] = 0;

        let pAp = 0;
        for (let i = 0; i < ndof; i++) pAp += p[i] * Ap[i];
        const alpha = rz / (pAp + EPSILON);

        let rz_new = 0;
        for (let i = 0; i < ndof; i++) {
            U[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            z[i] = invDiag[i] * r[i];
            rz_new += r[i] * z[i];
        }
        const beta = rz_new / (rz + EPSILON);
        for (let i = 0; i < ndof; i++) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    let compliance = 0;
    for (let i = 0; i < ndof; i++) compliance += F[i] * U[i];

    return { U, compliance, iters };
}

// ═══════════════════════════════════════════════════════════════════════
// Scale cube dimensions by +20% voxel count each step
// voxels = n^3, so n_new = n * (1.2)^(1/3) ≈ n * 1.0627
// ═══════════════════════════════════════════════════════════════════════
function nextCubeSize(n) {
    // Scale voxel count by 1.2x → side by 1.2^(1/3) ≈ 1.0627
    return Math.ceil(n * Math.pow(1.2, 1 / 3));
}

// ═══════════════════════════════════════════════════════════════════════
// Estimate GPU solver performance
// ═══════════════════════════════════════════════════════════════════════
// These estimates are based on typical GPU vs CPU speedup ratios for
// each CG operation. The dominant cost is the EbE mat-vec (applyA).
//
// GPU advantages:
//   - applyA: Each element's 24x24 mat-vec runs in 1 workgroup (24 threads),
//     all nel elements execute in parallel → ~10-20x speedup over sequential CPU
//     for large nel (limited by atomic scatter contention)
//   - dotProduct / axpy / Jacobi: bandwidth-bound vector ops,
//     GPU has ~5-10x memory bandwidth advantage
//   - No per-iteration CPU readback (only dot-product partials, ~1KB)
//
// GPU disadvantages:
//   - f32 precision (vs f64 CPU) may require ~20% more CG iterations
//   - CAS-based atomic scatter adds contention at shared DOFs
//   - Per-iteration mapAsync latency for dot product readback (~0.1-0.5ms)
//   - Setup overhead (buffer upload, pipeline creation)
//
// Conservative estimate: GPU provides 8-15x speedup on applyA,
// but with iteration overhead and f32 penalty, net ~5-10x overall.

function estimateGPUTime(jsTimeMs, nel, ndof, cgIters) {
    // applyA dominates: ~85% of CG iteration time for large problems
    // Vector ops (dot, axpy, jacobi): ~15% of time
    const applyAFraction = 0.85;
    const vectorFraction = 0.15;

    // GPU speedup factors (conservative for mid-range GPU)
    const applyASpeedup = Math.min(15, 3 + nel / 10000); // scales with parallelism
    const vectorSpeedup = Math.min(8, 2 + ndof / 50000);

    // f32 precision penalty: ~20% more iterations
    const f32IterPenalty = 1.2;

    // Per-iteration GPU overhead: mapAsync latency for dot products
    // 3 dot products per iteration × ~0.2ms each
    const perIterOverheadMs = 3 * 0.2;

    const jsIterTime = jsTimeMs / Math.max(cgIters, 1);
    const gpuIterTime = jsIterTime * (
        applyAFraction / applyASpeedup +
        vectorFraction / vectorSpeedup
    ) + perIterOverheadMs;

    const gpuIters = Math.ceil(cgIters * f32IterPenalty);
    const gpuSetupMs = 5 + nel * 0.001; // buffer upload + pipeline creation

    return {
        totalMs: gpuSetupMs + gpuIterTime * gpuIters,
        iterTimeMs: gpuIterTime,
        estimatedIters: gpuIters,
        applyASpeedup,
        vectorSpeedup,
    };
}

// Estimate MGPCG time from Jacobi-PCG measurements.
// MGPCG uses multigrid V-cycle preconditioning which dramatically reduces
// CG iterations (typically 5-15x fewer), but each iteration costs more
// due to restriction/prolongation/smoothing across grid levels.
function estimateMGPCGTime(jsJacobiTimeMs, jsJacobiIters, nel) {
    // MGPCG typically needs 5-15x fewer iterations than Jacobi-PCG
    const iterReduction = Math.min(15, 5 + nel / 50000);
    const mgIters = Math.max(5, Math.ceil(jsJacobiIters / iterReduction));

    // But each MGPCG iteration is ~3-5x more expensive (V-cycle cost)
    const perIterCostMultiplier = 3.5;

    const jsIterTime = jsJacobiTimeMs / Math.max(jsJacobiIters, 1);
    const mgIterTime = jsIterTime * perIterCostMultiplier;
    const mgTotalMs = mgIterTime * mgIters;

    return { totalMs: mgTotalMs, iterTimeMs: mgIterTime, estimatedIters: mgIters };
}

// Estimate WASM Jacobi-PCG time.
// Based on measured WASM speedup ratios from benchmark-fea.js results.
// WASM eliminates per-iteration JS↔WASM overhead and benefits from
// SIMD and optimized memory access patterns.
function estimateWASMTime(jsTimeMs) {
    // WASM Jacobi-PCG is typically 1.5-3x faster than JS for 3D problems
    const wasmSpeedup = 2.0;
    return { totalMs: jsTimeMs / wasmSpeedup, speedup: wasmSpeedup };
}

// ═══════════════════════════════════════════════════════════════════════
// Main benchmark
// ═══════════════════════════════════════════════════════════════════════
console.log('');
console.log('╔' + '═'.repeat(88) + '╗');
console.log('║  GPUFEASolver Scaling Benchmark — Jacobi-PCG Scaling & GPU Performance Estimate    ║');
console.log('║  Starting at 20³, scaling voxel count by +20% until >30s per solve                 ║');
console.log('╚' + '═'.repeat(88) + '╝');
console.log('');

const KE = lk3D(NU);
const KEflat = flattenKE(KE);

const results = [];

// Warm up the JS engine with a small solve
{
    const warmEdof = precomputeEdofs3D(5, 5, 5);
    const warmProb = setupProblem3D(5, 5, 5);
    solveJacobiPCG(KEflat, warmEdof, 125, warmProb.ndof, warmProb.F, warmProb.fixedMask, PENAL);
}

let n = 20; // starting cube side length

while (true) {
    const nel = n * n * n;
    const ndof = 3 * (n + 1) * (n + 1) * (n + 1);

    console.log(`\n${'─'.repeat(88)}`);
    console.log(`  Cube: ${n}×${n}×${n}  (${nel.toLocaleString()} elements, ${ndof.toLocaleString()} DOFs)`);
    console.log('─'.repeat(88));

    // Setup mesh
    process.stdout.write('  Setting up mesh...');
    const t0 = performance.now();
    const edofArray = precomputeEdofs3D(n, n, n);
    const { fixedMask, freedofs, F } = setupProblem3D(n, n, n);
    const setupMs = performance.now() - t0;
    console.log(` done (${(setupMs / 1000).toFixed(2)}s)`);

    // Check memory constraints
    const memGB = (
        edofArray.byteLength + F.byteLength + fixedMask.byteLength +
        ndof * 8 * 5 + // U, r, z, p, Ap
        nel * 8 * 2 // E_vals, loc
    ) / (1024 * 1024 * 1024);
    console.log(`  Memory estimate: ${memGB.toFixed(2)} GB`);

    if (memGB > 4.0) {
        console.log('  ⚠ Memory limit reached (>4 GB). Stopping.');
        break;
    }

    // Solve
    process.stdout.write('  Solving Jacobi-PCG...');
    const solveStart = performance.now();
    const result = solveJacobiPCG(KEflat, edofArray, nel, ndof, F, fixedMask, PENAL);
    const solveMs = performance.now() - solveStart;
    console.log(` done (${(solveMs / 1000).toFixed(2)}s, ${result.iters} CG iterations)`);

    // Estimate GPU performance
    const gpuEst = estimateGPUTime(solveMs, nel, ndof, result.iters);
    const mgEst = estimateMGPCGTime(solveMs, result.iters, nel);
    const wasmEst = estimateWASMTime(solveMs);

    const gpuSpeedup = solveMs / gpuEst.totalMs;

    console.log(`  Compliance: ${result.compliance.toExponential(6)}`);
    console.log('');
    console.log('  ┌─────────────────────────────────────────────────────────────────┐');
    console.log('  │  Solver Comparison (measured JS Jacobi + estimates)             │');
    console.log('  ├─────────────────────┬──────────┬───────────┬────────────────────┤');
    console.log('  │ Solver              │ Time     │ CG Iters  │ vs JS Jacobi       │');
    console.log('  ├─────────────────────┼──────────┼───────────┼────────────────────┤');
    console.log(`  │ JS Jacobi-PCG       │ ${fmtTime(solveMs).padStart(8)} │ ${String(result.iters).padStart(9)} │ ${'1.00x (baseline)'.padStart(18)} │`);
    console.log(`  │ WASM Jacobi-PCG  ~  │ ${fmtTime(wasmEst.totalMs).padStart(8)} │ ${String(result.iters).padStart(9)} │ ${(wasmEst.speedup.toFixed(2) + 'x faster').padStart(18)} │`);
    console.log(`  │ JS MGPCG         ~  │ ${fmtTime(mgEst.totalMs).padStart(8)} │ ${String(mgEst.estimatedIters).padStart(9)} │ ${((solveMs / mgEst.totalMs).toFixed(2) + 'x faster').padStart(18)} │`);
    console.log(`  │ GPU Jacobi-PCG   ~  │ ${fmtTime(gpuEst.totalMs).padStart(8)} │ ${String(gpuEst.estimatedIters).padStart(9)} │ ${(gpuSpeedup.toFixed(2) + 'x faster').padStart(18)} │`);
    console.log('  └─────────────────────┴──────────┴───────────┴────────────────────┘');
    console.log(`    ~ = estimated  │  GPU applyA speedup: ${gpuEst.applyASpeedup.toFixed(1)}x  │  GPU vec ops: ${gpuEst.vectorSpeedup.toFixed(1)}x`);

    results.push({
        n, nel, ndof,
        jsMs: solveMs,
        jsIters: result.iters,
        gpuMs: gpuEst.totalMs,
        gpuIters: gpuEst.estimatedIters,
        gpuSpeedup,
        mgMs: mgEst.totalMs,
        mgIters: mgEst.estimatedIters,
        wasmMs: wasmEst.totalMs,
    });

    if (solveMs > TIMEOUT_SEC * 1000) {
        console.log(`\n  ⏱ JS Jacobi-PCG exceeded ${TIMEOUT_SEC}s timeout. Stopping.`);
        break;
    }

    n = nextCubeSize(n);
}

// ═══════════════════════════════════════════════════════════════════════
// Summary table
// ═══════════════════════════════════════════════════════════════════════
console.log('\n\n' + '═'.repeat(120));
console.log('  SCALING SUMMARY');
console.log('═'.repeat(120));
console.log(
    'Cube       │ Elements      │ DOFs          │ JS Jacobi     │ WASM Jacobi~  │ MGPCG~        │ GPU Jacobi~   │ GPU Speedup'
);
console.log('─'.repeat(120));

for (const r of results) {
    const cube = `${r.n}³`.padEnd(10);
    const elems = r.nel.toLocaleString().padStart(13);
    const dofs = r.ndof.toLocaleString().padStart(13);
    const js = fmtTime(r.jsMs).padStart(13);
    const wasm = fmtTime(r.wasmMs).padStart(13);
    const mg = fmtTime(r.mgMs).padStart(13);
    const gpu = fmtTime(r.gpuMs).padStart(13);
    const speedup = (r.gpuSpeedup.toFixed(2) + 'x').padStart(11);
    console.log(`${cube} │ ${elems} │ ${dofs} │ ${js} │ ${wasm} │ ${mg} │ ${gpu} │ ${speedup}`);
}

console.log('═'.repeat(120));

// Scaling analysis
if (results.length >= 2) {
    console.log('\n  SCALING ANALYSIS:');
    const first = results[0];
    const last = results[results.length - 1];
    const volumeRatio = last.nel / first.nel;
    const jsTimeRatio = last.jsMs / first.jsMs;

    const alpha = Math.log(jsTimeRatio) / Math.log(volumeRatio);
    console.log(`    Volume scaling: ${volumeRatio.toFixed(2)}x (${first.nel.toLocaleString()} → ${last.nel.toLocaleString()} elements)`);
    console.log(`    JS Jacobi-PCG time scaling: ${jsTimeRatio.toFixed(2)}x (${fmtTime(first.jsMs)} → ${fmtTime(last.jsMs)})`);
    console.log(`    Scaling exponent (time ~ N^α): α = ${alpha.toFixed(2)}`);
    console.log('');
    console.log('    Expected scaling behaviors:');
    console.log('      • JS/WASM Jacobi-PCG:  O(N^1.33) — EbE matvec O(N) × CG iters O(N^0.33)');
    console.log('      • MGPCG:               O(N)      — optimal multigrid: O(N) per V-cycle, O(1) iters');
    console.log('      • GPU Jacobi-PCG:      O(N^0.33) — parallel EbE matvec O(1) × CG iters O(N^0.33)');
    console.log('                             (until GPU saturates, then O(N^1.33) with smaller constant)');

    // Extrapolated estimates for 50³ and beyond
    console.log('\n  EXTRAPOLATED ESTIMATES (from measured scaling):');
    console.log('  ┌────────────────────────────────────────────────────────────────────────────────────┐');
    console.log('  │ Cube       │ Elements      │ JS Jacobi~    │ WASM~         │ MGPCG~        │ GPU~ │');
    console.log('  ├────────────┼───────────────┼───────────────┼───────────────┼───────────────┼──────┤');

    const extraSizes = [40, 50, 60, 70, 80, 100];
    for (const en of extraSizes) {
        const eNel = en * en * en;
        const eNdof = 3 * (en + 1) * (en + 1) * (en + 1);
        // Extrapolate JS time using measured scaling exponent
        const jsExtMs = last.jsMs * Math.pow(eNel / last.nel, alpha);
        const eGpu = estimateGPUTime(jsExtMs, eNel, eNdof, MAX_CG_ITERATIONS);
        const eMg = estimateMGPCGTime(jsExtMs, MAX_CG_ITERATIONS, eNel);
        const eWasm = estimateWASMTime(jsExtMs);
        const cubeStr = `${en}³`.padEnd(10);
        const nelStr = eNel.toLocaleString().padStart(13);
        console.log(`  │ ${cubeStr} │ ${nelStr} │ ${fmtTime(jsExtMs).padStart(13)} │ ${fmtTime(eWasm.totalMs).padStart(13)} │ ${fmtTime(eMg.totalMs).padStart(13)} │ ${fmtTime(eGpu.totalMs).padStart(4)} │`);
    }
    console.log('  └────────────┴───────────────┴───────────────┴───────────────┴───────────────┴──────┘');
    console.log('    All values marked ~ are estimates based on measured scaling trends');
}

// GPU-specific notes
console.log('\n  NOTES ON GPU ESTIMATES:');
console.log('    • GPU estimates assume a mid-range discrete GPU (e.g. RTX 3060 / RX 6700)');
console.log('    • applyA speedup: each element runs as 1 workgroup (24 threads), all elements parallel');
console.log('    • Scatter-add uses CAS-based atomic f32 add (contention at shared DOFs)');
console.log('    • f32 precision may require ~20% more CG iterations than f64');
console.log('    • Only dot-product partials are read back per iteration (~1 KB)');
console.log('    • Actual GPU performance varies significantly with hardware and driver');
console.log('    • To get real GPU timings, run benchmark-gpu.html in a WebGPU-capable browser');
console.log('');

function fmtTime(ms) {
    if (ms < 1000) return ms.toFixed(0) + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(2) + 's';
    return (ms / 60000).toFixed(1) + 'min';
}
