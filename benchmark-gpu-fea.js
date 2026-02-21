#!/usr/bin/env node
/**
 * Benchmark: GPUFEASolver vs JS Jacobi-PCG — Performance & Conformity
 *
 * Actually runs BOTH solvers side-by-side and compares:
 *   1. JS Jacobi-PCG (f64) — baseline reference
 *   2. GPU Jacobi-PCG (f32, GPUFEASolver) — WebGPU via Dawn in Node.js
 *
 * Measures from 20³ cubes, scaling voxel count by +20% each step until
 * the JS solver exceeds 30s. Includes WASM and MGPCG estimates from
 * measured JS data. Extrapolates to 50³, 70³, and 100³ using measured
 * scaling exponents.
 *
 * Usage:
 *   node benchmark-gpu-fea.js
 *   node benchmark-gpu-fea.js --max-cube=40
 */

import { performance } from 'perf_hooks';
import { GPUFEASolver } from './js/gpu-fea-solver.js';

const EPSILON = 1e-12;
const CG_TOLERANCE = 1e-8;
const MAX_CG_ITERATIONS = 2000;
const E0 = 1;
const Emin = 1e-9;
const NU = 0.3;
const PENAL = 3;
const TIMEOUT_SEC = 30;

// Parse CLI args
const args = process.argv.slice(2);
const maxCubeArg = args.find(a => a.startsWith('--max-cube='));
const MAX_CUBE = maxCubeArg ? parseInt(maxCubeArg.split('=')[1]) : Infinity;

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
                for (let nd = 0; nd < 8; nd++) { dN[nd][0] *= 0.125; dN[nd][1] *= 0.125; dN[nd][2] *= 0.125; }

                const B = Array.from({ length: 6 }, () => new Float64Array(24));
                for (let nd = 0; nd < 8; nd++) {
                    const col = nd * 3;
                    B[0][col] = dN[nd][0];
                    B[1][col + 1] = dN[nd][1];
                    B[2][col + 2] = dN[nd][2];
                    B[3][col] = dN[nd][1]; B[3][col + 1] = dN[nd][0];
                    B[4][col + 1] = dN[nd][2]; B[4][col + 2] = dN[nd][1];
                    B[5][col] = dN[nd][2]; B[5][col + 2] = dN[nd][0];
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

    // Fix left face (all nodes at x=0, all 3 DOFs)
    const fixeddofs = [];
    for (let j = 0; j <= nely; j++) {
        for (let k = 0; k <= nelz; k++) {
            const nd = 0 * nny * nnz + j * nnz + k;
            fixeddofs.push(3 * nd, 3 * nd + 1, 3 * nd + 2);
        }
    }

    const fixedMask = new Uint8Array(ndof);
    for (const d of fixeddofs) if (d >= 0 && d < ndof) fixedMask[d] = 1;

    let nFree = 0;
    for (let i = 0; i < ndof; i++) if (!fixedMask[i]) nFree++;
    const freedofs = new Int32Array(nFree);
    let fp = 0;
    for (let i = 0; i < ndof; i++) if (!fixedMask[i]) freedofs[fp++] = i;

    // Downward force at right-center node
    const F = new Float64Array(ndof);
    const n_rc = nelx * nny * nnz + Math.floor(nely / 2) * nnz + Math.floor(nelz / 2);
    F[3 * n_rc + 1] = -1.0;  // -Y direction

    return { nel, ndof, fixedMask, freedofs, F, fixeddofs };
}

// ═══════════════════════════════════════════════════════════════════════
// JS Jacobi-PCG solver (f64 reference)
// ═══════════════════════════════════════════════════════════════════════
function solveJacobiPCG(KEflat, edofArray, nel, ndof, F, fixedMask, penal) {
    const dE = E0 - Emin;
    const skipThreshold = Emin * 1000;

    // Precompute element stiffnesses (all-solid)
    const E_vals = new Float64Array(nel);
    const activeElements = [];
    for (let e = 0; e < nel; e++) {
        const E = Emin + Math.pow(1.0, penal) * dE;
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
// Solution conformity check (JS f64 vs GPU f32)
// ═══════════════════════════════════════════════════════════════════════
function compareResults(jsU, gpuU, ndof) {
    let maxDiff = 0, sumSqDiff = 0, sumSqRef = 0;
    let maxAbsJS = 0;
    for (let i = 0; i < ndof; i++) {
        const ref = jsU[i];
        const gpu = gpuU[i];
        const d = Math.abs(ref - gpu);
        if (d > maxDiff) maxDiff = d;
        sumSqDiff += d * d;
        sumSqRef += ref * ref;
        if (Math.abs(ref) > maxAbsJS) maxAbsJS = Math.abs(ref);
    }
    const rmsDiff = Math.sqrt(sumSqDiff / ndof);
    const relErr = Math.sqrt(sumSqRef) > 0 ? Math.sqrt(sumSqDiff / sumSqRef) : 0;
    return { maxDiff, rmsDiff, relErr, maxAbsJS };
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════
function nextCubeSize(n) {
    return Math.ceil(n * Math.pow(1.2, 1 / 3));
}

function fmtTime(ms) {
    if (ms < 1000) return ms.toFixed(0) + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(2) + 's';
    return (ms / 60000).toFixed(1) + 'min';
}

function fmtExp(v) { return v.toExponential(2); }

function estimateWASMTime(jsTimeMs) {
    return { totalMs: jsTimeMs / 2.0, speedup: 2.0 };
}

function estimateMGPCGTime(jsJacobiTimeMs, jsJacobiIters, nel) {
    const iterReduction = Math.min(15, 5 + nel / 50000);
    const mgIters = Math.max(5, Math.ceil(jsJacobiIters / iterReduction));
    const jsIterTime = jsJacobiTimeMs / Math.max(jsJacobiIters, 1);
    return { totalMs: jsIterTime * 3.5 * mgIters, estimatedIters: mgIters };
}

// ═══════════════════════════════════════════════════════════════════════
// Main benchmark
// ═══════════════════════════════════════════════════════════════════════
async function main() {
    console.log('');
    console.log('\u2554' + '\u2550'.repeat(92) + '\u2557');
    console.log('\u2551  GPUFEASolver Benchmark \u2014 JS Jacobi-PCG (f64) vs GPU Jacobi-PCG (f32)' + ' '.repeat(20) + '\u2551');
    console.log('\u2551  Cube sizes 20\u00b3\u219232\u00b3 (+20% voxels/step), comparing performance & solution conformity  \u2551');
    console.log('\u255a' + '\u2550'.repeat(92) + '\u255d');
    console.log('');

    // ── Initialize GPU ──
    const gpuSolver = new GPUFEASolver();
    let gpuAvailable = false;
    process.stdout.write('  Initializing WebGPU (Dawn)...');
    try {
        gpuAvailable = await gpuSolver.init();
    } catch (err) {
        console.log(` failed: ${err.message}`);
    }
    if (gpuAvailable) {
        console.log(' OK');
    } else {
        console.log(' not available \u2014 GPU columns will show N/A');
    }

    // ── Element stiffness matrix ──
    const KE = lk3D(NU);
    const KEflat = flattenKE(KE);
    const KEflat32 = new Float32Array(KEflat);

    // Warm up JS engine
    {
        const we = precomputeEdofs3D(5, 5, 5);
        const wp = setupProblem3D(5, 5, 5);
        solveJacobiPCG(KEflat, we, 125, wp.ndof, wp.F, wp.fixedMask, PENAL);
    }

    const results = [];
    let n = 20;
    let gpuTimedOut = false;

    while (n <= MAX_CUBE) {
        const nel = n * n * n;
        const ndof = 3 * (n + 1) * (n + 1) * (n + 1);

        console.log(`\n${'─'.repeat(92)}`);
        console.log(`  Cube: ${n}\u00d7${n}\u00d7${n}  (${nel.toLocaleString()} elements, ${ndof.toLocaleString()} DOFs)`);
        console.log('─'.repeat(92));

        // Setup mesh
        process.stdout.write('  Building mesh...');
        const t0 = performance.now();
        const edofArray = precomputeEdofs3D(n, n, n);
        const prob = setupProblem3D(n, n, n);
        const { fixedMask, freedofs, F } = prob;
        const meshMs = performance.now() - t0;
        console.log(` ${(meshMs / 1000).toFixed(2)}s`);

        // Memory estimate
        const memGB = (edofArray.byteLength + F.byteLength + fixedMask.byteLength +
            ndof * 8 * 5 + nel * 8 * 2) / (1024 ** 3);
        if (memGB > 4.0) {
            console.log(`  \u26a0 Memory limit reached (${memGB.toFixed(1)} GB > 4 GB). Stopping.`);
            break;
        }

        // ── JS Jacobi-PCG (f64) ──
        process.stdout.write('  JS Jacobi-PCG (f64)...');
        const jsStart = performance.now();
        const jsResult = solveJacobiPCG(KEflat, edofArray, nel, ndof, F, fixedMask, PENAL);
        const jsMs = performance.now() - jsStart;
        console.log(` ${fmtTime(jsMs)}  (${jsResult.iters} iters, compliance=${fmtExp(jsResult.compliance)})`);

        // ── GPU Jacobi-PCG (f32) ──
        let gpuMs = null, gpuIters = null, gpuCompliance = null, conformity = null;
        let gpuSetupMs = null, gpuConverged = null;

        if (gpuAvailable && !gpuTimedOut) {
            try {
                // Setup (upload buffers, create pipelines)
                process.stdout.write('  GPU setup...');
                const gsStart = performance.now();
                const densities = new Float32Array(nel);
                densities.fill(1.0);
                const F32 = new Float32Array(F);
                gpuSolver.setup({
                    KEflat: KEflat32, edofArray, densities, fixedMask, F: F32,
                    nel, ndof, E0, Emin, penal: PENAL,
                });
                gpuSetupMs = performance.now() - gsStart;
                console.log(` ${fmtTime(gpuSetupMs)}`);

                // Solve
                process.stdout.write('  GPU Jacobi-PCG (f32)...');
                const gpuStart = performance.now();
                const gpuResult = await gpuSolver.solve({
                    maxIterations: MAX_CG_ITERATIONS,
                    tolerance: CG_TOLERANCE,
                });
                gpuMs = performance.now() - gpuStart;
                gpuIters = gpuResult.iterations;
                gpuConverged = gpuResult.converged;

                // Compute compliance from GPU result
                gpuCompliance = 0;
                for (let i = 0; i < ndof; i++) gpuCompliance += F[i] * gpuResult.U[i];

                console.log(` ${fmtTime(gpuMs)}  (${gpuIters} iters, compliance=${fmtExp(gpuCompliance)}, converged=${gpuConverged})`);

                // Conformity check
                conformity = compareResults(jsResult.U, gpuResult.U, ndof);
                const relPct = (conformity.relErr * 100).toFixed(4);
                const cDiffPct = Math.abs(jsResult.compliance - gpuCompliance) / Math.abs(jsResult.compliance) * 100;
                console.log(`  Conformity: relErr=${fmtExp(conformity.relErr)} (${relPct}%), maxDiff=${fmtExp(conformity.maxDiff)}, rmsDiff=${fmtExp(conformity.rmsDiff)}`);
                console.log(`  Compliance diff: ${cDiffPct.toFixed(4)}%`);

                if (gpuMs > TIMEOUT_SEC * 1000) {
                    gpuTimedOut = true;
                    console.log(`  \u23f1 GPU solve exceeded ${TIMEOUT_SEC}s, skipping GPU for larger sizes.`);
                }
            } catch (err) {
                console.log(` failed: ${err.message}`);
                gpuMs = null;
            }
        } else if (!gpuAvailable) {
            console.log('  GPU: N/A (WebGPU not available)');
        } else {
            console.log('  GPU: skipped (timed out on previous size)');
        }

        // ── Estimates ──
        const wasmEst = estimateWASMTime(jsMs);
        const mgEst = estimateMGPCGTime(jsMs, jsResult.iters, nel);

        // ── Per-size comparison table ──
        const gpuSpeedup = gpuMs ? (jsMs / gpuMs) : null;
        console.log('');
        console.log('  \u250c' + '\u2500'.repeat(80) + '\u2510');
        console.log('  \u2502  Solver               \u2502 Time     \u2502 CG Iters  \u2502 vs JS Jacobi      \u2502 Status    \u2502');
        console.log('  \u251c' + '\u2500'.repeat(80) + '\u2524');
        console.log(`  \u2502 JS Jacobi-PCG (f64)   \u2502 ${fmtTime(jsMs).padStart(8)} \u2502 ${String(jsResult.iters).padStart(9)} \u2502 ${'1.00x (baseline)'.padStart(18)} \u2502 measured  \u2502`);
        if (gpuMs !== null) {
            const spStr = gpuSpeedup >= 1 ? `${gpuSpeedup.toFixed(2)}x faster` : `${(1 / gpuSpeedup).toFixed(2)}x slower`;
            console.log(`  \u2502 GPU Jacobi-PCG (f32)  \u2502 ${fmtTime(gpuMs).padStart(8)} \u2502 ${String(gpuIters).padStart(9)} \u2502 ${spStr.padStart(18)} \u2502 measured  \u2502`);
        } else {
            console.log(`  \u2502 GPU Jacobi-PCG (f32)  \u2502 ${'N/A'.padStart(8)} \u2502 ${'N/A'.padStart(9)} \u2502 ${'N/A'.padStart(18)} \u2502 ${'N/A'.padStart(9)} \u2502`);
        }
        console.log(`  \u2502 WASM Jacobi-PCG    ~  \u2502 ${fmtTime(wasmEst.totalMs).padStart(8)} \u2502 ${String(jsResult.iters).padStart(9)} \u2502 ${(wasmEst.speedup.toFixed(2) + 'x faster').padStart(18)} \u2502 estimated \u2502`);
        const mgSpeedup = jsMs / mgEst.totalMs;
        console.log(`  \u2502 JS MGPCG            ~ \u2502 ${fmtTime(mgEst.totalMs).padStart(8)} \u2502 ${String(mgEst.estimatedIters).padStart(9)} \u2502 ${(mgSpeedup.toFixed(2) + 'x faster').padStart(18)} \u2502 estimated \u2502`);
        console.log('  \u2514' + '\u2500'.repeat(80) + '\u2518');
        console.log('    ~ = estimated');

        results.push({
            n, nel, ndof,
            jsMs, jsIters: jsResult.iters, jsCompliance: jsResult.compliance,
            gpuMs, gpuIters, gpuCompliance, gpuSetupMs, gpuConverged,
            conformity,
            wasmMs: wasmEst.totalMs,
            mgMs: mgEst.totalMs, mgIters: mgEst.estimatedIters,
        });

        if (jsMs > TIMEOUT_SEC * 1000) {
            console.log(`\n  \u23f1 JS Jacobi-PCG exceeded ${TIMEOUT_SEC}s. Stopping measurement loop.`);
            break;
        }

        n = nextCubeSize(n);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Summary: Performance
    // ═════════════════════════════════════════════════════════════════════
    console.log('\n\n' + '\u2550'.repeat(132));
    console.log('  PERFORMANCE SCALING SUMMARY');
    console.log('\u2550'.repeat(132));
    console.log(
        'Cube       \u2502 Elements      \u2502 DOFs          \u2502 JS Jacobi     \u2502 GPU Jacobi    \u2502 GPU Speedup \u2502 WASM~         \u2502 MGPCG~'
    );
    console.log('\u2500'.repeat(132));

    for (const r of results) {
        const cube = `${r.n}\u00b3`.padEnd(10);
        const elems = r.nel.toLocaleString().padStart(13);
        const dofs = r.ndof.toLocaleString().padStart(13);
        const js = fmtTime(r.jsMs).padStart(13);
        const gpu = r.gpuMs !== null ? fmtTime(r.gpuMs).padStart(13) : 'N/A'.padStart(13);
        let sp;
        if (r.gpuMs !== null) {
            const ratio = r.jsMs / r.gpuMs;
            sp = ratio >= 1
                ? (ratio.toFixed(2) + 'x faster').padStart(11)
                : ((1 / ratio).toFixed(2) + 'x slower').padStart(11);
        } else {
            sp = 'N/A'.padStart(11);
        }
        const wasm = fmtTime(r.wasmMs).padStart(13);
        const mg = fmtTime(r.mgMs).padStart(13);
        console.log(`${cube} \u2502 ${elems} \u2502 ${dofs} \u2502 ${js} \u2502 ${gpu} \u2502 ${sp} \u2502 ${wasm} \u2502 ${mg}`);
    }
    console.log('\u2550'.repeat(132));

    // ═════════════════════════════════════════════════════════════════════
    // Summary: Conformity
    // ═════════════════════════════════════════════════════════════════════
    const conformityResults = results.filter(r => r.conformity);
    if (conformityResults.length > 0) {
        console.log('\n' + '\u2550'.repeat(115));
        console.log('  SOLUTION CONFORMITY: JS (f64) vs GPU (f32)');
        console.log('\u2550'.repeat(115));
        console.log(
            'Cube       \u2502 JS Compliance     \u2502 GPU Compliance    \u2502 Comp. Diff %  \u2502 Rel. Error    \u2502 Max |diff|    \u2502 RMS diff'
        );
        console.log('\u2500'.repeat(115));

        for (const r of conformityResults) {
            const cube = `${r.n}\u00b3`.padEnd(10);
            const jsc = fmtExp(r.jsCompliance).padStart(17);
            const gc = fmtExp(r.gpuCompliance).padStart(17);
            const cDiff = (Math.abs(r.jsCompliance - r.gpuCompliance) / Math.abs(r.jsCompliance) * 100).toFixed(4);
            const re = fmtExp(r.conformity.relErr).padStart(13);
            const md = fmtExp(r.conformity.maxDiff).padStart(13);
            const rd = fmtExp(r.conformity.rmsDiff).padStart(8);
            console.log(`${cube} \u2502 ${jsc} \u2502 ${gc} \u2502 ${cDiff.padStart(12)}% \u2502 ${re} \u2502 ${md} \u2502 ${rd}`);
        }
        console.log('\u2550'.repeat(115));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Summary: Scaling analysis & extrapolation
    // ═════════════════════════════════════════════════════════════════════
    if (results.length >= 2) {
        console.log('\n  SCALING ANALYSIS:');
        const first = results[0], last = results[results.length - 1];
        const volRatio = last.nel / first.nel;
        const jsRatio = last.jsMs / first.jsMs;
        const alpha = Math.log(jsRatio) / Math.log(volRatio);
        console.log(`    Volume scaling: ${volRatio.toFixed(2)}x (${first.nel.toLocaleString()} \u2192 ${last.nel.toLocaleString()} elements)`);
        console.log(`    JS Jacobi-PCG time scaling: ${jsRatio.toFixed(2)}x`);
        console.log(`    Scaling exponent (time ~ N^\u03b1): \u03b1 = ${alpha.toFixed(2)}`);

        const gpuResults = results.filter(r => r.gpuMs !== null);
        if (gpuResults.length >= 2) {
            const gFirst = gpuResults[0], gLast = gpuResults[gpuResults.length - 1];
            const gRatio = gLast.gpuMs / gFirst.gpuMs;
            const gAlpha = Math.log(gRatio) / Math.log(gLast.nel / gFirst.nel);
            console.log(`    GPU Jacobi-PCG time scaling: ${gRatio.toFixed(2)}x, \u03b1 = ${gAlpha.toFixed(2)}`);
        }

        console.log('');
        console.log('    Expected scaling behaviors:');
        console.log('      \u2022 JS/WASM Jacobi:  O(N^1.33)  \u2014 EbE matvec O(N) \u00d7 CG iters O(N^0.33)');
        console.log('      \u2022 MGPCG:           O(N)       \u2014 optimal MG: O(N) per V-cycle, O(1) iters');
        console.log('      \u2022 GPU Jacobi:      O(N^0.33)  \u2014 once GPU saturates, parallel EbE is O(1), CG iters O(N^0.33)');

        // Extrapolation
        const extraSizes = [40, 50, 60, 70, 80, 100].filter(s => s > last.n);
        if (extraSizes.length > 0) {
            const lastGPURatio = gpuResults.length > 0
                ? gpuResults[gpuResults.length - 1].gpuMs / gpuResults[gpuResults.length - 1].jsMs
                : null;

            console.log('\n  EXTRAPOLATED ESTIMATES (from measured JS \u03b1=' + alpha.toFixed(2) + '):');
            console.log('  ' + '\u2500'.repeat(100));
            console.log('  Cube       \u2502 Elements      \u2502 JS Jacobi~    \u2502 WASM~         \u2502 MGPCG~        \u2502 GPU~');
            console.log('  ' + '\u2500'.repeat(100));

            for (const en of extraSizes) {
                const eNel = en ** 3;
                const jsExtMs = last.jsMs * Math.pow(eNel / last.nel, alpha);
                const wEst = estimateWASMTime(jsExtMs);
                const mEst = estimateMGPCGTime(jsExtMs, MAX_CG_ITERATIONS, eNel);
                const gpuExt = lastGPURatio !== null ? fmtTime(jsExtMs * lastGPURatio) : 'N/A';
                console.log(`  ${(en + '\u00b3').padEnd(10)} \u2502 ${eNel.toLocaleString().padStart(13)} \u2502 ${fmtTime(jsExtMs).padStart(13)} \u2502 ${fmtTime(wEst.totalMs).padStart(13)} \u2502 ${fmtTime(mEst.totalMs).padStart(13)} \u2502 ${gpuExt.padStart(13)}`);
            }
            console.log('  ' + '\u2500'.repeat(100));
            console.log('    All ~ values are estimates based on measured scaling trends');
        }
    }

    // Notes
    console.log('\n  NOTES:');
    if (gpuAvailable) {
        console.log('    \u2022 GPU solver ran via Dawn (webgpu npm package) in Node.js');
        console.log('    \u2022 GPU uses f32 arithmetic, JS uses f64 \u2014 expect ~0.01-1% relative error');
        console.log('    \u2022 GPU applyA: 2-pass (local EbE mat-vec \u2192 CAS atomic scatter-add)');
        console.log('    \u2022 Only dot-product partials read back per CG iteration (~1 KB)');
        console.log('    \u2022 Final solution read back once after convergence');
    } else {
        console.log('    \u2022 GPU solver unavailable \u2014 install "webgpu" package for GPU comparison');
    }
    console.log('    \u2022 WASM and MGPCG columns are estimates from measured JS timing');
    console.log('    \u2022 All solves use uniform density=1.0 (cantilever beam, left-face fixed)');
    console.log('');

    // Cleanup
    gpuSolver.destroy();
}

main().catch(err => {
    console.error('Benchmark failed:', err);
    process.exit(1);
});
