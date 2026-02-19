#!/usr/bin/env node
/**
 * 3D Topology Optimization Benchmark
 *
 * Default problem: 50×50×50 mm cube, 5 mm voxel → 10×10×10 elements
 *   - Target material: 10 %
 *   - Max iterations: 30
 *   - Penalty factor: 20
 *   - Filter radius: 0.9
 *   - AMR: on  (min granule 0.02, max granule 2)
 *
 * After optimisation the script:
 *   1. Verifies actual material percentage vs target
 *   2. Runs a post-optimisation FEA on the final density field
 *   3. Reports per-element Von-Mises-equivalent stress
 *      (highest stress and top-5 % average)
 *   4. Cross-compares multiple solver configurations
 *
 * NOTE: For a high-level benchmark that uses the TopologySolver library API
 * (works in both Node.js and the browser), import TopologySolver instead:
 *
 *   import { TopologySolver } from './lib/topology-solver.js';
 *   const solver = new TopologySolver();
 *   const result = await solver.optimize(model, { solver: 'mgpcg', maxIterations: 30, ... }, onProgress);
 *
 * This benchmark runs the solver engine directly for detailed profiling.
 */

import { performance } from 'perf_hooks';

// ═══════════════════════════════════════════════════════════════════════
// Constants – mirror optimizer-worker-3d.js
// ═══════════════════════════════════════════════════════════════════════
const EPSILON = 1e-12;
const CG_TOL_START = 1e-3;
const CG_TOL_END = 1e-8;
const MAX_CG_ITERATIONS_MGPCG = 1000;
const MAX_CG_ITERATIONS_JACOBI = 2000;

const MG_MAX_LEVELS = 6;
const MG_NU1 = 2;
const MG_NU2 = 2;
const MG_OMEGA = 0.5;       // must be < 2/ρ(D⁻¹A) ≈ 0.645 for 3D hex ν=0.3
const MG_COARSE_ITERS = 30;
const GALERKIN_MAX_NDOF = 3000;  // Use dense Galerkin P^T A P for coarse levels with ndof ≤ this

const E0 = 1;
const Emin = 1e-9;
const NU = 0.3;
const DENSITY_THRESHOLD = 0.3;

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════
function _powDensity(x, p) {
    if (!Number.isFinite(p) || !Number.isInteger(p)) return Math.pow(x, p);
    switch (p) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return x * x;
        case 3: return x * x * x;
        default: return Math.pow(x, p);
    }
}

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
                    [(1 - eta) * (1 - zeta), -(1 + xi) * (1 - zeta), -(1 + xi) * (1 - eta)],
                    [(1 + eta) * (1 - zeta), (1 + xi) * (1 - zeta), -(1 + xi) * (1 + eta)],
                    [-(1 + eta) * (1 - zeta), (1 - xi) * (1 - zeta), -(1 - xi) * (1 + eta)],
                    [-(1 - eta) * (1 + zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - eta)],
                    [(1 - eta) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - eta)],
                    [(1 + eta) * (1 + zeta), (1 + xi) * (1 + zeta), (1 + xi) * (1 + eta)],
                    [-(1 + eta) * (1 + zeta), (1 - xi) * (1 + zeta), (1 - xi) * (1 + eta)]
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

                const CB = Array.from({ length: 6 }, () => new Float64Array(24));
                for (let m = 0; m < 6; m++)
                    for (let n = 0; n < 24; n++)
                        for (let p = 0; p < 6; p++)
                            CB[m][n] += C[m][p] * B[p][n];

                for (let m = 0; m < 24; m++)
                    for (let n = 0; n < 24; n++)
                        for (let p = 0; p < 6; p++)
                            KE[m][n] += B[p][m] * CB[p][n]; // detJ=1, weight=1
            }
        }
    }
    return KE;
}

function flattenKE(KE) {
    const flat = new Float64Array(24 * 24);
    for (let i = 0; i < 24; i++)
        for (let j = 0; j < 24; j++)
            flat[i * 24 + j] = KE[i][j];
    return flat;
}

// ═══════════════════════════════════════════════════════════════════════
// Edof precomputation
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

// ═══════════════════════════════════════════════════════════════════════
// Problem setup
// ═══════════════════════════════════════════════════════════════════════
function setupProblem3D(nelx, nely, nelz) {
    const nel = nelx * nely * nelz;
    const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
    const nny = nely + 1, nnz = nelz + 1;

    // Fix bottom-corners (4 corner nodes, all 3 DOFs each)
    const fixeddofs = [];
    const corners = [
        0 * nny * nnz + 0 * nnz + 0,
        nelx * nny * nnz + 0 * nnz + 0,
        0 * nny * nnz + nely * nnz + 0,
        nelx * nny * nnz + nely * nnz + 0
    ];
    for (const n of corners) {
        fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);
    }

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

    return { nel, ndof, fixeddofs, fixedMask, freedofs, F };
}

// ═══════════════════════════════════════════════════════════════════════
// Precompute stiffnesses
// ═══════════════════════════════════════════════════════════════════════
function precomputeStiffness(x, penal, nel) {
    const E_vals = new Float64Array(nel);
    const dE = E0 - Emin;
    const activeElements = [];
    const skipThreshold = Emin * 1000;
    for (let e = 0; e < nel; e++) {
        const E = Emin + _powDensity(x[e], penal) * dE;
        E_vals[e] = E;
        if (E > skipThreshold) activeElements.push(e);
    }
    return { E_vals, activeElements };
}

// ═══════════════════════════════════════════════════════════════════════
// Full-space EbE matvec
// ═══════════════════════════════════════════════════════════════════════
function fullSpaceMatVec(E_vals, activeElements, KEflat, edofArray, p, Ap, ndof) {
    Ap.fill(0);
    const loc = new Float64Array(24);
    for (let ae = 0; ae < activeElements.length; ae++) {
        const e = activeElements[ae];
        const E = E_vals[e];
        const eOff = e * 24;
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

// ═══════════════════════════════════════════════════════════════════════
// MG Preconditioner with Galerkin coarse operators
// ═══════════════════════════════════════════════════════════════════════
class MGPrecond3D {
    constructor(KEflat, wasmMod) {
        this.KEflat = KEflat; this.levels = []; this._tmpL = new Float64Array(24);
        this._wasmMod = wasmMod || null;
        this._wasmReady = false;
        this._wasmKeOff = 0;
        this._wasmScratchOff = 0;
    }

    static _downsampleFixedMaskBy2(fixedFine, nxF0, nyF0, nzF0, nxC0, nyC0, nzC0) {
        const nxF = nxF0 + 1, nyF = nyF0 + 1, nzF = nzF0 + 1;
        const nxC = nxC0 + 1, nyC = nyC0 + 1, nzC = nzC0 + 1;
        const fixedC = new Uint8Array(3 * nxC * nyC * nzC);
        for (let cz = 0; cz < nzC; cz++) {
            const fz = Math.min(cz * 2, nzF - 1);
            for (let cy = 0; cy < nyC; cy++) {
                const fy = Math.min(cy * 2, nyF - 1);
                for (let cx = 0; cx < nxC; cx++) {
                    const fx = Math.min(cx * 2, nxF - 1);
                    const nF = fx * nyF * nzF + fy * nzF + fz;
                    const nC = cx * nyC * nzC + cy * nzC + cz;
                    fixedC[3 * nC] = fixedFine[3 * nF];
                    fixedC[3 * nC + 1] = fixedFine[3 * nF + 1];
                    fixedC[3 * nC + 2] = fixedFine[3 * nF + 2];
                }
            }
        }
        return fixedC;
    }

    ensure(nelx, nely, nelz, edofArrayFine, fixedMaskFine, freeDofsFine) {
        if (this.levels.length > 0) {
            const L0 = this.levels[0];
            if (L0.nelx === nelx && L0.nely === nely && L0.nelz === nelz) return;
        }
        this.levels = [];
        this.levels.push(this._makeLevel(nelx, nely, nelz, edofArrayFine, fixedMaskFine, freeDofsFine));
        let cx = nelx, cy = nely, cz = nelz, fm = fixedMaskFine;
        for (let li = 1; li < MG_MAX_LEVELS; li++) {
            const nx = Math.floor(cx / 2), ny = Math.floor(cy / 2), nz = Math.floor(cz / 2);
            if (nx < 2 || ny < 2 || nz < 2) break;
            const edof = precomputeEdofs3D(nx, ny, nz);
            const fmC = MGPrecond3D._downsampleFixedMaskBy2(fm, cx, cy, cz, nx, ny, nz);
            let nFree = 0;
            const ndofC = 3 * (nx + 1) * (ny + 1) * (nz + 1);
            for (let i = 0; i < ndofC; i++) if (!fmC[i]) nFree++;
            const freeC = new Int32Array(nFree);
            let p = 0;
            for (let i = 0; i < ndofC; i++) if (!fmC[i]) freeC[p++] = i;
            this.levels.push(this._makeLevel(nx, ny, nz, edof, fmC, freeC));
            cx = nx; cy = ny; cz = nz; fm = fmC;
        }
        this._initWasmBuffers();
    }

    _initWasmBuffers() {
        const wasm = this._wasmMod;
        if (!wasm || !wasm.exports.applyAEbe3D) { this._wasmReady = false; return; }
        try {
            const mem = wasm.exports.memory;
            const align8 = (v) => (v + 7) & ~7;
            let totalBytes = 576 * 8 + 24 * 8;
            for (const level of this.levels) {
                totalBytes += level.nel * 24 * 4 + 8 + level.nel * 8 + level.nel * 4 + 8 + level.ndof * 8 + level.ndof * 8;
            }
            const neededPages = Math.ceil(totalBytes / 65536) + 1;
            const dataStart = mem.buffer.byteLength;
            if (mem.grow(neededPages) === -1) { this._wasmReady = false; return; }
            let offset = dataStart;
            this._wasmKeOff = offset; offset += 576 * 8;
            new Float64Array(mem.buffer, this._wasmKeOff, 576).set(this.KEflat);
            this._wasmScratchOff = offset; offset += 24 * 8;
            for (const level of this.levels) {
                const w = {};
                w.edofsOff = offset; offset += level.nel * 24 * 4;
                offset = align8(offset);
                w.evalsOff = offset; offset += level.nel * 8;
                w.activeOff = offset; offset += level.nel * 4;
                offset = align8(offset);
                w.pOff = offset; offset += level.ndof * 8;
                w.apOff = offset; offset += level.ndof * 8;
                new Int32Array(mem.buffer, w.edofsOff, level.edofArray.length).set(level.edofArray);
                level._wasm = w;
            }
            this._wasmReady = true;
        } catch (e) { this._wasmReady = false; }
    }

    _wasmSyncLevel(level) {
        if (!this._wasmReady || !level._wasm) return;
        const mem = this._wasmMod.exports.memory;
        const w = level._wasm;
        new Float64Array(mem.buffer, w.evalsOff, level.nel).set(level.E_vals);
        new Int32Array(mem.buffer, w.activeOff, level.activeCount).set(level.active.subarray(0, level.activeCount));
    }

    _makeLevel(nelx, nely, nelz, edofArray, fixedMask, freeDofs) {
        const nel = nelx * nely * nelz, ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        return { nelx, nely, nelz, nel, ndof, edofArray, fixedMask, freeDofs,
            dens: new Float32Array(nel), E_vals: new Float64Array(nel),
            active: new Int32Array(nel), activeCount: 0,
            diag: new Float64Array(ndof), invDiag: new Float64Array(ndof),
            Au: new Float64Array(ndof), res: new Float64Array(ndof),
            b: new Float64Array(ndof), x: new Float64Array(ndof),
            denseK: null };  // dense Galerkin matrix (if assembled)
    }

    // ─── Prolongation map: fine node → (coarse nodes, weights) ───
    _buildProlongMap(fine, coarse) {
        const nxF = fine.nelx + 1, nyF = fine.nely + 1, nzF = fine.nelz + 1;
        const nxC = coarse.nelx + 1, nyC = coarse.nely + 1, nzC = coarse.nelz + 1;
        const nnodesF = nxF * nyF * nzF;
        const pNodes = new Int32Array(nnodesF * 8);
        const pWeights = new Float64Array(nnodesF * 8);
        const pCount = new Uint8Array(nnodesF);
        for (let fz = 0; fz < nzF; fz++) {
            const cz0 = Math.min(fz >> 1, nzC - 1), cz1 = Math.min(cz0 + 1, nzC - 1);
            const tz = (cz0 !== cz1 && (fz & 1)) ? 0.5 : 0.0;
            const czs0 = cz0, czs1 = cz1, wz0 = 1 - tz, wz1 = tz;
            for (let fy = 0; fy < nyF; fy++) {
                const cy0 = Math.min(fy >> 1, nyC - 1), cy1 = Math.min(cy0 + 1, nyC - 1);
                const ty = (cy0 !== cy1 && (fy & 1)) ? 0.5 : 0.0;
                const cys0 = cy0, cys1 = cy1, wy0 = 1 - ty, wy1 = ty;
                for (let fx = 0; fx < nxF; fx++) {
                    const cx0 = Math.min(fx >> 1, nxC - 1), cx1 = Math.min(cx0 + 1, nxC - 1);
                    const tx = (cx0 !== cx1 && (fx & 1)) ? 0.5 : 0.0;
                    const cxs0 = cx0, cxs1 = cx1, wx0 = 1 - tx, wx1 = tx;
                    const nF = fx * nyF * nzF + fy * nzF + fz;
                    const off = nF * 8;
                    let cnt = 0;
                    const cxArr = [cxs0, cxs1], wxArr = [wx0, wx1];
                    const cyArr = [cys0, cys1], wyArr = [wy0, wy1];
                    const czArr = [czs0, czs1], wzArr = [wz0, wz1];
                    for (let dz = 0; dz < 2; dz++)
                        for (let dy = 0; dy < 2; dy++)
                            for (let dx = 0; dx < 2; dx++) {
                                const w = wxArr[dx] * wyArr[dy] * wzArr[dz];
                                if (w > 0) {
                                    pNodes[off + cnt] = cxArr[dx] * nyC * nzC + cyArr[dy] * nzC + czArr[dz];
                                    pWeights[off + cnt] = w;
                                    cnt++;
                                }
                            }
                    pCount[nF] = cnt;
                }
            }
        }
        return { pNodes, pWeights, pCount };
    }

    // ─── Assemble dense Galerkin matrix from fine-level elements: K_c = P^T K_f P ───
    _assembleGalerkinFromElements(li) {
        const fine = this.levels[li - 1], coarse = this.levels[li];
        const ndofC = coarse.ndof;
        const K = new Float64Array(ndofC * ndofC);
        const pMap = this._buildProlongMap(fine, coarse);
        const { pNodes, pWeights, pCount } = pMap;
        const KEflat = this.KEflat, edof = fine.edofArray;

        for (let ai = 0; ai < fine.activeCount; ai++) {
            const e = fine.active[ai], E = fine.E_vals[e], eO = e * 24;
            for (let i = 0; i < 24; i++) {
                const fDofI = edof[eO + i];
                const fNodeI = (fDofI / 3) | 0, dI = fDofI % 3;
                const offI = fNodeI * 8, cntI = pCount[fNodeI];
                for (let j = i; j < 24; j++) {
                    const kij = E * KEflat[i * 24 + j];
                    if (kij === 0) continue;
                    const fDofJ = edof[eO + j];
                    const fNodeJ = (fDofJ / 3) | 0, dJ = fDofJ % 3;
                    const offJ = fNodeJ * 8, cntJ = pCount[fNodeJ];
                    for (let pi = 0; pi < cntI; pi++) {
                        const cr = 3 * pNodes[offI + pi] + dI;
                        const wi = pWeights[offI + pi];
                        for (let pj = 0; pj < cntJ; pj++) {
                            const cc = 3 * pNodes[offJ + pj] + dJ;
                            const val = wi * kij * pWeights[offJ + pj];
                            K[cr * ndofC + cc] += val;
                            if (i !== j) K[cc * ndofC + cr] += val;
                        }
                    }
                }
            }
        }
        coarse.denseK = K;
        this._diagFromDense(coarse);
    }

    // ─── Assemble dense Galerkin from parent's dense matrix: K_c = P^T K_p P ───
    _assembleGalerkinFromMatrix(li) {
        const fine = this.levels[li - 1], coarse = this.levels[li];
        const ndofF = fine.ndof, ndofC = coarse.ndof;
        const pMap = this._buildProlongMap(fine, coarse);
        const { pNodes, pWeights, pCount } = pMap;
        const nnodesF = (fine.nelx + 1) * (fine.nely + 1) * (fine.nelz + 1);

        // Build dense prolongation P (ndofF × ndofC)
        const P = new Float64Array(ndofF * ndofC);
        for (let nF = 0; nF < nnodesF; nF++) {
            const off = nF * 8, cnt = pCount[nF];
            for (let pi = 0; pi < cnt; pi++) {
                const nC = pNodes[off + pi], w = pWeights[off + pi];
                for (let d = 0; d < 3; d++) P[(3 * nF + d) * ndofC + (3 * nC + d)] = w;
            }
        }
        const Kf = fine.denseK;
        // Temp = K_f * P  (ndofF × ndofC)
        const Temp = new Float64Array(ndofF * ndofC);
        for (let i = 0; i < ndofF; i++) {
            const rowF = i * ndofF, rowT = i * ndofC;
            for (let j = 0; j < ndofC; j++) {
                let s = 0;
                for (let k = 0; k < ndofF; k++) s += Kf[rowF + k] * P[k * ndofC + j];
                Temp[rowT + j] = s;
            }
        }
        // K_c = P^T * Temp  (ndofC × ndofC)
        const K = new Float64Array(ndofC * ndofC);
        for (let i = 0; i < ndofC; i++) {
            for (let j = 0; j < ndofC; j++) {
                let s = 0;
                for (let k = 0; k < ndofF; k++) s += P[k * ndofC + i] * Temp[k * ndofC + j];
                K[i * ndofC + j] = s;
            }
        }
        coarse.denseK = K;
        this._diagFromDense(coarse);
    }

    // ─── E-val averaging fallback for large levels ───
    _restrictEvalAvg(fine, coarse) {
        const nxF = fine.nelx, nyF = fine.nely, nzF = fine.nelz;
        const nxC = coarse.nelx, nyC = coarse.nely, nzC = coarse.nelz;
        const EF = fine.E_vals, EC = coarse.E_vals;
        const skipT = Emin * 1000;
        let ac = 0, idxC = 0;
        for (let cz = 0; cz < nzC; cz++) {
            const fz0 = cz * 2;
            for (let cy = 0; cy < nyC; cy++) {
                const fy0 = cy * 2;
                for (let cx = 0; cx < nxC; cx++, idxC++) {
                    const fx0 = cx * 2;
                    let sum = 0, count = 0;
                    for (let dz = 0; dz < 2; dz++) { const fz = fz0 + dz; if (fz >= nzF) continue;
                        for (let dy = 0; dy < 2; dy++) { const fy = fy0 + dy; if (fy >= nyF) continue;
                            for (let dx = 0; dx < 2; dx++) { const fx = fx0 + dx; if (fx >= nxF) continue;
                                sum += EF[fx + fy * nxF + fz * nxF * nyF]; count++;
                    }}}
                    const E = count > 0 ? (sum / count) * 2.0 : 0;
                    EC[idxC] = E;
                    if (E > skipT) coarse.active[ac++] = idxC;
                }
            }
        }
        coarse.activeCount = ac;
        coarse.diag.fill(0);
        const KEflat = this.KEflat, edof = coarse.edofArray;
        for (let ai = 0; ai < ac; ai++) {
            const e = coarse.active[ai], E = EC[e], eO = e * 24;
            for (let i = 0; i < 24; i++) coarse.diag[edof[eO + i]] += E * KEflat[i * 24 + i];
        }
        coarse.invDiag.fill(0);
        for (let i = 0; i < coarse.freeDofs.length; i++) {
            const d = coarse.freeDofs[i]; const v = coarse.diag[d];
            coarse.invDiag[d] = v > 1e-30 ? 1.0 / v : 0.0;
        }
        coarse.denseK = null;
    }

    // ─── Extract diagonal from dense K ───
    _diagFromDense(level) {
        const K = level.denseK, n = level.ndof;
        level.diag.fill(0);
        for (let i = 0; i < n; i++) level.diag[i] = K[i * n + i];
        level.invDiag.fill(0);
        for (let ii = 0; ii < level.freeDofs.length; ii++) {
            const d = level.freeDofs[ii];
            level.invDiag[d] = level.diag[d] > 1e-30 ? 1.0 / level.diag[d] : 0;
        }
    }

    // ─── Update operators from fine density field ───
    updateFromFine(xFine, penal) {
        if (this.levels.length === 0) return;
        // Level 0: element-by-element E_vals
        this.levels[0].dens.set(xFine);
        this._updateOp(this.levels[0], penal, 0);
        this._wasmSyncLevel(this.levels[0]);
        // Coarse levels: Galerkin or E-val-avg fallback
        for (let li = 1; li < this.levels.length; li++) {
            const coarse = this.levels[li];
            if (coarse.ndof <= GALERKIN_MAX_NDOF) {
                const parentHasDense = this.levels[li - 1].denseK;
                if (parentHasDense) {
                    this._assembleGalerkinFromMatrix(li);
                } else {
                    this._assembleGalerkinFromElements(li);
                }
            } else {
                // Fallback for large levels: E-val averaging
                this._restrictEvalAvg(this.levels[li - 1], this.levels[li]);
            }
            this._wasmSyncLevel(coarse);
        }
    }

    _updateOp(level, penal, levelIdx) {
        const hScale = 1 << levelIdx;
        const dE = E0 - Emin, skipT = Emin * 1000;
        let ac = 0;
        for (let e = 0; e < level.nel; e++) {
            const E = (Emin + _powDensity(level.dens[e], penal) * dE) * hScale;
            level.E_vals[e] = E;
            if (E > skipT) level.active[ac++] = e;
        }
        level.activeCount = ac;
        level.diag.fill(0);
        const KEflat = this.KEflat, edof = level.edofArray;
        for (let ai = 0; ai < ac; ai++) {
            const e = level.active[ai], E = level.E_vals[e], eO = e * 24;
            for (let i = 0; i < 24; i++) level.diag[edof[eO + i]] += E * KEflat[i * 24 + i];
        }
        level.invDiag.fill(0);
        for (let i = 0; i < level.freeDofs.length; i++) {
            const d = level.freeDofs[i]; const v = level.diag[d];
            level.invDiag[d] = v > 1e-30 ? 1.0 / v : 0.0;
        }
    }

    // ─── Matrix-vector product: dense or element-by-element ───
    applyA(li, p, Ap) {
        const level = this.levels[li];
        if (level.denseK) {
            // Dense Galerkin matvec
            const K = level.denseK, n = level.ndof;
            Ap.fill(0);
            for (let i = 0; i < n; i++) {
                let s = 0; const off = i * n;
                for (let j = 0; j < n; j++) s += K[off + j] * p[j];
                Ap[i] = s;
            }
            return;
        }
        // WASM-accelerated element-by-element
        if (this._wasmReady && level._wasm) {
            const mem = this._wasmMod.exports.memory;
            const w = level._wasm;
            new Float64Array(mem.buffer, w.pOff, level.ndof).set(p.subarray(0, level.ndof));
            this._wasmMod.exports.applyAEbe3D(
                this._wasmKeOff, w.edofsOff, w.evalsOff, w.activeOff, level.activeCount,
                w.pOff, w.apOff, level.ndof, this._wasmScratchOff
            );
            Ap.set(new Float64Array(mem.buffer, w.apOff, level.ndof));
            return;
        }
        // Fallback: JS element-by-element
        Ap.fill(0);
        const KEflat = this.KEflat, edof = level.edofArray;
        const loc = this._tmpL;
        for (let ai = 0; ai < level.activeCount; ai++) {
            const e = level.active[ai], E = level.E_vals[e], eO = e * 24;
            for (let j = 0; j < 24; j++) loc[j] = p[edof[eO + j]];
            for (let i = 0; i < 24; i++) {
                let sum = 0; const row = i * 24;
                for (let j = 0; j < 24; j++) sum += KEflat[row + j] * loc[j];
                Ap[edof[eO + i]] += E * sum;
            }
        }
    }

    vcycle(li, b, x) {
        const level = this.levels[li], Au = level.Au, invD = level.invDiag, free = level.freeDofs;
        // Pre-smooth
        for (let s = 0; s < MG_NU1; s++) {
            this.applyA(li, x, Au);
            for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; x[d] += MG_OMEGA * invD[d] * (b[d] - Au[d]); }
        }
        // Compute residual
        this.applyA(li, x, Au);
        const res = level.res; res.fill(0);
        for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; res[d] = b[d] - Au[d]; }
        // Coarsest: just smooth more
        if (li === this.levels.length - 1) {
            for (let s = 0; s < MG_COARSE_ITERS; s++) {
                this.applyA(li, x, Au);
                for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; x[d] += MG_OMEGA * invD[d] * (b[d] - Au[d]); }
            }
            return;
        }
        // Restrict residual (full-weighting = P^T)
        const coarse = this.levels[li + 1]; const bC = coarse.b; bC.fill(0);
        const nxF = level.nelx + 1, nyF = level.nely + 1, nzF = level.nelz + 1;
        const nxC = coarse.nelx + 1, nyC = coarse.nely + 1, nzC = coarse.nelz + 1;
        for (let fz = 0; fz < nzF; fz++) {
            const cz0 = Math.min(fz >> 1, nzC - 1), cz1 = Math.min(cz0 + 1, nzC - 1);
            const tz = (cz0 !== cz1 && (fz & 1)) ? 0.5 : 0.0;
            const wz0 = 1 - tz, wz1 = tz;
            for (let fy = 0; fy < nyF; fy++) {
                const cy0 = Math.min(fy >> 1, nyC - 1), cy1 = Math.min(cy0 + 1, nyC - 1);
                const ty = (cy0 !== cy1 && (fy & 1)) ? 0.5 : 0.0;
                const wy0 = 1 - ty, wy1 = ty;
                for (let fx = 0; fx < nxF; fx++) {
                    const cx0 = Math.min(fx >> 1, nxC - 1), cx1 = Math.min(cx0 + 1, nxC - 1);
                    const tx = (cx0 !== cx1 && (fx & 1)) ? 0.5 : 0.0;
                    const wx0 = 1 - tx, wx1 = tx;
                    const nF = fx * nyF * nzF + fy * nzF + fz;
                    const r0 = res[3*nF], r1 = res[3*nF+1], r2 = res[3*nF+2];
                    const w000 = wx0*wy0*wz0; if (w000) { const c3 = 3*(cx0*nyC*nzC+cy0*nzC+cz0); bC[c3]+=w000*r0; bC[c3+1]+=w000*r1; bC[c3+2]+=w000*r2; }
                    const w100 = wx1*wy0*wz0; if (w100) { const c3 = 3*(cx1*nyC*nzC+cy0*nzC+cz0); bC[c3]+=w100*r0; bC[c3+1]+=w100*r1; bC[c3+2]+=w100*r2; }
                    const w010 = wx0*wy1*wz0; if (w010) { const c3 = 3*(cx0*nyC*nzC+cy1*nzC+cz0); bC[c3]+=w010*r0; bC[c3+1]+=w010*r1; bC[c3+2]+=w010*r2; }
                    const w110 = wx1*wy1*wz0; if (w110) { const c3 = 3*(cx1*nyC*nzC+cy1*nzC+cz0); bC[c3]+=w110*r0; bC[c3+1]+=w110*r1; bC[c3+2]+=w110*r2; }
                    const w001 = wx0*wy0*wz1; if (w001) { const c3 = 3*(cx0*nyC*nzC+cy0*nzC+cz1); bC[c3]+=w001*r0; bC[c3+1]+=w001*r1; bC[c3+2]+=w001*r2; }
                    const w101 = wx1*wy0*wz1; if (w101) { const c3 = 3*(cx1*nyC*nzC+cy0*nzC+cz1); bC[c3]+=w101*r0; bC[c3+1]+=w101*r1; bC[c3+2]+=w101*r2; }
                    const w011 = wx0*wy1*wz1; if (w011) { const c3 = 3*(cx0*nyC*nzC+cy1*nzC+cz1); bC[c3]+=w011*r0; bC[c3+1]+=w011*r1; bC[c3+2]+=w011*r2; }
                    const w111 = wx1*wy1*wz1; if (w111) { const c3 = 3*(cx1*nyC*nzC+cy1*nzC+cz1); bC[c3]+=w111*r0; bC[c3+1]+=w111*r1; bC[c3+2]+=w111*r2; }
                }
            }
        }
        const fixedC = coarse.fixedMask;
        for (let i = 0; i < coarse.ndof; i++) if (fixedC[i]) bC[i] = 0;
        // Solve on coarse
        const xC = coarse.x; xC.fill(0);
        this.vcycle(li + 1, bC, xC);
        // Prolongate and add
        const fixedF = level.fixedMask;
        for (let fz = 0; fz < nzF; fz++) {
            const cz0 = Math.min(fz >> 1, nzC - 1), cz1 = Math.min(cz0 + 1, nzC - 1);
            const tz = (cz0 !== cz1 && (fz & 1)) ? 0.5 : 0.0; const wz0 = 1 - tz, wz1 = tz;
            for (let fy = 0; fy < nyF; fy++) {
                const cy0 = Math.min(fy >> 1, nyC - 1), cy1 = Math.min(cy0 + 1, nyC - 1);
                const ty = (cy0 !== cy1 && (fy & 1)) ? 0.5 : 0.0; const wy0 = 1 - ty, wy1 = ty;
                for (let fx = 0; fx < nxF; fx++) {
                    const cx0 = Math.min(fx >> 1, nxC - 1), cx1 = Math.min(cx0 + 1, nxC - 1);
                    const tx = (cx0 !== cx1 && (fx & 1)) ? 0.5 : 0.0; const wx0 = 1 - tx, wx1 = tx;
                    const nF = fx * nyF * nzF + fy * nzF + fz;
                    const n000 = cx0 * nyC * nzC + cy0 * nzC + cz0, n100 = cx1 * nyC * nzC + cy0 * nzC + cz0;
                    const n010 = cx0 * nyC * nzC + cy1 * nzC + cz0, n110 = cx1 * nyC * nzC + cy1 * nzC + cz0;
                    const n001 = cx0 * nyC * nzC + cy0 * nzC + cz1, n101 = cx1 * nyC * nzC + cy0 * nzC + cz1;
                    const n011 = cx0 * nyC * nzC + cy1 * nzC + cz1, n111 = cx1 * nyC * nzC + cy1 * nzC + cz1;
                    const w000 = wx0 * wy0 * wz0, w100 = wx1 * wy0 * wz0, w010 = wx0 * wy1 * wz0, w110 = wx1 * wy1 * wz0;
                    const w001 = wx0 * wy0 * wz1, w101 = wx1 * wy0 * wz1, w011 = wx0 * wy1 * wz1, w111 = wx1 * wy1 * wz1;
                    const bF = 3 * nF;
                    for (let d = 0; d < 3; d++) {
                        if (!fixedF[bF + d]) {
                            x[bF + d] += xC[3 * n000 + d] * w000 + xC[3 * n100 + d] * w100 + xC[3 * n010 + d] * w010 + xC[3 * n110 + d] * w110
                                        + xC[3 * n001 + d] * w001 + xC[3 * n101 + d] * w101 + xC[3 * n011 + d] * w011 + xC[3 * n111 + d] * w111;
                        }
                    }
                }
            }
        }
        // Post-smooth
        for (let s = 0; s < MG_NU2; s++) {
            this.applyA(li, x, Au);
            for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; x[d] += MG_OMEGA * invD[d] * (b[d] - Au[d]); }
        }
    }

    apply(r, z) { z.fill(0); this.vcycle(0, r, z); }
}

// ═══════════════════════════════════════════════════════════════════════
// Jacobi-PCG solver (full-space)
// ═══════════════════════════════════════════════════════════════════════
function solveJacobiPCG(KEflat, edofArray, x, penal, nel, ndof, F, freedofs, fixedMask, tolerance, maxIter) {
    const { E_vals, activeElements } = precomputeStiffness(x, penal, nel);

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

    const U = new Float64Array(ndof);
    const r = new Float64Array(ndof);
    const z = new Float64Array(ndof);
    const p = new Float64Array(ndof);
    const Ap = new Float64Array(ndof);

    for (let i = 0; i < ndof; i++) r[i] = fixedMask[i] ? 0 : F[i];
    let rz = 0;
    for (let i = 0; i < ndof; i++) { z[i] = invDiag[i] * r[i]; p[i] = z[i]; rz += r[i] * z[i]; }

    let r0n2 = 0;
    for (let i = 0; i < ndof; i++) r0n2 += r[i] * r[i];
    const tolSq = tolerance * tolerance * Math.max(r0n2, 1e-30);

    let iters = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        let rn2 = 0;
        for (let i = 0; i < ndof; i++) rn2 += r[i] * r[i];
        if (rn2 < tolSq) break;
        iters++;

        fullSpaceMatVec(E_vals, activeElements, KEflat, edofArray, p, Ap, ndof);
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
    return { U, iters };
}

// ═══════════════════════════════════════════════════════════════════════
// MGPCG solver (full-space)
// ═══════════════════════════════════════════════════════════════════════
function solveMGPCG(KEflat, edofArray, x, penal, nelx, nely, nelz, ndof, F, freedofs, fixedMask, tolerance, maxIter, mg) {
    mg.ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs);
    mg.updateFromFine(x, penal);

    const U = new Float64Array(ndof);
    const r = new Float64Array(ndof);
    const z = new Float64Array(ndof);
    const p = new Float64Array(ndof);
    const Ap = new Float64Array(ndof);

    for (let i = 0; i < ndof; i++) r[i] = fixedMask[i] ? 0 : F[i];
    mg.apply(r, z);
    for (let i = 0; i < ndof; i++) if (fixedMask[i]) z[i] = 0;
    p.set(z);

    let rz = 0;
    for (let i = 0; i < ndof; i++) rz += r[i] * z[i];

    let r0n2 = 0;
    for (let i = 0; i < ndof; i++) r0n2 += r[i] * r[i];
    const tolSq = tolerance * tolerance * Math.max(r0n2, 1e-30);

    let iters = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        let rn2 = 0;
        for (let i = 0; i < ndof; i++) rn2 += r[i] * r[i];
        if (rn2 < tolSq) break;
        iters++;

        mg.applyA(0, p, Ap);
        for (let i = 0; i < ndof; i++) if (fixedMask[i]) Ap[i] = 0;

        let pAp = 0;
        for (let i = 0; i < ndof; i++) pAp += p[i] * Ap[i];
        const alpha = rz / (pAp + EPSILON);

        for (let i = 0; i < ndof; i++) { U[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }

        mg.apply(r, z);
        for (let i = 0; i < ndof; i++) if (fixedMask[i]) z[i] = 0;

        let rz_new = 0;
        for (let i = 0; i < ndof; i++) rz_new += r[i] * z[i];
        const beta = rz_new / (rz + EPSILON);
        for (let i = 0; i < ndof; i++) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }
    return { U, iters };
}

// ═══════════════════════════════════════════════════════════════════════
// Filter
// ═══════════════════════════════════════════════════════════════════════
function prepareFilter3D(nelx, nely, nelz, rmin) {
    const iH = [], jH = [], sH = [];
    let k = 0;
    const rFloor = Math.floor(rmin);
    for (let i = 0; i < nelx; i++) for (let j = 0; j < nely; j++) for (let m = 0; m < nelz; m++) {
        const e1 = i + j * nelx + m * nelx * nely;
        for (let ki = Math.max(i - rFloor, 0); ki <= Math.min(i + rFloor, nelx - 1); ki++)
            for (let l = Math.max(j - rFloor, 0); l <= Math.min(j + rFloor, nely - 1); l++)
                for (let n = Math.max(m - rFloor, 0); n <= Math.min(m + rFloor, nelz - 1); n++) {
                    const e2 = ki + l * nelx + n * nelx * nely;
                    const dist = Math.sqrt((i - ki) ** 2 + (j - l) ** 2 + (m - n) ** 2);
                    if (dist <= rmin) { iH[k] = e1; jH[k] = e2; sH[k] = Math.max(0, rmin - dist); k++; }
                }
    }
    const nel = nelx * nely * nelz;
    const Hs = new Float32Array(nel);
    for (let i = 0; i < k; i++) Hs[iH[i]] += sH[i];
    return { iH, jH, sH, Hs, k };
}

function filterSens(dc, x, H, nel) {
    const dcn = new Float32Array(nel);
    for (let i = 0; i < H.k; i++) dcn[H.iH[i]] += H.sH[i] * x[H.jH[i]] * dc[H.jH[i]];
    for (let i = 0; i < nel; i++) dcn[i] /= (H.Hs[i] * Math.max(1e-3, x[i]));
    return dcn;
}

// ═══════════════════════════════════════════════════════════════════════
// OC update
// ═══════════════════════════════════════════════════════════════════════
function OC(nel, x, volfrac, dc, xnew) {
    const move = 0.2;
    let l1 = 0, l2 = 1e9;
    while ((l2 - l1) / (l2 + l1) > 1e-3) {
        const lmid = 0.5 * (l2 + l1);
        let sumX = 0;
        for (let i = 0; i < nel; i++) {
            const Be = Math.max(0, -dc[i] / lmid);
            const candidate = x[i] * Math.sqrt(Be);
            const val = Math.max(1e-9, Math.max(x[i] - move, Math.min(1, Math.min(x[i] + move, isNaN(candidate) ? 1e-9 : candidate))));
            xnew[i] = val;
            sumX += val;
        }
        if (sumX > volfrac * nel) l1 = lmid; else l2 = lmid;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Post-FEA stress analysis
// ═══════════════════════════════════════════════════════════════════════
function computeElementStresses(nelx, nely, nelz, x, U, KEflat, edofArray, penal) {
    const nel = nelx * nely * nelz;
    const stresses = new Float64Array(nel);
    const Ue = new Float64Array(24);

    for (let e = 0; e < nel; e++) {
        if (x[e] < DENSITY_THRESHOLD) continue;
        const eOff = e * 24;
        for (let i = 0; i < 24; i++) Ue[i] = U[edofArray[eOff + i]];

        // Strain energy density  u^T K_e u  –  Von-Mises-proportional for SIMP compliance
        let energy = 0;
        for (let i = 0; i < 24; i++) {
            const row = i * 24;
            for (let j = 0; j < 24; j++) energy += Ue[i] * KEflat[row + j] * Ue[j];
        }
        const Eval = Emin + _powDensity(x[e], penal) * (E0 - Emin);
        stresses[e] = Eval * energy;
    }
    return stresses;
}

function analyzeStress(stresses, x) {
    const solid = [];
    for (let i = 0; i < stresses.length; i++) {
        if (x[i] >= DENSITY_THRESHOLD) solid.push(stresses[i]);
    }
    if (solid.length === 0) return { maxStress: 0, top5PctAvg: 0, solidCount: 0 };

    solid.sort((a, b) => b - a);
    const maxStress = solid[0];
    const top5Count = Math.max(1, Math.ceil(solid.length * 0.05));
    let top5Sum = 0;
    for (let i = 0; i < top5Count; i++) top5Sum += solid[i];
    return { maxStress, top5PctAvg: top5Sum / top5Count, solidCount: solid.length };
}

function computeMaterialPct(x) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i];
    return sum / x.length;
}

function computeSolidPct(x) {
    let count = 0;
    for (let i = 0; i < x.length; i++) if (x[i] >= DENSITY_THRESHOLD) count++;
    return count / x.length;
}

// ═══════════════════════════════════════════════════════════════════════
// Full topology optimisation loop
// ═══════════════════════════════════════════════════════════════════════
function runTO(nelx, nely, nelz, config, solverType, timeoutMs = 120000) {
    const nel = nelx * nely * nelz;
    const KE = lk3D(NU);
    const KEflat = flattenKE(KE);
    const edofArray = precomputeEdofs3D(nelx, nely, nelz);
    const { ndof, fixeddofs, fixedMask, freedofs, F } = setupProblem3D(nelx, nely, nelz);

    const volfrac = config.volumeFraction;
    const penal = config.penaltyFactor;
    const maxIter = config.maxIterations;
    const rmin = config.filterRadius;

    const H = prepareFilter3D(nelx, nely, nelz, rmin);

    let x = new Float32Array(nel); x.fill(volfrac);
    let xnew = new Float32Array(nel);
    let xold = new Float32Array(nel);

    const mg = solverType === 'mgpcg' ? new MGPrecond3D(KEflat) : null;

    // Penalty continuation: ramp from 3 to target over iterations
    const penalStart = Math.min(3, penal);
    const penalEnd = penal;

    let loop = 0, change = 1, c = 0;
    const maxCG = solverType === 'mgpcg' ? MAX_CG_ITERATIONS_MGPCG : MAX_CG_ITERATIONS_JACOBI;
    const iterStats = [];
    const t0 = performance.now();

    while (change > 0.01 && loop < maxIter) {
        loop++;
        const tIter0 = performance.now();
        xold.set(x);

        // Current penalty via continuation
        const progress = maxIter > 1 ? (loop - 1) / (maxIter - 1) : 1;
        const curPenal = penalStart + (penalEnd - penalStart) * progress;

        // Scheduled tolerance
        const tol = Math.exp(Math.log(CG_TOL_START) + progress * (Math.log(CG_TOL_END) - Math.log(CG_TOL_START)));

        let U, cgIters;
        if (solverType === 'mgpcg') {
            const res = solveMGPCG(KEflat, edofArray, x, curPenal, nelx, nely, nelz, ndof, F, freedofs, fixedMask, tol, maxCG, mg);
            U = res.U; cgIters = res.iters;
        } else {
            const res = solveJacobiPCG(KEflat, edofArray, x, curPenal, nel, ndof, F, freedofs, fixedMask, tol, maxCG);
            U = res.U; cgIters = res.iters;
        }

        // Compliance
        c = 0;
        for (let i = 0; i < ndof; i++) c += F[i] * U[i];

        // Sensitivity
        const dc = new Float32Array(nel);
        const Ue = new Float64Array(24);
        for (let e = 0; e < nel; e++) {
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) Ue[i] = U[edofArray[eOff + i]];
            let energy = 0;
            for (let i = 0; i < 24; i++) { const row = i * 24; for (let j = 0; j < 24; j++) energy += Ue[i] * KEflat[row + j] * Ue[j]; }
            dc[e] = -curPenal * Math.pow(Math.max(x[e], 1e-12), curPenal - 1) * E0 * energy;
        }

        const dcn = filterSens(dc, x, H, nel);
        OC(nel, x, volfrac, dcn, xnew);

        change = 0;
        for (let i = 0; i < nel; i++) change = Math.max(change, Math.abs(xnew[i] - xold[i]));

        const tmp = x; x = xnew; xnew = tmp;

        const tIter1 = performance.now();
        iterStats.push({ iter: loop, compliance: c, change, cgIters, time: tIter1 - tIter0 });
        if (performance.now() - t0 > timeoutMs) break;
    }

    const totalTime = performance.now() - t0;

    // ── Post-optimisation FEA stress analysis ──────────────────────────
    let UPost;
    if (solverType === 'mgpcg') {
        UPost = solveMGPCG(KEflat, edofArray, x, penalEnd, nelx, nely, nelz, ndof, F, freedofs, fixedMask, CG_TOL_END, maxCG, mg).U;
    } else {
        UPost = solveJacobiPCG(KEflat, edofArray, x, penalEnd, nel, ndof, F, freedofs, fixedMask, CG_TOL_END, maxCG).U;
    }
    const stresses = computeElementStresses(nelx, nely, nelz, x, UPost, KEflat, edofArray, penalEnd);
    const stressReport = analyzeStress(stresses, x);
    const materialPct = computeMaterialPct(x);
    const solidPct = computeSolidPct(x);

    return {
        solver: solverType,
        nelx, nely, nelz, nel, ndof,
        iterations: loop,
        timedOut: totalTime > timeoutMs,
        finalCompliance: c,
        materialPct, solidPct,
        targetPct: volfrac,
        totalTime,
        avgIterTime: totalTime / loop,
        stressReport,
        iterStats
    };
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════
console.log('');
console.log('='.repeat(90));
console.log('  3D Topology Optimization Benchmark  --  MGPCG vs Jacobi-PCG');
console.log('='.repeat(90));

const defaultConfig = {
    volumeFraction: 0.10,
    maxIterations: 30,
    penaltyFactor: 20,
    filterRadius: 0.9
};

// 50x50x50 mm at 5 mm voxel = 10x10x10
const testCases = [
    { nelx: 10,  nely: 10,  nelz: 10,  label: '10x10x10 (1k elems, 50mm/5mm voxel)' },
    { nelx: 15,  nely: 15,  nelz: 15,  label: '15x15x15 (3.4k elems)' },
    { nelx: 20,  nely: 20,  nelz: 20,  label: '20x20x20 (8k elems)' },
    { nelx: 25,  nely: 25,  nelz: 25,  label: '25x25x25 (15.6k elems)' },
    { nelx: 30,  nely: 30,  nelz: 30,  label: '30x30x30 (27k elems)' },
];

const allResults = [];

for (const tc of testCases) {
    console.log('');
    console.log('='.repeat(90));
    console.log('  Mesh: ' + tc.label);
    console.log('='.repeat(90));

    const configs = [
        { solver: 'jacobi', label: 'Jacobi-PCG' },
        { solver: 'mgpcg',  label: 'MGPCG' },
    ];

    const caseResults = {};
    for (const cfg of configs) {
        process.stdout.write('  Running ' + cfg.label + '...');
        const result = runTO(tc.nelx, tc.nely, tc.nelz, defaultConfig, cfg.solver);
        process.stdout.write(' done (' + (result.totalTime / 1000).toFixed(2) + 's' + (result.timedOut ? ' TIMEOUT' : '') + ')\n');
        caseResults[cfg.solver] = result;
        allResults.push({ label: tc.label, ...result });
    }

    // Per-solver details
    for (const cfg of configs) {
        const r = caseResults[cfg.solver];
        console.log('');
        console.log('  ' + cfg.label + ':');
        console.log('    Iterations:      ' + r.iterations + (r.timedOut ? ' (TIMEOUT)' : ''));
        console.log('    Final compliance: ' + r.finalCompliance.toFixed(6));
        console.log('    Material %:      ' + (r.materialPct * 100).toFixed(2) + '%  (target: ' + (r.targetPct * 100).toFixed(1) + '%)');
        console.log('    Solid voxels:    ' + (r.solidPct * 100).toFixed(2) + '%  (threshold >= ' + DENSITY_THRESHOLD + ')');
        console.log('    Total time:      ' + (r.totalTime / 1000).toFixed(3) + 's');
        console.log('    Avg iter time:   ' + r.avgIterTime.toFixed(1) + 'ms');
        console.log('    Avg CG iters:    ' + (r.iterStats.reduce((s, it) => s + it.cgIters, 0) / r.iterations).toFixed(1));
        console.log('    Max stress:      ' + r.stressReport.maxStress.toExponential(4));
        console.log('    Top 5% avg:      ' + r.stressReport.top5PctAvg.toExponential(4));
        console.log('    Solid elements:  ' + r.stressReport.solidCount);
    }

    // Cross-compare
    const jac = caseResults['jacobi'];
    const mgr = caseResults['mgpcg'];
    if (jac && mgr) {
        console.log('');
        console.log('  CROSS-COMPARISON:');
        console.log('    Speedup (MGPCG vs Jacobi):     ' + (jac.totalTime / mgr.totalTime).toFixed(2) + 'x');
        console.log('    Compliance diff:                ' + (Math.abs(jac.finalCompliance - mgr.finalCompliance) / Math.abs(jac.finalCompliance) * 100).toFixed(4) + '%');
        console.log('    Material % (Jacobi):            ' + (jac.materialPct * 100).toFixed(2) + '%');
        console.log('    Material % (MGPCG):             ' + (mgr.materialPct * 100).toFixed(2) + '%');
        console.log('    Max stress ratio (MGPCG/Jac):   ' + (mgr.stressReport.maxStress / (jac.stressReport.maxStress + 1e-30)).toFixed(4));
        console.log('    Top 5% stress ratio:            ' + (mgr.stressReport.top5PctAvg / (jac.stressReport.top5PctAvg + 1e-30)).toFixed(4));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SUMMARY TABLE
// ═══════════════════════════════════════════════════════════════════════
console.log('');
console.log('');
console.log('='.repeat(140));
console.log('  SUMMARY');
console.log('='.repeat(140));
console.log('Mesh               | Solver     | Iters | Compliance    | Matl %  | Solid %  | Total (s) | Avg Iter (ms) | Max Stress    | Top 5% Avg');
console.log('-'.repeat(140));
for (const r of allResults) {
    const label = r.label.substring(0, 18).padEnd(18);
    const solver = r.solver.padEnd(10);
    const iters = String(r.iterations).padStart(5);
    const comp = r.finalCompliance.toFixed(6).padStart(13);
    const matPct = (r.materialPct * 100).toFixed(1).padStart(6) + '%';
    const solidPct = (r.solidPct * 100).toFixed(1).padStart(6) + '%';
    const total = (r.totalTime / 1000).toFixed(2).padStart(9);
    const avgI = r.avgIterTime.toFixed(1).padStart(13);
    const maxS = r.stressReport.maxStress.toExponential(3).padStart(13);
    const t5 = r.stressReport.top5PctAvg.toExponential(3).padStart(10);
    console.log(label + ' | ' + solver + ' | ' + iters + ' | ' + comp + ' | ' + matPct + ' | ' + solidPct + ' | ' + total + ' | ' + avgI + ' | ' + maxS + ' | ' + t5);
}
console.log('='.repeat(140));

// Material & stress verification table
console.log('');
console.log('='.repeat(110));
console.log('  MATERIAL & STRESS VERIFICATION');
console.log('='.repeat(110));
console.log('Mesh               | Solver     | Target % | Actual % | Delta   | Max Stress    | Top 5% Stress | Pass?');
console.log('-'.repeat(110));
for (const r of allResults) {
    const label = r.label.substring(0, 18).padEnd(18);
    const solver = r.solver.padEnd(10);
    const target = (r.targetPct * 100).toFixed(1).padStart(7) + '%';
    const actual = (r.materialPct * 100).toFixed(1).padStart(7) + '%';
    const delta = ((r.materialPct - r.targetPct) * 100).toFixed(2).padStart(6) + '%';
    const maxS = r.stressReport.maxStress.toExponential(3).padStart(13);
    const t5 = r.stressReport.top5PctAvg.toExponential(3).padStart(13);
    const pass = r.stressReport.maxStress > 0 && r.stressReport.solidCount > 0 ? '  YES' : '   NO';
    console.log(label + ' | ' + solver + ' | ' + target + ' | ' + actual + ' | ' + delta + ' | ' + maxS + ' | ' + t5 + ' | ' + pass);
}
console.log('='.repeat(110));

console.log('');
console.log('Done.');
console.log('');
