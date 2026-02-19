#!/usr/bin/env node
/**
 * Tests for the WASM-accelerated ebeMGPCG function.
 * Verifies that the full self-contained WASM MGPCG solver produces the same
 * results as the JavaScript reference MGPCG implementation.
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

// ─── MG constants (must match WASM) ───
const MG_MAX_LEVELS = 6;
const MG_NU1 = 2;
const MG_NU2 = 2;
const MG_OMEGA = 0.5;
const MG_COARSE_ITERS = 30;
const GALERKIN_MAX_NDOF = 3000;
const EPSILON = 1e-12;
const E0 = 1;
const Emin = 1e-9;

// ─── 3D element stiffness (8-node hex, 2×2×2 Gauss) ───
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
    for (let gi = 0; gi < 2; gi++) for (let gj = 0; gj < 2; gj++) for (let gk = 0; gk < 2; gk++) {
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
            B[0][col] = dN[n][0]; B[1][col + 1] = dN[n][1]; B[2][col + 2] = dN[n][2];
            B[3][col] = dN[n][1]; B[3][col + 1] = dN[n][0];
            B[4][col + 1] = dN[n][2]; B[4][col + 2] = dN[n][1];
            B[5][col] = dN[n][2]; B[5][col + 2] = dN[n][0];
        }
        const CB = Array.from({ length: 6 }, () => new Float64Array(24));
        for (let m = 0; m < 6; m++) for (let n = 0; n < 24; n++) for (let p = 0; p < 6; p++) CB[m][n] += C[m][p] * B[p][n];
        for (let m = 0; m < 24; m++) for (let n = 0; n < 24; n++) for (let p = 0; p < 6; p++) KE[m][n] += B[p][m] * CB[p][n];
    }
    return KE;
}

function flattenKE(KE) {
    const flat = new Float64Array(24 * 24);
    for (let i = 0; i < 24; i++) for (let j = 0; j < 24; j++) flat[i * 24 + j] = KE[i][j];
    return flat;
}

// ─── 3D mesh helpers ───
function precomputeEdofs3D(nelx, nely, nelz) {
    const nel = nelx * nely * nelz;
    const nny = nely + 1, nnz = nelz + 1;
    const edofArray = new Int32Array(nel * 24);
    for (let elz = 0; elz < nelz; elz++) for (let ely = 0; ely < nely; ely++) for (let elx = 0; elx < nelx; elx++) {
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
    return edofArray;
}

function setupProblem3D(nelx, nely, nelz) {
    const nel = nelx * nely * nelz;
    const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
    const nny = nely + 1, nnz = nelz + 1;

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

    const F = new Float64Array(ndof);
    const n_tc = Math.floor(nelx / 2) * nny * nnz + Math.floor(nely / 2) * nnz + nelz;
    F[3 * n_tc + 1] = -1.0;

    return { nel, ndof, fixedMask, freedofs, F };
}

// ─── JS reference MGPCG solver (minimal version) ───
function downsampleFixedMask(fixedFine, nxF0, nyF0, nzF0, nxC0, nyC0, nzC0) {
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

function fullSpaceMatVec(E_vals, activeElements, KEflat, edofArray, p, Ap, ndof) {
    Ap.fill(0);
    const loc = new Float64Array(24);
    for (let ae = 0; ae < activeElements.length; ae++) {
        const e = activeElements[ae];
        const E = E_vals[e], eOff = e * 24;
        for (let j = 0; j < 24; j++) loc[j] = p[edofArray[eOff + j]];
        for (let i = 0; i < 24; i++) {
            const gi = edofArray[eOff + i];
            let sum = 0;
            for (let j = 0; j < 24; j++) sum += KEflat[i * 24 + j] * loc[j];
            Ap[gi] += E * sum;
        }
    }
}

function solveMGPCG_JS(KEflat, edofArray, densities, penal, nelx, nely, nelz, ndof, F, freedofs, fixedMask, tolerance, maxIter) {
    const nel = nelx * nely * nelz;
    const dE = E0 - Emin;
    const skipT = Emin * 1000;

    // Build levels
    const levels = [];
    function makeLevel(nx, ny, nz, edof, fm, free) {
        const nel = nx * ny * nz, ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1);
        return {
            nelx: nx, nely: ny, nelz: nz, nel, ndof, edofArray: edof, fixedMask: fm, freeDofs: free,
            E_vals: new Float64Array(nel), active: [], activeCount: 0,
            diag: new Float64Array(ndof), invDiag: new Float64Array(ndof),
            Au: new Float64Array(ndof), res: new Float64Array(ndof),
            b: new Float64Array(ndof), x: new Float64Array(ndof), denseK: null
        };
    }

    levels.push(makeLevel(nelx, nely, nelz, edofArray, fixedMask, freedofs));
    let cx = nelx, cy = nely, cz = nelz, fm = fixedMask;
    for (let li = 1; li < MG_MAX_LEVELS; li++) {
        const nx = Math.floor(cx / 2), ny = Math.floor(cy / 2), nz = Math.floor(cz / 2);
        if (nx < 2 || ny < 2 || nz < 2) break;
        const edof = precomputeEdofs3D(nx, ny, nz);
        const fmC = downsampleFixedMask(fm, cx, cy, cz, nx, ny, nz);
        const ndofC = 3 * (nx + 1) * (ny + 1) * (nz + 1);
        let nFree = 0;
        for (let i = 0; i < ndofC; i++) if (!fmC[i]) nFree++;
        const freeC = new Int32Array(nFree);
        let p = 0;
        for (let i = 0; i < ndofC; i++) if (!fmC[i]) freeC[p++] = i;
        levels.push(makeLevel(nx, ny, nz, edof, fmC, freeC));
        cx = nx; cy = ny; cz = nz; fm = fmC;
    }

    // Update level 0 E_vals
    {
        const level = levels[0];
        let ac = 0;
        for (let e = 0; e < nel; e++) {
            const E = Emin + Math.pow(densities[e], penal) * dE;
            level.E_vals[e] = E;
            if (E > skipT) { level.active.push(e); ac++; }
        }
        level.activeCount = ac;
        level.diag.fill(0);
        for (const e of level.active) {
            const E = level.E_vals[e], eO = e * 24;
            for (let i = 0; i < 24; i++) level.diag[edofArray[eO + i]] += E * KEflat[i * 24 + i];
        }
        for (let i = 0; i < level.freeDofs.length; i++) {
            const d = level.freeDofs[i];
            level.invDiag[d] = level.diag[d] > 1e-30 ? 1.0 / level.diag[d] : 0;
        }
    }

    // Update coarse levels via E-val averaging
    for (let li = 1; li < levels.length; li++) {
        const fine = levels[li - 1], coarse = levels[li];
        const nxF = fine.nelx, nyF = fine.nely, nzF = fine.nelz;
        const nxC = coarse.nelx, nyC = coarse.nely, nzC = coarse.nelz;
        let ac = 0, idxC = 0;
        coarse.active = [];
        for (let czz = 0; czz < nzC; czz++) {
            const fz0 = czz * 2;
            for (let cyy = 0; cyy < nyC; cyy++) {
                const fy0 = cyy * 2;
                for (let cxx = 0; cxx < nxC; cxx++, idxC++) {
                    const fx0 = cxx * 2;
                    let sum = 0, count = 0;
                    for (let dz = 0; dz < 2; dz++) { const fz = fz0 + dz; if (fz >= nzF) continue;
                        for (let dy = 0; dy < 2; dy++) { const fy = fy0 + dy; if (fy >= nyF) continue;
                            for (let dx = 0; dx < 2; dx++) { const fx = fx0 + dx; if (fx >= nxF) continue;
                                sum += fine.E_vals[fx + fy * nxF + fz * nxF * nyF]; count++;
                    }}}
                    const E = count > 0 ? (sum / count) * 2.0 : 0;
                    coarse.E_vals[idxC] = E;
                    if (E > skipT) { coarse.active.push(idxC); ac++; }
                }
            }
        }
        coarse.activeCount = ac;
        coarse.diag.fill(0);
        for (const e of coarse.active) {
            const E = coarse.E_vals[e], eO = e * 24;
            for (let i = 0; i < 24; i++) coarse.diag[coarse.edofArray[eO + i]] += E * KEflat[i * 24 + i];
        }
        for (let i = 0; i < coarse.freeDofs.length; i++) {
            const d = coarse.freeDofs[i];
            coarse.invDiag[d] = coarse.diag[d] > 1e-30 ? 1.0 / coarse.diag[d] : 0;
        }
    }

    // applyA
    function applyA(li, p, Ap) {
        const level = levels[li];
        if (level.denseK) {
            const K = level.denseK, n = level.ndof;
            Ap.fill(0);
            for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += K[i * n + j] * p[j]; Ap[i] = s; }
            return;
        }
        fullSpaceMatVec(level.E_vals, level.active, KEflat, level.edofArray, p, Ap, level.ndof);
    }

    // vcycle
    function vcycle(li, b, x) {
        const level = levels[li], Au = level.Au, invD = level.invDiag, free = level.freeDofs;
        for (let s = 0; s < MG_NU1; s++) {
            applyA(li, x, Au);
            for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; x[d] += MG_OMEGA * invD[d] * (b[d] - Au[d]); }
        }
        applyA(li, x, Au);
        const res = level.res; res.fill(0);
        for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; res[d] = b[d] - Au[d]; }
        if (li === levels.length - 1) {
            for (let s = 0; s < MG_COARSE_ITERS; s++) {
                applyA(li, x, Au);
                for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; x[d] += MG_OMEGA * invD[d] * (b[d] - Au[d]); }
            }
            return;
        }
        const coarse = levels[li + 1]; const bC = coarse.b; bC.fill(0);
        const nxF = level.nelx + 1, nyF = level.nely + 1, nzF = level.nelz + 1;
        const nxC = coarse.nelx + 1, nyC = coarse.nely + 1, nzC = coarse.nelz + 1;
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
                    const r0 = res[3 * nF], r1 = res[3 * nF + 1], r2 = res[3 * nF + 2];
                    const w000 = wx0 * wy0 * wz0; if (w000) { const c3 = 3 * (cx0 * nyC * nzC + cy0 * nzC + cz0); bC[c3] += w000 * r0; bC[c3 + 1] += w000 * r1; bC[c3 + 2] += w000 * r2; }
                    const w100 = wx1 * wy0 * wz0; if (w100) { const c3 = 3 * (cx1 * nyC * nzC + cy0 * nzC + cz0); bC[c3] += w100 * r0; bC[c3 + 1] += w100 * r1; bC[c3 + 2] += w100 * r2; }
                    const w010 = wx0 * wy1 * wz0; if (w010) { const c3 = 3 * (cx0 * nyC * nzC + cy1 * nzC + cz0); bC[c3] += w010 * r0; bC[c3 + 1] += w010 * r1; bC[c3 + 2] += w010 * r2; }
                    const w110 = wx1 * wy1 * wz0; if (w110) { const c3 = 3 * (cx1 * nyC * nzC + cy1 * nzC + cz0); bC[c3] += w110 * r0; bC[c3 + 1] += w110 * r1; bC[c3 + 2] += w110 * r2; }
                    const w001 = wx0 * wy0 * wz1; if (w001) { const c3 = 3 * (cx0 * nyC * nzC + cy0 * nzC + cz1); bC[c3] += w001 * r0; bC[c3 + 1] += w001 * r1; bC[c3 + 2] += w001 * r2; }
                    const w101 = wx1 * wy0 * wz1; if (w101) { const c3 = 3 * (cx1 * nyC * nzC + cy0 * nzC + cz1); bC[c3] += w101 * r0; bC[c3 + 1] += w101 * r1; bC[c3 + 2] += w101 * r2; }
                    const w011 = wx0 * wy1 * wz1; if (w011) { const c3 = 3 * (cx0 * nyC * nzC + cy1 * nzC + cz1); bC[c3] += w011 * r0; bC[c3 + 1] += w011 * r1; bC[c3 + 2] += w011 * r2; }
                    const w111 = wx1 * wy1 * wz1; if (w111) { const c3 = 3 * (cx1 * nyC * nzC + cy1 * nzC + cz1); bC[c3] += w111 * r0; bC[c3 + 1] += w111 * r1; bC[c3 + 2] += w111 * r2; }
                }
            }
        }
        for (let i = 0; i < coarse.ndof; i++) if (coarse.fixedMask[i]) bC[i] = 0;
        const xC = coarse.x; xC.fill(0);
        vcycle(li + 1, bC, xC);
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
                    const nFn = fx * nyF * nzF + fy * nzF + fz;
                    const n000 = cx0 * nyC * nzC + cy0 * nzC + cz0, n100 = cx1 * nyC * nzC + cy0 * nzC + cz0;
                    const n010 = cx0 * nyC * nzC + cy1 * nzC + cz0, n110 = cx1 * nyC * nzC + cy1 * nzC + cz0;
                    const n001 = cx0 * nyC * nzC + cy0 * nzC + cz1, n101 = cx1 * nyC * nzC + cy0 * nzC + cz1;
                    const n011 = cx0 * nyC * nzC + cy1 * nzC + cz1, n111 = cx1 * nyC * nzC + cy1 * nzC + cz1;
                    const pw000 = wx0 * wy0 * wz0, pw100 = wx1 * wy0 * wz0, pw010 = wx0 * wy1 * wz0, pw110 = wx1 * wy1 * wz0;
                    const pw001 = wx0 * wy0 * wz1, pw101 = wx1 * wy0 * wz1, pw011 = wx0 * wy1 * wz1, pw111 = wx1 * wy1 * wz1;
                    const bF = 3 * nFn;
                    for (let d = 0; d < 3; d++) {
                        if (!fixedF[bF + d]) {
                            x[bF + d] += xC[3 * n000 + d] * pw000 + xC[3 * n100 + d] * pw100 + xC[3 * n010 + d] * pw010 + xC[3 * n110 + d] * pw110
                                + xC[3 * n001 + d] * pw001 + xC[3 * n101 + d] * pw101 + xC[3 * n011 + d] * pw011 + xC[3 * n111 + d] * pw111;
                        }
                    }
                }
            }
        }
        for (let s = 0; s < MG_NU2; s++) {
            applyA(li, x, Au);
            for (let ii = 0; ii < free.length; ii++) { const d = free[ii]; x[d] += MG_OMEGA * invD[d] * (b[d] - Au[d]); }
        }
    }

    // PCG outer loop
    const U = new Float64Array(ndof);
    const r = new Float64Array(ndof);
    const z = new Float64Array(ndof);
    const p = new Float64Array(ndof);
    const Ap = new Float64Array(ndof);

    for (let i = 0; i < ndof; i++) r[i] = fixedMask[i] ? 0 : F[i];
    z.fill(0); vcycle(0, r, z);
    for (let i = 0; i < ndof; i++) if (fixedMask[i]) z[i] = 0;
    p.set(z);

    let rz = 0, r0n2 = 0;
    for (let i = 0; i < ndof; i++) { rz += r[i] * z[i]; r0n2 += r[i] * r[i]; }
    const tolSq = tolerance * tolerance * Math.max(r0n2, 1e-30);

    let iters = 0;
    for (let iter = 0; iter < maxIter; iter++) {
        let rn2 = 0;
        for (let i = 0; i < ndof; i++) rn2 += r[i] * r[i];
        if (rn2 < tolSq) break;
        iters++;
        applyA(0, p, Ap);
        for (let i = 0; i < ndof; i++) if (fixedMask[i]) Ap[i] = 0;
        let pAp = 0;
        for (let i = 0; i < ndof; i++) pAp += p[i] * Ap[i];
        const alpha = rz / (pAp + EPSILON);
        for (let i = 0; i < ndof; i++) { U[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        z.fill(0); vcycle(0, r, z);
        for (let i = 0; i < ndof; i++) if (fixedMask[i]) z[i] = 0;
        let rz_new = 0;
        for (let i = 0; i < ndof; i++) rz_new += r[i] * z[i];
        const beta = rz_new / (rz + EPSILON);
        for (let i = 0; i < ndof; i++) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    let c = 0;
    for (let i = 0; i < ndof; i++) c += F[i] * U[i];
    return { U, compliance: c, iters, levels };
}

// ─── WASM MGPCG wrapper ───
function solveMGPCG_WASM(wasmMod, KEflat, edofArray, densities, penal, nelx, nely, nelz, ndof, F, freedofs, fixedMask, tolerance, maxIter) {
    // First solve with JS to get the level hierarchy data
    const jsResult = solveMGPCG_JS(KEflat, edofArray, densities, penal, nelx, nely, nelz, ndof, F, freedofs, fixedMask, tolerance, maxIter);
    const levels = jsResult.levels;

    const mem = wasmMod.exports.memory;
    const align8 = (v) => (v + 7) & ~7;

    // Calculate total memory needed
    const LEVEL_STRIDE = 80;
    let totalBytes = levels.length * LEVEL_STRIDE;  // level descriptors
    totalBytes += 576 * 8;  // KEflat
    totalBytes += ndof * 8;  // F
    totalBytes += ndof * 8;  // U
    totalBytes += ndof;      // fixedMask
    totalBytes += 4 * ndof * 8;  // CG work: r, z, p, Ap

    for (const level of levels) {
        totalBytes += level.nel * 24 * 4;  // edofs (i32)
        totalBytes += level.nel * 8;       // E_vals (f64)
        totalBytes += level.nel * 4;       // active (i32)
        totalBytes += level.ndof;          // fixedMask (u8)
        totalBytes += level.freeDofs.length * 4;  // freeDofs (i32)
        totalBytes += level.ndof * 8;      // invDiag (f64)
        totalBytes += level.ndof * 8;      // Au (f64)
        totalBytes += level.ndof * 8;      // res (f64)
        totalBytes += level.ndof * 8;      // b (f64)
        totalBytes += level.ndof * 8;      // x (f64)
        totalBytes += 24 * 8;             // scratch (f64)
    }
    totalBytes += 4096;  // padding

    const currentBytes = mem.buffer.byteLength;
    if (currentBytes < totalBytes + 65536) {
        const pages = Math.ceil((totalBytes + 65536 - currentBytes) / 65536);
        mem.grow(pages);
    }
    const dataStart = mem.buffer.byteLength - totalBytes - 256;

    let offset = align8(dataStart);

    // Level descriptors
    const levelsOff = offset;
    offset += levels.length * LEVEL_STRIDE;
    offset = align8(offset);

    // KEflat
    const keOff = offset;
    new Float64Array(mem.buffer, keOff, 576).set(KEflat);
    offset += 576 * 8;

    // F
    const fOff = offset;
    new Float64Array(mem.buffer, fOff, ndof).set(F);
    offset += ndof * 8;

    // U
    const uOff = offset;
    new Float64Array(mem.buffer, uOff, ndof).fill(0);
    offset += ndof * 8;

    // fixedMask
    const fmOff = offset;
    new Uint8Array(mem.buffer, fmOff, ndof).set(fixedMask);
    offset += ndof;
    offset = align8(offset);

    // CG work
    const cgWorkOff = offset;
    offset += 4 * ndof * 8;
    offset = align8(offset);

    // Per-level data
    for (let li = 0; li < levels.length; li++) {
        const level = levels[li];
        const descOff = levelsOff + li * LEVEL_STRIDE;

        // Edofs
        const edofsOff = offset;
        new Int32Array(mem.buffer, edofsOff, level.nel * 24).set(level.edofArray);
        offset += level.nel * 24 * 4;
        offset = align8(offset);

        // E_vals
        const evalsOff = offset;
        new Float64Array(mem.buffer, evalsOff, level.nel).set(level.E_vals);
        offset += level.nel * 8;

        // Active
        const activeOff = offset;
        const activeArr = new Int32Array(level.activeCount);
        for (let i = 0; i < level.activeCount; i++) activeArr[i] = level.active[i];
        new Int32Array(mem.buffer, activeOff, level.activeCount).set(activeArr);
        offset += level.nel * 4;
        offset = align8(offset);

        // FixedMask
        const fmLvlOff = offset;
        new Uint8Array(mem.buffer, fmLvlOff, level.ndof).set(level.fixedMask);
        offset += level.ndof;
        offset = align8(offset);

        // FreeDofs
        const freeOff = offset;
        new Int32Array(mem.buffer, freeOff, level.freeDofs.length).set(level.freeDofs);
        offset += level.freeDofs.length * 4;
        offset = align8(offset);

        // InvDiag
        const invDiagOff = offset;
        new Float64Array(mem.buffer, invDiagOff, level.ndof).set(level.invDiag);
        offset += level.ndof * 8;

        // Au
        const auOff = offset;
        offset += level.ndof * 8;

        // Res
        const resOff = offset;
        offset += level.ndof * 8;

        // b
        const bOff = offset;
        offset += level.ndof * 8;

        // x
        const xOff = offset;
        offset += level.ndof * 8;

        // Scratch
        const scratchOff = offset;
        offset += 24 * 8;
        offset = align8(offset);

        // Write level descriptor
        const dv = new DataView(mem.buffer);
        dv.setInt32(descOff + 0, level.nelx, true);
        dv.setInt32(descOff + 4, level.nely, true);
        dv.setInt32(descOff + 8, level.nelz, true);
        dv.setInt32(descOff + 12, level.ndof, true);
        dv.setInt32(descOff + 16, level.nel, true);
        dv.setInt32(descOff + 20, level.freeDofs.length, true);
        dv.setInt32(descOff + 24, level.activeCount, true);
        dv.setInt32(descOff + 28, level.denseK ? 1 : 0, true);
        dv.setUint32(descOff + 32, edofsOff, true);
        dv.setUint32(descOff + 36, evalsOff, true);
        dv.setUint32(descOff + 40, activeOff, true);
        dv.setUint32(descOff + 44, fmLvlOff, true);
        dv.setUint32(descOff + 48, freeOff, true);
        dv.setUint32(descOff + 52, invDiagOff, true);
        dv.setUint32(descOff + 56, 0, true);  // no dense K for now
        dv.setUint32(descOff + 60, auOff, true);
        dv.setUint32(descOff + 64, resOff, true);
        dv.setUint32(descOff + 68, bOff, true);
        dv.setUint32(descOff + 72, xOff, true);
        dv.setUint32(descOff + 76, scratchOff, true);
    }

    // Call WASM ebeMGPCG
    const iterations = wasmMod.exports.ebeMGPCG(
        levelsOff, levels.length, keOff, fOff, uOff,
        fmOff, ndof, maxIter, tolerance, cgWorkOff
    );

    const U = new Float64Array(ndof);
    U.set(new Float64Array(mem.buffer, uOff, ndof));

    let c = 0;
    for (let i = 0; i < ndof; i++) c += F[i] * U[i];
    return { U, compliance: c, iterations };
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════
async function runTests() {
    const wasmMod = await loadWasm();

    console.log('Test 1: WASM module exports ebeMGPCG');
    assert(typeof wasmMod.exports.ebeMGPCG === 'function', 'ebeMGPCG should be exported');

    const nu = 0.3;
    const penal = 3;
    const tolerance = 1e-6;
    const maxIter = 500;
    const KE = lk3D(nu);
    const KEflat = flattenKE(KE);

    // ─── Test 2: Small 3D mesh (4×4×4, all solid) ───
    console.log('Test 2: Small 3D mesh 4×4×4 (all solid)');
    {
        const nelx = 4, nely = 4, nelz = 4;
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const prob = setupProblem3D(nelx, nely, nelz);
        const densities = new Float64Array(prob.nel).fill(1.0);

        const jsResult = solveMGPCG_JS(KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);
        const wasmResult = solveMGPCG_WASM(wasmMod, KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);

        let maxDiff = 0;
        for (let i = 0; i < prob.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        const compDiff = Math.abs(jsResult.compliance - wasmResult.compliance);
        assert(maxDiff < 1e-4, `Displacement match, max diff = ${maxDiff.toExponential(3)}`);
        assert(compDiff / Math.abs(jsResult.compliance) < 1e-3,
            `Compliance match: JS=${jsResult.compliance.toExponential(3)} WASM=${wasmResult.compliance.toExponential(3)}`);
    }

    // ─── Test 3: Medium 3D mesh (6×6×6, all solid) ───
    console.log('Test 3: Medium 3D mesh 6×6×6 (all solid)');
    {
        const nelx = 6, nely = 6, nelz = 6;
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const prob = setupProblem3D(nelx, nely, nelz);
        const densities = new Float64Array(prob.nel).fill(1.0);

        const jsResult = solveMGPCG_JS(KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);
        const wasmResult = solveMGPCG_WASM(wasmMod, KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);

        let maxDiff = 0;
        for (let i = 0; i < prob.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        assert(maxDiff < 1e-4, `Displacement match, max diff = ${maxDiff.toExponential(3)}`);
        assert(Math.abs(jsResult.compliance - wasmResult.compliance) / Math.abs(jsResult.compliance) < 1e-3,
            `Compliance match: JS=${jsResult.compliance.toExponential(3)} WASM=${wasmResult.compliance.toExponential(3)}`);
    }

    // ─── Test 4: Mixed densities ───
    console.log('Test 4: Mixed densities (4×4×4)');
    {
        const nelx = 4, nely = 4, nelz = 4;
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const prob = setupProblem3D(nelx, nely, nelz);
        const densities = new Float64Array(prob.nel);
        for (let e = 0; e < prob.nel; e++) densities[e] = 0.3 + 0.7 * (e / prob.nel);

        const jsResult = solveMGPCG_JS(KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);
        const wasmResult = solveMGPCG_WASM(wasmMod, KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);

        let maxDiff = 0;
        for (let i = 0; i < prob.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        assert(maxDiff < 1e-3, `Mixed densities displacement match, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 5: Non-zero compliance ───
    console.log('Test 5: WASM MGPCG produces non-zero compliance');
    {
        const nelx = 4, nely = 4, nelz = 4;
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const prob = setupProblem3D(nelx, nely, nelz);
        const densities = new Float64Array(prob.nel).fill(1.0);

        const wasmResult = solveMGPCG_WASM(wasmMod, KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, prob.F, prob.freedofs, prob.fixedMask, tolerance, maxIter);

        assert(wasmResult.compliance !== 0, `Compliance is non-zero: ${wasmResult.compliance.toExponential(3)}`);
        assert(wasmResult.iterations > 0, `CG iterations > 0: ${wasmResult.iterations}`);
    }

    // ─── Test 6: Zero force produces zero displacement ───
    console.log('Test 6: Zero force produces zero displacement');
    {
        const nelx = 4, nely = 4, nelz = 4;
        const edofArray = precomputeEdofs3D(nelx, nely, nelz);
        const prob = setupProblem3D(nelx, nely, nelz);
        const densities = new Float64Array(prob.nel).fill(1.0);
        const zeroF = new Float64Array(prob.ndof);

        const wasmResult = solveMGPCG_WASM(wasmMod, KEflat, edofArray, densities, penal, nelx, nely, nelz, prob.ndof, zeroF, prob.freedofs, prob.fixedMask, tolerance, maxIter);

        let maxU = 0;
        for (let i = 0; i < prob.ndof; i++) maxU = Math.max(maxU, Math.abs(wasmResult.U[i]));
        assert(maxU < 1e-10, `Zero force gives zero displacement, max |U| = ${maxU.toExponential(3)}`);
    }

    // ─── Summary ───
    console.log(`\nResults: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
    console.error('Test runner error:', err);
    process.exit(1);
});
