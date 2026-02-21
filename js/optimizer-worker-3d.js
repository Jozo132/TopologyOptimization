// Web Worker for 3D topology optimization using SIMP algorithm with 8-node hexahedral elements
// This runs in a separate thread so the UI stays responsive.
// Supports both Web Worker (browser) and worker_threads (Node.js) environments.

import { GPUCompute } from './gpu-compute.js';
import { GPUFEASolver } from './gpu-fea-solver.js';
import { NonlinearSolver } from './nonlinear-solver.js';
import { createMaterial } from './material-models.js';

// Node.js / browser environment compatibility shim
if (typeof self === 'undefined') {
    const { parentPort } = await import('worker_threads');
    globalThis.self = globalThis;
    globalThis.postMessage = (data, transferList) => parentPort.postMessage(data, transferList);
    parentPort.on('message', (data) => {
        if (typeof globalThis.onmessage === 'function') globalThis.onmessage({ data });
    });
}

const EPSILON = 1e-12;

// Yield to the full event loop so pending messages (cancel/pause/updateConfig) can be processed.
// Uses setTimeout(0) to advance past the I/O phase in Node.js and browsers alike.
const _yieldToLoop = () => new Promise(r => setTimeout(r, 0));

// Linear solver settings
// Default solver for large 3D problems: multigrid-preconditioned CG (MGPCG).
// Available linear solvers: 'mgpcg', 'cg', 'petsc'
// 'petsc' uses KSP-style Krylov solvers with BDDC domain decomposition or multigrid preconditioning.
const DEFAULT_LINEAR_SOLVER = 'mgpcg';

// FEA solver backend: controls which compute backend runs the linear solve.
// 'auto' = best available (webgpu > wasm > js)
// 'webgpu' = GPU-resident Jacobi-PCG via GPUFEASolver
// 'wasm' = WASM-compiled solvers (ebePCG / ebeMGPCG / ebeKSP_BDDC)
// 'js' = pure JavaScript fallback
const DEFAULT_FEA_BACKEND = 'auto';

// CG tolerance is scheduled during the optimization (looser early, tighter late)
const CG_TOL_START = 1e-3;
const CG_TOL_END = 1e-8;
const CG_TOLERANCE = CG_TOL_END; // backward-compat alias

// With MGPCG, hundreds (not thousands) of iterations should be enough.
const MAX_CG_ITERATIONS = 400;

// Face-to-node mapping: faceIndex → indices into the 8-node array [n0..n7]
// Matches the face definitions in viewer.js for consistent face picking
const FACE_NODE_INDICES = [
    [0, 4, 7, 3], // fi=0: -X face
    [5, 1, 2, 6], // fi=1: +X face
    [0, 1, 5, 4], // fi=2: -Y face
    [3, 7, 6, 2], // fi=3: +Y face
    [0, 3, 2, 1], // fi=4: -Z face
    [4, 5, 6, 7], // fi=5: +Z face
];

// Geometric multigrid parameters (V-cycle)
const MG_MAX_LEVELS = 6;
const MG_NU1 = 2;           // pre-smoothing steps
const MG_NU2 = 2;           // post-smoothing steps
const MG_OMEGA = 0.5;       // damped Jacobi relaxation – must be < 2/ρ(D⁻¹A) ≈ 0.645 for 3D hex ν=0.3
const MG_COARSE_ITERS = 30;  // extra smoothing at coarsest level
const GALERKIN_MAX_NDOF = 3000;  // Use dense Galerkin P^T A P for coarse levels with ndof ≤ this

// WASM Module for high-performance operations
let wasmModule = null;
let wasmLoaded = false;

async function loadWasmModule() {
    try {
        // Resolve WASM path relative to this worker module (works in browser and Node.js)
        const wasmUrl = new URL('../wasm/matrix-ops.wasm', import.meta.url);
        let buffer;
        try {
            // fetch() works in browsers and Node.js 18+
            const response = await fetch(wasmUrl.href);
            buffer = await response.arrayBuffer();
        } catch (_fetchErr) {
            // Fallback for older Node.js: use fs
            const { readFile } = await import('fs/promises');
            const { fileURLToPath } = await import('url');
            buffer = (await readFile(fileURLToPath(wasmUrl))).buffer;
        }
        const module = await WebAssembly.compile(buffer);
        
        wasmModule = await WebAssembly.instantiate(module, {
            env: {
                abort: () => console.error('WASM abort called'),
                seed: () => Date.now()
            }
        });
        
        wasmLoaded = true;
        console.log('WASM module loaded in worker');
        return true;
    } catch (error) {
        console.warn('Failed to load WASM in worker, using pure JS:', error);
        wasmLoaded = false;
        return false;
    }
}

// Simple 3D AMR Manager for adaptive element grouping
class SimpleAMRManager3D {
    constructor(nelx, nely, nelz, useAMR, minSize, maxSize, amrInterval) {
        this.nelx = nelx;
        this.nely = nely;
        this.nelz = nelz;
        this.useAMR = useAMR;
        this.minSize = minSize || 0.5;
        this.maxSize = maxSize || 2;
        this.amrInterval = amrInterval || 3;
        this.groups = [];
        this.elementToGroup = null;
        this.refinementCount = 0;
        
        if (useAMR) {
            this.initializeGroups();
        }
    }
    
    initializeGroups() {
        const groupSize = Math.max(1, Math.floor(this.maxSize));
        const nel = this.nelx * this.nely * this.nelz;
        this.elementToGroup = new Int32Array(nel);
        this.groups = [];
        
        let groupId = 0;
        const groupsX = Math.ceil(this.nelx / groupSize);
        const groupsY = Math.ceil(this.nely / groupSize);
        const groupsZ = Math.ceil(this.nelz / groupSize);
        
        for (let gz = 0; gz < groupsZ; gz++) {
            for (let gy = 0; gy < groupsY; gy++) {
                for (let gx = 0; gx < groupsX; gx++) {
                    const elements = [];
                    const startX = gx * groupSize;
                    const startY = gy * groupSize;
                    const startZ = gz * groupSize;
                    const endX = Math.min(startX + groupSize, this.nelx);
                    const endY = Math.min(startY + groupSize, this.nely);
                    const endZ = Math.min(startZ + groupSize, this.nelz);
                    
                    for (let z = startZ; z < endZ; z++) {
                        for (let y = startY; y < endY; y++) {
                            for (let x = startX; x < endX; x++) {
                                const idx = x + y * this.nelx + z * this.nelx * this.nely;
                                this.elementToGroup[idx] = groupId;
                                elements.push(idx);
                            }
                        }
                    }
                    
                    this.groups.push({
                        id: groupId,
                        size: groupSize,
                        elements: elements,
                        stress: 0,
                        density: 0.5
                    });
                    
                    groupId++;
                }
            }
        }
    }
    
    updateAndRefine(elementEnergies, densities, iteration) {
        if (!this.useAMR || this.groups.length === 0) return;
        if (iteration % this.amrInterval !== 0) return;
        
        // Update group stresses
        for (const group of this.groups) {
            let totalStress = 0;
            let totalDensity = 0;
            
            for (const elemIdx of group.elements) {
                const energy = elementEnergies[elemIdx] || 0;
                const density = densities[elemIdx] || 0.5;
                const stiffness = 1e-9 + Math.pow(density, 3) * (1 - 1e-9);
                totalStress += stiffness * energy;
                totalDensity += density;
            }
            
            group.stress = totalStress / group.elements.length;
            group.density = totalDensity / group.elements.length;
        }
        
        // Adaptive refinement with progressive thresholds
        const progressFactor = Math.min(iteration / 50, 1.0);
        const stressThreshold = 0.15 + progressFactor * 0.10;
        
        const stresses = this.groups.map(g => g.stress).sort((a, b) => b - a);
        const highStressThreshold = stresses[Math.floor(stresses.length * stressThreshold)] || 0;
        
        const toRefine = this.groups.filter(g => 
            g.stress > highStressThreshold && 
            g.size > this.minSize * 2 &&
            g.elements.length > 8
        ).slice(0, Math.min(3, Math.max(1, Math.floor(6 - iteration / 15)))); // Fewer refinements in 3D
        
        for (const group of toRefine) {
            this.splitGroup(group);
        }
        
        this.refinementCount++;
    }
    
    splitGroup(group) {
        const idx = this.groups.indexOf(group);
        if (idx === -1 || group.elements.length <= 1) return;
        
        this.groups.splice(idx, 1);
        
        // Find bounding box
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        for (const elemIdx of group.elements) {
            const z = Math.floor(elemIdx / (this.nelx * this.nely));
            const rem = elemIdx % (this.nelx * this.nely);
            const y = Math.floor(rem / this.nelx);
            const x = rem % this.nelx;
            
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            minZ = Math.min(minZ, z);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            maxZ = Math.max(maxZ, z);
        }
        
        const midX = Math.floor((minX + maxX) / 2);
        const midY = Math.floor((minY + maxY) / 2);
        const midZ = Math.floor((minZ + maxZ) / 2);
        const halfSize = Math.max(1, Math.floor(group.size / 2));
        
        // Create 8 sub-groups
        const newGroups = [];
        for (let i = 0; i < 8; i++) {
            newGroups.push({
                id: this.groups.length + i,
                size: halfSize,
                elements: [],
                stress: group.stress,
                density: group.density
            });
        }
        
        // Assign elements to octants
        for (const elemIdx of group.elements) {
            const z = Math.floor(elemIdx / (this.nelx * this.nely));
            const rem = elemIdx % (this.nelx * this.nely);
            const y = Math.floor(rem / this.nelx);
            const x = rem % this.nelx;
            
            const subIdx = (x >= midX ? 1 : 0) + (y >= midY ? 2 : 0) + (z >= midZ ? 4 : 0);
            newGroups[subIdx].elements.push(elemIdx);
            this.elementToGroup[elemIdx] = newGroups[subIdx].id;
        }
        
        // Add non-empty groups
        for (const ng of newGroups) {
            if (ng.elements.length > 0) {
                this.groups.push(ng);
            }
        }
    }
    
    getStats() {
        if (!this.useAMR || this.groups.length === 0) {
            return null;
        }
        
        const sizes = this.groups.map(g => g.size);
        return {
            groupCount: this.groups.length,
            minGroupSize: Math.min(...sizes),
            maxGroupSize: Math.max(...sizes),
            avgGroupSize: sizes.reduce((a, b) => a + b, 0) / sizes.length,
            totalElements: this.nelx * this.nely * this.nelz,
            refinementCount: this.refinementCount
        };
    }
}


/**
 * Fast integer-power helper for SIMP penalization.
 * Falls back to Math.pow for non-integers / uncommon exponents.
 */
function _powDensity(x, p) {
    if (!Number.isFinite(p)) return Math.pow(x, p);
    if (!Number.isInteger(p)) return Math.pow(x, p);
    switch (p) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return x * x;
        case 3: return x * x * x;
        case 4: { const x2 = x * x; return x2 * x2; }
        case 5: { const x2 = x * x; return x2 * x2 * x; }
        case 6: { const x2 = x * x; return x2 * x2 * x2; }
        default: return Math.pow(x, p);
    }
}

/**
 * A lightweight geometric multigrid (GMG) preconditioner for 3D voxel-grid elasticity.
 * Intended as a practical MGPCG speed-up for large structured grids.
 *
 * Notes:
 *  - This uses rediscretization on coarser grids via restricted densities (2x coarsening),
 *    which is simple and effective as a preconditioner for compliance-minimization TO.
 *  - Restriction uses injection; prolongation uses trilinear interpolation.
 */
class MGPrecond3D {
    constructor(KEflat) {
        this.KEflat = KEflat;
        this.levels = [];
        this._tmpLocal = new Float64Array(24);
        this._wasmReady = false;
        this._wasmKeOff = 0;
        this._wasmScratchOff = 0;
    }

    static _precomputeEdofs3D(nelx, nely, nelz) {
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
                    edofArray[offset]      = 3 * n0;
                    edofArray[offset + 1]  = 3 * n0 + 1;
                    edofArray[offset + 2]  = 3 * n0 + 2;
                    edofArray[offset + 3]  = 3 * n1;
                    edofArray[offset + 4]  = 3 * n1 + 1;
                    edofArray[offset + 5]  = 3 * n1 + 2;
                    edofArray[offset + 6]  = 3 * n2;
                    edofArray[offset + 7]  = 3 * n2 + 1;
                    edofArray[offset + 8]  = 3 * n2 + 2;
                    edofArray[offset + 9]  = 3 * n3;
                    edofArray[offset + 10] = 3 * n3 + 1;
                    edofArray[offset + 11] = 3 * n3 + 2;
                    edofArray[offset + 12] = 3 * n4;
                    edofArray[offset + 13] = 3 * n4 + 1;
                    edofArray[offset + 14] = 3 * n4 + 2;
                    edofArray[offset + 15] = 3 * n5;
                    edofArray[offset + 16] = 3 * n5 + 1;
                    edofArray[offset + 17] = 3 * n5 + 2;
                    edofArray[offset + 18] = 3 * n6;
                    edofArray[offset + 19] = 3 * n6 + 1;
                    edofArray[offset + 20] = 3 * n6 + 2;
                    edofArray[offset + 21] = 3 * n7;
                    edofArray[offset + 22] = 3 * n7 + 1;
                    edofArray[offset + 23] = 3 * n7 + 2;
                }
            }
        }
        return edofArray;
    }

    static _buildFreeDofsFromFixedMask(fixedMask) {
        let nFree = 0;
        for (let i = 0; i < fixedMask.length; i++) if (!fixedMask[i]) nFree++;
        const free = new Int32Array(nFree);
        let p = 0;
        for (let i = 0; i < fixedMask.length; i++) if (!fixedMask[i]) free[p++] = i;
        return free;
    }

    static _downsampleFixedMaskBy2(fixedFine, nelxFine, nelyFine, nelzFine, nelxCoarse, nelyCoarse, nelzCoarse) {
        const nxF = nelxFine + 1, nyF = nelyFine + 1, nzF = nelzFine + 1;
        const nxC = nelxCoarse + 1, nyC = nelyCoarse + 1, nzC = nelzCoarse + 1;
        const fixedC = new Uint8Array(3 * nxC * nyC * nzC);

        for (let cz = 0; cz < nzC; cz++) {
            const fz = Math.min(cz * 2, nzF - 1);
            for (let cy = 0; cy < nyC; cy++) {
                const fy = Math.min(cy * 2, nyF - 1);
                for (let cx = 0; cx < nxC; cx++) {
                    const fx = Math.min(cx * 2, nxF - 1);
                    const nF = fx * nyF * nzF + fy * nzF + fz;
                    const nC = cx * nyC * nzC + cy * nzC + cz;
                    const baseF = 3 * nF;
                    const baseC = 3 * nC;
                    fixedC[baseC]     = fixedFine[baseF];
                    fixedC[baseC + 1] = fixedFine[baseF + 1];
                    fixedC[baseC + 2] = fixedFine[baseF + 2];
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

        // Level 0 (fine)
        const level0 = this._makeLevel(nelx, nely, nelz, edofArrayFine, fixedMaskFine, freeDofsFine);
        this.levels.push(level0);

        // Coarser levels via factor-2 coarsening
        let cx = nelx, cy = nely, cz = nelz;
        let fixedMask = fixedMaskFine;

        for (let li = 1; li < MG_MAX_LEVELS; li++) {
            const nx = Math.floor(cx / 2);
            const ny = Math.floor(cy / 2);
            const nz = Math.floor(cz / 2);
            if (nx < 2 || ny < 2 || nz < 2) break;

            const edof = MGPrecond3D._precomputeEdofs3D(nx, ny, nz);
            const fixedC = MGPrecond3D._downsampleFixedMaskBy2(fixedMask, cx, cy, cz, nx, ny, nz);
            const freeC = MGPrecond3D._buildFreeDofsFromFixedMask(fixedC);

            const level = this._makeLevel(nx, ny, nz, edof, fixedC, freeC);
            this.levels.push(level);

            cx = nx; cy = ny; cz = nz;
            fixedMask = fixedC;
        }

        this._initWasmBuffers();
    }

    /**
     * Allocate WASM memory buffers for accelerated applyA.
     * Per-level regions hold edofs, E_vals, active, p, and Ap.
     * KEflat and a scratch buffer are shared across levels.
     */
    _initWasmBuffers() {
        if (!wasmModule || !wasmModule.exports.applyAEbe3D) {
            this._wasmReady = false;
            return;
        }
        try {
            const mem = wasmModule.exports.memory;
            const align8 = (v) => (v + 7) & ~7;
            // Calculate total needed memory (with alignment padding)
            const keSize = 576 * 8;       // KEflat: 24*24 f64
            const scratchSize = 24 * 8;   // scratch: 24 f64
            let totalBytes = keSize + scratchSize;

            for (const level of this.levels) {
                totalBytes += level.nel * 24 * 4 + 8;  // edofs (i32) + align
                totalBytes += level.nel * 8;       // E_vals (f64)
                totalBytes += level.nel * 4 + 8;   // active (i32) + align
                totalBytes += level.ndof * 8;      // p (f64)
                totalBytes += level.ndof * 8;      // Ap (f64)
            }

            const neededPages = Math.ceil(totalBytes / 65536) + 1;
            const dataStart = mem.buffer.byteLength;
            const growResult = mem.grow(neededPages);
            if (growResult === -1) {
                this._wasmReady = false;
                return;
            }

            let offset = dataStart;

            // Shared KEflat
            this._wasmKeOff = offset; offset += keSize;
            new Float64Array(mem.buffer, this._wasmKeOff, 576).set(this.KEflat);

            // Shared scratch
            this._wasmScratchOff = offset; offset += scratchSize;

            // Per-level allocations
            for (const level of this.levels) {
                const w = {};
                w.edofsOff = offset; offset += level.nel * 24 * 4;
                offset = align8(offset);
                w.evalsOff = offset; offset += level.nel * 8;
                w.activeOff = offset; offset += level.nel * 4;
                offset = align8(offset);
                w.pOff = offset; offset += level.ndof * 8;
                w.apOff = offset; offset += level.ndof * 8;

                // Copy static edofs (don't change during solve)
                new Int32Array(mem.buffer, w.edofsOff, level.edofArray.length).set(level.edofArray);
                level._wasm = w;
            }

            this._wasmReady = true;
        } catch (e) {
            this._wasmReady = false;
        }
    }

    /**
     * Sync density-derived data (E_vals, active list) to WASM memory for a level.
     */
    _wasmSyncLevel(level) {
        if (!this._wasmReady || !level._wasm) return;
        const mem = wasmModule.exports.memory;
        const w = level._wasm;
        new Float64Array(mem.buffer, w.evalsOff, level.nel).set(level.E_vals);
        new Int32Array(mem.buffer, w.activeOff, level.activeCount).set(
            level.active.subarray(0, level.activeCount)
        );
    }

    _makeLevel(nelx, nely, nelz, edofArray, fixedMask, freeDofs) {
        const nel = nelx * nely * nelz;
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);

        return {
            nelx, nely, nelz,
            nel, ndof,
            edofArray,
            fixedMask,
            freeDofs,

            // Densities live in Float32 arrays
            dens: new Float32Array(nel),

            // Element stiffness scalars and active list
            E_vals: new Float64Array(nel),
            active: new Int32Array(nel),
            activeCount: 0,

            // Jacobi inverse diagonal (for smoothing)
            diag: new Float64Array(ndof),
            invDiag: new Float64Array(ndof),

            // Work vectors
            Au: new Float64Array(ndof),
            res: new Float64Array(ndof),
            b: new Float64Array(ndof),
            x: new Float64Array(ndof),
            denseK: null  // dense Galerkin matrix (if assembled)
        };
    }

    /**
     * Update densities and build per-level operators.
     * Level 0: element-by-element. Coarse levels: Galerkin P^T A P or E-val avg fallback.
     */
    updateFromFine(xFine, penal, E0, Emin) {
        if (this.levels.length === 0) return;

        // Level 0: E_vals from densities
        const L0 = this.levels[0];
        L0.dens.set(xFine);
        this._updateOperatorLevel(L0, penal, E0, Emin);
        this._wasmSyncLevel(L0);

        // Coarse levels: Galerkin or E-val averaging fallback
        for (let li = 1; li < this.levels.length; li++) {
            const coarse = this.levels[li];
            if (coarse.ndof <= GALERKIN_MAX_NDOF) {
                const parentHasDense = this.levels[li - 1].denseK;
                if (parentHasDense) {
                    this._assembleGalerkinFromMatrix(li);
                } else {
                    this._assembleGalerkinFromElements(li, E0, Emin);
                }
            } else {
                this._restrictEvalAvg(this.levels[li - 1], this.levels[li], E0, Emin);
            }
            this._wasmSyncLevel(coarse);
        }
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

    // ─── Assemble dense Galerkin matrix from fine-level elements ───
    _assembleGalerkinFromElements(li, E0, Emin) {
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

    // ─── Assemble dense Galerkin from parent's dense matrix ───
    _assembleGalerkinFromMatrix(li) {
        const fine = this.levels[li - 1], coarse = this.levels[li];
        const ndofF = fine.ndof, ndofC = coarse.ndof;
        const pMap = this._buildProlongMap(fine, coarse);
        const { pNodes, pWeights, pCount } = pMap;
        const nnodesF = (fine.nelx + 1) * (fine.nely + 1) * (fine.nelz + 1);
        const P = new Float64Array(ndofF * ndofC);
        for (let nF = 0; nF < nnodesF; nF++) {
            const off = nF * 8, cnt = pCount[nF];
            for (let pi = 0; pi < cnt; pi++) {
                const nC = pNodes[off + pi], w = pWeights[off + pi];
                for (let d = 0; d < 3; d++) P[(3 * nF + d) * ndofC + (3 * nC + d)] = w;
            }
        }
        const Kf = fine.denseK;
        const Temp = new Float64Array(ndofF * ndofC);
        for (let i = 0; i < ndofF; i++) {
            const rowF = i * ndofF, rowT = i * ndofC;
            for (let j = 0; j < ndofC; j++) {
                let s = 0;
                for (let k = 0; k < ndofF; k++) s += Kf[rowF + k] * P[k * ndofC + j];
                Temp[rowT + j] = s;
            }
        }
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

    // ─── E-val averaging fallback for large coarse levels ───
    _restrictEvalAvg(fine, coarse, E0, Emin) {
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

    _updateOperatorLevel(level, penal, E0, Emin) {
        const nel = level.nel;
        const E_vals = level.E_vals;
        const active = level.active;

        const dE = E0 - Emin;
        const skipThreshold = Emin * 1000;

        // Active elements + stiffness
        let aCount = 0;
        const dens = level.dens;
        for (let e = 0; e < nel; e++) {
            const rho = dens[e];
            const rhoP = _powDensity(rho, penal);
            const E = Emin + rhoP * dE;
            E_vals[e] = E;
            if (E > skipThreshold) active[aCount++] = e;
        }
        level.activeCount = aCount;

        // Diagonal for Jacobi smoother / preconditioner
        const diag = level.diag;
        diag.fill(0);
        const edofArray = level.edofArray;
        const KEflat = this.KEflat;

        for (let ai = 0; ai < aCount; ai++) {
            const e = active[ai];
            const E = E_vals[e];
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) {
                diag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
            }
        }

        const invDiag = level.invDiag;
        invDiag.fill(0);
        const free = level.freeDofs;
        for (let i = 0; i < free.length; i++) {
            const dof = free[i];
            const d = diag[dof];
            invDiag[dof] = d > 1e-30 ? 1.0 / d : 0.0;
        }
    }

    /**
     * Apply A = K(x) to vector p (full-space, fixed dofs are expected to be zero).
     */
    applyA(levelIdx, p, Ap) {
        const level = this.levels[levelIdx];
        // Dense Galerkin matvec for coarse levels with assembled operator
        if (level.denseK) {
            const K = level.denseK, n = level.ndof;
            Ap.fill(0);
            for (let i = 0; i < n; i++) {
                let s = 0; const off = i * n;
                for (let j = 0; j < n; j++) s += K[off + j] * p[j];
                Ap[i] = s;
            }
            return;
        }

        // WASM-accelerated element-by-element matvec
        if (this._wasmReady && level._wasm) {
            const mem = wasmModule.exports.memory;
            const w = level._wasm;
            // Copy input vector p into WASM memory
            new Float64Array(mem.buffer, w.pOff, level.ndof).set(p.subarray(0, level.ndof));
            // Call WASM kernel
            wasmModule.exports.applyAEbe3D(
                this._wasmKeOff, w.edofsOff, w.evalsOff, w.activeOff, level.activeCount,
                w.pOff, w.apOff, level.ndof, this._wasmScratchOff
            );
            // Read result back
            Ap.set(new Float64Array(mem.buffer, w.apOff, level.ndof));
            return;
        }

        // Fallback: JS element-by-element matvec
        Ap.fill(0);

        const KEflat = this.KEflat;
        const edofArray = level.edofArray;
        const E_vals = level.E_vals;
        const active = level.active;
        const aCount = level.activeCount;

        const loc = this._tmpLocal;

        for (let ai = 0; ai < aCount; ai++) {
            const e = active[ai];
            const E = E_vals[e];
            const eOff = e * 24;

            // Gather local dofs once (reduces global memory traffic)
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

    /**
     * V-cycle: approximately solve A x = b.
     */
    vcycle(levelIdx, b, x) {
        const level = this.levels[levelIdx];
        const Au = level.Au;
        const invDiag = level.invDiag;
        const free = level.freeDofs;

        // Pre-smoothing: damped Jacobi
        for (let s = 0; s < MG_NU1; s++) {
            this.applyA(levelIdx, x, Au);
            for (let ii = 0; ii < free.length; ii++) {
                const dof = free[ii];
                const r = b[dof] - Au[dof];
                x[dof] += MG_OMEGA * invDiag[dof] * r;
            }
        }

        // Residual r = b - A x
        this.applyA(levelIdx, x, Au);
        const res = level.res;
        res.fill(0);
        for (let ii = 0; ii < free.length; ii++) {
            const dof = free[ii];
            res[dof] = b[dof] - Au[dof];
        }

        // Coarsest level: extra smoothing iterations
        if (levelIdx === this.levels.length - 1) {
            for (let s = 0; s < MG_COARSE_ITERS; s++) {
                this.applyA(levelIdx, x, Au);
                for (let ii = 0; ii < free.length; ii++) {
                    const dof = free[ii];
                    const r = b[dof] - Au[dof];
                    x[dof] += MG_OMEGA * invDiag[dof] * r;
                }
            }
            return;
        }

        // Restrict residual to coarse RHS (full-weighting = P^T)
        const coarse = this.levels[levelIdx + 1];
        const bC = coarse.b;
        bC.fill(0);

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
        // Zero fixed DOFs on coarse grid
        const fixedC = coarse.fixedMask;
        for (let i = 0; i < coarse.ndof; i++) if (fixedC[i]) bC[i] = 0;

        // Solve on coarse: recurse
        const xC = coarse.x;
        xC.fill(0);
        this.vcycle(levelIdx + 1, bC, xC);

        // Prolongate and correct fine solution (trilinear interpolation)
        const fixedF = level.fixedMask;
        for (let fz = 0; fz < nzF; fz++) {
            const cz0 = Math.min(fz >> 1, nzC - 1);
            const cz1 = Math.min(cz0 + 1, nzC - 1);
            const tz = (cz0 !== cz1 && (fz & 1)) ? 0.5 : 0.0;
            const wz0 = 1.0 - tz, wz1 = tz;

            for (let fy = 0; fy < nyF; fy++) {
                const cy0 = Math.min(fy >> 1, nyC - 1);
                const cy1 = Math.min(cy0 + 1, nyC - 1);
                const ty = (cy0 !== cy1 && (fy & 1)) ? 0.5 : 0.0;
                const wy0 = 1.0 - ty, wy1 = ty;

                for (let fx = 0; fx < nxF; fx++) {
                    const cx0 = Math.min(fx >> 1, nxC - 1);
                    const cx1 = Math.min(cx0 + 1, nxC - 1);
                    const tx = (cx0 !== cx1 && (fx & 1)) ? 0.5 : 0.0;
                    const wx0 = 1.0 - tx, wx1 = tx;

                    const nF = fx * nyF * nzF + fy * nzF + fz;

                    const n000 = cx0 * nyC * nzC + cy0 * nzC + cz0;
                    const n100 = cx1 * nyC * nzC + cy0 * nzC + cz0;
                    const n010 = cx0 * nyC * nzC + cy1 * nzC + cz0;
                    const n110 = cx1 * nyC * nzC + cy1 * nzC + cz0;
                    const n001 = cx0 * nyC * nzC + cy0 * nzC + cz1;
                    const n101 = cx1 * nyC * nzC + cy0 * nzC + cz1;
                    const n011 = cx0 * nyC * nzC + cy1 * nzC + cz1;
                    const n111 = cx1 * nyC * nzC + cy1 * nzC + cz1;

                    const w000 = wx0 * wy0 * wz0;
                    const w100 = wx1 * wy0 * wz0;
                    const w010 = wx0 * wy1 * wz0;
                    const w110 = wx1 * wy1 * wz0;
                    const w001 = wx0 * wy0 * wz1;
                    const w101 = wx1 * wy0 * wz1;
                    const w011 = wx0 * wy1 * wz1;
                    const w111 = wx1 * wy1 * wz1;

                    const baseF = 3 * nF;
                    if (!fixedF[baseF]) {
                        x[baseF] +=
                            xC[3 * n000] * w000 + xC[3 * n100] * w100 + xC[3 * n010] * w010 + xC[3 * n110] * w110 +
                            xC[3 * n001] * w001 + xC[3 * n101] * w101 + xC[3 * n011] * w011 + xC[3 * n111] * w111;
                    }
                    if (!fixedF[baseF + 1]) {
                        x[baseF + 1] +=
                            xC[3 * n000 + 1] * w000 + xC[3 * n100 + 1] * w100 + xC[3 * n010 + 1] * w010 + xC[3 * n110 + 1] * w110 +
                            xC[3 * n001 + 1] * w001 + xC[3 * n101 + 1] * w101 + xC[3 * n011 + 1] * w011 + xC[3 * n111 + 1] * w111;
                    }
                    if (!fixedF[baseF + 2]) {
                        x[baseF + 2] +=
                            xC[3 * n000 + 2] * w000 + xC[3 * n100 + 2] * w100 + xC[3 * n010 + 2] * w010 + xC[3 * n110 + 2] * w110 +
                            xC[3 * n001 + 2] * w001 + xC[3 * n101 + 2] * w101 + xC[3 * n011 + 2] * w011 + xC[3 * n111 + 2] * w111;
                    }
                }
            }
        }

        // Post-smoothing
        for (let s = 0; s < MG_NU2; s++) {
            this.applyA(levelIdx, x, Au);
            for (let ii = 0; ii < free.length; ii++) {
                const dof = free[ii];
                const r = b[dof] - Au[dof];
                x[dof] += MG_OMEGA * invDiag[dof] * r;
            }
        }
    }

    /**
     * Apply the multigrid preconditioner: z = A^{-1} r (approximately)
     */
    apply(r, z) {
        z.fill(0);
        this.vcycle(0, r, z);
    }
}

// ─── KSP Solver parameters (PETSc-style) ─────────────────────────────────────
// BDDC (Balancing Domain Decomposition by Constraints) configuration
const KSP_BDDC_OVERLAP = 1;            // subdomain overlap layers
const KSP_BDDC_COARSE_ITERS = 30;      // iterations for coarse-level solve
const KSP_BDDC_SMOOTHER_ITERS = 3;     // local Jacobi smoothing steps per apply
const KSP_BDDC_MIN_SUBDOMAINS = 2;     // minimum number of subdomains per axis

/**
 * PETSc-style KSP (Krylov Subspace) solver with pluggable preconditioners.
 *
 * Supports:
 *  - PCBDDC: Balancing Domain Decomposition by Constraints
 *  - PCMG:   Multigrid (delegates to MGPrecond3D)
 *  - PCJACOBI: Diagonal (Jacobi) preconditioner fallback
 *
 * The solver uses Preconditioned Conjugate Gradient (PCG) as the default
 * Krylov method, matching PETSc's KSPCG for symmetric positive-definite systems.
 */
class KSPSolver {
    /**
     * @param {Float64Array} KEflat - Flattened 24×24 element stiffness matrix
     * @param {object} [options]
     * @param {string} [options.pc='bddc'] - Preconditioner: 'bddc', 'mg', or 'jacobi'
     */
    constructor(KEflat, options = {}) {
        this.KEflat = KEflat;
        this.pcType = options.pc || 'bddc';

        // BDDC state
        this._subdomains = null;
        this._coarseMap = null;
        this._localInvDiag = null;

        // MG delegate (when pcType === 'mg')
        this._mg = null;

        // Grid dimensions (set during ensure())
        this._nelx = 0;
        this._nely = 0;
        this._nelz = 0;
        this._ndof = 0;

        // Cached operator data
        this._E_vals = null;
        this._activeElements = null;
        this._edofArray = null;
        this._fixedMask = null;
        this._freedofs = null;
        this._invDiag = null;

        // Work buffers
        this._Ap_buf = null;
    }

    /**
     * Ensure the solver hierarchy is built for the given grid.
     * Analogous to KSPSetUp in PETSc.
     */
    ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs) {
        if (this._nelx === nelx && this._nely === nely && this._nelz === nelz) return;

        this._nelx = nelx;
        this._nely = nely;
        this._nelz = nelz;
        this._ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        this._edofArray = edofArray;
        this._fixedMask = fixedMask;
        this._freedofs = freedofs;
        this._Ap_buf = new Float64Array(this._ndof);

        if (this.pcType === 'mg') {
            if (!this._mg) this._mg = new MGPrecond3D(this.KEflat);
            this._mg.ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs);
        } else if (this.pcType === 'bddc') {
            this._buildSubdomains(nelx, nely, nelz);
        }
    }

    /**
     * Update operators from current density field.
     * Analogous to updating the matrix in PETSc before KSPSolve.
     */
    updateOperators(x, penal, E0, Emin) {
        const nel = this._nelx * this._nely * this._nelz;
        const dE = E0 - Emin;

        // Build E_vals and active element list
        if (!this._E_vals || this._E_vals.length !== nel) {
            this._E_vals = new Float64Array(nel);
        }
        const E_vals = this._E_vals;
        const active = [];
        const skipThreshold = Emin * 1000;

        for (let e = 0; e < nel; e++) {
            const E = Emin + Math.pow(x[e], penal) * dE;
            E_vals[e] = E;
            if (E > skipThreshold) active.push(e);
        }
        this._activeElements = active;

        // Build global inverse diagonal (Jacobi)
        if (!this._invDiag || this._invDiag.length !== this._ndof) {
            this._invDiag = new Float64Array(this._ndof);
        }
        const diag = new Float64Array(this._ndof);
        const KEflat = this.KEflat;
        const edofArray = this._edofArray;
        for (let ai = 0; ai < active.length; ai++) {
            const e = active[ai];
            const E = E_vals[e];
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) {
                diag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
            }
        }
        const invDiag = this._invDiag;
        invDiag.fill(0);
        for (let i = 0; i < this._ndof; i++) {
            if (!this._fixedMask[i] && diag[i] > 1e-30) {
                invDiag[i] = 1.0 / diag[i];
            }
        }

        // Update preconditioner
        if (this.pcType === 'mg' && this._mg) {
            this._mg.updateFromFine(x, penal, E0, Emin);
        } else if (this.pcType === 'bddc' && this._subdomains) {
            this._updateBDDC(diag);
        }
    }

    /**
     * Partition the grid into non-overlapping subdomains for BDDC.
     * Each subdomain is a contiguous block of elements along each axis.
     */
    _buildSubdomains(nelx, nely, nelz) {
        // Determine number of subdomains per axis (target ~8-64 total subdomains)
        const totalElements = nelx * nely * nelz;
        const targetSubdomains = Math.max(8, Math.min(64, Math.ceil(Math.cbrt(totalElements / 100))));
        const nsx = Math.max(KSP_BDDC_MIN_SUBDOMAINS, Math.min(nelx, Math.round(Math.cbrt(targetSubdomains * (nelx * nelx) / (nely * nelz)))));
        const nsy = Math.max(KSP_BDDC_MIN_SUBDOMAINS, Math.min(nely, Math.round(Math.cbrt(targetSubdomains * (nely * nely) / (nelx * nelz)))));
        const nsz = Math.max(KSP_BDDC_MIN_SUBDOMAINS, Math.min(nelz, Math.round(Math.cbrt(targetSubdomains * (nelz * nelz) / (nelx * nely)))));

        const subdomains = [];
        for (let sz = 0; sz < nsz; sz++) {
            const z0 = Math.floor(sz * nelz / nsz);
            const z1 = Math.floor((sz + 1) * nelz / nsz);
            for (let sy = 0; sy < nsy; sy++) {
                const y0 = Math.floor(sy * nely / nsy);
                const y1 = Math.floor((sy + 1) * nely / nsy);
                for (let sx = 0; sx < nsx; sx++) {
                    const x0 = Math.floor(sx * nelx / nsx);
                    const x1 = Math.floor((sx + 1) * nelx / nsx);
                    const elements = [];
                    for (let ez = z0; ez < z1; ez++) {
                        for (let ey = y0; ey < y1; ey++) {
                            for (let ex = x0; ex < x1; ex++) {
                                elements.push(ex + ey * nelx + ez * nelx * nely);
                            }
                        }
                    }
                    if (elements.length > 0) {
                        subdomains.push({
                            elements,
                            x0, x1, y0, y1, z0, z1,
                            // DOFs touched by this subdomain (set in _updateBDDC)
                            dofs: null,
                            // Local inverse diagonal for subdomain Jacobi smoothing
                            localInvDiag: null
                        });
                    }
                }
            }
        }
        this._subdomains = subdomains;

        // Build coarse space: one constraint DOF per subdomain vertex (corner nodes)
        // These are the "primal" DOFs in BDDC terminology
        const coarseDofSet = new Set();
        const nny = nely + 1;
        const nnz = nelz + 1;
        for (const sub of subdomains) {
            // Corner nodes of this subdomain block
            const corners = [
                [sub.x0, sub.y0, sub.z0], [sub.x1, sub.y0, sub.z0],
                [sub.x0, sub.y1, sub.z0], [sub.x1, sub.y1, sub.z0],
                [sub.x0, sub.y0, sub.z1], [sub.x1, sub.y0, sub.z1],
                [sub.x0, sub.y1, sub.z1], [sub.x1, sub.y1, sub.z1],
            ];
            for (const [cx, cy, cz] of corners) {
                const clampedX = Math.min(cx, nelx);
                const clampedY = Math.min(cy, nely);
                const clampedZ = Math.min(cz, nelz);
                const nodeIdx = clampedX * nny * nnz + clampedY * nnz + clampedZ;
                coarseDofSet.add(3 * nodeIdx);
                coarseDofSet.add(3 * nodeIdx + 1);
                coarseDofSet.add(3 * nodeIdx + 2);
            }
        }
        this._coarseMap = [...coarseDofSet].sort((a, b) => a - b);
    }

    /**
     * Update BDDC preconditioner with current diagonal.
     */
    _updateBDDC(globalDiag) {
        if (!this._subdomains) return;

        const edofArray = this._edofArray;
        const fixedMask = this._fixedMask;
        const E_vals = this._E_vals;
        const KEflat = this.KEflat;
        const ndof = this._ndof;

        for (const sub of this._subdomains) {
            // Collect DOFs for this subdomain
            const dofSet = new Set();
            for (const e of sub.elements) {
                const eOff = e * 24;
                for (let i = 0; i < 24; i++) {
                    dofSet.add(edofArray[eOff + i]);
                }
            }
            sub.dofs = [...dofSet].sort((a, b) => a - b);

            // Build local inverse diagonal for subdomain Jacobi smoothing
            const localDiag = new Float64Array(ndof);
            for (const e of sub.elements) {
                const E = E_vals[e];
                if (E < 1e-30) continue;
                const eOff = e * 24;
                for (let i = 0; i < 24; i++) {
                    localDiag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
                }
            }
            const localInvDiag = new Float64Array(ndof);
            for (const d of sub.dofs) {
                if (!fixedMask[d] && localDiag[d] > 1e-30) {
                    localInvDiag[d] = 1.0 / localDiag[d];
                }
            }
            sub.localInvDiag = localInvDiag;
        }
    }

    /**
     * Apply the global stiffness operator K*p → Ap (element-by-element).
     */
    applyA(p, Ap) {
        Ap.fill(0);
        const KEflat = this.KEflat;
        const edofArray = this._edofArray;
        const E_vals = this._E_vals;
        const active = this._activeElements;
        const loc = new Float64Array(24);

        for (let ai = 0; ai < active.length; ai++) {
            const e = active[ai];
            const E = E_vals[e];
            const eOff = e * 24;
            for (let j = 0; j < 24; j++) loc[j] = p[edofArray[eOff + j]];
            for (let i = 0; i < 24; i++) {
                const gi = edofArray[eOff + i];
                let sum = 0;
                const row = i * 24;
                for (let j = 0; j < 24; j++) sum += KEflat[row + j] * loc[j];
                Ap[gi] += E * sum;
            }
        }
    }

    /**
     * Apply BDDC preconditioner: z ≈ K⁻¹ r.
     *
     * BDDC algorithm:
     *  1. Local subdomain solves (additive Schwarz with Jacobi smoothing)
     *  2. Coarse-level correction via primal (corner) DOFs
     *  3. Weighted assembly of local + coarse corrections
     */
    _applyBDDC(r, z) {
        const ndof = this._ndof;
        const fixedMask = this._fixedMask;

        z.fill(0);

        // Step 1: Additive Schwarz — local subdomain Jacobi smoothing
        // Each subdomain applies local inverse diagonal to the residual
        const localCorrection = new Float64Array(ndof);
        const weight = new Float64Array(ndof);

        for (const sub of this._subdomains) {
            const invD = sub.localInvDiag;
            for (const d of sub.dofs) {
                if (!fixedMask[d]) {
                    localCorrection[d] += invD[d] * r[d];
                    weight[d] += 1.0;
                }
            }
        }

        // Average contributions from overlapping subdomain solves
        for (let i = 0; i < ndof; i++) {
            if (weight[i] > 0) {
                localCorrection[i] /= weight[i];
            }
        }

        // Apply additional smoothing sweeps for better local approximation
        const Ap = this._Ap_buf;
        for (let sweep = 0; sweep < KSP_BDDC_SMOOTHER_ITERS - 1; sweep++) {
            this.applyA(localCorrection, Ap);
            for (const sub of this._subdomains) {
                const invD = sub.localInvDiag;
                for (const d of sub.dofs) {
                    if (!fixedMask[d]) {
                        localCorrection[d] += invD[d] * (r[d] - Ap[d]);
                    }
                }
            }
        }

        // Step 2: Coarse-level correction via primal DOFs
        // The coarse space enforces continuity at subdomain corner nodes.
        // Apply Jacobi correction restricted to the primal (corner) DOFs.
        const invDiag = this._invDiag;
        const coarseCorrection = new Float64Array(ndof);
        this.applyA(localCorrection, Ap);
        const coarseMap = this._coarseMap;
        for (let ci = 0; ci < coarseMap.length; ci++) {
            const i = coarseMap[ci];
            if (fixedMask[i]) continue;
            const coarseRes = r[i] - Ap[i];
            coarseCorrection[i] = invDiag[i] * coarseRes;
        }

        // Step 3: Combine local and coarse corrections
        for (let i = 0; i < ndof; i++) {
            z[i] = localCorrection[i] + coarseCorrection[i];
        }

        // Zero fixed DOFs
        for (let i = 0; i < ndof; i++) {
            if (fixedMask[i]) z[i] = 0;
        }
    }

    /**
     * Apply the selected preconditioner: z ≈ K⁻¹ r.
     * Dispatches to the appropriate PC implementation.
     */
    apply(r, z) {
        if (this.pcType === 'mg' && this._mg) {
            this._mg.apply(r, z);
        } else if (this.pcType === 'bddc' && this._subdomains) {
            this._applyBDDC(r, z);
        } else {
            // PCJACOBI fallback
            const invDiag = this._invDiag;
            for (let i = 0; i < this._ndof; i++) {
                z[i] = invDiag[i] * r[i];
            }
        }
    }
}

class TopologyOptimizerWorker3D {
    constructor() {
        this.rmin = 1.5;
        this.penal = 3;
        this.E0 = 1;
        this.Emin = 1e-9;
        this.nu = 0.3;
        this.cancelled = false;
        this.useWasm = false;
        // Pause / resume state
        this._paused = false;
        this._resumeResolve = null;
        // Pending config update (applied at next iteration boundary)
        this._pendingConfig = null;

        // Linear solve state (warm-start + reusable buffers)
        this.linearSolver = DEFAULT_LINEAR_SOLVER;
        this._mg = null;           // MGPrecond3D instance (built lazily)
        this._ksp = null;          // KSPSolver instance (built lazily for 'petsc')
        this._U_prev = null;       // previous displacement (warm start)
        this._pcg_r = null;
        this._pcg_z = null;
        this._pcg_p = null;
        this._pcg_Ap = null;
        this._pcg_Au = null;
        this._U = null;

        // Optional WebGPU acceleration for element-energy kernel
        this.useGPU = false;
        this.gpuCompute = null;
        this._KEflat32 = null;

        // FEA solver backend selection (auto/webgpu/wasm/js)
        this.feaSolverBackend = DEFAULT_FEA_BACKEND;
        this._resolvedBackend = null; // actual backend after auto-detection
        this.gpuFEASolver = null;     // GPUFEASolver instance (lazy init)
        this._gpuFEAInitAttempted = false;
        this._gpuFEAAvailable = false;

        // Volumetric stress snapshot state
        this._volumetricOutputMode = 'on-stop';
        this._lastVolumetricStress = null;
        this._lastVolumetricMaxStress = 0;
        this._lastVolumetricIteration = 0;
        this._lastVolumetricDims = null;
    }

    _flattenKE(KE, size) {
        const flat = new Float64Array(size * size);
        for (let i = 0; i < size; i++)
            for (let j = 0; j < size; j++)
                flat[i * size + j] = KE[i][j];
        return flat;
    }

    _precomputeEdofs3D(nelx, nely, nelz) {
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
                    edofArray[offset]      = 3 * n0;
                    edofArray[offset + 1]  = 3 * n0 + 1;
                    edofArray[offset + 2]  = 3 * n0 + 2;
                    edofArray[offset + 3]  = 3 * n1;
                    edofArray[offset + 4]  = 3 * n1 + 1;
                    edofArray[offset + 5]  = 3 * n1 + 2;
                    edofArray[offset + 6]  = 3 * n2;
                    edofArray[offset + 7]  = 3 * n2 + 1;
                    edofArray[offset + 8]  = 3 * n2 + 2;
                    edofArray[offset + 9]  = 3 * n3;
                    edofArray[offset + 10] = 3 * n3 + 1;
                    edofArray[offset + 11] = 3 * n3 + 2;
                    edofArray[offset + 12] = 3 * n4;
                    edofArray[offset + 13] = 3 * n4 + 1;
                    edofArray[offset + 14] = 3 * n4 + 2;
                    edofArray[offset + 15] = 3 * n5;
                    edofArray[offset + 16] = 3 * n5 + 1;
                    edofArray[offset + 17] = 3 * n5 + 2;
                    edofArray[offset + 18] = 3 * n6;
                    edofArray[offset + 19] = 3 * n6 + 1;
                    edofArray[offset + 20] = 3 * n6 + 2;
                    edofArray[offset + 21] = 3 * n7;
                    edofArray[offset + 22] = 3 * n7 + 1;
                    edofArray[offset + 23] = 3 * n7 + 2;
                }
            }
        }
        return edofArray;
    }

    _precomputeStiffness(x, penal, nel) {
        const E_vals = new Float64Array(nel);
        const E0 = this.E0;
        const Emin = this.Emin;
        const dE = E0 - Emin;
        const activeElements = [];
        const skipThreshold = Emin * 1000;
        for (let e = 0; e < nel; e++) {
            const E = Emin + Math.pow(x[e], penal) * dE;
            E_vals[e] = E;
            if (E > skipThreshold) {
                activeElements.push(e);
            }
        }
        return { E_vals, activeElements };
    }

    _ebeMatVec(E_vals, activeElements, KEflat, edofArray, nel, edofSize, p_reduced, Ap_reduced, freedofs, ndof) {
        const p_full = this._p_full;
        const Ap_full = this._Ap_full;

        p_full.fill(0);
        for (let i = 0, len = freedofs.length; i < len; i++) {
            p_full[freedofs[i]] = p_reduced[i];
        }

        Ap_full.fill(0);
        for (let ae = 0, aeLen = activeElements.length; ae < aeLen; ae++) {
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

        for (let i = 0, len = freedofs.length; i < len; i++) {
            Ap_reduced[i] = Ap_full[freedofs[i]];
        }
    }

    _computeDiagonal(E_vals, activeElements, KEflat, edofArray, nel, edofSize, freedofs, ndof) {
        const diag = new Float64Array(ndof);
        for (let ae = 0, aeLen = activeElements.length; ae < aeLen; ae++) {
            const e = activeElements[ae];
            const E = E_vals[e];
            const eOff = e * edofSize;
            for (let i = 0; i < edofSize; i++) {
                diag[edofArray[eOff + i]] += E * KEflat[i * edofSize + i];
            }
        }
        const invDiag = new Float64Array(freedofs.length);
        for (let i = 0, len = freedofs.length; i < len; i++) {
            const d = diag[freedofs[i]];
            invDiag[i] = d > 1e-30 ? 1.0 / d : 0.0;
        }
        return invDiag;
    }

    _computeElementEnergyFlat(KEflat, Ue, size) {
        let energy = 0;
        for (let i = 0; i < size; i++) {
            const keRow = i * size;
            const ui = Ue[i];
            for (let j = 0; j < size; j++) {
                energy += ui * KEflat[keRow + j] * Ue[j];
            }
        }
        return energy;
    }

    _normalizeVolumetricOutputMode(mode) {
        if (mode === 'every-iteration') return 'every-iteration';
        if (mode === 'off') return 'off';
        return 'on-stop';
    }

    _buildVolumetricSnapshot(iteration) {
        if (!this._lastVolumetricStress || !this._lastVolumetricDims) return null;
        const stressCopy = new Float32Array(this._lastVolumetricStress);
        return {
            payload: {
                nx: this._lastVolumetricDims.nx,
                ny: this._lastVolumetricDims.ny,
                nz: this._lastVolumetricDims.nz,
                iteration,
                maxStress: this._lastVolumetricMaxStress,
                stress: stressCopy
            },
            transfer: [stressCopy.buffer]
        };
    }

    postLatestVolumetricSnapshot(reason = 'on-demand') {
        const iteration = this._lastVolumetricIteration || 0;
        const snapshot = this._buildVolumetricSnapshot(iteration);
        if (snapshot) {
            postMessage({ type: 'volumetric', reason, volumetricData: snapshot.payload }, snapshot.transfer);
        } else {
            postMessage({ type: 'volumetric', reason, volumetricData: null });
        }
    }

    async optimize(model, config) {
        // Try to load WASM module if not already loaded
        if (!wasmLoaded && !this.wasmLoadAttempted) {
            this.wasmLoadAttempted = true;
            await loadWasmModule();
            this.useWasm = wasmLoaded;
        }

        // Route to genetic optimizer if requested
        if (config.optimizer === 'genetic') {
            return this.geneticOptimize(model, config);
        }
        
        const { nx, ny, nz } = model;
        const nelx = nx;
        const nely = ny;
        const nelz = nz;
        // Declared as `let` so updateConfig() can mutate them during optimization
        let volfrac = config.volumeFraction;
        let maxIterations = config.maxIterations;

        this.penal = config.penaltyFactor;
        this.rmin = config.filterRadius;
        this.cancelled = false;
        this._paused = false;
        this.useGPU = !!config.useGPU;
        this._volumetricOutputMode = this._normalizeVolumetricOutputMode(config.volumetricOutputMode);
        this._lastVolumetricStress = null;
        this._lastVolumetricMaxStress = 0;
        this._lastVolumetricIteration = 0;
        this._lastVolumetricDims = { nx: nelx, ny: nely, nz: nelz };
        // Allow selecting the linear solver via config (e.g. 'mgpcg' or 'cg')
        if (config.linearSolver) this.linearSolver = config.linearSolver;
        // FEA solver backend selection (auto/webgpu/wasm/js)
        if (config.feaSolverBackend) this.feaSolverBackend = config.feaSolverBackend;

        // Apply material properties from config if provided
        if (config.youngsModulus) {
            this.E0 = config.youngsModulus;
        }
        if (config.poissonsRatio !== undefined) {
            this.nu = config.poissonsRatio;
        }

        const nel = nelx * nely * nelz;

        // ── FEA-only mode ─────────────────────────────────────────────────
        if (config.feaOnly) {
            const x = new Float32Array(nel);
            if (model.elements) {
                for (let i = 0; i < nel; i++) {
                    x[i] = model.elements[i] > 0.5 ? 1.0 : this.Emin;
                }
            } else {
                x.fill(1.0);
            }

            let fixeddofs = this.getFixedDOFs(nelx, nely, nelz, config.constraintPosition, config.constraintDOFs);
            let F = this.getLoadVector(nelx, nely, nelz, config.forceDirection, config.forceMagnitude, config.forceVector);

            if (config.paintedConstraints && config.paintedConstraints.length > 0) {
                fixeddofs = this.getFixedDOFsFromPaint(nelx, nely, nelz, config.paintedConstraints, config.constraintDOFs);
            }
            if (config.paintedForces && config.paintedForces.length > 0) {
                F = this.getLoadVectorFromPaint(nelx, nely, nelz, config.paintedForces, config.forceDirection, config.forceMagnitude, config.forceVector);
            }

            // ── Nonlinear FEA sub-mode ────────────────────────────────────
            if (config.nonlinearMode) {
                const startTime = performance.now();
                const nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
                const nodeCount = nnx * nny * nnz;
                const elemCount = nel;

                // Build mesh descriptor for NonlinearSolver
                const mesh = {
                    nelx, nely, nelz,
                    nodeCount,
                    elemCount,
                    getElementNodes: (e) => {
                        // 8 corner nodes for hex element e
                        const ez = Math.floor(e / (nelx * nely));
                        const ey = Math.floor((e % (nelx * nely)) / nelx);
                        const ex = e % nelx;
                        const n = (iz, iy, ix) => iz * nny * nnx + iy * nnx + ix;
                        return [
                            n(ez,   ey,   ex),   n(ez,   ey,   ex+1),
                            n(ez,   ey+1, ex+1), n(ez,   ey+1, ex),
                            n(ez+1, ey,   ex),   n(ez+1, ey,   ex+1),
                            n(ez+1, ey+1, ex+1), n(ez+1, ey+1, ex)
                        ];
                    },
                    getNodeCoords: (n) => {
                        const nz = Math.floor(n / (nny * nnx));
                        const ny = Math.floor((n % (nny * nnx)) / nnx);
                        const nx = n % nnx;
                        return [nx, ny, nz];
                    }
                };

                // Create material model
                const E = config.youngsModulus || this.E0;
                const nu = config.poissonsRatio !== undefined ? config.poissonsRatio : this.nu;
                const sigma_y = config.yieldStrength || 0;
                const matType = sigma_y > 0 ? 'j2-plasticity' : 'neo-hookean';
                const material = createMaterial(matType, { E, nu, sigY: sigma_y, H: 0 });

                // Build full-DOF load vector (Float64)
                const loads = new Float64Array(nodeCount * 3);
                for (let i = 0; i < Math.min(F.length, loads.length); i++) loads[i] = F[i];

                // Fixed DOFs as Int32Array
                const constraints = new Int32Array(fixeddofs.length);
                for (let i = 0; i < fixeddofs.length; i++) constraints[i] = fixeddofs[i];

                // Run nonlinear solver
                const nlSolver = new NonlinearSolver({
                    numLoadSteps: config.nonlinearLoadSteps || 10,
                    maxNewtonIter: config.nonlinearMaxNewtonIter || 20,
                    residualTol: config.nonlinearTolerance || 1e-6,
                    incrementTol: config.nonlinearTolerance || 1e-6
                });

                let nlStep = 0;
                const nlResult = nlSolver.solve(mesh, material, constraints, loads, (info) => {
                    if (info.step !== nlStep) {
                        nlStep = info.step;
                        const progress = (info.step / (config.nonlinearLoadSteps || 10)) * 100;
                        postMessage({
                            type: 'progress',
                            iteration: info.step,
                            compliance: info.residualNorm || 0,
                            meshData: null,
                            timing: null,
                            maxStress: 0
                        });
                    }
                });

                // Extract von Mises stress per element
                const elementStress = nlResult.vonMisesStress;
                let maxStress = 0;
                for (let e = 0; e < elemCount; e++) {
                    if (elementStress[e] > maxStress) maxStress = elementStress[e];
                }

                const edofArray = this._precomputeEdofs3D(nelx, nely, nelz);
                const elementEnergies = nlResult.strainEnergy || new Float32Array(elemCount);
                const elementForces = this.computeElementForces(nelx, nely, nelz, F);
                const meshData = this.buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces, null);

                const displacementU = Float32Array.from(nlResult.displacement);
                const totalTime = performance.now() - startTime;

                this._lastVolumetricStress = Float32Array.from(elementStress);
                this._lastVolumetricMaxStress = maxStress;
                this._lastVolumetricIteration = 1;

                this._cleanupGPU();
                postMessage({
                    type: 'complete',
                    result: {
                        densities: x,
                        finalCompliance: 0,
                        iterations: nlResult.totalIterations || 0,
                        volumeFraction: 1.0,
                        nx: nelx,
                        ny: nely,
                        nz: nelz,
                        meshData,
                        maxStress,
                        elementStress: Float32Array.from(elementStress),
                        feaOnly: true,
                        nonlinearMode: true,
                        converged: nlResult.converged,
                        displacementU,
                        timing: {
                            totalTime,
                            avgIterationTime: totalTime,
                            iterationTimes: [totalTime],
                            usingWasm: false
                        }
                    }
                });
                return;
            }
            // ─────────────────────────────────────────────────────────────
            const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
            const fixedMask = new Uint8Array(ndof);
            for (let i = 0; i < fixeddofs.length; i++) {
                const dof = fixeddofs[i];
                if (dof >= 0 && dof < ndof) fixedMask[dof] = 1;
            }
            let nFree = 0;
            for (let i = 0; i < ndof; i++) if (!fixedMask[i]) nFree++;
            const freedofs = new Int32Array(nFree);
            let fp = 0;
            for (let i = 0; i < ndof; i++) if (!fixedMask[i]) freedofs[fp++] = i;

            const KE = this.lk();
            const KEflat = this._flattenKE(KE, 24);
            const edofArray = this._precomputeEdofs3D(nelx, nely, nelz);
            const elementForces = this.computeElementForces(nelx, nely, nelz, F);

            const penal = config.penaltyFactor || 3;
            const startTime = performance.now();
            const { U, c: compliance } = await this.FE(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, config, 1, 1, fixeddofs);

            const elementEnergies = new Float32Array(nel);
            const elementStress = new Float32Array(nel);
            const Ue = new Float64Array(24);
            let maxStress = 0;
            for (let e = 0; e < nel; e++) {
                const eOff = e * 24;
                for (let i = 0; i < 24; i++) {
                    Ue[i] = U[edofArray[eOff + i]];
                }
                const energy = this._computeElementEnergyFlat(KEflat, Ue, 24);
                elementEnergies[e] = energy;
                const stiffness = this.Emin + Math.pow(x[e], penal) * (this.E0 - this.Emin);
                elementStress[e] = stiffness * energy;
                if (elementStress[e] > maxStress) maxStress = elementStress[e];
            }

            // Fatigue risk computation (S-N curve approach)
            // When fatigueMode is enabled, replace stress values with fatigue risk (0=safe, 1=failure)
            // Uses S-N curve (Wöhler curve) approach: risk ∝ σ^k where k is the S-N slope exponent.
            // k=3 is a conservative default valid for many metals and engineering polymers
            // (typical range: k=3-5 for metals, k=9-12 for composites).
            let displayStress = elementStress;
            let displayMaxStress = maxStress;
            if (config.fatigueMode && maxStress > 0) {
                const k = config.snExponent || 3; // S-N slope exponent (configurable, default 3)
                const fatigueRisk = new Float32Array(nel);
                let maxRisk = 0;
                for (let e = 0; e < nel; e++) {
                    const s = elementStress[e] / maxStress; // normalized 0-1
                    fatigueRisk[e] = Math.pow(s, k);
                    if (fatigueRisk[e] > maxRisk) maxRisk = fatigueRisk[e];
                }
                // Re-normalize fatigueRisk to 0-1 range
                if (maxRisk > 0) {
                    for (let e = 0; e < nel; e++) fatigueRisk[e] /= maxRisk;
                }
                displayStress = fatigueRisk;
                displayMaxStress = 1.0;
            }

            // Downsample displacement vector from Float64 to Float32 for efficient transfer
            const displacementU = Float32Array.from(U);

            const meshData = this.buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces, null);
            const totalTime = performance.now() - startTime;

            this._lastVolumetricStress = displayStress;
            this._lastVolumetricMaxStress = displayMaxStress;
            this._lastVolumetricIteration = 1;

            this._cleanupGPU();
            postMessage({
                type: 'complete',
                result: {
                    densities: x,
                    finalCompliance: compliance,
                    iterations: 0,
                    volumeFraction: 1.0,
                    nx: nelx,
                    ny: nely,
                    nz: nelz,
                    meshData: meshData,
                    maxStress: displayMaxStress,
                    elementStress: displayStress,
                    feaOnly: true,
                    fatigueMode: !!config.fatigueMode,
                    displacementU: displacementU,
                    timing: {
                        totalTime: totalTime,
                        avgIterationTime: totalTime,
                        iterationTimes: [totalTime],
                        usingWasm: this.useWasm
                    }
                }
            });
            return;
        }
        
        // Initialize AMR manager if enabled
        const amrManager = config.useAMR ? 
            new SimpleAMRManager3D(nelx, nely, nelz, true, config.minGranuleSize, config.maxGranuleSize, config.amrInterval) : 
            null;

        let x = new Float32Array(nel);
        x.fill(volfrac);
        let xnew = new Float32Array(nel);
        let xold = new Float32Array(nel);

        // For blended-curvature meshes, use the model's element densities as
        // the initial density distribution so the solver starts from the
        // curvature-adaptive surface representation.
        const isBlendedCurvature = model.meshType === 'blended-curvature';
        if (isBlendedCurvature && model.elements) {
            for (let i = 0; i < nel; i++) {
                x[i] = model.elements[i] > 0 ? Math.max(volfrac, model.elements[i]) : 0;
            }
        }

        const { H, Hs } = this.prepareFilter(nelx, nely, nelz, this.rmin);
        let fixeddofs = this.getFixedDOFs(nelx, nely, nelz, config.constraintPosition, config.constraintDOFs);
        let F = this.getLoadVector(nelx, nely, nelz, config.forceDirection, config.forceMagnitude, config.forceVector);

        // Apply painted constraints (override dropdown if painted faces exist)
        if (config.paintedConstraints && config.paintedConstraints.length > 0) {
            fixeddofs = this.getFixedDOFsFromPaint(nelx, nely, nelz, config.paintedConstraints, config.constraintDOFs);
        }

        // Apply painted forces (override dropdown if painted faces exist)
        if (config.paintedForces && config.paintedForces.length > 0) {
            F = this.getLoadVectorFromPaint(nelx, nely, nelz, config.paintedForces, config.forceDirection, config.forceMagnitude, config.forceVector);
        }

        // Build set of element indices that must stay solid (constraint/force surfaces)
        const preservedElements = new Set();
        const allPaintedKeys = [
            ...(config.paintedConstraints || []),
            ...(config.paintedForces || []),
            ...(config.paintedKeep || [])
        ];
        for (const key of allPaintedKeys) {
            const parts = key.split(',');
            if (parts.length < 3) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            const vz = parseInt(parts[2], 10);
            if (!isNaN(vx) && !isNaN(vy) && !isNaN(vz) && 
                vx >= 0 && vx < nelx && vy >= 0 && vy < nely && vz >= 0 && vz < nelz) {
                preservedElements.add(vx + vy * nelx + vz * nelx * nely);
            }
        }
        // Also preserve elements along dropdown-selected constraint/force positions
        if ((!config.paintedConstraints || config.paintedConstraints.length === 0)) {
            const constraintElems = this.getConstraintElements(nelx, nely, nelz, config.constraintPosition);
            for (const idx of constraintElems) preservedElements.add(idx);
        }
        if ((!config.paintedForces || config.paintedForces.length === 0)) {
            const forceElems = this.getForceElements(nelx, nely, nelz, config.forceDirection);
            for (const idx of forceElems) preservedElements.add(idx);
        }

        // Initialize preserved elements to full density
        for (const idx of preservedElements) {
            x[idx] = 1.0;
        }

        // Build set of void element indices (elements outside initial solid space)
        const voidElements = new Set();
        if (config.constrainToSolid && model.elements) {
            for (let i = 0; i < nel; i++) {
                if (model.elements[i] < 0.5) {
                    voidElements.add(i);
                    x[i] = 0.0;
                }
            }
        }

        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);

        // Fixed / free DOF masks (TypedArrays to avoid huge JS arrays + Sets)
        const fixedMask = new Uint8Array(ndof);
        for (let i = 0; i < fixeddofs.length; i++) {
            const dof = fixeddofs[i];
            if (dof >= 0 && dof < ndof) fixedMask[dof] = 1;
        }

        // Build free DOF list (Int32Array)
        let nFree = 0;
        for (let i = 0; i < ndof; i++) if (!fixedMask[i]) nFree++;
        const freedofs = new Int32Array(nFree);
        let fp = 0;
        for (let i = 0; i < ndof; i++) if (!fixedMask[i]) freedofs[fp++] = i;

        // Convert Sets to fast masks for hot inner loops (OC + constraints)
        const preservedMask = new Uint8Array(nel);
        for (const idx of preservedElements) preservedMask[idx] = 1;

        const voidMask = new Uint8Array(nel);
        for (const idx of voidElements) voidMask[idx] = 1;

        const KE = this.lk();
        const KEflat = this._flattenKE(KE, 24);
        this._KEflat32 = this._KEflat32 && this._KEflat32.length === KEflat.length
            ? this._KEflat32
            : new Float32Array(KEflat.length);
        for (let i = 0; i < KEflat.length; i++) this._KEflat32[i] = KEflat[i];
        const edofArray = this._precomputeEdofs3D(nelx, nely, nelz);

        if (this.useGPU) {
            try {
                if (!this.gpuCompute) this.gpuCompute = new GPUCompute();
                const ok = await this.gpuCompute.init();
                if (!ok || !this.gpuCompute.isAvailable()) {
                    this.useGPU = false;
                }
            } catch (_gpuInitError) {
                this.useGPU = false;
            }
        }

        // ── Initialize GPU FEA solver if backend is 'auto' or 'webgpu' ──
        await this._initGPUFEASolver();
        this._resolvedBackend = this._resolveBackend();

        // Compute per-element force magnitudes for adaptive mesh info
        const elementForces = this.computeElementForces(nelx, nely, nelz, F);

        let loop = 0;
        let change = 1;
        let c = 0;
        let lastElementEnergies = null;

        // Penalization continuation: ramp penal from penalStart to penalTarget
        const penalTarget = this.penal;
        const penalStart = config.penalStart != null ? config.penalStart : Math.min(penalTarget, 1.5);
        const continuationIters = config.continuationIters != null ? config.continuationIters : Math.max(20, Math.floor(maxIterations / 3));

        // Heaviside projection parameters
        const useProjection = config.useProjection !== false;
        const betaMax = config.betaMax != null ? config.betaMax : 64;
        const betaInterval = config.betaInterval != null ? config.betaInterval : 5;

        // Benchmark timing
        const iterationTimes = [];
        let iterTimeSum = 0;
        const startTime = performance.now();

        // UI update cadence (building & transferring mesh data can dominate runtime for large 3D grids)
        const uiUpdateInterval = (config && config.uiUpdateInterval) ? config.uiUpdateInterval : (nel >= 100000 ? 5 : 1);

        while (change > 0.01 && loop < maxIterations) {
            // Yield to the event loop so pending messages (cancel/pause/updateConfig) are processed
            await _yieldToLoop();

            if (this.cancelled) {
                postMessage({ type: 'cancelled', iteration: loop });
                return;
            }

            // Apply pending config updates at the start of each iteration
            if (this._pendingConfig) {
                const patch = this._pendingConfig;
                this._pendingConfig = null;
                if (patch.penaltyFactor !== undefined) this.penal = patch.penaltyFactor;
                if (patch.filterRadius !== undefined) this.rmin = patch.filterRadius;
                if (patch.volumeFraction !== undefined) volfrac = patch.volumeFraction;
                if (patch.maxIterations !== undefined) maxIterations = patch.maxIterations;
                if (patch.linearSolver !== undefined) this.linearSolver = patch.linearSolver;
                if (patch.feaSolverBackend !== undefined) {
                    this.feaSolverBackend = patch.feaSolverBackend;
                    await this._initGPUFEASolver();
                    this._resolvedBackend = this._resolveBackend();
                }
                if (patch.volumetricOutputMode !== undefined) {
                    this._volumetricOutputMode = this._normalizeVolumetricOutputMode(patch.volumetricOutputMode);
                }
                if (patch.useGPU !== undefined) {
                    this.useGPU = !!patch.useGPU;
                    if (this.useGPU) {
                        try {
                            if (!this.gpuCompute) this.gpuCompute = new GPUCompute();
                            const ok = await this.gpuCompute.init();
                            if (!ok || !this.gpuCompute.isAvailable()) this.useGPU = false;
                        } catch (_gpuInitError) {
                            this.useGPU = false;
                        }
                    }
                }
            }

            // Pause between iterations if requested
            if (this._paused) {
                const snapshot = this._volumetricOutputMode !== 'off' ? this._buildVolumetricSnapshot(loop) : null;
                if (snapshot && snapshot.payload) {
                    postMessage({ type: 'paused', iteration: loop, volumetricData: snapshot.payload }, snapshot.transfer);
                } else {
                    postMessage({ type: 'paused', iteration: loop, volumetricData: null });
                }
                await new Promise(resolve => { this._resumeResolve = resolve; });
            }

            loop++;
            const iterStartTime = performance.now();
            xold.set(x);

            // Penalization continuation: ramp penal from penalStart → penalTarget
            const currentPenal = penalStart + (penalTarget - penalStart) * Math.min(1.0, (loop - 1) / continuationIters);

            // Heaviside projection with beta-continuation (beta doubles every betaInterval iterations)
            const beta = useProjection ? Math.min(betaMax, Math.pow(2, Math.floor((loop - 1) / betaInterval))) : 1;
            const xPhys = useProjection ? this._heavisideProject(x, beta) : x;

            const { U, c: compliance, solverStats } = await this.FE(nelx, nely, nelz, xPhys, currentPenal, KEflat, edofArray, F, freedofs, fixedMask, config, loop, maxIterations, fixeddofs);
            c = compliance;

            const dc = new Float32Array(nel);
            const elementEnergies = new Float32Array(nel);
            const elementStress = new Float32Array(nel);
            let usedGpuThisIter = false;

            if (this.useGPU && this.gpuCompute && this.gpuCompute.isAvailable()) {
                try {
                    const gpuEnergies = await this.gpuCompute.computeElementEnergies(
                        new Float32Array(U),
                        this._KEflat32,
                        edofArray,
                        nel,
                        24
                    );
                    if (gpuEnergies && gpuEnergies.length === nel) {
                        elementEnergies.set(gpuEnergies);
                        usedGpuThisIter = true;
                    }
                } catch (_gpuIterError) {
                    this.useGPU = false;
                    usedGpuThisIter = false;
                }
            }

            if (!usedGpuThisIter) {
                const Ue = new Float64Array(24);
                for (let e = 0; e < nel; e++) {
                    const eOff = e * 24;
                    for (let i = 0; i < 24; i++) {
                        Ue[i] = U[edofArray[eOff + i]];
                    }
                    const energy = this._computeElementEnergyFlat(KEflat, Ue, 24);
                    elementEnergies[e] = energy;
                }
            }

            // Precompute Heaviside chain-rule denominator for this iteration
            const tanhBeta = Math.tanh(beta * 0.5);
            const heavisideDenom = 2 * tanhBeta; // tanh(b*0.5) + tanh(b*0.5)
            let iterMaxStress = 0;
            for (let e = 0; e < nel; e++) {
                const energy = elementEnergies[e];
                elementEnergies[e] = energy;
                const stiffness = this.Emin + Math.pow(xPhys[e], currentPenal) * (this.E0 - this.Emin);
                const stress = stiffness * energy;
                elementStress[e] = stress;
                if (stress > iterMaxStress) iterMaxStress = stress;
                // Sensitivity w.r.t. xPhys
                const dc_phys = -currentPenal * Math.pow(xPhys[e], currentPenal - 1) * this.E0 * energy;
                // Chain rule: d(xPhys)/d(x) via Heaviside (1.0 when projection disabled)
                const th = Math.tanh(beta * (x[e] - 0.5));
                const dPhys_dx = useProjection ? beta * (1 - th * th) / heavisideDenom : 1.0;
                dc[e] = dc_phys * dPhys_dx;
            }

            const dcn = this.filterSensitivities(dc, x, H, Hs, nelx, nely, nelz);
            
            // Apply AMR-weighted filtering if enabled
            let dcnWeighted = dcn;
            if (amrManager && amrManager.groups.length > 0) {
                // Weight sensitivities by group size (smaller groups = more refinement = higher weight)
                dcnWeighted = new Float32Array(nel);
                for (const group of amrManager.groups) {
                    // Inverse size weighting: smaller size = higher weight
                    const weight = amrManager.maxSize / Math.max(group.size, 0.5);
                    for (const elemIdx of group.elements) {
                        dcnWeighted[elemIdx] = dcn[elemIdx] * weight;
                    }
                }
                
                // Normalize to maintain similar scale
                let sumOriginal = 0, sumWeighted = 0;
                for (let i = 0; i < nel; i++) {
                    sumOriginal += Math.abs(dcn[i]);
                    sumWeighted += Math.abs(dcnWeighted[i]);
                }
                if (sumWeighted > 0) {
                    const normFactor = sumOriginal / sumWeighted;
                    for (let i = 0; i < nel; i++) {
                        dcnWeighted[i] *= normFactor;
                    }
                }
            }
            
            this.OC(nelx, nely, nelz, x, volfrac, dcnWeighted, preservedMask, voidMask, xnew);

            // Fill internal voids if enabled
            if (config.preventVoids) {
                this._fillInternalVoids(xnew, nelx, nely, nelz);
            }

            // Apply manufacturing constraints if enabled
            if (config.manufacturingConstraint) {
                if (config.manufacturingAngle != null) {
                    this._applyOverhangConstraint(xnew, nelx, nely, nelz, config.manufacturingAngle, 0.3, preservedMask);
                    this._enforceToolAccessibility(xnew, nelx, nely, nelz, config.manufacturingAngle, 0.3, preservedMask);
                }
                if (config.manufacturingMaxDepth > 0) {
                    this._applyMaxDepthConstraint(xnew, nelx, nely, nelz, config.manufacturingMaxDepth, 0.3, preservedMask);
                }
                if (config.manufacturingMinRadius > 0) {
                    this._applyMinRadiusConstraint(xnew, nelx, nely, nelz, config.manufacturingMinRadius, 0.3, preservedMask);
                }
            }

            change = 0;
            for (let i = 0; i < nel; i++) {
                change = Math.max(change, Math.abs(xnew[i] - xold[i]));
            }

            // Swap buffers (avoid per-iteration allocations)
            const tmpX = x;
            x = xnew;
            xnew = tmpX;
            lastElementEnergies = elementEnergies;
            this._lastVolumetricStress = elementStress;
            this._lastVolumetricMaxStress = iterMaxStress;
            this._lastVolumetricIteration = loop;
            
            // Perform AMR refinement if enabled
            if (amrManager) {
                amrManager.updateAndRefine(elementEnergies, x, loop);
            }

            // Build adaptive mesh data only on UI update iterations (large 3D grids get expensive)
            const shouldUpdateUI = (uiUpdateInterval <= 1) || (loop === 1) || (loop % uiUpdateInterval === 0) || (change <= 0.01) || (loop === maxIterations);
            const meshData = shouldUpdateUI
                ? this.buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces, amrManager)
                : null;

            // Track iteration timing
            const iterEndTime = performance.now();
            const iterTime = iterEndTime - iterStartTime;
            iterationTimes.push(iterTime);
            iterTimeSum += iterTime;

            // Calculate average time per iteration
            const avgIterTime = iterTimeSum / iterationTimes.length;
            const elapsedTime = iterEndTime - startTime;

            if (shouldUpdateUI) {
                const volumetricSnapshot = this._volumetricOutputMode === 'every-iteration'
                    ? this._buildVolumetricSnapshot(loop)
                    : null;
                postMessage({
                    type: 'progress',
                    iteration: loop,
                    compliance: c,
                    meshData: meshData,
                    maxStress: meshData ? meshData.maxStress : 0,
                    volumetricData: volumetricSnapshot ? volumetricSnapshot.payload : null,
                    penal: currentPenal,
                    beta: beta,
                    timing: {
                        iterationTime: iterTime,
                        avgIterationTime: avgIterTime,
                        elapsedTime: elapsedTime,
                        usingWasm: this.useWasm,
                        usingGPU: usedGpuThisIter
                    }
                }, volumetricSnapshot ? volumetricSnapshot.transfer : undefined);
            }
        }

        if (this.cancelled) {
            postMessage({ type: 'cancelled', iteration: loop });
            return;
        }

        const finalMesh = this.buildAdaptiveMesh(nelx, nely, nelz, x, lastElementEnergies, elementForces, amrManager);
        
        // Final timing statistics
        const totalTime = performance.now() - startTime;
        const avgIterTime = iterationTimes.length > 0 
            ? iterationTimes.reduce((a, b) => a + b, 0) / iterationTimes.length 
            : 0;
        
        // Get AMR statistics if enabled
        const amrStats = amrManager ? amrManager.getStats() : null;

        const finalVolumetric = this._volumetricOutputMode !== 'off'
            ? this._buildVolumetricSnapshot(loop)
            : null;

        const completePayload = {
            type: 'complete',
            result: {
                densities: x,
                finalCompliance: c,
                iterations: loop,
                volumeFraction: volfrac,
                nx: nelx,
                ny: nely,
                nz: nelz,
                meshData: finalMesh,
                maxStress: finalMesh ? finalMesh.maxStress : 0,
                volumetricData: finalVolumetric ? finalVolumetric.payload : null,
                amrStats: amrStats,
                timing: {
                    totalTime: totalTime,
                    avgIterationTime: avgIterTime,
                    iterationTimes: iterationTimes,
                    usingWasm: this.useWasm
                }
            }
        };

        this._cleanupGPU();
        if (finalVolumetric) {
            postMessage(completePayload, finalVolumetric.transfer);
        } else {
            postMessage(completePayload);
        }
    }

    // Get DOF indices for an 8-node hexahedral element
    // Node numbering (local):
    //   Bottom face (z=0): 0(x,y), 1(x+1,y), 2(x+1,y+1), 3(x,y+1)
    //   Top face (z=1):    4(x,y), 5(x+1,y), 6(x+1,y+1), 7(x,y+1)
    getElementDOFs(elx, ely, elz, nelx, nely, nelz) {
        const nny = nely + 1;
        const nnz = nelz + 1;
        const n0 = elx * nny * nnz + ely * nnz + elz;
        const n1 = (elx + 1) * nny * nnz + ely * nnz + elz;
        const n2 = (elx + 1) * nny * nnz + (ely + 1) * nnz + elz;
        const n3 = elx * nny * nnz + (ely + 1) * nnz + elz;
        const n4 = elx * nny * nnz + ely * nnz + (elz + 1);
        const n5 = (elx + 1) * nny * nnz + ely * nnz + (elz + 1);
        const n6 = (elx + 1) * nny * nnz + (ely + 1) * nnz + (elz + 1);
        const n7 = elx * nny * nnz + (ely + 1) * nnz + (elz + 1);

        return [
            3*n0, 3*n0+1, 3*n0+2,
            3*n1, 3*n1+1, 3*n1+2,
            3*n2, 3*n2+1, 3*n2+2,
            3*n3, 3*n3+1, 3*n3+2,
            3*n4, 3*n4+1, 3*n4+2,
            3*n5, 3*n5+1, 3*n5+2,
            3*n6, 3*n6+1, 3*n6+2,
            3*n7, 3*n7+1, 3*n7+2
        ];
    }

    /**
     * Genetic algorithm optimizer for 3D topology optimization.
     * Uses tournament selection, uniform crossover, and bit-flip mutation
     * with volume-fraction penalty to evolve element density distributions.
     */
    async geneticOptimize(model, config) {
        const { nx, ny, nz } = model;
        const nelx = nx;
        const nely = ny;
        const nelz = nz;
        const volfrac = config.volumeFraction || 0.5;
        const maxGenerations = config.maxIterations || 50;
        const populationSize = config.populationSize || 20;
        const eliteCount = config.eliteCount || 2;
        const mutationRate = config.mutationRate || 0.02;
        const crossoverRate = config.crossoverRate || 0.8;
        const tournamentSize = config.tournamentSize || 3;
        const penaltyWeight = config.volumePenalty || 2.0;

        this.penal = config.penaltyFactor || 3;
        this.rmin = config.filterRadius || 1.5;
        this.cancelled = false;
        this._paused = false;
        if (config.linearSolver) this.linearSolver = config.linearSolver;

        if (config.youngsModulus) this.E0 = config.youngsModulus;
        if (config.poissonsRatio !== undefined) this.nu = config.poissonsRatio;

        const nel = nelx * nely * nelz;

        let fixeddofs = this.getFixedDOFs(nelx, nely, nelz, config.constraintPosition, config.constraintDOFs);
        let F = this.getLoadVector(nelx, nely, nelz, config.forceDirection, config.forceMagnitude, config.forceVector);

        if (config.paintedConstraints && config.paintedConstraints.length > 0) {
            fixeddofs = this.getFixedDOFsFromPaint(nelx, nely, nelz, config.paintedConstraints, config.constraintDOFs);
        }
        if (config.paintedForces && config.paintedForces.length > 0) {
            F = this.getLoadVectorFromPaint(nelx, nely, nelz, config.paintedForces, config.forceDirection, config.forceMagnitude, config.forceVector);
        }

        // Build preserved/void element sets
        const preservedElements = new Set();
        const allPaintedKeys = [
            ...(config.paintedConstraints || []),
            ...(config.paintedForces || []),
            ...(config.paintedKeep || [])
        ];
        for (const key of allPaintedKeys) {
            const parts = key.split(',');
            if (parts.length < 3) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            const vz = parseInt(parts[2], 10);
            if (!isNaN(vx) && !isNaN(vy) && !isNaN(vz) &&
                vx >= 0 && vx < nelx && vy >= 0 && vy < nely && vz >= 0 && vz < nelz) {
                preservedElements.add(vx + vy * nelx + vz * nelx * nely);
            }
        }
        if (!config.paintedConstraints || config.paintedConstraints.length === 0) {
            const constraintElems = this.getConstraintElements(nelx, nely, nelz, config.constraintPosition);
            for (const idx of constraintElems) preservedElements.add(idx);
        }
        if (!config.paintedForces || config.paintedForces.length === 0) {
            const forceElems = this.getForceElements(nelx, nely, nelz, config.forceDirection);
            for (const idx of forceElems) preservedElements.add(idx);
        }

        const voidElements = new Set();
        if (config.constrainToSolid && model.elements) {
            for (let i = 0; i < nel; i++) {
                if (model.elements[i] < 0.5) voidElements.add(i);
            }
        }

        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const fixedMask = new Uint8Array(ndof);
        for (let i = 0; i < fixeddofs.length; i++) {
            const dof = fixeddofs[i];
            if (dof >= 0 && dof < ndof) fixedMask[dof] = 1;
        }
        let nFree = 0;
        for (let i = 0; i < ndof; i++) if (!fixedMask[i]) nFree++;
        const freedofs = new Int32Array(nFree);
        let fp = 0;
        for (let i = 0; i < ndof; i++) if (!fixedMask[i]) freedofs[fp++] = i;

        const KE = this.lk();
        const KEflat = this._flattenKE(KE, 24);
        const edofArray = this._precomputeEdofs3D(nelx, nely, nelz);
        const elementForces = this.computeElementForces(nelx, nely, nelz, F);

        // Seeded random number generator
        let _seed = 42;
        const _random = () => { _seed = (_seed * 1103515245 + 12345) & 0x7fffffff; return _seed / 0x7fffffff; };

        const createIndividual = () => {
            const x = new Float32Array(nel);
            for (let i = 0; i < nel; i++) {
                if (preservedElements.has(i)) { x[i] = 1.0; }
                else if (voidElements.has(i)) { x[i] = 0.0; }
                else { x[i] = _random() < volfrac ? 1.0 : 0.0; }
            }
            return x;
        };

        const evaluate = async (x) => {
            // Use Emin floor for void elements to avoid singular stiffness matrix
            const xFEA = new Float32Array(nel);
            for (let i = 0; i < nel; i++) {
                xFEA[i] = x[i] > 0.5 ? x[i] : 0.001;
            }

            const { U, c: compliance } = await this.FE(nelx, nely, nelz, xFEA, this.penal, KEflat, edofArray, F, freedofs, fixedMask, config, 1, 1, fixeddofs);

            // Guard against invalid FEA results (diverged solver)
            if (!isFinite(compliance) || compliance <= 0) {
                return { compliance: 1e20, rawCompliance: 1e20, volFrac: 1.0, U };
            }

            let vol = 0;
            for (let i = 0; i < nel; i++) vol += x[i];
            const volFrac = vol / nel;
            const volViolation = Math.max(0, volFrac - volfrac);
            const penalty = penaltyWeight * compliance * volViolation * volViolation;
            return { compliance: compliance + penalty, rawCompliance: compliance, volFrac, U };
        };

        const tournamentSelect = (population, fitnesses) => {
            let bestIdx = Math.floor(_random() * population.length);
            for (let t = 1; t < tournamentSize; t++) {
                const idx = Math.floor(_random() * population.length);
                if (fitnesses[idx] < fitnesses[bestIdx]) bestIdx = idx;
            }
            return bestIdx;
        };

        const crossover = (parent1, parent2) => {
            const child = new Float32Array(nel);
            for (let i = 0; i < nel; i++) {
                if (preservedElements.has(i)) { child[i] = 1.0; }
                else if (voidElements.has(i)) { child[i] = 0.0; }
                else { child[i] = _random() < 0.5 ? parent1[i] : parent2[i]; }
            }
            return child;
        };

        const mutate = (individual, rate) => {
            for (let i = 0; i < nel; i++) {
                if (preservedElements.has(i) || voidElements.has(i)) continue;
                if (_random() < rate) {
                    individual[i] = individual[i] > 0.5 ? 0.0 : 1.0;
                }
            }
        };

        let population = [];
        for (let i = 0; i < populationSize; i++) {
            population.push(createIndividual());
        }

        const startTime = performance.now();
        const iterationTimes = [];
        let bestCompliance = Infinity;
        let bestIndividual = null;

        for (let gen = 0; gen < maxGenerations; gen++) {
            await _yieldToLoop();

            if (this.cancelled) {
                postMessage({ type: 'cancelled', iteration: gen });
                return;
            }

            if (this._paused) {
                postMessage({ type: 'paused', iteration: gen });
                await new Promise(resolve => { this._resumeResolve = resolve; });
            }

            const iterStartTime = performance.now();

            const results = await Promise.all(population.map(ind => evaluate(ind)));
            const fitnesses = results.map(r => r.compliance);

            for (let i = 0; i < populationSize; i++) {
                if (fitnesses[i] < bestCompliance) {
                    bestCompliance = fitnesses[i];
                    bestIndividual = Float32Array.from(population[i]);
                }
            }

            const indices = Array.from({ length: populationSize }, (_, i) => i);
            indices.sort((a, b) => fitnesses[a] - fitnesses[b]);

            const newPop = [];
            for (let i = 0; i < eliteCount && i < populationSize; i++) {
                newPop.push(Float32Array.from(population[indices[i]]));
            }

            while (newPop.length < populationSize) {
                if (_random() < crossoverRate) {
                    const p1 = tournamentSelect(population, fitnesses);
                    const p2 = tournamentSelect(population, fitnesses);
                    const child = crossover(population[p1], population[p2]);
                    mutate(child, mutationRate);
                    newPop.push(child);
                } else {
                    const p = tournamentSelect(population, fitnesses);
                    const child = Float32Array.from(population[p]);
                    mutate(child, mutationRate);
                    newPop.push(child);
                }
            }

            population = newPop;

            // Build mesh for visualization
            const bestEval = await evaluate(bestIndividual);
            const elementEnergies = new Float32Array(nel);
            const Ue = new Float64Array(24);
            for (let e = 0; e < nel; e++) {
                const eOff = e * 24;
                for (let i = 0; i < 24; i++) {
                    Ue[i] = bestEval.U[edofArray[eOff + i]];
                }
                elementEnergies[e] = this._computeElementEnergyFlat(KEflat, Ue, 24);
            }

            const meshData = this.buildAdaptiveMesh(nelx, nely, nelz, bestIndividual, elementEnergies, elementForces, null);

            const iterEndTime = performance.now();
            const iterTime = iterEndTime - iterStartTime;
            iterationTimes.push(iterTime);

            const avgIterTime = iterationTimes.reduce((a, b) => a + b, 0) / iterationTimes.length;
            const elapsedTime = iterEndTime - startTime;

            postMessage({
                type: 'progress',
                iteration: gen + 1,
                compliance: bestCompliance,
                meshData: meshData,
                maxStress: meshData ? meshData.maxStress : 0,
                penal: this.penal,
                beta: 1,
                timing: {
                    iterationTime: iterTime,
                    avgIterationTime: avgIterTime,
                    elapsedTime: elapsedTime,
                    usingWasm: this.useWasm
                }
            });
        }

        if (this.cancelled) {
            postMessage({ type: 'cancelled', iteration: maxGenerations });
            return;
        }

        // Final evaluation
        const finalEval = await evaluate(bestIndividual);
        const finalEnergies = new Float32Array(nel);
        const Ue = new Float64Array(24);
        for (let e = 0; e < nel; e++) {
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) {
                Ue[i] = finalEval.U[edofArray[eOff + i]];
            }
            finalEnergies[e] = this._computeElementEnergyFlat(KEflat, Ue, 24);
        }

        const finalMesh = this.buildAdaptiveMesh(nelx, nely, nelz, bestIndividual, finalEnergies, elementForces, null);

        const totalTime = performance.now() - startTime;
        const avgIterTime = iterationTimes.length > 0
            ? iterationTimes.reduce((a, b) => a + b, 0) / iterationTimes.length
            : 0;

        this._cleanupGPU();
        postMessage({
            type: 'complete',
            result: {
                densities: bestIndividual,
                finalCompliance: bestCompliance,
                iterations: maxGenerations,
                volumeFraction: volfrac,
                nx: nelx,
                ny: nely,
                nz: nelz,
                meshData: finalMesh,
                maxStress: finalMesh ? finalMesh.maxStress : 0,
                geneticOptimization: true,
                timing: {
                    totalTime: totalTime,
                    avgIterationTime: avgIterTime,
                    iterationTimes: iterationTimes,
                    usingWasm: this.useWasm
                }
            }
        });
    }

    // Compute per-element force magnitude based on the global load vector
    computeElementForces(nelx, nely, nelz, F) {
        const nel = nelx * nely * nelz;
        const forces = new Float32Array(nel);
        for (let elz = 0; elz < nelz; elz++) {
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const edof = this.getElementDOFs(elx, ely, elz, nelx, nely, nelz);
                    let mag = 0;
                    for (let d = 0; d < edof.length; d++) {
                        mag += F[edof[d]] * F[edof[d]];
                    }
                    forces[elx + ely * nelx + elz * nelx * nely] = Math.sqrt(mag);
                }
            }
        }
        return forces;
    }

    /**
     * Build adaptive mesh data.
     * When AMR is active each AMR group is rendered as a merged block whose face size
     * matches the group bounding box — coarse groups produce large faces (low stress)
     * and fine groups produce small faces (high stress), making the grain size
     * variation visually obvious.  When AMR is disabled every element is rendered
     * individually (original behaviour).
     * Returns an array of { vertices, normal, density, strain, blockSize } triangle
     * objects for rendering.
     */
    buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces, amrManager) {
        // Duplicated from constants.js since workers cannot use ES module imports
        const DENSITY_THRESHOLD = 0.3;
        const triangles = [];
        const amrCells = [];

        // Compute stress-based metric: stiffness × strain energy per element
        let maxStress = 0;
        const elementStress = elementEnergies ? new Float32Array(elementEnergies.length) : null;
        if (elementEnergies) {
            for (let i = 0; i < elementEnergies.length; i++) {
                const stiffness = this.Emin + Math.pow(x[i], this.penal) * (this.E0 - this.Emin);
                elementStress[i] = stiffness * elementEnergies[i];
                if (elementStress[i] > maxStress) maxStress = elementStress[i];
            }
        }

        // Helper – is an element solid?
        const isSolid = (ex, ey, ez) => {
            if (ex < 0 || ex >= nelx || ey < 0 || ey >= nely || ez < 0 || ez >= nelz) return false;
            return x[ex + ey * nelx + ez * nelx * nely] > DENSITY_THRESHOLD;
        };

        // Face directions: [dx,dy,dz] for each of the 6 faces
        const faceNeighbors = [
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1],
        ];

        // ── AMR-aware rendering ──────────────────────────────────────────────
        const rendered = new Uint8Array(nelx * nely * nelz);

        if (amrManager && amrManager.groups && amrManager.groups.length > 0) {
            for (const group of amrManager.groups) {
                if (group.elements.length === 0) continue;

                let minGX = nelx, minGY = nely, minGZ = nelz;
                let maxGX = -1, maxGY = -1, maxGZ = -1;
                let totalDensity = 0, totalStress = 0, solidCount = 0;

                for (const idx of group.elements) {
                    const elz = Math.floor(idx / (nelx * nely));
                    const rem = idx % (nelx * nely);
                    const ely = Math.floor(rem / nelx);
                    const elx = rem % nelx;
                    const density = x[idx];
                    if (density > DENSITY_THRESHOLD) {
                        solidCount++;
                        totalDensity += density;
                        totalStress += elementStress ? elementStress[idx] : 0;
                        if (elx < minGX) minGX = elx;
                        if (ely < minGY) minGY = ely;
                        if (elz < minGZ) minGZ = elz;
                        if (elx > maxGX) maxGX = elx;
                        if (ely > maxGY) maxGY = ely;
                        if (elz > maxGZ) maxGZ = elz;
                    }
                    rendered[idx] = 1;
                }

                if (solidCount === 0) continue;

                const avgDensity = totalDensity / solidCount;
                const avgStress = totalStress / solidCount;
                const strain = maxStress > 0 ? avgStress / maxStress : 0;

                const bx0 = minGX, bx1 = maxGX + 1;
                const by0 = minGY, by1 = maxGY + 1;
                const bz0 = minGZ, bz1 = maxGZ + 1;
                const bw = bx1 - bx0;

                // AMR cuboid cell payload for crack-free AMR surface meshing in viewer
                amrCells.push({
                    x: bx0,
                    y: by0,
                    z: bz0,
                    size: bw,
                    density: avgDensity,
                    stress: strain
                });

                // Is a neighbouring slab of the block fully solid outside?
                const isNeighbourSolid = (dx, dy, dz) => {
                    if (dx === -1) {
                        if (bx0 === 0) return false;
                        for (let gy = by0; gy < by1; gy++) {
                            for (let gz = bz0; gz < bz1; gz++) {
                                if (!isSolid(bx0 - 1, gy, gz)) return false;
                            }
                        }
                        return true;
                    }
                    if (dx === 1) {
                        if (bx1 >= nelx) return false;
                        for (let gy = by0; gy < by1; gy++) {
                            for (let gz = bz0; gz < bz1; gz++) {
                                if (!isSolid(bx1, gy, gz)) return false;
                            }
                        }
                        return true;
                    }
                    if (dy === -1) {
                        if (by0 === 0) return false;
                        for (let gx = bx0; gx < bx1; gx++) {
                            for (let gz = bz0; gz < bz1; gz++) {
                                if (!isSolid(gx, by0 - 1, gz)) return false;
                            }
                        }
                        return true;
                    }
                    if (dy === 1) {
                        if (by1 >= nely) return false;
                        for (let gx = bx0; gx < bx1; gx++) {
                            for (let gz = bz0; gz < bz1; gz++) {
                                if (!isSolid(gx, by1, gz)) return false;
                            }
                        }
                        return true;
                    }
                    if (dz === -1) {
                        if (bz0 === 0) return false;
                        for (let gx = bx0; gx < bx1; gx++) {
                            for (let gy = by0; gy < by1; gy++) {
                                if (!isSolid(gx, gy, bz0 - 1)) return false;
                            }
                        }
                        return true;
                    }
                    if (dz === 1) {
                        if (bz1 >= nelz) return false;
                        for (let gx = bx0; gx < bx1; gx++) {
                            for (let gy = by0; gy < by1; gy++) {
                                if (!isSolid(gx, gy, bz1)) return false;
                            }
                        }
                        return true;
                    }
                    return false;
                };

                // Emit one merged quad per exposed boundary face of the block
                // Face 0: X- face
                if (!isNeighbourSolid(-1, 0, 0)) {
                    triangles.push({ vertices: [[bx0, by0, bz0], [bx0, by1, bz0], [bx0, by1, bz1]], normal: [-1, 0, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by0, bz0], [bx0, by1, bz1], [bx0, by0, bz1]], normal: [-1, 0, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 1: X+ face
                if (!isNeighbourSolid(1, 0, 0)) {
                    triangles.push({ vertices: [[bx1, by1, bz0], [bx1, by0, bz0], [bx1, by0, bz1]], normal: [1, 0, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx1, by1, bz0], [bx1, by0, bz1], [bx1, by1, bz1]], normal: [1, 0, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 2: Y- face
                if (!isNeighbourSolid(0, -1, 0)) {
                    triangles.push({ vertices: [[bx0, by0, bz0], [bx0, by0, bz1], [bx1, by0, bz1]], normal: [0, -1, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by0, bz0], [bx1, by0, bz1], [bx1, by0, bz0]], normal: [0, -1, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 3: Y+ face
                if (!isNeighbourSolid(0, 1, 0)) {
                    triangles.push({ vertices: [[bx0, by1, bz1], [bx0, by1, bz0], [bx1, by1, bz0]], normal: [0, 1, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by1, bz1], [bx1, by1, bz0], [bx1, by1, bz1]], normal: [0, 1, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 4: Z- face
                if (!isNeighbourSolid(0, 0, -1)) {
                    triangles.push({ vertices: [[bx0, by0, bz0], [bx1, by0, bz0], [bx1, by1, bz0]], normal: [0, 0, -1], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by0, bz0], [bx1, by1, bz0], [bx0, by1, bz0]], normal: [0, 0, -1], density: avgDensity, strain, blockSize: bw });
                }
                // Face 5: Z+ face
                if (!isNeighbourSolid(0, 0, 1)) {
                    triangles.push({ vertices: [[bx1, by0, bz1], [bx0, by0, bz1], [bx0, by1, bz1]], normal: [0, 0, 1], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx1, by0, bz1], [bx0, by1, bz1], [bx1, by1, bz1]], normal: [0, 0, 1], density: avgDensity, strain, blockSize: bw });
                }
            }
        }

        // ── Per-element fallback for elements not covered by any AMR group ───
        for (let elz = 0; elz < nelz; elz++) {
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const idx = elx + ely * nelx + elz * nelx * nely;
                    if (rendered[idx]) continue;
                    const density = x[idx];
                    if (density <= DENSITY_THRESHOLD) continue;

                    const visibleFaces = [];
                    for (let fi = 0; fi < 6; fi++) {
                        const [dx, dy, dz] = faceNeighbors[fi];
                        if (!isSolid(elx + dx, ely + dy, elz + dz)) visibleFaces.push(fi);
                    }
                    if (visibleFaces.length === 0) continue;

                    const strain = (maxStress > 0 && elementStress) ? elementStress[idx] / maxStress : 0;
                    amrCells.push({ x: elx, y: ely, z: elz, size: 1, density, stress: strain });
                    this.addSubdividedElement(triangles, elx, ely, elz, density, 1, visibleFaces, strain);
                }
            }
        }

        triangles.maxStress = maxStress;
        triangles.amrCells = amrCells;
        return triangles;
    }

    /**
     * Add an 8-node hexahedral element as triangulated faces.
     * Only emits faces listed in visibleFaces (indices 0-5).
     */
    addSubdividedElement(triangles, ex, ey, ez, density, subdivLevel, visibleFaces, strain) {
        const n = subdivLevel;
        const step = 1.0 / n;

        const baseX = ex;
        const baseY = ey;
        const baseZ = ez;

        for (const fi of visibleFaces) {
            switch (fi) {
                case 0: // Front face (X = baseX)
                    for (let sy = 0; sy < n; sy++) {
                        for (let sz = 0; sz < n; sz++) {
                            const y0 = baseY + sy * step;
                            const z0 = baseZ + sz * step;
                            const y1 = y0 + step;
                            const z1 = z0 + step;
                            triangles.push({ vertices: [[baseX, y0, z0], [baseX, y1, z0], [baseX, y1, z1]], normal: [-1, 0, 0], density, strain, blockSize: 1 });
                            triangles.push({ vertices: [[baseX, y0, z0], [baseX, y1, z1], [baseX, y0, z1]], normal: [-1, 0, 0], density, strain, blockSize: 1 });
                        }
                    }
                    break;
                case 1: // Back face (X = baseX + 1)
                    for (let sy = 0; sy < n; sy++) {
                        for (let sz = 0; sz < n; sz++) {
                            const y0 = baseY + sy * step;
                            const z0 = baseZ + sz * step;
                            const y1 = y0 + step;
                            const z1 = z0 + step;
                            triangles.push({ vertices: [[baseX + 1, y1, z0], [baseX + 1, y0, z0], [baseX + 1, y0, z1]], normal: [1, 0, 0], density, strain, blockSize: 1 });
                            triangles.push({ vertices: [[baseX + 1, y1, z0], [baseX + 1, y0, z1], [baseX + 1, y1, z1]], normal: [1, 0, 0], density, strain, blockSize: 1 });
                        }
                    }
                    break;
                case 2: // Left face (Y = baseY)
                    for (let sx = 0; sx < n; sx++) {
                        for (let sz = 0; sz < n; sz++) {
                            const x0 = baseX + sx * step;
                            const z0 = baseZ + sz * step;
                            const x1 = x0 + step;
                            const z1 = z0 + step;
                            triangles.push({ vertices: [[x0, baseY, z0], [x0, baseY, z1], [x1, baseY, z1]], normal: [0, -1, 0], density, strain, blockSize: 1 });
                            triangles.push({ vertices: [[x0, baseY, z0], [x1, baseY, z1], [x1, baseY, z0]], normal: [0, -1, 0], density, strain, blockSize: 1 });
                        }
                    }
                    break;
                case 3: // Right face (Y = baseY + 1)
                    for (let sx = 0; sx < n; sx++) {
                        for (let sz = 0; sz < n; sz++) {
                            const x0 = baseX + sx * step;
                            const z0 = baseZ + sz * step;
                            const x1 = x0 + step;
                            const z1 = z0 + step;
                            triangles.push({ vertices: [[x0, baseY + 1, z1], [x0, baseY + 1, z0], [x1, baseY + 1, z0]], normal: [0, 1, 0], density, strain, blockSize: 1 });
                            triangles.push({ vertices: [[x0, baseY + 1, z1], [x1, baseY + 1, z0], [x1, baseY + 1, z1]], normal: [0, 1, 0], density, strain, blockSize: 1 });
                        }
                    }
                    break;
                case 4: // Bottom face (Z = baseZ)
                    for (let sx = 0; sx < n; sx++) {
                        for (let sy = 0; sy < n; sy++) {
                            const x0 = baseX + sx * step;
                            const y0 = baseY + sy * step;
                            const x1 = x0 + step;
                            const y1 = y0 + step;
                            triangles.push({ vertices: [[x0, y0, baseZ], [x1, y0, baseZ], [x1, y1, baseZ]], normal: [0, 0, -1], density, strain, blockSize: 1 });
                            triangles.push({ vertices: [[x0, y0, baseZ], [x1, y1, baseZ], [x0, y1, baseZ]], normal: [0, 0, -1], density, strain, blockSize: 1 });
                        }
                    }
                    break;
                case 5: // Top face (Z = baseZ + 1)
                    for (let sx = 0; sx < n; sx++) {
                        for (let sy = 0; sy < n; sy++) {
                            const x0 = baseX + sx * step;
                            const y0 = baseY + sy * step;
                            const x1 = x0 + step;
                            const y1 = y0 + step;
                            triangles.push({ vertices: [[x1, y0, baseZ + 1], [x0, y0, baseZ + 1], [x0, y1, baseZ + 1]], normal: [0, 0, 1], density, strain, blockSize: 1 });
                            triangles.push({ vertices: [[x1, y0, baseZ + 1], [x0, y1, baseZ + 1], [x1, y1, baseZ + 1]], normal: [0, 0, 1], density, strain, blockSize: 1 });
                        }
                    }
                    break;
            }
        }
    }

    prepareFilter(nelx, nely, nelz, rmin) {
        const iH = [];
        const jH = [];
        const sH = [];
        let k = 0;

        for (let i = 0; i < nelx; i++) {
            for (let j = 0; j < nely; j++) {
                for (let m = 0; m < nelz; m++) {
                    const e1 = i + j * nelx + m * nelx * nely;

                    for (let k_iter = Math.max(i - Math.floor(rmin), 0);
                         k_iter <= Math.min(i + Math.floor(rmin), nelx - 1);
                         k_iter++) {
                        for (let l = Math.max(j - Math.floor(rmin), 0);
                             l <= Math.min(j + Math.floor(rmin), nely - 1);
                             l++) {
                            for (let n = Math.max(m - Math.floor(rmin), 0);
                                 n <= Math.min(m + Math.floor(rmin), nelz - 1);
                                 n++) {
                                const e2 = k_iter + l * nelx + n * nelx * nely;
                                const dist = Math.sqrt((i - k_iter) ** 2 + (j - l) ** 2 + (m - n) ** 2);

                                if (dist <= rmin) {
                                    iH[k] = e1;
                                    jH[k] = e2;
                                    sH[k] = Math.max(0, rmin - dist);
                                    k++;
                                }
                            }
                        }
                    }
                }
            }
        }

        const H = { i: iH, j: jH, s: sH };
        const Hs = new Float32Array(nelx * nely * nelz);

        for (let i = 0; i < k; i++) {
            Hs[iH[i]] += sH[i];
        }

        return { H, Hs };
    }

    filterSensitivities(dc, x, H, Hs, nelx, nely, nelz) {
        const dcn = new Float32Array(nelx * nely * nelz);

        for (let i = 0; i < H.i.length; i++) {
            dcn[H.i[i]] += H.s[i] * x[H.j[i]] * dc[H.j[i]];
        }

        for (let i = 0; i < nelx * nely * nelz; i++) {
            dcn[i] = dcn[i] / (Hs[i] * Math.max(1e-3, x[i]));
        }

        return dcn;
    }

    OC(nelx, nely, nelz, x, volfrac, dc, preservedMask, voidMask, xnew) {
        const nel = nelx * nely * nelz;
        const move = 0.2;

        let l1 = 0;
        let l2 = 1e9;

        while ((l2 - l1) / (l2 + l1) > 1e-3) {
            const lmid = 0.5 * (l2 + l1);

            for (let i = 0; i < nel; i++) {
                if (preservedMask && preservedMask[i]) {
                    xnew[i] = 1.0;
                } else if (voidMask && voidMask[i]) {
                    xnew[i] = 0.0;
                } else {
                    const Be = -dc[i] / lmid;
                    xnew[i] = Math.max(0.0,
                              Math.max(x[i] - move,
                              Math.min(1.0,
                              Math.min(x[i] + move, x[i] * Math.sqrt(Be)))));
                }
            }

            let sumXnew = 0;
            for (let i = 0; i < nel; i++) {
                sumXnew += xnew[i];
            }

            if (sumXnew > volfrac * nel) {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }
    }

    /**
     * Fill internal voids (enclosed air bubbles) in the 3D density field.
     * Uses flood-fill from boundary void elements to identify exterior voids;
     * any void not reachable from the boundary is considered internal and filled.
     * @param {Float32Array} x - density array (modified in place)
     * @param {number} nelx
     * @param {number} nely
     * @param {number} nelz
     * @param {number} [threshold=0.3] - density below this is considered void
     */
    _fillInternalVoids(x, nelx, nely, nelz, threshold = 0.3) {
        const nel = nelx * nely * nelz;
        const visited = new Uint8Array(nel);
        const queue = [];

        // Helper to convert (ex, ey, ez) to flat index (x-major: x + y*nelx + z*nelx*nely)
        const idx3 = (ex, ey, ez) => ex + ey * nelx + ez * nelx * nely;

        // Seed: all boundary elements that are void
        for (let ex = 0; ex < nelx; ex++) {
            for (let ey = 0; ey < nely; ey++) {
                for (let ez = 0; ez < nelz; ez++) {
                    if (ex === 0 || ex === nelx - 1 ||
                        ey === 0 || ey === nely - 1 ||
                        ez === 0 || ez === nelz - 1) {
                        const i = idx3(ex, ey, ez);
                        if (x[i] < threshold) {
                            visited[i] = 1;
                            queue.push(i);
                        }
                    }
                }
            }
        }

        // Flood-fill from boundary voids
        while (queue.length > 0) {
            const i = queue.pop();
            const ez = Math.floor(i / (nelx * nely));
            const rem = i % (nelx * nely);
            const ey = Math.floor(rem / nelx);
            const ex = rem % nelx;
            const neighbors = [
                ex > 0 ? idx3(ex - 1, ey, ez) : -1,
                ex < nelx - 1 ? idx3(ex + 1, ey, ez) : -1,
                ey > 0 ? idx3(ex, ey - 1, ez) : -1,
                ey < nely - 1 ? idx3(ex, ey + 1, ez) : -1,
                ez > 0 ? idx3(ex, ey, ez - 1) : -1,
                ez < nelz - 1 ? idx3(ex, ey, ez + 1) : -1,
            ];
            for (const ni of neighbors) {
                if (ni >= 0 && !visited[ni] && x[ni] < threshold) {
                    visited[ni] = 1;
                    queue.push(ni);
                }
            }
        }

        // Fill internal voids: any void element not visited from boundary
        for (let i = 0; i < nel; i++) {
            if (x[i] < threshold && !visited[i]) {
                x[i] = 1.0;
            }
        }
    }

    /**
     * Compute the horizontal span (in grid cells) for a given overhang angle.
     * At 90° the span is 0 (vertical only). Lower angles allow more horizontal reach.
     */
    _overhangSpan(angleDeg, maxExtent) {
        const angleRad = angleDeg * Math.PI / 180;
        const reach = Math.min(Math.tan(Math.PI / 2 - angleRad), maxExtent);
        return Math.round(reach);
    }

    /**
     * Manufacturing overhang constraint (3D).
     * Sweeps bottom-to-top (build direction = +Y). For each layer, an element
     * is only allowed to be solid if it has support from the layer below
     * within the permitted overhang cone defined by `angleDeg`.
     *  - 90° = CNC machining (no overhangs).
     *  - 88° = mold tooling (very slight overhang).
     *  - Lower angles allow steeper overhangs.
     *
     * Y-axis points up: ey=0 is the physical bottom (self-supported ground),
     * ey=nely-1 is the physical top.
     *
     * Uses x-major indexing: idx = ex + ey * nelx + ez * nelx * nely.
     */
    _applyOverhangConstraint(x, nelx, nely, nelz, angleDeg, threshold = 0.3, preservedMask = null) {
        const span = this._overhangSpan(angleDeg, Math.max(nelx, nelz));

        const idx3 = (ex, ey, ez) => ex + ey * nelx + ez * nelx * nely;

        // Sweep from ey=1 to ey=nely-1; ey=0 (ground) is self-supported
        for (let ey = 1; ey < nely; ey++) {
            const belowRow = ey - 1;
            for (let ez = 0; ez < nelz; ez++) {
                for (let ex = 0; ex < nelx; ex++) {
                    const idx = idx3(ex, ey, ez);
                    if (x[idx] < threshold) continue;
                    if (preservedMask && preservedMask[idx]) continue;

                    // Check support cone in the layer below
                    let supported = false;
                    for (let dz = -span; dz <= span && !supported; dz++) {
                        const sz = ez + dz;
                        if (sz < 0 || sz >= nelz) continue;
                        for (let dx = -span; dx <= span; dx++) {
                            const sx = ex + dx;
                            if (sx < 0 || sx >= nelx) continue;
                            if (x[idx3(sx, belowRow, sz)] >= threshold) {
                                supported = true;
                                break;
                            }
                        }
                    }

                    if (!supported) {
                        x[idx] = 0.0;
                    }
                }
            }
        }
    }

    /**
     * Enforce CNC tool accessibility from the top surface (3D).
     * For 3-axis CNC, the tool approaches from above (ey=nely-1). Any void element
     * must be reachable from the top surface within the tool's angular reach.
     * At 90° (reach=0), each column must have void elements contiguous from the top;
     * once solid material is encountered, everything below must be solid.
     *
     * Y-axis points up: ey=0 is the physical bottom, ey=nely-1 is the physical
     * top where the CNC tool enters.
     *
     * Uses x-major indexing: idx = ex + ey * nelx + ez * nelx * nely.
     */
    _enforceToolAccessibility(x, nelx, nely, nelz, angleDeg, threshold = 0.3, preservedMask = null) {
        const span = this._overhangSpan(angleDeg, Math.max(nelx, nelz));
        const nel = nelx * nely * nelz;
        const accessible = new Uint8Array(nel);

        const idx3 = (ex, ey, ez) => ex + ey * nelx + ez * nelx * nely;

        // Seed: all void elements on the top layer (ey=nely-1) are accessible
        for (let ez = 0; ez < nelz; ez++) {
            for (let ex = 0; ex < nelx; ex++) {
                const idx = idx3(ex, nely - 1, ez);
                if (x[idx] < threshold) {
                    accessible[idx] = 1;
                }
            }
        }

        // Sweep top-to-bottom: a void element is accessible if an accessible
        // element exists in the layer above within the tool's angular reach
        for (let ey = nely - 2; ey >= 0; ey--) {
            const aboveRow = ey + 1;
            for (let ez = 0; ez < nelz; ez++) {
                for (let ex = 0; ex < nelx; ex++) {
                    const idx = idx3(ex, ey, ez);
                    if (x[idx] >= threshold) continue;

                    let found = false;
                    for (let dz = -span; dz <= span && !found; dz++) {
                        const sz = ez + dz;
                        if (sz < 0 || sz >= nelz) continue;
                        for (let dx = -span; dx <= span; dx++) {
                            const sx = ex + dx;
                            if (sx < 0 || sx >= nelx) continue;
                            if (accessible[idx3(sx, aboveRow, sz)]) {
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found) accessible[idx] = 1;
                }
            }
        }

        // Fill inaccessible voids
        for (let i = 0; i < nel; i++) {
            if (x[i] < threshold && !accessible[i]) {
                x[i] = 1.0;
            }
        }
    }

    /**
     * Enforce maximum milling depth from the top surface (3D).
     * Elements deeper than maxDepth layers from the top (ey=nely-1) are forced
     * solid, since the CNC tool cannot reach them.
     *
     * Y-axis points up: ey=0 is the physical bottom, ey=nely-1 is the physical
     * top where the CNC tool enters.
     *
     * Uses x-major indexing: idx = ex + ey * nelx + ez * nelx * nely.
     */
    _applyMaxDepthConstraint(x, nelx, nely, nelz, maxDepth, threshold = 0.3, preservedMask = null) {
        const limit = Math.floor(maxDepth);
        const idx3 = (ex, ey, ez) => ex + ey * nelx + ez * nelx * nely;

        // Force solid for layers below the tool's reach from the top
        const cutoff = nely - limit;
        for (let ey = 0; ey < cutoff; ey++) {
            for (let ez = 0; ez < nelz; ez++) {
                for (let ex = 0; ex < nelx; ex++) {
                    const idx = idx3(ex, ey, ez);
                    if (x[idx] < threshold) {
                        x[idx] = 1.0;
                    }
                }
            }
        }
    }

    /**
     * Enforce minimum radius on pocket corners (3D).
     * Applies morphological closing (dilate then erode) on the solid region
     * with a spherical structuring element of the specified radius.
     * This rounds all internal (concave) corners of pockets to at least the
     * given radius, matching the CNC tool's minimum corner radius.
     *
     * Uses x-major indexing: idx = ex + ey * nelx + ez * nelx * nely.
     */
    _applyMinRadiusConstraint(x, nelx, nely, nelz, radius, threshold = 0.3, preservedMask = null) {
        const nel = nelx * nely * nelz;
        const r = Math.ceil(radius);
        const r2 = radius * radius;

        // Build spherical structuring element offsets
        const offsets = [];
        for (let dz = -r; dz <= r; dz++) {
            for (let dy = -r; dy <= r; dy++) {
                for (let dx = -r; dx <= r; dx++) {
                    if (dx * dx + dy * dy + dz * dz <= r2) {
                        offsets.push([dx, dy, dz]);
                    }
                }
            }
        }

        const idx3 = (ex, ey, ez) => ex + ey * nelx + ez * nelx * nely;

        // Step 1: Dilate solid region
        const dilated = new Float32Array(nel);
        for (let i = 0; i < nel; i++) dilated[i] = x[i];

        for (let ez = 0; ez < nelz; ez++) {
            for (let ey = 0; ey < nely; ey++) {
                for (let ex = 0; ex < nelx; ex++) {
                    if (x[idx3(ex, ey, ez)] >= threshold) {
                        for (const [dx, dy, dz] of offsets) {
                            const sx = ex + dx, sy = ey + dy, sz = ez + dz;
                            if (sx >= 0 && sx < nelx && sy >= 0 && sy < nely &&
                                sz >= 0 && sz < nelz) {
                                const sidx = idx3(sx, sy, sz);
                                if (dilated[sidx] < threshold) dilated[sidx] = 1.0;
                            }
                        }
                    }
                }
            }
        }

        // Step 2: Erode the dilated result
        const eroded = new Float32Array(nel);
        for (let i = 0; i < nel; i++) eroded[i] = dilated[i];

        for (let ez = 0; ez < nelz; ez++) {
            for (let ey = 0; ey < nely; ey++) {
                for (let ex = 0; ex < nelx; ex++) {
                    const idx = idx3(ex, ey, ez);
                    if (dilated[idx] >= threshold) {
                        for (const [dx, dy, dz] of offsets) {
                            const sx = ex + dx, sy = ey + dy, sz = ez + dz;
                            if (sx < 0 || sx >= nelx || sy < 0 || sy >= nely ||
                                sz < 0 || sz >= nelz ||
                                dilated[idx3(sx, sy, sz)] < threshold) {
                                eroded[idx] = 0.0;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Apply: only fill void elements that became solid after closing
        for (let i = 0; i < nel; i++) {
            if (x[i] < threshold && eroded[i] >= threshold) {
                x[i] = eroded[i];
            }
        }
    }

    // 3D element stiffness matrix for 8-node hexahedral element
    // Using analytical integration for efficiency
    lk() {
        const E = 1.0;
        const nu = this.nu;
        
        // Constitutive matrix for 3D elasticity (full 3D formulation)
        const C = [];
        const fact = E / ((1 + nu) * (1 - 2 * nu));
        C[0] = [fact * (1 - nu), fact * nu, fact * nu, 0, 0, 0];
        C[1] = [fact * nu, fact * (1 - nu), fact * nu, 0, 0, 0];
        C[2] = [fact * nu, fact * nu, fact * (1 - nu), 0, 0, 0];
        C[3] = [0, 0, 0, fact * (1 - 2 * nu) / 2, 0, 0];
        C[4] = [0, 0, 0, 0, fact * (1 - 2 * nu) / 2, 0];
        C[5] = [0, 0, 0, 0, 0, fact * (1 - 2 * nu) / 2];

        const KE = Array(24).fill(0).map(() => Array(24).fill(0));
        
        // 2x2x2 Gauss integration
        const gaussPoints = [-1 / Math.sqrt(3), 1 / Math.sqrt(3)];
        const gaussWeights = [1, 1];

        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                for (let k = 0; k < 2; k++) {
                    const xi = gaussPoints[i];
                    const eta = gaussPoints[j];
                    const zeta = gaussPoints[k];
                    const weight = gaussWeights[i] * gaussWeights[j] * gaussWeights[k];

                    // Shape function derivatives in natural coordinates
                    const dN = [
                        [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
                        [ (1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
                        [ (1+eta)*(1-zeta),  (1+xi)*(1-zeta), -(1+xi)*(1+eta)],
                        [-(1+eta)*(1-zeta),  (1-xi)*(1-zeta), -(1-xi)*(1+eta)],
                        [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta),  (1-xi)*(1-eta)],
                        [ (1-eta)*(1+zeta), -(1+xi)*(1+zeta),  (1+xi)*(1-eta)],
                        [ (1+eta)*(1+zeta),  (1+xi)*(1+zeta),  (1+xi)*(1+eta)],
                        [-(1+eta)*(1+zeta),  (1-xi)*(1+zeta),  (1-xi)*(1+eta)]
                    ];

                    for (let n = 0; n < 8; n++) {
                        dN[n][0] *= 0.125;
                        dN[n][1] *= 0.125;
                        dN[n][2] *= 0.125;
                    }

                    // Jacobian matrix (for unit cube element, J is identity)
                    const J = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
                    const detJ = 1.0;

                    // B matrix (strain-displacement) - 6 x 24
                    const B = Array(6).fill(0).map(() => Array(24).fill(0));
                    for (let n = 0; n < 8; n++) {
                        const col = n * 3;
                        B[0][col] = dN[n][0];        // εxx = du/dx
                        B[1][col + 1] = dN[n][1];    // εyy = dv/dy
                        B[2][col + 2] = dN[n][2];    // εzz = dw/dz
                        B[3][col] = dN[n][1];        // γxy = du/dy + dv/dx
                        B[3][col + 1] = dN[n][0];
                        B[4][col + 1] = dN[n][2];    // γyz = dv/dz + dw/dy
                        B[4][col + 2] = dN[n][1];
                        B[5][col] = dN[n][2];        // γxz = du/dz + dw/dx
                        B[5][col + 2] = dN[n][0];
                    }

                    // KE += B^T * C * B * detJ * weight
                    const CB = Array(6).fill(0).map(() => Array(24).fill(0));
                    for (let m = 0; m < 6; m++) {
                        for (let n = 0; n < 24; n++) {
                            for (let p = 0; p < 6; p++) {
                                CB[m][n] += C[m][p] * B[p][n];
                            }
                        }
                    }

                    for (let m = 0; m < 24; m++) {
                        for (let n = 0; n < 24; n++) {
                            for (let p = 0; p < 6; p++) {
                                KE[m][n] += B[p][m] * CB[p][n] * detJ * weight;
                            }
                        }
                    }
                }
            }
        }

        return KE;
    }

    /**
     * Heaviside projection: maps design variable x → physical density xPhys.
     * @param {Float32Array} x - design variables
     * @param {number} beta - projection sharpness
     * @param {number} [eta=0.5] - threshold level
     * @returns {Float32Array} projected physical densities
     */
    _heavisideProject(x, beta, eta = 0.5) {
        const tanhBeta = Math.tanh(beta * eta);
        const tanhB1Eta = Math.tanh(beta * (1 - eta));
        const denom = tanhBeta + tanhB1Eta;
        const xPhys = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            xPhys[i] = (tanhBeta + Math.tanh(beta * (x[i] - eta))) / denom;
        }
        return xPhys;
    }

    /**
     * Allocate WASM memory buffers for full-solve ebePCG.
     * Re-allocates only when problem dimensions change.
     */
    _initWasmPCGBuffers(nel, ndof, nfree) {
        if (this._wasmPCG &&
            this._wasmPCG.nel === nel &&
            this._wasmPCG.ndof === ndof &&
            this._wasmPCG.nfree === nfree) return true;

        if (!wasmModule || !wasmModule.exports.ebePCG) return false;

        try {
            const mem = wasmModule.exports.memory;
            const edofSize = 24;
            const align8 = (v) => (v + 7) & ~7;

            // Input buffers
            const densSize    = nel * 8;           // f64[nel]
            const keSize      = 576 * 8;           // f64[24*24]
            const edofsSize   = nel * 24 * 4;      // i32[nel*24]
            const fSize       = ndof * 8;          // f64[ndof]
            const uSize       = ndof * 8;          // f64[ndof]
            const freedofsSize = nfree * 4;         // i32[nfree]

            // Workspace: E_vals[nel] + active[nel as i32] + diag[ndof] +
            //   Uf[nfree] + r[nfree] + z[nfree] + p[nfree] + Ap[nfree] +
            //   pfull[ndof] + apfull[ndof] + scratch[edofSize] + invDiagSafe[nfree]
            const workSize = nel * 8 + nel * 4 + ndof * 8 +
                nfree * 8 * 5 + ndof * 8 * 2 + edofSize * 8 + nfree * 8;

            const totalBytes = densSize + keSize + edofsSize + fSize + uSize + freedofsSize + workSize + 1024;

            const neededPages = Math.ceil(totalBytes / 65536) + 1;
            const dataStart = mem.buffer.byteLength;
            const growResult = mem.grow(neededPages);
            if (growResult === -1) return false;

            let off = dataStart;

            const densOff = off;      off += densSize;
            const keOff = off;        off += keSize;
            const edofsOff = off;     off += edofsSize; off = align8(off);
            const fOff = off;         off += fSize;
            const uOff = off;         off += uSize;
            const freedofsOff = off;  off += freedofsSize; off = align8(off);
            const workOff = off;

            this._wasmPCG = { nel, ndof, nfree, densOff, keOff, edofsOff, fOff, uOff, freedofsOff, workOff };
            return true;
        } catch (e) {
            console.warn('Failed to allocate WASM PCG buffers:', e);
            return false;
        }
    }

    /**
     * Full FEA solve using the WASM ebePCG kernel.
     * The entire Jacobi-PCG loop runs in WASM with zero per-iteration
     * JS↔WASM boundary crossings.
     */
    _solveWithWasmPCG(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, iteration, maxIterations) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const nel = nelx * nely * nelz;
        const nfree = freedofs.length;

        if (!this._initWasmPCGBuffers(nel, ndof, nfree)) return null;

        const mem = wasmModule.exports.memory;
        const w = this._wasmPCG;

        // Copy inputs to WASM memory
        new Float64Array(mem.buffer, w.densOff, nel).set(x.subarray(0, nel));
        new Float64Array(mem.buffer, w.keOff, 576).set(KEflat);
        new Int32Array(mem.buffer, w.edofsOff, nel * 24).set(edofArray);
        new Float64Array(mem.buffer, w.fOff, ndof).set(F);
        new Int32Array(mem.buffer, w.freedofsOff, nfree).set(freedofs);

        // Warm-start: copy previous U (or zero if first iteration)
        const uWasm = new Float64Array(mem.buffer, w.uOff, ndof);
        if (iteration > 1 && this._U_prev) {
            uWasm.set(this._U_prev);
        } else {
            uWasm.fill(0);
        }

        // Schedule CG tolerance
        const progress = maxIterations > 1 ? (iteration - 1) / (maxIterations - 1) : 1;
        const tolerance = Math.exp(
            Math.log(CG_TOL_START) + progress * (Math.log(CG_TOL_END) - Math.log(CG_TOL_START))
        );
        const maxIter = Math.min(nfree, MAX_CG_ITERATIONS);

        // Call full WASM FEA solver
        const cgIters = wasmModule.exports.ebePCG(
            w.densOff, w.keOff, w.edofsOff, w.fOff, w.uOff, w.freedofsOff,
            nel, 24, ndof, nfree,
            this.Emin, this.E0, penal, maxIter, tolerance, w.workOff
        );

        // Allocate / reuse output buffers
        if (!this._U || this._U.length !== ndof) {
            this._U = new Float64Array(ndof);
            this._U_prev = new Float64Array(ndof);
        }

        // Read U back from WASM
        const U = this._U;
        U.set(new Float64Array(mem.buffer, w.uOff, ndof));

        // Save for warm-start
        this._U_prev.set(U);

        // Compute compliance
        let c = 0;
        for (let i = 0; i < ndof; i++) c += F[i] * U[i];

        return { U, c, solverStats: { cgIterations: cgIters, tolerance } };
    }

    /**
     * Serialize JS multigrid levels into WASM linear memory and call ebeMGPCG.
     * Returns null if WASM MGPCG is unavailable.
     *
     * WASM level descriptor layout (80 bytes / level):
     *   0: nelx   4: nely   8: nelz   12: ndof  16: nel  20: nfree
     *  24: activeCount  28: hasDenseK  32: edofsPtr  36: evalsPtr
     *  40: activePtr    44: fixedMaskPtr 48: freeDofsPtr 52: invDiagPtr
     *  56: denseKPtr    60: AuPtr       64: resPtr      68: bPtr
     *  72: xPtr         76: scratchPtr
     */
    _solveWithWasmMGPCG(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, iteration, maxIterations) {
        if (!wasmLoaded || !wasmModule?.exports?.ebeMGPCG) return null;

        // Ensure MG hierarchy is built & updated
        if (!this._mg) {
            this._mg = new MGPrecond3D(KEflat);
        }
        this._mg.ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs);
        this._mg.updateFromFine(x, penal, this.E0, this.Emin);

        const levels = this._mg.levels;
        const numLevels = levels.length;
        const ndof0 = levels[0].ndof;
        const MG_LEVEL_STRIDE = 80;
        const align8 = (v) => (v + 7) & ~7;

        // ── Calculate total WASM memory needed ──
        let totalBytes = 0;
        totalBytes += 576 * 8;                     // KEflat (shared)
        totalBytes += MG_LEVEL_STRIDE * numLevels; // level descriptors
        totalBytes += ndof0 * 8 * 4;               // CG workspace: r, z, p, Ap
        totalBytes += ndof0 * 8;                   // F (force vector)
        totalBytes += ndof0 * 8;                   // U (displacement)
        totalBytes += ndof0;                       // fixedMask (u8)
        totalBytes += 64;                          // alignment padding

        for (const level of levels) {
            totalBytes += level.nel * 24 * 4 + 8;          // edofs + align
            totalBytes += level.nel * 8;                    // E_vals
            totalBytes += level.nel * 4 + 8;                // active + align
            totalBytes += level.ndof + 8;                   // fixedMask + align
            totalBytes += level.freeDofs.length * 4 + 8;    // freeDofs + align
            totalBytes += level.ndof * 8;                   // invDiag
            totalBytes += level.ndof * 8 * 4;               // Au, res, b, x
            totalBytes += 24 * 8;                           // scratch
            if (level.denseK) {
                totalBytes += level.ndof * level.ndof * 8;
            }
        }

        const mem = wasmModule.exports.memory;
        const neededPages = Math.ceil(totalBytes / 65536) + 2;
        const dataStart = mem.buffer.byteLength;
        const growResult = mem.grow(neededPages);
        if (growResult === -1) return null;

        let off = dataStart;

        // ── Write KEflat ──
        const keOff = off; off += 576 * 8;
        new Float64Array(mem.buffer, keOff, 576).set(KEflat);

        // ── Write F ──
        const fOff = off; off += ndof0 * 8;
        new Float64Array(mem.buffer, fOff, ndof0).set(F);

        // ── Write U ──
        const uOff = off; off += ndof0 * 8;
        new Float64Array(mem.buffer, uOff, ndof0).fill(0);

        // ── Write finest fixedMask (u8) ──
        const fixedMask0Off = off; off += ndof0; off = align8(off);
        new Uint8Array(mem.buffer, fixedMask0Off, ndof0).set(fixedMask);

        // ── CG workspace ──
        const cgWorkOff = off; off += ndof0 * 8 * 4;

        // ── Level descriptors ──
        const levelsOff = off; off += MG_LEVEL_STRIDE * numLevels;

        // ── Per-level data ──
        for (let li = 0; li < numLevels; li++) {
            const level = levels[li];
            const descOff = levelsOff + li * MG_LEVEL_STRIDE;

            // Allocate per-level arrays
            const edofsOff = off; off += level.nel * 24 * 4; off = align8(off);
            const evalsOff = off; off += level.nel * 8;
            const activeOff = off; off += level.nel * 4; off = align8(off);
            const fmOff = off; off += level.ndof; off = align8(off);
            const freeOff = off; off += level.freeDofs.length * 4; off = align8(off);
            const invDiagOff = off; off += level.ndof * 8;
            let denseKOff = 0;
            if (level.denseK) {
                denseKOff = off; off += level.ndof * level.ndof * 8;
            }
            const auOff = off; off += level.ndof * 8;
            const resOff = off; off += level.ndof * 8;
            const bOff = off; off += level.ndof * 8;
            const xOff = off; off += level.ndof * 8;
            const scratchOff = off; off += 24 * 8;

            // Copy data
            new Int32Array(mem.buffer, edofsOff, level.edofArray.length).set(level.edofArray);
            new Float64Array(mem.buffer, evalsOff, level.nel).set(level.E_vals);
            new Int32Array(mem.buffer, activeOff, level.activeCount).set(level.active.subarray(0, level.activeCount));
            new Uint8Array(mem.buffer, fmOff, level.ndof).set(level.fixedMask);
            new Int32Array(mem.buffer, freeOff, level.freeDofs.length).set(level.freeDofs);
            new Float64Array(mem.buffer, invDiagOff, level.ndof).set(level.invDiag);
            if (level.denseK) {
                new Float64Array(mem.buffer, denseKOff, level.ndof * level.ndof).set(level.denseK);
            }
            // Zero work vectors
            new Float64Array(mem.buffer, auOff, level.ndof).fill(0);
            new Float64Array(mem.buffer, resOff, level.ndof).fill(0);
            new Float64Array(mem.buffer, bOff, level.ndof).fill(0);
            new Float64Array(mem.buffer, xOff, level.ndof).fill(0);

            // Write level descriptor (80 bytes)
            const desc = new DataView(mem.buffer, descOff, MG_LEVEL_STRIDE);
            desc.setInt32(0, level.nelx, true);
            desc.setInt32(4, level.nely, true);
            desc.setInt32(8, level.nelz, true);
            desc.setInt32(12, level.ndof, true);
            desc.setInt32(16, level.nel, true);
            desc.setInt32(20, level.freeDofs.length, true);
            desc.setInt32(24, level.activeCount, true);
            desc.setInt32(28, level.denseK ? 1 : 0, true);
            desc.setUint32(32, edofsOff, true);
            desc.setUint32(36, evalsOff, true);
            desc.setUint32(40, activeOff, true);
            desc.setUint32(44, fmOff, true);
            desc.setUint32(48, freeOff, true);
            desc.setUint32(52, invDiagOff, true);
            desc.setUint32(56, denseKOff, true);
            desc.setUint32(60, auOff, true);
            desc.setUint32(64, resOff, true);
            desc.setUint32(68, bOff, true);
            desc.setUint32(72, xOff, true);
            desc.setUint32(76, scratchOff, true);
        }

        // Schedule CG tolerance
        const progress = maxIterations > 1 ? (iteration - 1) / (maxIterations - 1) : 1;
        const tolerance = Math.exp(
            Math.log(CG_TOL_START) + progress * (Math.log(CG_TOL_END) - Math.log(CG_TOL_START))
        );
        const nfree = freedofs.length;
        const maxIter = Math.min(nfree, MAX_CG_ITERATIONS);

        // Call full WASM MGPCG solver
        const cgIters = wasmModule.exports.ebeMGPCG(
            levelsOff, numLevels, keOff,
            fOff, uOff, fixedMask0Off,
            ndof0, maxIter, tolerance, cgWorkOff
        );

        // Allocate / reuse output buffers
        if (!this._U || this._U.length !== ndof0) {
            this._U = new Float64Array(ndof0);
            this._U_prev = new Float64Array(ndof0);
        }

        // Read U back from WASM
        const U = this._U;
        U.set(new Float64Array(mem.buffer, uOff, ndof0));

        // Save for warm-start
        this._U_prev.set(U);

        // Compute compliance
        let c2 = 0;
        for (let i = 0; i < ndof0; i++) c2 += F[i] * U[i];

        return { U, c: c2, solverStats: { cgIterations: cgIters, tolerance } };
    }

    /**
     * Full FEA solve using the WASM ebeKSP_BDDC kernel.
     * The entire PCG loop with BDDC domain decomposition preconditioner
     * runs in WASM with zero per-iteration JS↔WASM boundary crossings.
     */
    _solveWithWasmKSP_BDDC(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, iteration, maxIterations) {
        if (!wasmLoaded || !wasmModule?.exports?.ebeKSP_BDDC) return null;

        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const nel = nelx * nely * nelz;
        const nfree = freedofs.length;
        const dE = this.E0 - this.Emin;
        const skipThreshold = this.Emin * 1000;

        // Build E_vals and active elements
        const E_vals = new Float64Array(nel);
        const active = [];
        for (let e = 0; e < nel; e++) {
            const E = this.Emin + Math.pow(x[e], penal) * dE;
            E_vals[e] = E;
            if (E > skipThreshold) active.push(e);
        }
        const activeCount = active.length;

        // Build global diagonal and inverse diagonal
        const diag = new Float64Array(ndof);
        for (let ai = 0; ai < activeCount; ai++) {
            const e = active[ai];
            const E = E_vals[e];
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) {
                diag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
            }
        }
        const invDiag = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) {
            if (!fixedMask[i] && diag[i] > 1e-30) invDiag[i] = 1.0 / diag[i];
        }

        // Build KSP solver (for its subdomain partitioning)
        if (!this._ksp) {
            this._ksp = new KSPSolver(KEflat, { pc: 'bddc' });
        }
        this._ksp.ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs);
        this._ksp.updateOperators(x, penal, this.E0, this.Emin);

        const subdomains = this._ksp._subdomains;
        const coarseMap = this._ksp._coarseMap;
        if (!subdomains || !coarseMap) return null;

        const numSubs = subdomains.length;
        const numCoarse = coarseMap.length;
        const BDDC_SUB_STRIDE = 32;
        const align8 = (v) => (v + 7) & ~7;

        // Calculate total WASM memory needed
        let totalBytes = 0;
        totalBytes += 576 * 8;               // KEflat
        totalBytes += nel * 24 * 4 + 8;      // edofs + align
        totalBytes += nel * 8;               // evals
        totalBytes += activeCount * 4 + 8;   // active + align
        totalBytes += ndof + 8;              // fixedMask + align
        totalBytes += ndof * 8;              // invDiag
        totalBytes += ndof * 8;              // F
        totalBytes += ndof * 8;              // U
        totalBytes += numCoarse * 4 + 8;     // coarseMap + align
        totalBytes += BDDC_SUB_STRIDE * numSubs; // subdomain descriptors
        // Per-subdomain data
        for (const sub of subdomains) {
            totalBytes += sub.elements.length * 4 + 8;  // elements + align
            totalBytes += sub.dofs.length * 4 + 8;      // dofs + align
            totalBytes += ndof * 8;                     // localInvDiag
        }
        // Workspace: 7 * ndof * 8 + 192 (scratch24)
        totalBytes += 7 * ndof * 8 + 192 + 1024;

        const mem = wasmModule.exports.memory;
        const neededPages = Math.ceil(totalBytes / 65536) + 2;
        const dataStart = mem.buffer.byteLength;
        const growResult = mem.grow(neededPages);
        if (growResult === -1) return null;

        let off = dataStart;

        // Write KEflat
        const keOff = off; off += 576 * 8;
        new Float64Array(mem.buffer, keOff, 576).set(KEflat);

        // Write edofs
        const edofsOff = off; off += nel * 24 * 4; off = align8(off);
        new Int32Array(mem.buffer, edofsOff, nel * 24).set(edofArray);

        // Write evals
        const evalsOff = off; off += nel * 8;
        new Float64Array(mem.buffer, evalsOff, nel).set(E_vals);

        // Write active elements
        const activeOff = off; off += activeCount * 4; off = align8(off);
        new Int32Array(mem.buffer, activeOff, activeCount).set(new Int32Array(active));

        // Write fixedMask
        const fixedOff = off; off += ndof; off = align8(off);
        new Uint8Array(mem.buffer, fixedOff, ndof).set(fixedMask);

        // Write invDiag
        const invDiagOff = off; off += ndof * 8;
        new Float64Array(mem.buffer, invDiagOff, ndof).set(invDiag);

        // Write F
        const fOff = off; off += ndof * 8;
        new Float64Array(mem.buffer, fOff, ndof).set(F);

        // Write U
        const uOff = off; off += ndof * 8;
        new Float64Array(mem.buffer, uOff, ndof).fill(0);

        // Write coarse map
        const coarseOff = off; off += numCoarse * 4; off = align8(off);
        new Int32Array(mem.buffer, coarseOff, numCoarse).set(new Int32Array(coarseMap));

        // Write subdomain descriptors and data
        const subsOff = off; off += BDDC_SUB_STRIDE * numSubs;

        for (let si = 0; si < numSubs; si++) {
            const sub = subdomains[si];
            const descOff = subsOff + si * BDDC_SUB_STRIDE;

            // Write subdomain elements
            const elemOff = off; off += sub.elements.length * 4; off = align8(off);
            new Int32Array(mem.buffer, elemOff, sub.elements.length).set(new Int32Array(sub.elements));

            // Write subdomain DOFs
            const dofsOff = off; off += sub.dofs.length * 4; off = align8(off);
            new Int32Array(mem.buffer, dofsOff, sub.dofs.length).set(new Int32Array(sub.dofs));

            // Write local inverse diagonal
            const localInvDiagOff = off; off += ndof * 8;
            new Float64Array(mem.buffer, localInvDiagOff, ndof).set(sub.localInvDiag);

            // Write descriptor
            const desc = new DataView(mem.buffer, descOff, BDDC_SUB_STRIDE);
            desc.setInt32(0, sub.elements.length, true);
            desc.setInt32(4, sub.dofs.length, true);
            desc.setUint32(8, elemOff, true);
            desc.setUint32(12, dofsOff, true);
            desc.setUint32(16, localInvDiagOff, true);
            desc.setInt32(20, 0, true);
            desc.setInt32(24, 0, true);
            desc.setInt32(28, 0, true);
        }

        // Workspace
        const workOff = off;

        // Schedule CG tolerance
        const progress = maxIterations > 1 ? (iteration - 1) / (maxIterations - 1) : 1;
        const tolerance = Math.exp(
            Math.log(CG_TOL_START) + progress * (Math.log(CG_TOL_END) - Math.log(CG_TOL_START))
        );
        const maxIter = Math.min(nfree, MAX_CG_ITERATIONS);

        // Call WASM KSP BDDC solver
        const cgIters = wasmModule.exports.ebeKSP_BDDC(
            subsOff, numSubs, coarseOff, numCoarse,
            keOff, edofsOff, evalsOff,
            activeOff, activeCount,
            fixedOff, invDiagOff,
            fOff, uOff,
            ndof, maxIter, tolerance,
            KSP_BDDC_SMOOTHER_ITERS, workOff
        );

        // Allocate / reuse output buffers
        if (!this._U || this._U.length !== ndof) {
            this._U = new Float64Array(ndof);
            this._U_prev = new Float64Array(ndof);
        }

        // Read U back from WASM
        const U = this._U;
        U.set(new Float64Array(mem.buffer, uOff, ndof));

        // Save for warm-start
        this._U_prev.set(U);

        // Compute compliance
        let c = 0;
        for (let i = 0; i < ndof; i++) c += F[i] * U[i];

        return { U, c, solverStats: { cgIterations: cgIters, tolerance } };
    }

    // ═══════════════════════════════════════════════════════════════════
    // FEA solver backend selection & GPU FEA integration
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Initialize GPUFEASolver lazily (only once per process).
     * Called when feaSolverBackend is 'auto' or 'webgpu'.
     */
    async _initGPUFEASolver() {
        if (this._gpuFEAInitAttempted) return;
        if (this.feaSolverBackend !== 'auto' && this.feaSolverBackend !== 'webgpu') return;
        this._gpuFEAInitAttempted = true;
        try {
            this.gpuFEASolver = new GPUFEASolver();
            const ok = await this.gpuFEASolver.init();
            this._gpuFEAAvailable = ok;
            if (ok) {
                console.log('GPU FEA solver initialized (GPUFEASolver ready)');
            } else {
                console.log('GPU FEA solver probe failed — will use WASM/JS');
                this.gpuFEASolver = null;
            }
        } catch (err) {
            console.warn('GPU FEA solver init error:', err.message);
            this._gpuFEAAvailable = false;
            this.gpuFEASolver = null;
        }
    }

    /**
     * Release GPU resources (Dawn device) before the worker exits.
     * Must be called before posting the 'complete' message so the next worker
     * can safely create a new Dawn instance on the same GPU (only one Dawn
     * instance per D3D12 device is allowed at a time).
     */
    _cleanupGPU() {
        if (this.gpuFEASolver) {
            this.gpuFEASolver.destroy();
            this.gpuFEASolver = null;
        }
        if (this.gpuCompute) {
            if (typeof this.gpuCompute.destroy === 'function') this.gpuCompute.destroy();
            this.gpuCompute = null;
        }
        this._gpuFEAAvailable = false;
        this._gpuFEAInitAttempted = false;
    }

    /**
     * Resolve the actual backend to use based on feaSolverBackend setting
     * and what's actually available.
     * Order for 'auto': webgpu > wasm > js
     * @returns {'webgpu'|'wasm'|'js'}
     */
    _resolveBackend() {
        const pref = this.feaSolverBackend || 'auto';
        if (pref === 'webgpu') {
            this._resolvedBackend = this._gpuFEAAvailable ? 'webgpu' : (wasmLoaded ? 'wasm' : 'js');
        } else if (pref === 'wasm') {
            this._resolvedBackend = wasmLoaded ? 'wasm' : 'js';
        } else if (pref === 'js') {
            this._resolvedBackend = 'js';
        } else {
            // 'auto': best available
            if (this._gpuFEAAvailable) {
                this._resolvedBackend = 'webgpu';
            } else if (wasmLoaded) {
                this._resolvedBackend = 'wasm';
            } else {
                this._resolvedBackend = 'js';
            }
        }
        return this._resolvedBackend;
    }

    /**
     * Solve using GPUFEASolver (GPU-resident Jacobi-PCG).
     * Note: GPU solver always uses Jacobi preconditioner regardless of linearSolver setting
     * (MGPCG/KSP preconditioners are CPU-only algorithms).
     * Returns null if the solve fails, allowing fallback to WASM/JS.
     */
    async _solveWithGPUFEA(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, config, iteration, maxIterations, fixeddofs) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const nel = nelx * nely * nelz;

        try {
            // Schedule CG tolerance
            const progress = maxIterations > 1 ? (iteration - 1) / (maxIterations - 1) : 1;
            const tolerance = Math.exp(Math.log(CG_TOL_START) + progress * (Math.log(CG_TOL_END) - Math.log(CG_TOL_START)));

            // Convert inputs to f32 for GPU
            const KEflat32 = this._KEflat32 || new Float32Array(KEflat);
            const densities32 = (x instanceof Float32Array) ? x : new Float32Array(x);
            const fixedMaskU8 = (fixedMask instanceof Uint8Array) ? fixedMask : new Uint8Array(ndof);
            if (!(fixedMask instanceof Uint8Array)) {
                for (let i = 0; i < ndof; i++) fixedMaskU8[i] = fixedMask[i] ? 1 : 0;
            }
            const F32 = new Float32Array(ndof);
            for (let i = 0; i < ndof; i++) F32[i] = F[i];

            // Setup GPU buffers (re-uploads every iteration since densities change)
            this.gpuFEASolver.setup({
                KEflat: KEflat32,
                edofArray: new Int32Array(edofArray),
                densities: densities32,
                F: F32,
                fixedMask: fixedMaskU8,
                nel, ndof,
                E0: this.E0,
                Emin: this.Emin,
                penal
            });

            // Warm-start from previous solution
            let warmStart = null;
            if (iteration > 1 && this._U_prev && this._U_prev.length === ndof) {
                warmStart = new Float32Array(ndof);
                for (let i = 0; i < ndof; i++) warmStart[i] = this._U_prev[i];
            }

            const result = await this.gpuFEASolver.solve({
                maxIterations: Math.min(freedofs.length, MAX_CG_ITERATIONS),
                tolerance,
                warmStart
            });

            // Convert solution back to f64 for compatibility
            const U = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) U[i] = result.U[i];

            // Zero fixed DOFs
            if (fixeddofs) {
                for (const dof of fixeddofs) {
                    if (dof >= 0 && dof < ndof) U[dof] = 0;
                }
            }

            // Save for warm-start
            if (!this._U_prev || this._U_prev.length !== ndof) {
                this._U_prev = new Float64Array(ndof);
            }
            this._U_prev.set(U);

            // Compute compliance
            let c = 0;
            for (let i = 0; i < ndof; i++) c += F[i] * U[i];

            return { U, c, solverStats: { cgIterations: result.iterations, tolerance, backend: 'webgpu', converged: result.converged } };
        } catch (err) {
            console.warn('GPU FEA solve failed, falling back:', err.message);
            return null;
        }
    }

    async FE(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, config, iteration, maxIterations, fixeddofs) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const nel = nelx * nely * nelz;

        const backend = this._resolvedBackend || this._resolveBackend();

        // ── GPU FEA solve (GPU-resident Jacobi-PCG via GPUFEASolver) ──
        if (backend === 'webgpu' && this._gpuFEAAvailable && this.gpuFEASolver) {
            const result = await this._solveWithGPUFEA(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, config, iteration, maxIterations, fixeddofs);
            if (result) return result;
            // Fall through to WASM/JS if GPU solve failed
        }

        // ── Full WASM MGPCG solve (entire V-cycle + CG in WASM, zero per-iteration crossings) ──
        if (backend !== 'js' && this.linearSolver === 'mgpcg' && wasmLoaded && wasmModule?.exports?.ebeMGPCG) {
            const result = this._solveWithWasmMGPCG(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, iteration, maxIterations);
            if (result) return result;
            // Fall through to JS MGPCG if WASM allocation failed
        }

        // ── Full WASM KSP BDDC solve (BDDC domain decomposition in WASM, zero per-iteration crossings) ──
        if (backend !== 'js' && this.linearSolver === 'petsc' && (!config?.petscPC || config.petscPC === 'bddc') && wasmLoaded && wasmModule?.exports?.ebeKSP_BDDC) {
            const result = this._solveWithWasmKSP_BDDC(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, fixedMask, iteration, maxIterations);
            if (result) return result;
            // Fall through to JS KSP solver if WASM allocation failed
        }

        // ── Full WASM FEA solve (Jacobi-PCG, zero per-iteration JS↔WASM crossings) ──
        if (backend !== 'js' && this.linearSolver !== 'mgpcg' && this.linearSolver !== 'petsc' && wasmLoaded && wasmModule?.exports?.ebePCG) {
            const result = this._solveWithWasmPCG(nelx, nely, nelz, x, penal, KEflat, edofArray, F, freedofs, iteration, maxIterations);
            if (result) return result;
            // Fall through to JS solver if WASM allocation failed
        }

        // Schedule CG tolerance: looser early, tighter late
        const progress = maxIterations > 1 ? (iteration - 1) / (maxIterations - 1) : 1;
        const logStart = Math.log(CG_TOL_START);
        const logEnd = Math.log(CG_TOL_END);
        const tolerance = Math.exp(logStart + progress * (logEnd - logStart));

        // Lazily build / update multigrid preconditioner
        if (this.linearSolver === 'mgpcg') {
            if (!this._mg) {
                this._mg = new MGPrecond3D(KEflat);
            }
            this._mg.ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs);
            this._mg.updateFromFine(x, penal, this.E0, this.Emin);
        }

        // Lazily build / update KSP (PETSc-style) solver with BDDC or MG preconditioning
        if (this.linearSolver === 'petsc') {
            const pcType = config.petscPC || 'bddc';
            if (!this._ksp) {
                this._ksp = new KSPSolver(KEflat, { pc: pcType });
            }
            this._ksp.ensure(nelx, nely, nelz, edofArray, fixedMask, freedofs);
            this._ksp.updateOperators(x, penal, this.E0, this.Emin);
        }

        // Allocate / reuse solver buffers
        if (!this._U || this._U.length !== ndof) {
            this._U = new Float64Array(ndof);
            this._U_prev = new Float64Array(ndof);
            this._pcg_r = new Float64Array(ndof);
            this._pcg_z = new Float64Array(ndof);
            this._pcg_p = new Float64Array(ndof);
            this._pcg_Ap = new Float64Array(ndof);
        }

        const U = this._U;

        // Warm-start: copy previous solution
        if (iteration > 1 && this._U_prev) {
            U.set(this._U_prev);
        } else {
            U.fill(0);
        }

        // Zero out fixed DOFs
        if (fixeddofs) {
            for (let i = 0; i < fixeddofs.length; i++) {
                const dof = fixeddofs[i];
                if (dof >= 0 && dof < ndof) U[dof] = 0;
            }
        }

        // Precompute element stiffnesses
        const { E_vals, activeElements } = this._precomputeStiffness(x, penal, nel);

        const r = this._pcg_r;
        const z = this._pcg_z;
        const p = this._pcg_p;
        const Ap = this._pcg_Ap;

        // Compute initial residual: r = F - K*U
        if (this.linearSolver === 'mgpcg' && this._mg) {
            this._mg.applyA(0, U, r);
        } else if (this.linearSolver === 'petsc' && this._ksp) {
            this._ksp.applyA(U, r);
        } else {
            this._fullSpaceMatVec(E_vals, activeElements, KEflat, edofArray, nel, U, r, ndof);
        }

        for (let i = 0; i < ndof; i++) {
            r[i] = F[i] - r[i];
        }

        // Zero fixed DOFs in residual
        if (fixedMask) {
            for (let i = 0; i < ndof; i++) {
                if (fixedMask[i]) r[i] = 0;
            }
        }

        // Apply preconditioner: z = M^{-1} r
        if (this.linearSolver === 'mgpcg' && this._mg) {
            this._mg.apply(r, z);
        } else if (this.linearSolver === 'petsc' && this._ksp) {
            this._ksp.apply(r, z);
        } else {
            // Jacobi preconditioner fallback
            if (!this._fullInvDiag || this._fullInvDiag.length !== ndof) {
                this._fullInvDiag = new Float64Array(ndof);
            }
            this._computeFullDiagonal(E_vals, activeElements, KEflat, edofArray, nel, ndof, fixedMask, this._fullInvDiag);
            for (let i = 0; i < ndof; i++) z[i] = this._fullInvDiag[i] * r[i];
        }

        // Zero fixed DOFs
        if (fixedMask) {
            for (let i = 0; i < ndof; i++) {
                if (fixedMask[i]) z[i] = 0;
            }
        }

        // p = z
        p.set(z);

        // rz = r^T z
        let rz = 0;
        for (let i = 0; i < ndof; i++) rz += r[i] * z[i];

        // Initial residual norm for relative convergence check
        let r0norm2 = 0;
        for (let i = 0; i < ndof; i++) r0norm2 += r[i] * r[i];
        const tolSq = tolerance * tolerance * Math.max(r0norm2, 1e-30);

        let cgIters = 0;
        const maxIter = Math.min(freedofs.length, MAX_CG_ITERATIONS);

        for (let iter = 0; iter < maxIter; iter++) {
            // Check convergence
            let rnorm2 = 0;
            for (let i = 0; i < ndof; i++) rnorm2 += r[i] * r[i];
            if (rnorm2 < tolSq) break;
            cgIters = iter + 1;

            // Ap = K * p
            if (this.linearSolver === 'mgpcg' && this._mg) {
                this._mg.applyA(0, p, Ap);
            } else if (this.linearSolver === 'petsc' && this._ksp) {
                this._ksp.applyA(p, Ap);
            } else {
                this._fullSpaceMatVec(E_vals, activeElements, KEflat, edofArray, nel, p, Ap, ndof);
            }
            // Zero fixed DOFs
            if (fixedMask) {
                for (let i = 0; i < ndof; i++) {
                    if (fixedMask[i]) Ap[i] = 0;
                }
            }

            let pAp = 0;
            for (let i = 0; i < ndof; i++) pAp += p[i] * Ap[i];
            const alpha = rz / (pAp + EPSILON);

            for (let i = 0; i < ndof; i++) {
                U[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            // Apply preconditioner
            if (this.linearSolver === 'mgpcg' && this._mg) {
                this._mg.apply(r, z);
            } else if (this.linearSolver === 'petsc' && this._ksp) {
                this._ksp.apply(r, z);
            } else {
                for (let i = 0; i < ndof; i++) z[i] = this._fullInvDiag[i] * r[i];
            }
            // Zero fixed DOFs
            if (fixedMask) {
                for (let i = 0; i < ndof; i++) {
                    if (fixedMask[i]) z[i] = 0;
                }
            }

            let rz_new = 0;
            for (let i = 0; i < ndof; i++) rz_new += r[i] * z[i];

            const beta = rz_new / (rz + EPSILON);
            for (let i = 0; i < ndof; i++) {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }

        // Save for warm-start
        this._U_prev.set(U);

        // Compute compliance
        let c = 0;
        for (let i = 0; i < ndof; i++) {
            c += F[i] * U[i];
        }

        return { U, c, solverStats: { cgIterations: cgIters, tolerance } };
    }

    /**
     * Full-space element-by-element matrix-vector product (fallback for non-MGPCG path).
     */
    _fullSpaceMatVec(E_vals, activeElements, KEflat, edofArray, nel, p, Ap, ndof) {
        Ap.fill(0);
        const loc = new Float64Array(24);
        for (let ae = 0, aeLen = activeElements.length; ae < aeLen; ae++) {
            const e = activeElements[ae];
            const E = E_vals[e];
            const eOff = e * 24;
            for (let j = 0; j < 24; j++) {
                loc[j] = p[edofArray[eOff + j]];
            }
            for (let i = 0; i < 24; i++) {
                const gi = edofArray[eOff + i];
                let sum = 0;
                const keRow = i * 24;
                for (let j = 0; j < 24; j++) {
                    sum += KEflat[keRow + j] * loc[j];
                }
                Ap[gi] += E * sum;
            }
        }
    }

    /**
     * Compute full-space inverse diagonal for Jacobi preconditioner (fallback).
     */
    _computeFullDiagonal(E_vals, activeElements, KEflat, edofArray, nel, ndof, fixedMask, invDiag) {
        const diag = new Float64Array(ndof);
        for (let ae = 0, aeLen = activeElements.length; ae < aeLen; ae++) {
            const e = activeElements[ae];
            const E = E_vals[e];
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) {
                diag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
            }
        }
        invDiag.fill(0);
        for (let i = 0; i < ndof; i++) {
            if (!fixedMask[i] && diag[i] > 1e-30) {
                invDiag[i] = 1.0 / diag[i];
            }
        }
    }

    computeElementEnergy(KE, Ue) {
        let energy = 0;
        for (let i = 0; i < 24; i++) {
            for (let j = 0; j < 24; j++) {
                energy += Ue[i] * KE[i][j] * Ue[j];
            }
        }
        return energy;
    }

    _addNodeDOFs(fixeddofs, n, constraintDOFs) {
        const mode = constraintDOFs || 'all';
        if (mode === 'all' || mode.includes('x')) fixeddofs.push(3 * n);
        if (mode === 'all' || mode.includes('y')) fixeddofs.push(3 * n + 1);
        if (mode === 'all' || mode.includes('z')) fixeddofs.push(3 * n + 2);
    }

    getFixedDOFs(nelx, nely, nelz, position, constraintDOFs) {
        const fixeddofs = [];
        const nny = nely + 1;
        const nnz = nelz + 1;

        switch (position) {
            case 'left':
                // Fix all nodes at x=0
                for (let j = 0; j <= nely; j++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = 0 * nny * nnz + j * nnz + k;
                        this._addNodeDOFs(fixeddofs, n, constraintDOFs);
                    }
                }
                break;
            case 'right':
                // Fix all nodes at x=nelx
                for (let j = 0; j <= nely; j++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = nelx * nny * nnz + j * nnz + k;
                        this._addNodeDOFs(fixeddofs, n, constraintDOFs);
                    }
                }
                break;
            case 'bottom':
                // Fix all nodes at y=0
                for (let i = 0; i <= nelx; i++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = i * nny * nnz + 0 * nnz + k;
                        this._addNodeDOFs(fixeddofs, n, constraintDOFs);
                    }
                }
                break;
            case 'top':
                // Fix all nodes at y=nely
                for (let i = 0; i <= nelx; i++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = i * nny * nnz + nely * nnz + k;
                        this._addNodeDOFs(fixeddofs, n, constraintDOFs);
                    }
                }
                break;
            case 'bottom-corners':
                // Fix 4 corners at bottom (z=0, y=0)
                const corners = [
                    0 * nny * nnz + 0 * nnz + 0,              // (0,0,0)
                    nelx * nny * nnz + 0 * nnz + 0,           // (nelx,0,0)
                    0 * nny * nnz + nely * nnz + 0,           // (0,nely,0)
                    nelx * nny * nnz + nely * nnz + 0         // (nelx,nely,0)
                ];
                for (const n of corners) {
                    this._addNodeDOFs(fixeddofs, n, constraintDOFs);
                }
                break;
        }

        return fixeddofs;
    }

    getLoadVector(nelx, nely, nelz, direction, magnitude, forceVector) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const F = new Float32Array(ndof);
        const nny = nely + 1;
        const nnz = nelz + 1;

        if (forceVector && Array.isArray(forceVector) && forceVector.length >= 3) {
            // Custom force vector: apply at top-right corner
            const len = Math.sqrt(forceVector[0] ** 2 + forceVector[1] ** 2 + forceVector[2] ** 2);
            if (len > 0) {
                const n = nelx * nny * nnz + nely * nnz + nelz;
                F[3 * n]     = (forceVector[0] / len) * magnitude;
                F[3 * n + 1] = (forceVector[1] / len) * magnitude;
                F[3 * n + 2] = (forceVector[2] / len) * magnitude;
            }
            return F;
        }

        switch (direction) {
            case 'down': {
                // Apply force downward at top-right corner (nelx, nely, nelz)
                const n_down = nelx * nny * nnz + nely * nnz + nelz;
                F[3 * n_down + 1] = -magnitude;
                break;
            }
            case 'up': {
                // Apply force upward at top-right corner
                const n_up = nelx * nny * nnz + nely * nnz + nelz;
                F[3 * n_up + 1] = magnitude;
                break;
            }
            case 'left': {
                // Apply force leftward at right-middle
                const n_left = nelx * nny * nnz + Math.floor(nely / 2) * nnz + Math.floor(nelz / 2);
                F[3 * n_left] = -magnitude;
                break;
            }
            case 'right': {
                // Apply force rightward at left-middle
                const n_right = 0 * nny * nnz + Math.floor(nely / 2) * nnz + Math.floor(nelz / 2);
                F[3 * n_right] = magnitude;
                break;
            }
            case 'top-center': {
                // Apply force upward at center of top face (z = nelz)
                const n_top = Math.floor(nelx / 2) * nny * nnz + Math.floor(nely / 2) * nnz + nelz;
                F[3 * n_top + 2] = magnitude;
                break;
            }
        }

        return F;
    }

    /**
     * Get element indices along the constraint face/edge.
     */
    getConstraintElements(nelx, nely, nelz, position) {
        const elems = [];
        switch (position) {
            case 'left':
                for (let ez = 0; ez < nelz; ez++) {
                    for (let ey = 0; ey < nely; ey++) {
                        elems.push(0 + ey * nelx + ez * nelx * nely);
                    }
                }
                break;
            case 'right':
                for (let ez = 0; ez < nelz; ez++) {
                    for (let ey = 0; ey < nely; ey++) {
                        elems.push((nelx - 1) + ey * nelx + ez * nelx * nely);
                    }
                }
                break;
            case 'bottom':
                for (let ez = 0; ez < nelz; ez++) {
                    for (let ex = 0; ex < nelx; ex++) {
                        elems.push(ex + 0 * nelx + ez * nelx * nely);
                    }
                }
                break;
            case 'top':
                for (let ez = 0; ez < nelz; ez++) {
                    for (let ex = 0; ex < nelx; ex++) {
                        elems.push(ex + (nely - 1) * nelx + ez * nelx * nely);
                    }
                }
                break;
            case 'bottom-corners':
                // 4 corner elements at z=0
                elems.push(0 + 0 * nelx + 0 * nelx * nely);                      // (0,0,0)
                elems.push((nelx - 1) + 0 * nelx + 0 * nelx * nely);             // (nelx-1,0,0)
                elems.push(0 + (nely - 1) * nelx + 0 * nelx * nely);             // (0,nely-1,0)
                elems.push((nelx - 1) + (nely - 1) * nelx + 0 * nelx * nely);    // (nelx-1,nely-1,0)
                break;
        }
        return elems;
    }

    /**
     * Get element indices where the default force is applied.
     */
    getForceElements(nelx, nely, nelz, direction) {
        const elems = [];
        switch (direction) {
            case 'down':
            case 'up':
                // Force at top-right corner — nearest element (nelx-1, nely-1, nelz-1)
                elems.push((nelx - 1) + (nely - 1) * nelx + (nelz - 1) * nelx * nely);
                break;
            case 'left':
                // Force at right-middle — element at (nelx-1, floor(nely/2), floor(nelz/2))
                elems.push((nelx - 1) + Math.floor(nely / 2) * nelx + Math.floor(nelz / 2) * nelx * nely);
                break;
            case 'right':
                // Force at left-middle — element at (0, floor(nely/2), floor(nelz/2))
                elems.push(0 + Math.floor(nely / 2) * nelx + Math.floor(nelz / 2) * nelx * nely);
                break;
            case 'top-center':
                // Force at center top — element at (floor(nelx/2), floor(nely/2), nelz-1)
                elems.push(Math.floor(nelx / 2) + Math.floor(nely / 2) * nelx + (nelz - 1) * nelx * nely);
                break;
        }
        return elems;
    }

    /**
     * Convert painted constraint face keys to fixed DOFs.
     * Face keys are "x,y,z,faceIndex" where x,y,z are voxel coordinates.
     * Maps to 3D nodes at the voxel corners.
     */
    getFixedDOFsFromPaint(nelx, nely, nelz, paintedKeys, constraintDOFs) {
        const dofSet = new Set();
        const nny = nely + 1;
        const nnz = nelz + 1;
        const mode = constraintDOFs || 'all';

        for (const key of paintedKeys) {
            const parts = key.split(',');
            if (parts.length < 3) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            const vz = parseInt(parts[2], 10);
            if (isNaN(vx) || isNaN(vy) || isNaN(vz)) continue;

            // Map voxel (vx, vy, vz) to its 8 corner nodes
            const n0 = vx * nny * nnz + vy * nnz + vz;
            const n1 = (vx + 1) * nny * nnz + vy * nnz + vz;
            const n2 = (vx + 1) * nny * nnz + (vy + 1) * nnz + vz;
            const n3 = vx * nny * nnz + (vy + 1) * nnz + vz;
            const n4 = vx * nny * nnz + vy * nnz + (vz + 1);
            const n5 = (vx + 1) * nny * nnz + vy * nnz + (vz + 1);
            const n6 = (vx + 1) * nny * nnz + (vy + 1) * nnz + (vz + 1);
            const n7 = vx * nny * nnz + (vy + 1) * nnz + (vz + 1);
            const allNodes = [n0, n1, n2, n3, n4, n5, n6, n7];

            // Use face-specific nodes when faceIndex is available
            const fi = parts.length >= 4 ? parseInt(parts[3], 10) : -1;
            const nodes = (fi >= 0 && fi < 6)
                ? FACE_NODE_INDICES[fi].map(i => allNodes[i])
                : allNodes;

            const maxNode = (nelx + 1) * (nely + 1) * (nelz + 1);
            for (const n of nodes) {
                if (n >= 0 && n < maxNode) {
                    if (mode === 'all' || mode.includes('x')) dofSet.add(3 * n);
                    if (mode === 'all' || mode.includes('y')) dofSet.add(3 * n + 1);
                    if (mode === 'all' || mode.includes('z')) dofSet.add(3 * n + 2);
                }
            }
        }
        return Array.from(dofSet);
    }

    /**
     * Convert painted force face keys to a load vector.
     * Distributes force evenly across all painted face nodes.
     */
    getLoadVectorFromPaint(nelx, nely, nelz, paintedKeys, direction, magnitude, forceVector) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const F = new Float32Array(ndof);
        const nny = nely + 1;
        const nnz = nelz + 1;

        // Determine force direction components
        let fx = 0, fy = 0, fz = 0;
        if (forceVector && Array.isArray(forceVector) && forceVector.length >= 3) {
            const len = Math.sqrt(forceVector[0] ** 2 + forceVector[1] ** 2 + forceVector[2] ** 2);
            if (len > 0) {
                fx = forceVector[0] / len;
                fy = forceVector[1] / len;
                fz = forceVector[2] / len;
            }
        } else {
            switch (direction) {
                case 'down':       fy = -1; break;
                case 'up':         fy = 1; break;
                case 'left':       fx = -1; break;
                case 'right':      fx = 1; break;
                case 'top-center': fz = 1; break;
                default:           fy = -1;
            }
        }

        // Collect unique nodes from painted faces, using face-specific nodes when available
        const nodeSet = new Set();
        for (const key of paintedKeys) {
            const parts = key.split(',');
            if (parts.length < 3) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            const vz = parseInt(parts[2], 10);
            if (isNaN(vx) || isNaN(vy) || isNaN(vz)) continue;

            const n0 = vx * nny * nnz + vy * nnz + vz;
            const n1 = (vx + 1) * nny * nnz + vy * nnz + vz;
            const n2 = (vx + 1) * nny * nnz + (vy + 1) * nnz + vz;
            const n3 = vx * nny * nnz + (vy + 1) * nnz + vz;
            const n4 = vx * nny * nnz + vy * nnz + (vz + 1);
            const n5 = (vx + 1) * nny * nnz + vy * nnz + (vz + 1);
            const n6 = (vx + 1) * nny * nnz + (vy + 1) * nnz + (vz + 1);
            const n7 = vx * nny * nnz + (vy + 1) * nnz + (vz + 1);
            const allNodes = [n0, n1, n2, n3, n4, n5, n6, n7];

            // Use face-specific nodes when faceIndex is available
            const fi = parts.length >= 4 ? parseInt(parts[3], 10) : -1;
            const nodes = (fi >= 0 && fi < 6)
                ? FACE_NODE_INDICES[fi].map(i => allNodes[i])
                : allNodes;

            const maxNode = (nelx + 1) * (nely + 1) * (nelz + 1);
            for (const n of nodes) {
                if (n >= 0 && n < maxNode) {
                    nodeSet.add(n);
                }
            }
        }

        // Distribute force evenly across all unique nodes
        const nodeCount = nodeSet.size;
        if (nodeCount > 0) {
            const forcePerNode = magnitude / nodeCount;
            for (const n of nodeSet) {
                F[3 * n] += fx * forcePerNode;
                F[3 * n + 1] += fy * forcePerNode;
                F[3 * n + 2] += fz * forcePerNode;
            }
        }

        return F;
    }

    /**
     * WASM Memory Management Helpers
     * AssemblyScript uses a custom memory layout with headers for typed arrays
     */
    _copyToWasm(arr) {
        if (!wasmModule) throw new Error('WASM not loaded');
        
        // AssemblyScript TypedArray layout:
        // - Buffer: allocated with __new(byteLength, 1) for ArrayBuffer
        // - Array header: __new(12, typeId) with pointers to buffer
        // For Float64Array: typeId = 4, align = 3 (8 bytes per element)
        
        const byteLength = arr.length * 8; // 8 bytes per Float64
        const buffer = wasmModule.exports.__pin(wasmModule.exports.__new(byteLength, 1));
        const header = wasmModule.exports.__pin(wasmModule.exports.__new(12, 4)); // 4 = Float64Array type ID, also pinned
        
        // Set up the header (ArrayBufferView structure)
        const memory = wasmModule.exports.memory;
        const view = new DataView(memory.buffer);
        view.setUint32(header, buffer, true);  // buffer pointer
        view.setUint32(header + 4, buffer, true); // dataStart pointer  
        view.setUint32(header + 8, byteLength, true); // byteLength
        
        // Copy data to buffer
        new Float64Array(memory.buffer, buffer, arr.length).set(arr);
        wasmModule.exports.__unpin(buffer);
        
        return header;
    }

    _readFromWasm(ptr, length) {
        if (!wasmModule) throw new Error('WASM not loaded');
        
        const memory = wasmModule.exports.memory;
        const view = new DataView(memory.buffer);
        
        // Read the buffer pointer from the array header
        const buffer = view.getUint32(ptr, true);
        
        // Read data from buffer
        const result = new Float64Array(length);
        result.set(new Float64Array(memory.buffer, buffer, length));
        
        return result;
    }

    _freeWasm(ptr) {
        // AssemblyScript uses reference counting via __pin/__unpin
        // The GC will collect the memory when reference count reaches zero
        if (wasmModule && wasmModule.exports.__unpin) {
            wasmModule.exports.__unpin(ptr);
        }
    }
}

// Worker message handler
const optimizer = new TopologyOptimizerWorker3D();

self.onmessage = async function(e) {
    const { type, model, config } = e.data;

    if (type === 'start') {
        optimizer.cancelled = false;
        optimizer._paused = false;
        await optimizer.optimize(model, config);
    } else if (type === 'cancel') {
        optimizer.cancelled = true;
        // Unblock any pending pause so the cancellation check is reached
        if (optimizer._resumeResolve) { optimizer._resumeResolve(); optimizer._resumeResolve = null; }
    } else if (type === 'pause') {
        optimizer._paused = true;
    } else if (type === 'resume') {
        optimizer._paused = false;
        if (optimizer._resumeResolve) { optimizer._resumeResolve(); optimizer._resumeResolve = null; }
    } else if (type === 'updateConfig') {
        optimizer._pendingConfig = e.data.config;
    } else if (type === 'requestVolumetric') {
        optimizer.postLatestVolumetricSnapshot('on-demand');
    }
};
