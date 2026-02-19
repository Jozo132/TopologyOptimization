// Web Worker for topology optimization using SIMP algorithm
// This runs in a separate thread so the UI stays responsive.
// Supports both Web Worker (browser) and worker_threads (Node.js) environments.

// Node.js / browser environment compatibility shim
if (typeof self === 'undefined') {
    const { parentPort } = await import('worker_threads');
    globalThis.self = globalThis;
    globalThis.postMessage = (data) => parentPort.postMessage(data);
    parentPort.on('message', (data) => {
        if (typeof globalThis.onmessage === 'function') globalThis.onmessage({ data });
    });
}

const EPSILON = 1e-12;
const CG_TOLERANCE = 1e-8;
const CG_TOLERANCE_COARSE = 1e-2;
const MAX_CG_ITERATIONS = 2000;

// Yield to the full event loop so pending messages (cancel/pause/updateConfig) can be processed.
// Uses setTimeout(0) to advance past the I/O phase in Node.js and browsers alike.
const _yieldToLoop = () => new Promise(r => setTimeout(r, 0));

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

// Simple AMR Manager for adaptive element grouping
class SimpleAMRManager {
    constructor(nelx, nely, useAMR, minSize, maxSize, amrInterval) {
        this.nelx = nelx;
        this.nely = nely;
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
        const nel = this.nelx * this.nely;
        this.elementToGroup = new Int32Array(nel);
        this.groups = [];
        
        let groupId = 0;
        const groupsX = Math.ceil(this.nelx / groupSize);
        const groupsY = Math.ceil(this.nely / groupSize);
        
        for (let gy = 0; gy < groupsY; gy++) {
            for (let gx = 0; gx < groupsX; gx++) {
                const elements = [];
                const startX = gx * groupSize;
                const startY = gy * groupSize;
                const endX = Math.min(startX + groupSize, this.nelx);
                const endY = Math.min(startY + groupSize, this.nely);
                
                for (let y = startY; y < endY; y++) {
                    for (let x = startX; x < endX; x++) {
                        const idx = y + x * this.nely;
                        this.elementToGroup[idx] = groupId;
                        elements.push(idx);
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
    
    updateAndRefine(elementEnergies, densities, iteration) {
        if (!this.useAMR || this.groups.length === 0) return;
        if (iteration % this.amrInterval !== 0) return; // Refine every N iterations
        
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
        
        // Adaptive refinement: split high-stress large groups
        // Use more aggressive thresholds as optimization progresses
        const progressFactor = Math.min(iteration / 50, 1.0); // 0 to 1 over 50 iterations
        const stressThreshold = 0.15 + progressFactor * 0.10; // 15% to 25%
        
        const stresses = this.groups.map(g => g.stress).sort((a, b) => b - a);
        const highStressThreshold = stresses[Math.floor(stresses.length * stressThreshold)] || 0;
        
        const toRefine = this.groups.filter(g => 
            g.stress > highStressThreshold && 
            g.size > this.minSize * 2 &&
            g.elements.length > 4
        ).slice(0, Math.min(5, Math.max(2, Math.floor(10 - iteration / 10)))); // Reduce refinements over time
        
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
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const elemIdx of group.elements) {
            const y = elemIdx % this.nely;
            const x = Math.floor(elemIdx / this.nely);
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
        
        const midX = Math.floor((minX + maxX) / 2);
        const midY = Math.floor((minY + maxY) / 2);
        const halfSize = Math.max(1, Math.floor(group.size / 2));
        
        // Create 4 sub-groups
        const newGroups = [
            { id: this.groups.length, size: halfSize, elements: [], stress: group.stress, density: group.density },
            { id: this.groups.length + 1, size: halfSize, elements: [], stress: group.stress, density: group.density },
            { id: this.groups.length + 2, size: halfSize, elements: [], stress: group.stress, density: group.density },
            { id: this.groups.length + 3, size: halfSize, elements: [], stress: group.stress, density: group.density }
        ];
        
        // Assign elements to quadrants
        for (const elemIdx of group.elements) {
            const y = elemIdx % this.nely;
            const x = Math.floor(elemIdx / this.nely);
            const subIdx = (x >= midX ? 1 : 0) + (y >= midY ? 2 : 0);
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
            totalElements: this.nelx * this.nely,
            refinementCount: this.refinementCount
        };
    }
}

class TopologyOptimizerWorker {
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
    }

    _flattenKE(KE, size) {
        const flat = new Float64Array(size * size);
        for (let i = 0; i < size; i++)
            for (let j = 0; j < size; j++)
                flat[i * size + j] = KE[i][j];
        return flat;
    }

    _precomputeEdofs2D(nelx, nely) {
        const nel = nelx * nely;
        const edofArray = new Int32Array(nel * 8);
        for (let ely = 0; ely < nely; ely++) {
            for (let elx = 0; elx < nelx; elx++) {
                const idx = ely + elx * nely;
                const offset = idx * 8;
                const n1 = (nely + 1) * elx + ely;
                const n2 = (nely + 1) * (elx + 1) + ely;
                edofArray[offset]     = 2 * n1;
                edofArray[offset + 1] = 2 * n1 + 1;
                edofArray[offset + 2] = 2 * n2;
                edofArray[offset + 3] = 2 * n2 + 1;
                edofArray[offset + 4] = 2 * n2 + 2;
                edofArray[offset + 5] = 2 * n2 + 3;
                edofArray[offset + 6] = 2 * n1 + 2;
                edofArray[offset + 7] = 2 * n1 + 3;
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

    async optimize(model, config) {
        // Try to load WASM module if not already loaded
        if (!wasmLoaded && !this.wasmLoadAttempted) {
            this.wasmLoadAttempted = true;
            await loadWasmModule();
            this.useWasm = wasmLoaded;
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

        // Apply material properties from config if provided
        if (config.youngsModulus) {
            this.E0 = config.youngsModulus;
        }
        if (config.poissonsRatio !== undefined) {
            this.nu = config.poissonsRatio;
        }

        const nel = nelx * nely;
        
        // Initialize AMR manager if enabled
        const amrManager = config.useAMR ? 
            new SimpleAMRManager(nelx, nely, true, config.minGranuleSize, config.maxGranuleSize, config.amrInterval) : 
            null;

        let x = new Float32Array(nel).fill(volfrac);
        let xnew = new Float32Array(nel);
        let xold = new Float32Array(nel).fill(1);

        const { H, Hs } = this.prepareFilter(nelx, nely, this.rmin);
        let fixeddofs = this.getFixedDOFs(nelx, nely, config.constraintPosition);
        let F = this.getLoadVector(nelx, nely, config.forceDirection, config.forceMagnitude);

        // Apply painted constraints (override dropdown if painted faces exist)
        if (config.paintedConstraints && config.paintedConstraints.length > 0) {
            fixeddofs = this.getFixedDOFsFromPaint(nelx, nely, config.paintedConstraints);
        }

        // Apply painted forces (override dropdown if painted faces exist)
        if (config.paintedForces && config.paintedForces.length > 0) {
            F = this.getLoadVectorFromPaint(nelx, nely, config.paintedForces, config.forceDirection, config.forceMagnitude);
        }

        // Build set of element indices that must stay solid (constraint/force surfaces)
        const preservedElements = new Set();
        const allPaintedKeys = [
            ...(config.paintedConstraints || []),
            ...(config.paintedForces || [])
        ];
        for (const key of allPaintedKeys) {
            const parts = key.split(',');
            if (parts.length < 2) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            if (!isNaN(vx) && !isNaN(vy) && vx >= 0 && vx < nelx && vy >= 0 && vy < nely) {
                preservedElements.add(vy + vx * nely);
            }
        }
        // Also preserve elements along dropdown-selected constraint/force positions
        if ((!config.paintedConstraints || config.paintedConstraints.length === 0)) {
            const constraintElems = this.getConstraintElements(nelx, nely, config.constraintPosition);
            for (const idx of constraintElems) preservedElements.add(idx);
        }
        if ((!config.paintedForces || config.paintedForces.length === 0)) {
            const forceElems = this.getForceElements(nelx, nely, config.forceDirection);
            for (const idx of forceElems) preservedElements.add(idx);
        }

        // Initialize preserved elements to full density
        for (const idx of preservedElements) {
            x[idx] = 1.0;
        }

        // Build set of void element indices (elements outside initial solid space)
        // 2D worker uses column-major indexing (ey + ex * nely), while
        // the importer uses row-major (ix + iy * nx) where nelx = nx.
        const voidElements = new Set();
        if (config.constrainToSolid && model.elements) {
            for (let ex = 0; ex < nelx; ex++) {
                for (let ey = 0; ey < nely; ey++) {
                    const idx2D = ey + ex * nely; // 2D worker indexing
                    const idxImporter = ex + ey * nelx; // importer row-major indexing (z=0)
                    if (model.elements[idxImporter] < 0.5) {
                        voidElements.add(idx2D);
                        x[idx2D] = 0.0;
                    }
                }
            }
        }

        const ndof = 2 * (nelx + 1) * (nely + 1);
        const alldofs = Array.from({ length: ndof }, (_, i) => i);
        const fixedSet = new Set(fixeddofs);
        const freedofs = alldofs.filter(dof => !fixedSet.has(dof));

        const KE = this.lk();
        const KEflat = this._flattenKE(KE, 8);
        const edofArray = this._precomputeEdofs2D(nelx, nely);

        // Compute per-element force magnitudes for adaptive mesh info
        const elementForces = this.computeElementForces(nelx, nely, F);

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
        const startTime = performance.now();

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
            }

            // Pause between iterations if requested
            if (this._paused) {
                postMessage({ type: 'paused', iteration: loop });
                await new Promise(resolve => { this._resumeResolve = resolve; });
            }

            loop++;
            const iterStartTime = performance.now();
            xold = Float32Array.from(x);

            // Accuracy scheduling: loosen CG tolerance when change is large (early iterations)
            const iterTolerance = Math.max(CG_TOLERANCE, change * CG_TOLERANCE_COARSE);

            // Penalization continuation: ramp penal from penalStart → penalTarget
            const currentPenal = penalStart + (penalTarget - penalStart) * Math.min(1.0, (loop - 1) / continuationIters);

            // Heaviside projection with beta-continuation (beta doubles every betaInterval iterations)
            const beta = useProjection ? Math.min(betaMax, Math.pow(2, Math.floor((loop - 1) / betaInterval))) : 1;
            const xPhys = useProjection ? this._heavisideProject(x, beta) : x;

            const { U, c: compliance } = this.FE(nelx, nely, xPhys, currentPenal, KEflat, edofArray, F, freedofs, iterTolerance);
            c = compliance;

            const dc = new Float32Array(nel);
            const elementEnergies = new Float32Array(nel);
            const Ue = new Float64Array(8);
            // Precompute Heaviside chain-rule denominator for this iteration
            const tanhBeta = Math.tanh(beta * 0.5);
            const heavisideDenom = 2 * tanhBeta; // tanh(b*0.5) + tanh(b*0.5)
            for (let e = 0; e < nel; e++) {
                const eOff = e * 8;
                for (let i = 0; i < 8; i++) {
                    Ue[i] = U[edofArray[eOff + i]];
                }
                const energy = this._computeElementEnergyFlat(KEflat, Ue, 8);
                elementEnergies[e] = energy;
                // Sensitivity w.r.t. xPhys
                const dc_phys = -currentPenal * Math.pow(xPhys[e], currentPenal - 1) * this.E0 * energy;
                // Chain rule: d(xPhys)/d(x) via Heaviside (1.0 when projection disabled)
                const th = Math.tanh(beta * (x[e] - 0.5));
                const dPhys_dx = useProjection ? beta * (1 - th * th) / heavisideDenom : 1.0;
                dc[e] = dc_phys * dPhys_dx;
            }

            const dcn = this.filterSensitivities(dc, x, H, Hs, nelx, nely);
            
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
            
            xnew = this.OC(nelx, nely, x, volfrac, dcnWeighted, preservedElements, voidElements);

            change = 0;
            for (let i = 0; i < nel; i++) {
                change = Math.max(change, Math.abs(xnew[i] - xold[i]));
            }

            x = Float32Array.from(xnew);
            lastElementEnergies = elementEnergies;
            
            // Perform AMR refinement if enabled
            if (amrManager) {
                amrManager.updateAndRefine(elementEnergies, x, loop);
            }

            // Build adaptive mesh data for this iteration
            const meshData = this.buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces, amrManager);

            // Track iteration timing
            const iterEndTime = performance.now();
            const iterTime = iterEndTime - iterStartTime;
            iterationTimes.push(iterTime);
            
            // Calculate average time per iteration
            const avgIterTime = iterationTimes.reduce((a, b) => a + b, 0) / iterationTimes.length;
            const elapsedTime = iterEndTime - startTime;

            postMessage({
                type: 'progress',
                iteration: loop,
                compliance: c,
                meshData: meshData,
                penal: currentPenal,
                beta: beta,
                timing: {
                    iterationTime: iterTime,
                    avgIterationTime: avgIterTime,
                    elapsedTime: elapsedTime,
                    usingWasm: this.useWasm
                }
            });
        }

        if (this.cancelled) {
            postMessage({ type: 'cancelled', iteration: loop });
            return;
        }

        const finalMesh = this.buildAdaptiveMesh(nelx, nely, nelz, x, lastElementEnergies, elementForces, amrManager);

        // Also build the flat densities3D for export compatibility
        const densities3D = new Float32Array(nelx * nely * nelz);
        for (let z = 0; z < nelz; z++) {
            for (let y = 0; y < nely; y++) {
                for (let xpos = 0; xpos < nelx; xpos++) {
                    const idx2D = y + xpos * nely;
                    const idx3D = xpos + y * nelx + z * nelx * nely;
                    densities3D[idx3D] = x[idx2D];
                }
            }
        }
        
        // Final timing statistics
        const totalTime = performance.now() - startTime;
        const avgIterTime = iterationTimes.length > 0 
            ? iterationTimes.reduce((a, b) => a + b, 0) / iterationTimes.length 
            : 0;
        
        // Get AMR statistics if enabled
        const amrStats = amrManager ? amrManager.getStats() : null;

        postMessage({
            type: 'complete',
            result: {
                densities: densities3D,
                finalCompliance: c,
                iterations: loop,
                volumeFraction: volfrac,
                nx: nelx,
                ny: nely,
                nz: nelz,
                meshData: finalMesh,
                amrStats: amrStats,
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
    computeElementForces(nelx, nely, F) {
        const nel = nelx * nely;
        const forces = new Float32Array(nel);
        for (let ely = 0; ely < nely; ely++) {
            for (let elx = 0; elx < nelx; elx++) {
                const n1 = (nely + 1) * elx + ely;
                const n2 = (nely + 1) * (elx + 1) + ely;
                const edof = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n2 + 2, 2 * n2 + 3,
                    2 * n1 + 2, 2 * n1 + 3
                ];
                let mag = 0;
                for (let d = 0; d < edof.length; d++) {
                    mag += F[edof[d]] * F[edof[d]];
                }
                forces[ely + elx * nely] = Math.sqrt(mag);
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

        // Helper – is an element solid? (2D index: ely + elx * nely)
        const isSolid2D = (elx, ely) => {
            if (elx < 0 || elx >= nelx || ely < 0 || ely >= nely) return false;
            return x[ely + elx * nely] > DENSITY_THRESHOLD;
        };

        // ── AMR-aware rendering ──────────────────────────────────────────────
        // Build a set of element indices that have already been emitted so that
        // per-element fallback doesn't double-render anything covered by a group.
        const rendered = new Uint8Array(nelx * nely);

        if (amrManager && amrManager.groups && amrManager.groups.length > 0) {
            for (const group of amrManager.groups) {
                if (group.elements.length === 0) continue;

                // Compute group bounding box in element coordinates
                let minGX = nelx, minGY = nely, maxGX = -1, maxGY = -1;
                let totalDensity = 0;
                let totalStress = 0;
                let solidCount = 0;

                for (const idx of group.elements) {
                    const ely = idx % nely;
                    const elx = Math.floor(idx / nely);
                    const density = x[idx];
                    if (density > DENSITY_THRESHOLD) {
                        solidCount++;
                        totalDensity += density;
                        totalStress += elementStress ? elementStress[idx] : 0;
                        if (elx < minGX) minGX = elx;
                        if (ely < minGY) minGY = ely;
                        if (elx > maxGX) maxGX = elx;
                        if (ely > maxGY) maxGY = ely;
                    }
                    rendered[idx] = 1;
                }

                if (solidCount === 0) continue;

                const avgDensity = totalDensity / solidCount;
                const avgStress = totalStress / solidCount;
                const strain = maxStress > 0 ? avgStress / maxStress : 0;

                // Block spans the group bounding box
                const bx0 = minGX, bx1 = maxGX + 1;
                const by0 = minGY, by1 = maxGY + 1;
                const bw = bx1 - bx0; // block width  in elements
                const bh = by1 - by0; // block height in elements

                // Helper – is the block's neighbour in direction (dx,dy) solid?
                // A neighbouring block boundary is void if ANY element on the edge
                // outside the group is void (conservative: show face whenever boundary).
                const isNeighbourSolid = (dx, dy) => {
                    if (dx === -1) {
                        if (bx0 === 0) return false;
                        for (let gy = by0; gy < by1; gy++) {
                            if (!isSolid2D(bx0 - 1, gy)) return false;
                        }
                        return true;
                    }
                    if (dx === 1) {
                        if (bx1 >= nelx) return false;
                        for (let gy = by0; gy < by1; gy++) {
                            if (!isSolid2D(bx1, gy)) return false;
                        }
                        return true;
                    }
                    if (dy === -1) {
                        if (by0 === 0) return false;
                        for (let gx = bx0; gx < bx1; gx++) {
                            if (!isSolid2D(gx, by0 - 1)) return false;
                        }
                        return true;
                    }
                    if (dy === 1) {
                        if (by1 >= nely) return false;
                        for (let gx = bx0; gx < bx1; gx++) {
                            if (!isSolid2D(gx, by1)) return false;
                        }
                        return true;
                    }
                    return false;
                };

                // Emit one merged quad per boundary face of the block
                // Front / back (z faces) are always visible for 2D extrusion
                // Face 0: Front (z = 0)
                triangles.push({ vertices: [[bx0, by0, 0], [bx1, by0, 0], [bx1, by1, 0]], normal: [0, 0, -1], density: avgDensity, strain, blockSize: bw });
                triangles.push({ vertices: [[bx0, by0, 0], [bx1, by1, 0], [bx0, by1, 0]], normal: [0, 0, -1], density: avgDensity, strain, blockSize: bw });
                // Face 1: Back (z = 1)
                triangles.push({ vertices: [[bx1, by0, 1], [bx0, by0, 1], [bx0, by1, 1]], normal: [0, 0, 1], density: avgDensity, strain, blockSize: bw });
                triangles.push({ vertices: [[bx1, by0, 1], [bx0, by1, 1], [bx1, by1, 1]], normal: [0, 0, 1], density: avgDensity, strain, blockSize: bw });
                // Face 2: Bottom (y = by0)
                if (!isNeighbourSolid(0, -1)) {
                    triangles.push({ vertices: [[bx0, by0, 0], [bx1, by0, 0], [bx1, by0, 1]], normal: [0, -1, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by0, 0], [bx1, by0, 1], [bx0, by0, 1]], normal: [0, -1, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 3: Top (y = by1)
                if (!isNeighbourSolid(0, 1)) {
                    triangles.push({ vertices: [[bx0, by1, 1], [bx1, by1, 1], [bx1, by1, 0]], normal: [0, 1, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by1, 1], [bx1, by1, 0], [bx0, by1, 0]], normal: [0, 1, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 4: Left (x = bx0)
                if (!isNeighbourSolid(-1, 0)) {
                    triangles.push({ vertices: [[bx0, by0, 1], [bx0, by1, 1], [bx0, by1, 0]], normal: [-1, 0, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx0, by0, 1], [bx0, by1, 0], [bx0, by0, 0]], normal: [-1, 0, 0], density: avgDensity, strain, blockSize: bw });
                }
                // Face 5: Right (x = bx1)
                if (!isNeighbourSolid(1, 0)) {
                    triangles.push({ vertices: [[bx1, by0, 0], [bx1, by1, 0], [bx1, by1, 1]], normal: [1, 0, 0], density: avgDensity, strain, blockSize: bw });
                    triangles.push({ vertices: [[bx1, by0, 0], [bx1, by1, 1], [bx1, by0, 1]], normal: [1, 0, 0], density: avgDensity, strain, blockSize: bw });
                }
            }
        }

        // ── Per-element fallback for elements not covered by any AMR group ───
        const xyNeighbors = [
            { fi: 2, dx: 0, dy: -1 },
            { fi: 3, dx: 0, dy: 1 },
            { fi: 4, dx: -1, dy: 0 },
            { fi: 5, dx: 1, dy: 0 },
        ];

        for (let z = 0; z < nelz; z++) {
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const idx2D = ely + elx * nely;
                    if (rendered[idx2D]) continue;
                    const density = x[idx2D];
                    if (density <= DENSITY_THRESHOLD) continue;

                    // Front (z=0) and back (z=1) are always boundary for 2D single-layer extrusion
                    const visibleFaces = [0, 1];
                    for (const { fi, dx, dy } of xyNeighbors) {
                        if (!isSolid2D(elx + dx, ely + dy)) visibleFaces.push(fi);
                    }

                    const strain = (maxStress > 0 && elementStress) ? elementStress[idx2D] / maxStress : 0;
                    this.addSubdividedElement(triangles, elx, ely, z, density, 1, visibleFaces, strain);
                }
            }
        }

        return triangles;
    }

    /**
     * Add an element as two triangles per visible face (simple quad split).
     * visibleFaces: 0=front(z-), 1=back(z+), 2=bottom(y-), 3=top(y+), 4=left(x-), 5=right(x+).
     */
    addSubdividedElement(triangles, ex, ey, ez, density, subdivLevel, visibleFaces, strain) {
        const n = subdivLevel;
        const step = 1.0 / n;

        const baseX = ex;
        const baseY = ey;
        const baseZ = ez;

        for (const fi of visibleFaces) {
            switch (fi) {
                case 0: // Front face (z = baseZ)
                case 1: // Back face (z = baseZ + 1)
                    for (let sy = 0; sy < n; sy++) {
                        for (let sx = 0; sx < n; sx++) {
                            const x0 = baseX + sx * step;
                            const y0 = baseY + sy * step;
                            const x1 = x0 + step;
                            const y1 = y0 + step;
                            if (fi === 0) {
                                triangles.push({ vertices: [[x0, y0, baseZ], [x1, y0, baseZ], [x1, y1, baseZ]], normal: [0, 0, -1], density, strain, blockSize: 1 });
                                triangles.push({ vertices: [[x0, y0, baseZ], [x1, y1, baseZ], [x0, y1, baseZ]], normal: [0, 0, -1], density, strain, blockSize: 1 });
                            } else {
                                triangles.push({ vertices: [[x1, y0, baseZ + 1], [x0, y0, baseZ + 1], [x0, y1, baseZ + 1]], normal: [0, 0, 1], density, strain, blockSize: 1 });
                                triangles.push({ vertices: [[x1, y0, baseZ + 1], [x0, y1, baseZ + 1], [x1, y1, baseZ + 1]], normal: [0, 0, 1], density, strain, blockSize: 1 });
                            }
                        }
                    }
                    break;
                case 2: // Bottom face (y = baseY)
                    for (let s = 0; s < n; s++) {
                        const t0 = s * step;
                        const t1 = t0 + step;
                        triangles.push({ vertices: [[baseX + t0, baseY, baseZ], [baseX + t1, baseY, baseZ], [baseX + t1, baseY, baseZ + 1]], normal: [0, -1, 0], density, strain, blockSize: 1 });
                        triangles.push({ vertices: [[baseX + t0, baseY, baseZ], [baseX + t1, baseY, baseZ + 1], [baseX + t0, baseY, baseZ + 1]], normal: [0, -1, 0], density, strain, blockSize: 1 });
                    }
                    break;
                case 3: // Top face (y = baseY + 1)
                    for (let s = 0; s < n; s++) {
                        const t0 = s * step;
                        const t1 = t0 + step;
                        triangles.push({ vertices: [[baseX + t0, baseY + 1, baseZ + 1], [baseX + t1, baseY + 1, baseZ + 1], [baseX + t1, baseY + 1, baseZ]], normal: [0, 1, 0], density, strain, blockSize: 1 });
                        triangles.push({ vertices: [[baseX + t0, baseY + 1, baseZ + 1], [baseX + t1, baseY + 1, baseZ], [baseX + t0, baseY + 1, baseZ]], normal: [0, 1, 0], density, strain, blockSize: 1 });
                    }
                    break;
                case 4: // Left face (x = baseX)
                    for (let s = 0; s < n; s++) {
                        const t0 = s * step;
                        const t1 = t0 + step;
                        triangles.push({ vertices: [[baseX, baseY + t0, baseZ + 1], [baseX, baseY + t1, baseZ + 1], [baseX, baseY + t1, baseZ]], normal: [-1, 0, 0], density, strain, blockSize: 1 });
                        triangles.push({ vertices: [[baseX, baseY + t0, baseZ + 1], [baseX, baseY + t1, baseZ], [baseX, baseY + t0, baseZ]], normal: [-1, 0, 0], density, strain, blockSize: 1 });
                    }
                    break;
                case 5: // Right face (x = baseX + 1)
                    for (let s = 0; s < n; s++) {
                        const t0 = s * step;
                        const t1 = t0 + step;
                        triangles.push({ vertices: [[baseX + 1, baseY + t0, baseZ], [baseX + 1, baseY + t1, baseZ], [baseX + 1, baseY + t1, baseZ + 1]], normal: [1, 0, 0], density, strain, blockSize: 1 });
                        triangles.push({ vertices: [[baseX + 1, baseY + t0, baseZ], [baseX + 1, baseY + t1, baseZ + 1], [baseX + 1, baseY + t0, baseZ + 1]], normal: [1, 0, 0], density, strain, blockSize: 1 });
                    }
                    break;
            }
        }
    }

    prepareFilter(nelx, nely, rmin) {
        const iH = [];
        const jH = [];
        const sH = [];
        let k = 0;

        for (let i = 0; i < nelx; i++) {
            for (let j = 0; j < nely; j++) {
                const e1 = i * nely + j;

                for (let k_iter = Math.max(i - Math.floor(rmin), 0);
                     k_iter <= Math.min(i + Math.floor(rmin), nelx - 1);
                     k_iter++) {
                    for (let l = Math.max(j - Math.floor(rmin), 0);
                         l <= Math.min(j + Math.floor(rmin), nely - 1);
                         l++) {
                        const e2 = k_iter * nely + l;
                        const dist = Math.sqrt((i - k_iter) ** 2 + (j - l) ** 2);

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

        const H = { i: iH, j: jH, s: sH };
        const Hs = new Float32Array(nelx * nely);

        for (let i = 0; i < k; i++) {
            Hs[iH[i]] += sH[i];
        }

        return { H, Hs };
    }

    filterSensitivities(dc, x, H, Hs, nelx, nely) {
        const dcn = new Float32Array(nelx * nely);

        for (let i = 0; i < H.i.length; i++) {
            dcn[H.i[i]] += H.s[i] * x[H.j[i]] * dc[H.j[i]];
        }

        for (let i = 0; i < nelx * nely; i++) {
            dcn[i] = dcn[i] / (Hs[i] * Math.max(1e-3, x[i]));
        }

        return dcn;
    }

    OC(nelx, nely, x, volfrac, dc, preservedElements, voidElements) {
        const nel = nelx * nely;
        const xnew = new Float32Array(nel);
        const move = 0.2;

        let l1 = 0;
        let l2 = 1e9;

        while ((l2 - l1) / (l2 + l1) > 1e-3) {
            const lmid = 0.5 * (l2 + l1);

            for (let i = 0; i < nel; i++) {
                if (preservedElements && preservedElements.has(i)) {
                    xnew[i] = 1.0;
                } else if (voidElements && voidElements.has(i)) {
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

        return xnew;
    }

    lk() {
        const nu = this.nu;
        const k = [
            1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
            -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8
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

        return KE.map(row => row.map(val => val * 1 / (1 - nu * nu)));
    }

    /**
     * Heaviside projection: maps design variable x → physical density xPhys.
     * Provides crisp 0/1 designs; beta controls sharpness (1=smooth, 64+=near binary).
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

    FE(nelx, nely, x, penal, KEflat, edofArray, F, freedofs, tolerance = CG_TOLERANCE) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const nel = nelx * nely;
        const U = new Float64Array(ndof);

        const Uf = this.solveCG(x, penal, KEflat, edofArray, nel, 8, F, freedofs, ndof, tolerance);

        for (let i = 0; i < freedofs.length; i++) {
            U[freedofs[i]] = Uf[i];
        }

        let c = 0;
        for (let i = 0; i < ndof; i++) {
            c += F[i] * U[i];
        }

        return { U, c };
    }

    solveCG(x, penal, KEflat, edofArray, nel, edofSize, F, freedofs, ndof, tolerance = CG_TOLERANCE) {
        const n = freedofs.length;

        // Allocate full-space work buffers for EbE matvec
        if (!this._p_full || this._p_full.length !== ndof) {
            this._p_full = new Float64Array(ndof);
            this._Ap_full = new Float64Array(ndof);
        }

        // Precompute element stiffnesses once per solve (avoids Math.pow per CG iteration)
        const { E_vals, activeElements } = this._precomputeStiffness(x, penal, nel);

        // Compute Jacobi preconditioner (inverse diagonal of K)
        const invDiag = this._computeDiagonal(E_vals, activeElements, KEflat, edofArray, nel, edofSize, freedofs, ndof);

        // Reuse CG work arrays across calls when dimensions match
        if (!this._cgUf || this._cgUf.length !== n) {
            this._cgUf = new Float64Array(n);
            this._cgR = new Float64Array(n);
            this._cgZ = new Float64Array(n);
            this._cgP = new Float64Array(n);
            this._cgAp = new Float64Array(n);
        }
        const Uf = this._cgUf; Uf.fill(0);
        const r = this._cgR;
        const z = this._cgZ;
        const p = this._cgP;
        const Ap = this._cgAp;

        // r = F_free (initial residual since U_0 = 0)
        for (let i = 0; i < n; i++) {
            r[i] = F[freedofs[i]];
        }

        // z = M^{-1} r; p = z; rz = r^T z
        let rz = 0;
        for (let i = 0; i < n; i++) {
            z[i] = invDiag[i] * r[i];
            p[i] = z[i];
            rz += r[i] * z[i];
        }

        const maxIter = Math.min(n, MAX_CG_ITERATIONS);
        const tolSq = tolerance * tolerance;

        for (let iter = 0; iter < maxIter; iter++) {
            let rnorm2 = 0;
            for (let i = 0; i < n; i++) rnorm2 += r[i] * r[i];
            if (rnorm2 < tolSq) break;

            // Ap = K * p (element-by-element, using precomputed stiffness)
            this._ebeMatVec(E_vals, activeElements, KEflat, edofArray, nel, edofSize, p, Ap, freedofs, ndof);

            let pAp = 0;
            for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
            const alpha = rz / (pAp + EPSILON);

            let rz_new = 0;
            for (let i = 0; i < n; i++) {
                Uf[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                z[i] = invDiag[i] * r[i];
                rz_new += r[i] * z[i];
            }

            const beta = rz_new / (rz + EPSILON);
            for (let i = 0; i < n; i++) {
                p[i] = z[i] + beta * p[i];
            }

            rz = rz_new;
        }

        return Uf;
    }

    computeElementEnergy(KE, Ue) {
        let energy = 0;
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                energy += Ue[i] * KE[i][j] * Ue[j];
            }
        }
        return energy;
    }

    getFixedDOFs(nelx, nely, position) {
        const fixeddofs = [];

        switch (position) {
            case 'left':
                for (let j = 0; j <= nely; j++) {
                    fixeddofs.push(2 * j, 2 * j + 1);
                }
                break;
            case 'right':
                for (let j = 0; j <= nely; j++) {
                    const n = (nely + 1) * nelx + j;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
            case 'bottom':
                for (let i = 0; i <= nelx; i++) {
                    const n = (nely + 1) * i;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
            case 'top':
                for (let i = 0; i <= nelx; i++) {
                    const n = (nely + 1) * i + nely;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
        }

        return fixeddofs;
    }

    getLoadVector(nelx, nely, direction, magnitude) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const F = new Float32Array(ndof);

        switch (direction) {
            case 'down': {
                const n_down = (nely + 1) * nelx + nely;
                F[2 * n_down + 1] = -magnitude;
                break;
            }
            case 'up': {
                const n_up = (nely + 1) * nelx;
                F[2 * n_up + 1] = magnitude;
                break;
            }
            case 'left': {
                const n_left = (nely + 1) * nelx + Math.floor(nely / 2);
                F[2 * n_left] = -magnitude;
                break;
            }
            case 'right': {
                const n_right = Math.floor(nely / 2);
                F[2 * n_right] = magnitude;
                break;
            }
        }

        return F;
    }

    /**
     * Get element indices along the constraint edge.
     */
    getConstraintElements(nelx, nely, position) {
        const elems = [];
        switch (position) {
            case 'left':
                for (let ey = 0; ey < nely; ey++) elems.push(ey); // elx=0
                break;
            case 'right':
                for (let ey = 0; ey < nely; ey++) elems.push(ey + (nelx - 1) * nely);
                break;
            case 'bottom':
                for (let ex = 0; ex < nelx; ex++) elems.push(ex * nely); // ely=0
                break;
            case 'top':
                for (let ex = 0; ex < nelx; ex++) elems.push((nely - 1) + ex * nely);
                break;
        }
        return elems;
    }

    /**
     * Get element indices where the default force is applied.
     */
    getForceElements(nelx, nely, direction) {
        const elems = [];
        switch (direction) {
            case 'down':
                // Force applied downward at right edge bottom node — nearest element (nelx-1, nely-1)
                elems.push((nely - 1) + (nelx - 1) * nely);
                break;
            case 'up':
                // Force applied upward at right edge top node — nearest element (nelx-1, 0)
                elems.push((nelx - 1) * nely);
                break;
            case 'left':
                // Force at right-middle — element at (nelx-1, floor(nely/2))
                elems.push(Math.floor(nely / 2) + (nelx - 1) * nely);
                break;
            case 'right':
                // Force at left-middle — element at (0, floor(nely/2))
                elems.push(Math.floor(nely / 2));
                break;
        }
        return elems;
    }

    /**
     * Convert painted constraint face keys to fixed DOFs.
     * Face keys are "x,y,z,faceIndex" where x,y are voxel coordinates.
     * Maps to 2D nodes at the voxel corners.
     */
    getFixedDOFsFromPaint(nelx, nely, paintedKeys) {
        const dofSet = new Set();
        for (const key of paintedKeys) {
            const parts = key.split(',');
            if (parts.length < 2) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            if (isNaN(vx) || isNaN(vy)) continue;
            // Map voxel (vx, vy) to its 4 corner nodes in 2D
            const n1 = (nely + 1) * vx + vy;
            const n2 = (nely + 1) * (vx + 1) + vy;
            const nodes = [n1, n2, n2 + 1, n1 + 1];
            for (const n of nodes) {
                if (n >= 0 && n < (nelx + 1) * (nely + 1)) {
                    dofSet.add(2 * n);
                    dofSet.add(2 * n + 1);
                }
            }
        }
        return Array.from(dofSet);
    }

    /**
     * Convert painted force face keys to a load vector.
     * Distributes force evenly across all painted face nodes.
     */
    getLoadVectorFromPaint(nelx, nely, paintedKeys, direction, magnitude) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const F = new Float32Array(ndof);

        // Determine force direction components
        let fx = 0, fy = 0;
        switch (direction) {
            case 'down':  fy = -1; break;
            case 'up':    fy = 1; break;
            case 'left':  fx = -1; break;
            case 'right': fx = 1; break;
            default:      fy = -1;
        }

        // Collect unique nodes from painted faces
        const nodeSet = new Set();
        for (const key of paintedKeys) {
            const parts = key.split(',');
            if (parts.length < 2) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            if (isNaN(vx) || isNaN(vy)) continue;
            const n1 = (nely + 1) * vx + vy;
            const n2 = (nely + 1) * (vx + 1) + vy;
            const nodes = [n1, n2, n2 + 1, n1 + 1];
            for (const n of nodes) {
                if (n >= 0 && n < (nelx + 1) * (nely + 1)) {
                    nodeSet.add(n);
                }
            }
        }

        // Distribute force evenly across all unique nodes
        const nodeCount = nodeSet.size;
        if (nodeCount > 0) {
            const forcePerNode = magnitude / nodeCount;
            for (const n of nodeSet) {
                F[2 * n] += fx * forcePerNode;
                F[2 * n + 1] += fy * forcePerNode;
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
const optimizer = new TopologyOptimizerWorker();

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
    }
};
