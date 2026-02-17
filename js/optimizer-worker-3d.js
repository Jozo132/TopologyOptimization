// Web Worker for 3D topology optimization using SIMP algorithm with 8-node hexahedral elements
// This runs in a separate thread so the UI stays responsive.

const EPSILON = 1e-12;
const CG_TOLERANCE = 1e-8;

// WASM Module for high-performance operations
let wasmModule = null;
let wasmLoaded = false;

async function loadWasmModule() {
    try {
        // Resolve WASM path relative to the document root, not the worker script location.
        // Workers resolve fetch() URLs relative to their own script URL (inside js/),
        // so we go up one level to reach the project root where wasm/ lives.
        const baseUrl = new URL('..', self.location.href).href;
        const response = await fetch(new URL('wasm/matrix-ops.wasm', baseUrl).href);
        const buffer = await response.arrayBuffer();
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
    constructor(nelx, nely, nelz, useAMR, minSize, maxSize) {
        this.nelx = nelx;
        this.nely = nely;
        this.nelz = nelz;
        this.useAMR = useAMR;
        this.minSize = minSize || 0.5;
        this.maxSize = maxSize || 2;
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
        if (iteration % 10 !== 0) return;
        
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

class TopologyOptimizerWorker3D {
    constructor() {
        this.rmin = 1.5;
        this.penal = 3;
        this.E0 = 1;
        this.Emin = 1e-9;
        this.nu = 0.3;
        this.cancelled = false;
        this.useWasm = false;
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
        const volfrac = config.volumeFraction;
        const maxIterations = config.maxIterations;

        this.penal = config.penaltyFactor;
        this.rmin = config.filterRadius;
        this.cancelled = false;

        const nel = nelx * nely * nelz;
        
        // Initialize AMR manager if enabled
        const amrManager = config.useAMR ? 
            new SimpleAMRManager3D(nelx, nely, nelz, true, config.minGranuleSize, config.maxGranuleSize) : 
            null;

        let x = new Float32Array(nel).fill(volfrac);
        let xnew = new Float32Array(nel);
        let xold = new Float32Array(nel).fill(1);

        const { H, Hs } = this.prepareFilter(nelx, nely, nelz, this.rmin);
        let fixeddofs = this.getFixedDOFs(nelx, nely, nelz, config.constraintPosition);
        let F = this.getLoadVector(nelx, nely, nelz, config.forceDirection, config.forceMagnitude);

        // Apply painted constraints (override dropdown if painted faces exist)
        if (config.paintedConstraints && config.paintedConstraints.length > 0) {
            fixeddofs = this.getFixedDOFsFromPaint(nelx, nely, nelz, config.paintedConstraints);
        }

        // Apply painted forces (override dropdown if painted faces exist)
        if (config.paintedForces && config.paintedForces.length > 0) {
            F = this.getLoadVectorFromPaint(nelx, nely, nelz, config.paintedForces, config.forceDirection, config.forceMagnitude);
        }

        // Build set of element indices that must stay solid (constraint/force surfaces)
        const preservedElements = new Set();
        const allPaintedKeys = [
            ...(config.paintedConstraints || []),
            ...(config.paintedForces || [])
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

        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const alldofs = Array.from({ length: ndof }, (_, i) => i);
        const fixedSet = new Set(fixeddofs);
        const freedofs = alldofs.filter(dof => !fixedSet.has(dof));

        const KE = this.lk();

        // Compute per-element force magnitudes for adaptive mesh info
        const elementForces = this.computeElementForces(nelx, nely, nelz, F);

        let loop = 0;
        let change = 1;
        let c = 0;
        let lastElementEnergies = null;
        
        // Benchmark timing
        const iterationTimes = [];
        const startTime = performance.now();

        while (change > 0.01 && loop < maxIterations) {
            if (this.cancelled) {
                postMessage({ type: 'cancelled', iteration: loop });
                return;
            }

            loop++;
            const iterStartTime = performance.now();
            xold = Float32Array.from(x);

            const { U, c: compliance } = this.FE(nelx, nely, nelz, x, this.penal, KE, F, freedofs, fixeddofs);
            c = compliance;

            const dc = new Float32Array(nel);
            const elementEnergies = new Float32Array(nel);
            for (let elz = 0; elz < nelz; elz++) {
                for (let ely = 0; ely < nely; ely++) {
                    for (let elx = 0; elx < nelx; elx++) {
                        const edof = this.getElementDOFs(elx, ely, elz, nelx, nely, nelz);
                        const Ue = edof.map(dof => U[dof] || 0);
                        const idx = elx + ely * nelx + elz * nelx * nely;
                        const energy = this.computeElementEnergy(KE, Ue);
                        elementEnergies[idx] = energy;

                        dc[idx] = -this.penal * Math.pow(x[idx], this.penal - 1) *
                                  this.E0 * energy;
                    }
                }
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
            
            xnew = this.OC(nelx, nely, nelz, x, volfrac, dcnWeighted, preservedElements);

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
            const meshData = this.buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces);

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

        const finalMesh = this.buildAdaptiveMesh(nelx, nely, nelz, x, lastElementEnergies, elementForces);
        
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
                densities: x,
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
     * Elements with higher stress or near applied forces get subdivided
     * into smaller triangles, while low-stress regions use coarser triangles.
     * Uses stiffness-weighted strain energy (stiffness × raw energy) for auto-resizing.
     * Only boundary faces (adjacent to void or domain boundary) are emitted.
     * Returns an array of { vertices, density, strain } triangle objects for rendering.
     */
    buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces) {
        // Duplicated from constants.js since workers cannot use ES module imports
        const DENSITY_THRESHOLD = 0.3;
        const triangles = [];

        // Compute stress-based metric: scale energy by element stiffness (density^penal)
        let maxStress = 0;
        let maxForce = 0;
        const elementStress = elementEnergies ? new Float32Array(elementEnergies.length) : null;
        if (elementEnergies) {
            for (let i = 0; i < elementEnergies.length; i++) {
                const stiffness = this.Emin + Math.pow(x[i], this.penal) * (this.E0 - this.Emin);
                elementStress[i] = stiffness * elementEnergies[i];
                if (elementStress[i] > maxStress) maxStress = elementStress[i];
            }
        }
        if (elementForces) {
            for (let i = 0; i < elementForces.length; i++) {
                if (elementForces[i] > maxForce) maxForce = elementForces[i];
            }
        }

        // Helper to check if neighbor is solid (above density threshold)
        const isSolid = (ex, ey, ez) => {
            if (ex < 0 || ex >= nelx || ey < 0 || ey >= nely || ez < 0 || ez >= nelz) return false;
            return x[ex + ey * nelx + ez * nelx * nely] > DENSITY_THRESHOLD;
        };

        // Face directions: [dx, dy, dz] for each of the 6 faces
        const faceNeighbors = [
            [-1, 0, 0], // Front face (X = baseX)
            [1, 0, 0],  // Back face (X = baseX + 1)
            [0, -1, 0], // Left face (Y = baseY)
            [0, 1, 0],  // Right face (Y = baseY + 1)
            [0, 0, -1], // Bottom face (Z = baseZ)
            [0, 0, 1],  // Top face (Z = baseZ + 1)
        ];

        for (let elz = 0; elz < nelz; elz++) {
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const idx = elx + ely * nelx + elz * nelx * nely;
                    const density = x[idx];

                    if (density <= DENSITY_THRESHOLD) continue;

                    // Determine which faces are boundary (neighbor is void or outside domain)
                    const visibleFaces = [];
                    for (let fi = 0; fi < 6; fi++) {
                        const [dx, dy, dz] = faceNeighbors[fi];
                        if (!isSolid(elx + dx, ely + dy, elz + dz)) {
                            visibleFaces.push(fi);
                        }
                    }

                    // Skip fully interior elements
                    if (visibleFaces.length === 0) continue;

                    // Determine subdivision level based on stress / force ratio
                    let subdivLevel = 1; // default: 2 triangles per face (1x1 subdivision)
                    if (maxStress > 0 && elementStress) {
                        const stressRatio = elementStress[idx] / maxStress;
                        const forceRatio = maxForce > 0 ? elementForces[idx] / maxForce : 0;
                        const ratio = Math.max(stressRatio, forceRatio);
                        if (ratio > 0.6) {
                            subdivLevel = 4; // 4x4 subdivision for high-stress areas
                        } else if (ratio > 0.3) {
                            subdivLevel = 2; // 2x2 subdivision for medium areas
                        }
                    }

                    // Normalized strain for this element (0..1)
                    const strain = (maxStress > 0 && elementStress) ? elementStress[idx] / maxStress : 0;

                    // Generate subdivided mesh for visible faces only
                    this.addSubdividedElement(triangles, elx, ely, elz, density, subdivLevel, visibleFaces, strain);
                }
            }
        }

        return triangles;
    }

    /**
     * Add an 8-node hexahedral element as triangulated faces with adaptive subdivision.
     * Only emits faces listed in visibleFaces (indices 0-5).
     * subdivLevel=1 means 2 triangles per visible face (standard).
     * subdivLevel=2 means 4x2=8 triangles per face (2x2 grid).
     * subdivLevel=4 means 16x2=32 triangles per face (4x4 grid).
     * @param {number} strain - Normalized strain value (0..1) for this element.
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
                            triangles.push({
                                vertices: [[baseX, y0, z0], [baseX, y1, z0], [baseX, y1, z1]],
                                normal: [-1, 0, 0], density, strain
                            });
                            triangles.push({
                                vertices: [[baseX, y0, z0], [baseX, y1, z1], [baseX, y0, z1]],
                                normal: [-1, 0, 0], density, strain
                            });
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
                            triangles.push({
                                vertices: [[baseX + 1, y1, z0], [baseX + 1, y0, z0], [baseX + 1, y0, z1]],
                                normal: [1, 0, 0], density, strain
                            });
                            triangles.push({
                                vertices: [[baseX + 1, y1, z0], [baseX + 1, y0, z1], [baseX + 1, y1, z1]],
                                normal: [1, 0, 0], density, strain
                            });
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
                            triangles.push({
                                vertices: [[x0, baseY, z0], [x0, baseY, z1], [x1, baseY, z1]],
                                normal: [0, -1, 0], density, strain
                            });
                            triangles.push({
                                vertices: [[x0, baseY, z0], [x1, baseY, z1], [x1, baseY, z0]],
                                normal: [0, -1, 0], density, strain
                            });
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
                            triangles.push({
                                vertices: [[x0, baseY + 1, z1], [x0, baseY + 1, z0], [x1, baseY + 1, z0]],
                                normal: [0, 1, 0], density, strain
                            });
                            triangles.push({
                                vertices: [[x0, baseY + 1, z1], [x1, baseY + 1, z0], [x1, baseY + 1, z1]],
                                normal: [0, 1, 0], density, strain
                            });
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
                            triangles.push({
                                vertices: [[x0, y0, baseZ], [x1, y0, baseZ], [x1, y1, baseZ]],
                                normal: [0, 0, -1], density, strain
                            });
                            triangles.push({
                                vertices: [[x0, y0, baseZ], [x1, y1, baseZ], [x0, y1, baseZ]],
                                normal: [0, 0, -1], density, strain
                            });
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
                            triangles.push({
                                vertices: [[x1, y0, baseZ + 1], [x0, y0, baseZ + 1], [x0, y1, baseZ + 1]],
                                normal: [0, 0, 1], density, strain
                            });
                            triangles.push({
                                vertices: [[x1, y0, baseZ + 1], [x0, y1, baseZ + 1], [x1, y1, baseZ + 1]],
                                normal: [0, 0, 1], density, strain
                            });
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

    OC(nelx, nely, nelz, x, volfrac, dc, preservedElements) {
        const nel = nelx * nely * nelz;
        const xnew = new Float32Array(nel);
        const move = 0.2;

        let l1 = 0;
        let l2 = 1e9;

        while ((l2 - l1) / (l2 + l1) > 1e-3) {
            const lmid = 0.5 * (l2 + l1);

            for (let i = 0; i < nel; i++) {
                if (preservedElements && preservedElements.has(i)) {
                    xnew[i] = 1.0;
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

    FE(nelx, nely, nelz, x, penal, KE, F, freedofs, fixeddofs) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const U = new Float32Array(ndof);

        const K = this.assembleK(nelx, nely, nelz, x, penal, KE);
        const Uf = this.solveCG(K, F, freedofs, fixeddofs);

        freedofs.forEach((dof, i) => {
            U[dof] = Uf[i];
        });

        let c = 0;
        for (let i = 0; i < ndof; i++) {
            c += F[i] * U[i];
        }

        return { U, c };
    }

    assembleK(nelx, nely, nelz, x, penal, KE) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const K = Array(ndof).fill(0).map(() => Array(ndof).fill(0));

        for (let elz = 0; elz < nelz; elz++) {
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const edof = this.getElementDOFs(elx, ely, elz, nelx, nely, nelz);
                    const idx = elx + ely * nelx + elz * nelx * nely;
                    const E = this.Emin + Math.pow(x[idx], penal) * (this.E0 - this.Emin);

                    for (let i = 0; i < 24; i++) {
                        for (let j = 0; j < 24; j++) {
                            K[edof[i]][edof[j]] += E * KE[i][j];
                        }
                    }
                }
            }
        }

        return K;
    }

    solveCG(K, F, freedofs, fixeddofs) {
        const n = freedofs.length;
        const Uf = new Float32Array(n);

        // Try WASM accelerated CG solver
        if (this.useWasm && wasmModule) {
            try {
                // Extract reduced K matrix and F vector for free DOFs
                const Kf = new Float64Array(n * n);
                const Ff = new Float64Array(n);
                
                for (let i = 0; i < n; i++) {
                    Ff[i] = F[freedofs[i]];
                    for (let j = 0; j < n; j++) {
                        Kf[i * n + j] = K[freedofs[i]][freedofs[j]];
                    }
                }
                
                const Uf64 = new Float64Array(n);
                const maxIter = Math.min(n, 1000);
                
                // Call WASM CG solver
                const ptrK = this._copyToWasm(Kf);
                const ptrF = this._copyToWasm(Ff);
                const ptrU = this._copyToWasm(Uf64);
                
                wasmModule.exports.conjugateGradient(ptrK, ptrF, ptrU, n, maxIter, CG_TOLERANCE);
                
                // Read result back
                const result = this._readFromWasm(ptrU, n);
                for (let i = 0; i < n; i++) {
                    Uf[i] = result[i];
                }
                
                // Free WASM memory
                this._freeWasm(ptrK);
                this._freeWasm(ptrF);
                this._freeWasm(ptrU);
                
                return Uf;
            } catch (error) {
                console.warn('WASM CG solver failed, falling back to JS:', error);
                // Fall through to JS implementation
            }
        }

        // Pure JavaScript CG solver fallback
        const r = new Float32Array(n);
        const p = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            r[i] = F[freedofs[i]];
        }

        const maxIter = Math.min(n, 1000);

        let rho = 0;
        for (let i = 0; i < n; i++) {
            rho += r[i] * r[i];
            p[i] = r[i];
        }

        for (let iter = 0; iter < maxIter; iter++) {
            if (Math.sqrt(rho) < CG_TOLERANCE) break;

            const Ap = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Ap[i] += K[freedofs[i]][freedofs[j]] * p[j];
                }
            }

            let pAp = 0;
            for (let i = 0; i < n; i++) {
                pAp += p[i] * Ap[i];
            }
            const alpha = rho / (pAp + EPSILON);

            let rho_new = 0;
            for (let i = 0; i < n; i++) {
                Uf[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rho_new += r[i] * r[i];
            }

            const beta = rho_new / (rho + EPSILON);
            for (let i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
            }

            rho = rho_new;
        }

        return Uf;
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

    getFixedDOFs(nelx, nely, nelz, position) {
        const fixeddofs = [];
        const nny = nely + 1;
        const nnz = nelz + 1;

        switch (position) {
            case 'left':
                // Fix all nodes at x=0
                for (let j = 0; j <= nely; j++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = 0 * nny * nnz + j * nnz + k;
                        fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);
                    }
                }
                break;
            case 'right':
                // Fix all nodes at x=nelx
                for (let j = 0; j <= nely; j++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = nelx * nny * nnz + j * nnz + k;
                        fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);
                    }
                }
                break;
            case 'bottom':
                // Fix all nodes at y=0
                for (let i = 0; i <= nelx; i++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = i * nny * nnz + 0 * nnz + k;
                        fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);
                    }
                }
                break;
            case 'top':
                // Fix all nodes at y=nely
                for (let i = 0; i <= nelx; i++) {
                    for (let k = 0; k <= nelz; k++) {
                        const n = i * nny * nnz + nely * nnz + k;
                        fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);
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
                    fixeddofs.push(3 * n, 3 * n + 1, 3 * n + 2);
                }
                break;
        }

        return fixeddofs;
    }

    getLoadVector(nelx, nely, nelz, direction, magnitude) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const F = new Float32Array(ndof);
        const nny = nely + 1;
        const nnz = nelz + 1;

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
    getFixedDOFsFromPaint(nelx, nely, nelz, paintedKeys) {
        const dofSet = new Set();
        const nny = nely + 1;
        const nnz = nelz + 1;

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
            const nodes = [n0, n1, n2, n3, n4, n5, n6, n7];

            const maxNode = (nelx + 1) * (nely + 1) * (nelz + 1);
            for (const n of nodes) {
                if (n >= 0 && n < maxNode) {
                    dofSet.add(3 * n);
                    dofSet.add(3 * n + 1);
                    dofSet.add(3 * n + 2);
                }
            }
        }
        return Array.from(dofSet);
    }

    /**
     * Convert painted force face keys to a load vector.
     * Distributes force evenly across all painted face nodes.
     */
    getLoadVectorFromPaint(nelx, nely, nelz, paintedKeys, direction, magnitude) {
        const ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);
        const F = new Float32Array(ndof);
        const nny = nely + 1;
        const nnz = nelz + 1;

        // Determine force direction components
        let fx = 0, fy = 0, fz = 0;
        switch (direction) {
            case 'down':       fy = -1; break;
            case 'up':         fy = 1; break;
            case 'left':       fx = -1; break;
            case 'right':      fx = 1; break;
            case 'top-center': fz = 1; break;
            default:           fy = -1;
        }

        // Collect unique nodes from painted faces
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
            const nodes = [n0, n1, n2, n3, n4, n5, n6, n7];

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
        await optimizer.optimize(model, config);
    } else if (type === 'cancel') {
        optimizer.cancelled = true;
    }
};
