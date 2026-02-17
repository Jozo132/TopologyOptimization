// AMR Solver Integration
// Provides adaptive mesh refinement integration with existing uniform grid solver
// Uses a hybrid approach: AMR guides element grouping/splitting while maintaining uniform FEA

/**
 * Adaptive Element Manager
 * Maps between uniform FEA grid and variable-resolution AMR representation
 */
export class AdaptiveElementManager {
    constructor(nelx, nely, nelz, minSize, maxSize, useAMR) {
        this.nelx = nelx;
        this.nely = nely;
        this.nelz = nelz;
        this.minSize = minSize;
        this.maxSize = maxSize;
        this.useAMR = useAMR;
        this.is3D = nelz > 1;
        
        // Element grouping map: uniform grid element -> AMR group ID
        this.elementToGroup = null;
        // Group properties: group ID -> { size, elements[], stress, density }
        this.groups = null;
        
        this.initialize();
    }
    
    initialize() {
        const nel = this.nelx * this.nely * (this.is3D ? this.nelz : 1);
        
        if (!this.useAMR) {
            // No AMR: each element is its own group
            this.elementToGroup = new Int32Array(nel).map((_, i) => i);
            this.groups = new Array(nel).fill(0).map((_, i) => ({
                id: i,
                size: 1,
                elements: [i],
                stress: 0,
                density: 0.5
            }));
            return;
        }
        
        // Initialize with coarse grouping
        this.createInitialGrouping();
    }
    
    createInitialGrouping() {
        const groupSize = Math.max(1, Math.floor(this.maxSize));
        const nel = this.nelx * this.nely * (this.is3D ? this.nelz : 1);
        
        this.elementToGroup = new Int32Array(nel);
        this.groups = [];
        
        let groupId = 0;
        
        if (this.is3D) {
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
        } else {
            // 2D grouping
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
    }
    
    /**
     * Update group stresses from element energies
     */
    updateStresses(elementEnergies, densities) {
        for (const group of this.groups) {
            let totalStress = 0;
            let totalDensity = 0;
            
            for (const elemIdx of group.elements) {
                const energy = elementEnergies[elemIdx] || 0;
                const density = densities[elemIdx] || 0.5;
                // Stress = stiffness * energy
                const stiffness = this.Emin + Math.pow(density, 3) * (1 - this.Emin);
                totalStress += stiffness * energy;
                totalDensity += density;
            }
            
            group.stress = totalStress / group.elements.length;
            group.density = totalDensity / group.elements.length;
        }
        
        // Store for AMR
        this.Emin = 1e-9;
    }
    
    /**
     * Perform adaptive refinement/coarsening
     * Refines high-stress groups, coarsens low-stress groups
     */
    adaptMesh(targetGroupCount) {
        if (!this.useAMR || this.groups.length === 0) return;
        
        // Calculate stress thresholds
        const stresses = this.groups.map(g => g.stress).sort((a, b) => b - a);
        const highStressThreshold = stresses[Math.floor(stresses.length * 0.2)] || 0;
        const lowStressThreshold = stresses[Math.floor(stresses.length * 0.8)] || 0;
        
        // Find refinement candidates (high stress, can split)
        const refineCandidates = this.groups.filter(g => 
            g.stress > highStressThreshold && 
            g.size > this.minSize * 2
        );
        
        // Find coarsening candidates (low stress, small groups that can merge)
        const coarsenCandidates = this.groups.filter(g =>
            g.stress < lowStressThreshold &&
            g.size < this.maxSize
        );
        
        const currentCount = this.groups.length;
        const targetRefine = Math.min(
            refineCandidates.length,
            Math.floor((targetGroupCount - currentCount) / 7) // Each split adds ~7 groups
        );
        const targetCoarsen = Math.min(
            Math.floor(coarsenCandidates.length / 4), // Merge 4 into 1
            Math.floor(targetRefine * 7 / 4) // Balance refinement
        );
        
        // Sort by stress
        refineCandidates.sort((a, b) => b.stress - a.stress);
        coarsenCandidates.sort((a, b) => a.stress - b.stress);
        
        // Coarsen first to make room
        for (let i = 0; i < targetCoarsen; i += 4) {
            if (i + 3 < coarsenCandidates.length) {
                this.mergeGroups([
                    coarsenCandidates[i],
                    coarsenCandidates[i + 1],
                    coarsenCandidates[i + 2],
                    coarsenCandidates[i + 3]
                ]);
            }
        }
        
        // Then refine high-stress areas
        for (let i = 0; i < targetRefine; i++) {
            this.splitGroup(refineCandidates[i]);
        }
    }
    
    /**
     * Split a group into smaller groups
     */
    splitGroup(group) {
        if (!group || group.elements.length <= 1) return;
        
        // Remove old group
        const idx = this.groups.indexOf(group);
        if (idx === -1) return;
        this.groups.splice(idx, 1);
        
        // Split elements into sub-groups (2x2x2 for 3D, 2x2 for 2D)
        const halfSize = Math.max(1, Math.floor(group.size / 2));
        const newGroups = [];
        
        // Get bounding box of group elements
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        for (const elemIdx of group.elements) {
            let x, y, z;
            if (this.is3D) {
                z = Math.floor(elemIdx / (this.nelx * this.nely));
                const rem = elemIdx % (this.nelx * this.nely);
                y = Math.floor(rem / this.nelx);
                x = rem % this.nelx;
            } else {
                y = elemIdx % this.nely;
                x = Math.floor(elemIdx / this.nely);
                z = 0;
            }
            
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            minZ = Math.min(minZ, z);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            maxZ = Math.max(maxZ, z);
        }
        
        const midX = Math.floor((minX + maxX) / 2);
        const midY = Math.floor((minY + maxY) / 2);
        const midZ = this.is3D ? Math.floor((minZ + maxZ) / 2) : 0;
        
        // Create 4 or 8 sub-groups
        const subdivs = this.is3D ? 8 : 4;
        for (let i = 0; i < subdivs; i++) {
            newGroups.push({
                id: this.groups.length + i,
                size: halfSize,
                elements: [],
                stress: group.stress,
                density: group.density
            });
        }
        
        // Assign elements to sub-groups
        for (const elemIdx of group.elements) {
            let x, y, z;
            if (this.is3D) {
                z = Math.floor(elemIdx / (this.nelx * this.nely));
                const rem = elemIdx % (this.nelx * this.nely);
                y = Math.floor(rem / this.nelx);
                x = rem % this.nelx;
            } else {
                y = elemIdx % this.nely;
                x = Math.floor(elemIdx / this.nely);
                z = 0;
            }
            
            const subIdx = (x >= midX ? 1 : 0) +
                          (y >= midY ? 2 : 0) +
                          (this.is3D && z >= midZ ? 4 : 0);
            
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
    
    /**
     * Merge multiple groups into one
     */
    mergeGroups(groupsToMerge) {
        if (groupsToMerge.length === 0) return;
        
        const mergedElements = [];
        let totalStress = 0;
        let totalDensity = 0;
        
        for (const group of groupsToMerge) {
            mergedElements.push(...group.elements);
            totalStress += group.stress * group.elements.length;
            totalDensity += group.density * group.elements.length;
            
            // Remove from groups array
            const idx = this.groups.indexOf(group);
            if (idx !== -1) {
                this.groups.splice(idx, 1);
            }
        }
        
        const newGroup = {
            id: this.groups.length,
            size: Math.min(this.maxSize, groupsToMerge[0].size * 2),
            elements: mergedElements,
            stress: totalStress / mergedElements.length,
            density: totalDensity / mergedElements.length
        };
        
        // Update element to group mapping
        for (const elemIdx of mergedElements) {
            this.elementToGroup[elemIdx] = newGroup.id;
        }
        
        this.groups.push(newGroup);
    }
    
    /**
     * Get AMR statistics for benchmarking
     */
    getStats() {
        const sizes = this.groups.map(g => g.size);
        return {
            groupCount: this.groups.length,
            minGroupSize: Math.min(...sizes),
            maxGroupSize: Math.max(...sizes),
            avgGroupSize: sizes.reduce((a, b) => a + b, 0) / sizes.length,
            totalElements: this.nelx * this.nely * (this.is3D ? this.nelz : 1)
        };
    }
}
