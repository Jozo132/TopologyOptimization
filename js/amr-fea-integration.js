// AMR-FEA Integration Module
// Integrates Adaptive Mesh Refinement with the FEA solver
// Uses a practical approach: uniform FEA with adaptive element weighting

import { AdaptiveElementManager } from './amr-solver-integration.js';

/**
 * AMR-Enhanced FEA Solver Wrapper
 * Provides adaptive mesh capabilities while maintaining uniform grid FEA
 */
export class AMRFEASolver {
    constructor(nelx, nely, nelz, config) {
        this.nelx = nelx;
        this.nely = nely;
        this.nelz = nelz;
        this.config = config;
        this.is3D = nelz > 1;
        
        // Initialize AMR manager if enabled
        this.amrManager = null;
        if (config.useAMR) {
            this.amrManager = new AdaptiveElementManager(
                nelx, nely, nelz,
                config.minGranuleSize || 0.5,
                config.maxGranuleSize || 2,
                true
            );
        }
        
        // Refinement frequency (refine every N iterations)
        this.refinementInterval = 10;
        this.lastRefinementIteration = 0;
    }
    
    /**
     * Check if refinement should occur this iteration
     */
    shouldRefine(iteration) {
        if (!this.amrManager) return false;
        return (iteration - this.lastRefinementIteration) >= this.refinementInterval;
    }
    
    /**
     * Perform adaptive refinement based on element stresses
     */
    refineAdaptively(iteration, elementEnergies, densities) {
        if (!this.amrManager || !this.shouldRefine(iteration)) {
            return false;
        }
        
        this.lastRefinementIteration = iteration;
        
        // Update stress values in AMR manager
        this.amrManager.updateStresses(elementEnergies, densities);
        
        // Target group count (try to maintain similar count)
        const currentCount = this.amrManager.groups.length;
        const targetCount = Math.max(
            Math.floor(currentCount * 0.9),
            Math.ceil((this.nelx * this.nely * (this.is3D ? this.nelz : 1)) / 10)
        );
        
        // Perform adaptation
        this.amrManager.adaptMesh(targetCount);
        
        return true;
    }
    
    /**
     * Get element weights based on AMR grouping
     * Elements in smaller groups (high stress) get higher weight
     */
    getElementWeights() {
        if (!this.amrManager) {
            const nel = this.nelx * this.nely * (this.is3D ? this.nelz : 1);
            return new Float32Array(nel).fill(1.0);
        }
        
        const nel = this.nelx * this.nely * (this.is3D ? this.nelz : 1);
        const weights = new Float32Array(nel);
        
        // Calculate weights: smaller groups = higher weight (more refined = more important)
        for (const group of this.amrManager.groups) {
            const weight = this.amrManager.maxSize / Math.max(group.size, 0.1);
            for (const elemIdx of group.elements) {
                weights[elemIdx] = weight;
            }
        }
        
        // Normalize weights
        let sumWeights = 0;
        for (let i = 0; i < nel; i++) {
            sumWeights += weights[i];
        }
        const avgWeight = sumWeights / nel;
        for (let i = 0; i < nel; i++) {
            weights[i] /= avgWeight;
        }
        
        return weights;
    }
    
    /**
     * Apply AMR weighting to sensitivity filtering
     * Elements in refined regions have more influence
     */
    applyAMRWeighting(dc, weights) {
        if (!this.amrManager || !weights) {
            return dc;
        }
        
        const dcWeighted = new Float32Array(dc.length);
        for (let i = 0; i < dc.length; i++) {
            dcWeighted[i] = dc[i] * weights[i];
        }
        
        return dcWeighted;
    }
    
    /**
     * Get AMR visualization data for rendering
     */
    getAMRVisualizationData() {
        if (!this.amrManager) {
            return null;
        }
        
        const groups = [];
        for (const group of this.amrManager.groups) {
            groups.push({
                id: group.id,
                size: group.size,
                elementCount: group.elements.length,
                stress: group.stress,
                density: group.density,
                elements: group.elements.slice() // Copy array
            });
        }
        
        return {
            groups: groups,
            stats: this.amrManager.getStats()
        };
    }
    
    /**
     * Get benchmarking statistics
     */
    getBenchmarkStats() {
        if (!this.amrManager) {
            return {
                enabled: false,
                totalElements: this.nelx * this.nely * (this.is3D ? this.nelz : 1)
            };
        }
        
        const stats = this.amrManager.getStats();
        return {
            enabled: true,
            ...stats,
            refinementInterval: this.refinementInterval,
            minGranuleSize: this.config.minGranuleSize,
            maxGranuleSize: this.config.maxGranuleSize,
            sizeRatio: stats.maxGroupSize / stats.minGroupSize
        };
    }
}
