// Adaptive Mesh Refinement (AMR) Grid Implementation
// Supports dynamic refinement/coarsening of voxels based on stress

/**
 * Octree node for 3D AMR
 */
class OctreeNode {
    constructor(x, y, z, size, level = 0) {
        this.x = x;          // Position
        this.y = y;
        this.z = z;
        this.size = size;    // Size of this node
        this.level = level;  // Refinement level (0 = coarsest)
        
        this.density = 1.0;  // Material density
        this.stress = 0;     // Stress metric for refinement decisions
        this.children = null; // 8 children if refined
        this.isLeaf = true;
    }
    
    /**
     * Refine this node into 8 children
     */
    refine() {
        if (!this.isLeaf) return;
        
        this.isLeaf = false;
        this.children = [];
        const halfSize = this.size / 2;
        const nextLevel = this.level + 1;
        
        // Create 8 children in octree pattern
        for (let iz = 0; iz < 2; iz++) {
            for (let iy = 0; iy < 2; iy++) {
                for (let ix = 0; ix < 2; ix++) {
                    const child = new OctreeNode(
                        this.x + ix * halfSize,
                        this.y + iy * halfSize,
                        this.z + iz * halfSize,
                        halfSize,
                        nextLevel
                    );
                    child.density = this.density;
                    this.children.push(child);
                }
            }
        }
    }
    
    /**
     * Coarsen this node (remove children)
     */
    coarsen() {
        if (this.isLeaf) return;
        
        // Average density from children
        if (this.children && this.children.length > 0) {
            this.density = this.children.reduce((sum, c) => sum + c.density, 0) / this.children.length;
        }
        
        this.children = null;
        this.isLeaf = true;
    }
    
    /**
     * Get all leaf nodes under this node
     */
    getLeaves() {
        if (this.isLeaf) {
            return [this];
        }
        
        const leaves = [];
        for (const child of this.children) {
            leaves.push(...child.getLeaves());
        }
        return leaves;
    }
}

/**
 * Quadtree node for 2D AMR
 */
class QuadtreeNode {
    constructor(x, y, size, level = 0) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.level = level;
        
        this.density = 1.0;
        this.stress = 0;
        this.children = null;
        this.isLeaf = true;
    }
    
    refine() {
        if (!this.isLeaf) return;
        
        this.isLeaf = false;
        this.children = [];
        const halfSize = this.size / 2;
        const nextLevel = this.level + 1;
        
        // Create 4 children
        for (let iy = 0; iy < 2; iy++) {
            for (let ix = 0; ix < 2; ix++) {
                const child = new QuadtreeNode(
                    this.x + ix * halfSize,
                    this.y + iy * halfSize,
                    halfSize,
                    nextLevel
                );
                child.density = this.density;
                this.children.push(child);
            }
        }
    }
    
    coarsen() {
        if (this.isLeaf) return;
        
        if (this.children && this.children.length > 0) {
            this.density = this.children.reduce((sum, c) => sum + c.density, 0) / this.children.length;
        }
        
        this.children = null;
        this.isLeaf = true;
    }
    
    getLeaves() {
        if (this.isLeaf) {
            return [this];
        }
        
        const leaves = [];
        for (const child of this.children) {
            leaves.push(...child.getLeaves());
        }
        return leaves;
    }
}

/**
 * AMR Grid Manager
 */
export class AMRGrid {
    constructor(nx, ny, nz, minSize, maxSize, is3D = false) {
        this.nx = nx;
        this.ny = ny;
        this.nz = nz || 1;
        this.is3D = is3D;
        this.minSize = minSize;
        this.maxSize = maxSize;
        
        // Calculate levels based on size ratio
        this.maxLevel = Math.ceil(Math.log2(maxSize / minSize));
        
        // Create root nodes
        this.initializeGrid();
    }
    
    initializeGrid() {
        this.roots = [];
        
        // Determine base grid size from max size
        const baseSize = this.maxSize;
        const numX = Math.ceil(this.nx / baseSize);
        const numY = Math.ceil(this.ny / baseSize);
        
        if (this.is3D) {
            const numZ = Math.ceil(this.nz / baseSize);
            for (let iz = 0; iz < numZ; iz++) {
                for (let iy = 0; iy < numY; iy++) {
                    for (let ix = 0; ix < numX; ix++) {
                        const node = new OctreeNode(
                            ix * baseSize,
                            iy * baseSize,
                            iz * baseSize,
                            baseSize,
                            0
                        );
                        this.roots.push(node);
                    }
                }
            }
        } else {
            for (let iy = 0; iy < numY; iy++) {
                for (let ix = 0; ix < numX; ix++) {
                    const node = new QuadtreeNode(
                        ix * baseSize,
                        iy * baseSize,
                        baseSize,
                        0
                    );
                    this.roots.push(node);
                }
            }
        }
    }
    
    /**
     * Get all leaf nodes (actual elements)
     */
    getLeaves() {
        const leaves = [];
        for (const root of this.roots) {
            leaves.push(...root.getLeaves());
        }
        return leaves;
    }
    
    /**
     * Refine grid based on stress distribution
     * @param {Float32Array} stressValues - Stress at each current element
     * @param {number} targetCount - Try to maintain this many elements
     */
    adaptRefinement(stressValues, targetCount) {
        const leaves = this.getLeaves();
        
        // Map stress values to nodes
        if (stressValues && stressValues.length >= leaves.length) {
            for (let i = 0; i < leaves.length; i++) {
                leaves[i].stress = stressValues[i] || 0;
            }
        }
        
        // Calculate stress thresholds
        const stresses = leaves.map(n => n.stress).sort((a, b) => b - a);
        const highStressThreshold = stresses[Math.floor(stresses.length * 0.3)] || 0;
        const lowStressThreshold = stresses[Math.floor(stresses.length * 0.7)] || 0;
        
        // Count refinable and coarsenable nodes
        const refineCandidates = leaves.filter(n => 
            n.stress > highStressThreshold && 
            n.size > this.minSize &&
            n.isLeaf
        );
        
        const coarsenCandidates = [];
        for (const root of this.roots) {
            this.findCoarsenCandidates(root, lowStressThreshold, coarsenCandidates);
        }
        
        // Balance: coarsen low-stress to make room for high-stress refinement
        const currentCount = leaves.length;
        let refineCount = Math.min(refineCandidates.length, Math.max(0, targetCount - currentCount));
        let coarsenCount = Math.max(0, currentCount + refineCount * 7 - targetCount); // Each refine adds ~7 nodes
        
        // Coarsen first to make room
        const coarsenSorted = coarsenCandidates.sort((a, b) => a.maxStress - b.maxStress);
        for (let i = 0; i < Math.min(coarsenCount, coarsenSorted.length); i++) {
            coarsenSorted[i].node.coarsen();
        }
        
        // Then refine high-stress areas
        const refineSorted = refineCandidates.sort((a, b) => b.stress - a.stress);
        for (let i = 0; i < refineCount; i++) {
            refineSorted[i].refine();
        }
    }
    
    /**
     * Find parent nodes that can be coarsened
     */
    findCoarsenCandidates(node, stressThreshold, candidates) {
        if (node.isLeaf) return;
        
        // Check if all children are leaves and have low stress
        const allChildrenLeaves = node.children.every(c => c.isLeaf);
        if (allChildrenLeaves && node.size < this.maxSize) {
            const maxChildStress = Math.max(...node.children.map(c => c.stress));
            if (maxChildStress < stressThreshold) {
                candidates.push({ node, maxStress: maxChildStress });
            }
        }
        
        // Recurse
        if (node.children) {
            for (const child of node.children) {
                this.findCoarsenCandidates(child, stressThreshold, candidates);
            }
        }
    }
    
    /**
     * Convert AMR grid to flat arrays for FEA solver
     */
    toFlatArrays() {
        const leaves = this.getLeaves();
        const n = leaves.length;
        
        const x = new Float32Array(n);
        const y = new Float32Array(n);
        const z = new Float32Array(n);
        const sizes = new Float32Array(n);
        const densities = new Float32Array(n);
        const levels = new Int32Array(n);
        
        for (let i = 0; i < n; i++) {
            const leaf = leaves[i];
            x[i] = leaf.x;
            y[i] = leaf.y;
            z[i] = leaf.z || 0;
            sizes[i] = leaf.size;
            densities[i] = leaf.density;
            levels[i] = leaf.level;
        }
        
        return { x, y, z, sizes, densities, levels, count: n };
    }
    
    /**
     * Update densities from optimization
     */
    updateDensities(densityArray) {
        const leaves = this.getLeaves();
        for (let i = 0; i < Math.min(leaves.length, densityArray.length); i++) {
            leaves[i].density = densityArray[i];
        }
    }
}
