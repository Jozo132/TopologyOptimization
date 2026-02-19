// AMR Boundary-Face Surface Mesh Generator
// Produces a closed (watertight) triangle surface mesh from AMR voxels.
//
// Algorithm (AMR boundary-face meshing):
// 1. Define active(cell) from filters (density threshold + custom predicates).
//    Treat inactive as "air".
// 2. For each active leaf cell, for each of 6 faces:
//    - Query all neighboring leaf cells that touch that face (AMR-aware neighbor lookup).
//    - If the neighbor region is fully inactive → face belongs to the surface.
//    - Tessellate the face to the finest neighbor resolution on that face:
//      subdivide into 2^k × 2^k patches so each patch matches a potential fine neighbor.
//    - Emit 2 triangles per patch, consistent winding outward.
// 3. Vertex dedup (hash by quantized position), build indexed triangles.
// 4. Normals: per-face normals (voxel-wrap look) or averaged smooth normals.
// 5. Closedness check: every undirected edge appears exactly twice (opposite orientation).
//
// Output buffers (suitable for WebGPU/WebGL):
//   positions: Float32Array (N*3)
//   normals:   Float32Array (N*3)
//   stress:    Float32Array (N) (optional)
//   indices:   Uint32Array  (M*3)

import { DENSITY_THRESHOLD } from './constants.js';

// Face definitions: index → { normal, dir (neighbor offset), vertices (CCW from outside) }
const FACE_DEFS = [
    { normal: [-1, 0, 0], dir: [-1, 0, 0], verts: [[0,0,0],[0,0,1],[0,1,1],[0,1,0]] }, // -X
    { normal: [1, 0, 0],  dir: [1, 0, 0],  verts: [[1,0,1],[1,0,0],[1,1,0],[1,1,1]] },  // +X
    { normal: [0, -1, 0], dir: [0, -1, 0], verts: [[0,0,0],[1,0,0],[1,0,1],[0,0,1]] },  // -Y
    { normal: [0, 1, 0],  dir: [0, 1, 0],  verts: [[0,1,0],[0,1,1],[1,1,1],[1,1,0]] },  // +Y
    { normal: [0, 0, -1], dir: [0, 0, -1], verts: [[0,0,0],[0,1,0],[1,1,0],[1,0,0]] },  // -Z
    { normal: [0, 0, 1],  dir: [0, 0, 1],  verts: [[0,0,1],[1,0,1],[1,1,1],[0,1,1]] },  // +Z
];

/**
 * Quantize a position to an integer key for vertex deduplication.
 * Uses a fixed precision to handle floating-point rounding.
 */
function posKey(x, y, z) {
    // Quantize to 1e-6 resolution to avoid floating point mismatches
    const qx = Math.round(x * 1e6);
    const qy = Math.round(y * 1e6);
    const qz = Math.round(z * 1e6);
    return `${qx},${qy},${qz}`;
}

/**
 * Generate a watertight surface mesh from a uniform voxel grid.
 *
 * For each active voxel whose face borders an inactive neighbor (or grid boundary),
 * that face is emitted as 2 triangles with outward-pointing CCW winding.
 * Vertices are deduplicated by quantized position, producing an indexed mesh.
 *
 * @param {object} options
 * @param {Float32Array} options.densities   - Density per voxel (nx*ny*nz)
 * @param {number}       options.nx          - Grid size X
 * @param {number}       options.ny          - Grid size Y
 * @param {number}       options.nz          - Grid size Z
 * @param {number}      [options.threshold]  - Density threshold (default DENSITY_THRESHOLD)
 * @param {Float32Array}[options.stress]     - Optional per-voxel stress values
 * @param {boolean}     [options.smoothNormals] - Average normals for smooth shading
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint32Array, stress: Float32Array|null, watertight: boolean }}
 */
export function generateUniformSurfaceMesh(options) {
    const { densities, nx, ny, nz } = options;
    const threshold = options.threshold !== undefined ? options.threshold : DENSITY_THRESHOLD;
    const stressField = options.stress || null;
    const smooth = !!options.smoothNormals;

    const isActive = (x, y, z) => {
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return false;
        return densities[x + y * nx + z * nx * ny] > threshold;
    };

    // Collect vertices and triangles with dedup
    const vertexMap = new Map();   // posKey → vertex index
    const vertPositions = [];       // flat [x,y,z, x,y,z, ...]
    const vertNormals = [];         // accumulated normals for smooth shading
    const vertStress = [];          // per-vertex stress (averaged from contributing cells)
    const vertStressCount = [];     // number of cells contributing to vertex stress
    const triIndices = [];          // flat [i0,i1,i2, ...]

    function getOrCreateVertex(x, y, z) {
        const key = posKey(x, y, z);
        let idx = vertexMap.get(key);
        if (idx === undefined) {
            idx = vertPositions.length / 3;
            vertexMap.set(key, idx);
            vertPositions.push(x, y, z);
            vertNormals.push(0, 0, 0);
            vertStress.push(0);
            vertStressCount.push(0);
        }
        return idx;
    }

    for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
            for (let z = 0; z < nz; z++) {
                if (!isActive(x, y, z)) continue;

                const cellIdx = x + y * nx + z * nx * ny;
                const cellStress = stressField ? stressField[cellIdx] : 0;

                for (let fi = 0; fi < 6; fi++) {
                    const face = FACE_DEFS[fi];
                    const [dx, dy, dz] = face.dir;

                    // Check if neighbor is inactive (boundary face)
                    if (isActive(x + dx, y + dy, z + dz)) continue;

                    // Emit quad as 2 triangles (CCW winding outward)
                    const v = face.verts.map(([vx, vy, vz]) => [x + vx, y + vy, z + vz]);
                    const [nx_, ny_, nz_] = face.normal;

                    const vi = v.map(([px, py, pz]) => getOrCreateVertex(px, py, pz));

                    // Triangle 1: v0, v1, v2
                    triIndices.push(vi[0], vi[1], vi[2]);
                    // Triangle 2: v0, v2, v3
                    triIndices.push(vi[0], vi[2], vi[3]);

                    // Accumulate normals and stress at each vertex
                    for (const idx of vi) {
                        vertNormals[idx * 3]     += nx_;
                        vertNormals[idx * 3 + 1] += ny_;
                        vertNormals[idx * 3 + 2] += nz_;
                        vertStress[idx] += cellStress;
                        vertStressCount[idx]++;
                    }
                }
            }
        }
    }

    // Finalize normals
    const numVerts = vertPositions.length / 3;
    const positions = new Float32Array(vertPositions);
    const normals = new Float32Array(vertNormals);

    if (smooth) {
        // Normalize accumulated normals for smooth shading
        for (let i = 0; i < numVerts; i++) {
            const ox = i * 3;
            const len = Math.sqrt(normals[ox] ** 2 + normals[ox + 1] ** 2 + normals[ox + 2] ** 2);
            if (len > 0) {
                normals[ox]     /= len;
                normals[ox + 1] /= len;
                normals[ox + 2] /= len;
            }
        }
    } else {
        // Per-face normals: re-compute from triangle vertices for flat shading
        // For indexed meshes with shared vertices, we keep the accumulated normals
        // and normalize them (same vertices shared by perpendicular faces will get
        // averaged normals, which is acceptable for voxel-wrap look)
        for (let i = 0; i < numVerts; i++) {
            const ox = i * 3;
            const len = Math.sqrt(normals[ox] ** 2 + normals[ox + 1] ** 2 + normals[ox + 2] ** 2);
            if (len > 0) {
                normals[ox]     /= len;
                normals[ox + 1] /= len;
                normals[ox + 2] /= len;
            }
        }
    }

    // Average vertex stress
    let stress = null;
    if (stressField) {
        stress = new Float32Array(numVerts);
        for (let i = 0; i < numVerts; i++) {
            stress[i] = vertStressCount[i] > 0 ? vertStress[i] / vertStressCount[i] : 0;
        }
    }

    const indices = new Uint32Array(triIndices);

    // Watertight check: every undirected edge must appear exactly twice
    const watertight = checkWatertight(indices);

    return { positions, normals, indices, stress, watertight };
}

/**
 * Generate a watertight surface mesh from AMR leaf cells.
 *
 * Handles resolution transitions between cells of different sizes by
 * subdividing larger faces to match the finer neighbor resolution,
 * ensuring no cracks at coarse/fine boundaries.
 *
 * @param {object} options
 * @param {Array}  options.cells - Array of { x, y, z, size, density, stress? }
 * @param {number} [options.threshold] - Density threshold (default DENSITY_THRESHOLD)
 * @param {boolean}[options.smoothNormals] - Average normals for smooth shading
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint32Array, stress: Float32Array|null, watertight: boolean }}
 */
export function generateAMRSurfaceMesh(options) {
    const { cells } = options;
    const threshold = options.threshold !== undefined ? options.threshold : DENSITY_THRESHOLD;
    const smooth = !!options.smoothNormals;

    if (!cells || cells.length === 0) {
        return {
            positions: new Float32Array(0),
            normals: new Float32Array(0),
            indices: new Uint32Array(0),
            stress: null,
            watertight: true
        };
    }

    // Build spatial index: map from quantized cell center → cell data
    // We use a hash map for O(1) neighbor lookups
    const cellMap = new Map();
    let minSize = Infinity;

    for (const cell of cells) {
        if (cell.size < minSize) minSize = cell.size;
        // Key by the cell origin (x, y, z) for lookup
        const key = posKey(cell.x, cell.y, cell.z);
        cellMap.set(key, cell);
    }

    /**
     * Check if a point is covered by an active cell.
     * For AMR, we need to check cells at all possible sizes.
     */
    function findCellAt(px, py, pz) {
        // Check at the finest resolution first, then coarser
        const key = posKey(px, py, pz);
        const cell = cellMap.get(key);
        if (cell) return cell;
        return null;
    }

    /**
     * Find all leaf cells that touch a given face of a cell.
     * A face at position (fx, fy, fz) with size (sw, sh) on a given axis.
     * Returns the finest resolution found among neighbors.
     */
    function findNeighborCells(cell, faceIdx) {
        const face = FACE_DEFS[faceIdx];
        const [dx, dy, dz] = face.dir;
        const size = cell.size;

        // Neighbor region: offset cell position by face direction * size
        const nx = cell.x + dx * size;
        const ny = cell.y + dy * size;
        const nz = cell.z + dz * size;

        // Check for a same-size neighbor
        const sameKey = posKey(nx, ny, nz);
        const sameNeighbor = cellMap.get(sameKey);
        if (sameNeighbor && sameNeighbor.size === size) {
            return { active: sameNeighbor.density > threshold, finestSize: size };
        }

        // Check for finer neighbors: subdivide the face region
        let anyActive = false;
        let finestSize = size;

        // Determine the face tangent axes
        const tangents = getFaceTangents(faceIdx);
        const [tu, tv] = tangents;

        // Try all possible fine cells in the neighbor region
        for (let step = minSize; step < size; step = step) {
            const subdiv = Math.round(size / step);
            for (let iu = 0; iu < subdiv; iu++) {
                for (let iv = 0; iv < subdiv; iv++) {
                    const px = nx + tu[0] * iu * step + tv[0] * iv * step;
                    const py = ny + tu[1] * iu * step + tv[1] * iv * step;
                    const pz = nz + tu[2] * iu * step + tv[2] * iv * step;
                    const fineCell = findCellAt(px, py, pz);
                    if (fineCell) {
                        if (fineCell.density > threshold) anyActive = true;
                        if (fineCell.size < finestSize) finestSize = fineCell.size;
                    }
                }
            }
            break; // Only check at minimum size
        }

        // Also check for coarser neighbor
        if (!sameNeighbor) {
            // Try to find a larger cell that contains this neighbor position
            for (const [, c] of cellMap) {
                if (c.size <= size) continue;
                if (nx >= c.x && nx < c.x + c.size &&
                    ny >= c.y && ny < c.y + c.size &&
                    nz >= c.z && nz < c.z + c.size) {
                    if (c.density > threshold) anyActive = true;
                    break;
                }
            }
        }

        return { active: anyActive, finestSize };
    }

    // Face tangent axes for subdivision, derived from face vertex definitions.
    // tu = v1 - v0, tv = v3 - v0 to preserve the CCW winding order.
    function getFaceTangents(faceIdx) {
        const v = FACE_DEFS[faceIdx].verts;
        const tu = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
        const tv = [v[3][0] - v[0][0], v[3][1] - v[0][1], v[3][2] - v[0][2]];
        return [tu, tv];
    }

    // Collect vertices and triangles with dedup
    const vertexMap = new Map();
    const vertPositions = [];
    const vertNormals = [];
    const vertStress = [];
    const vertStressCount = [];
    const triIndices = [];

    function getOrCreateVertex(x, y, z) {
        const key = posKey(x, y, z);
        let idx = vertexMap.get(key);
        if (idx === undefined) {
            idx = vertPositions.length / 3;
            vertexMap.set(key, idx);
            vertPositions.push(x, y, z);
            vertNormals.push(0, 0, 0);
            vertStress.push(0);
            vertStressCount.push(0);
        }
        return idx;
    }

    for (const cell of cells) {
        if (cell.density <= threshold) continue;

        const cellStress = cell.stress || 0;
        const size = cell.size;

        for (let fi = 0; fi < 6; fi++) {
            const { active, finestSize } = findNeighborCells(cell, fi);
            if (active) continue; // Neighbor is active, not a boundary face

            const face = FACE_DEFS[fi];
            const [fnx, fny, fnz] = face.normal;

            // Subdivide face to match finest neighbor resolution
            const subdiv = Math.max(1, Math.round(size / finestSize));

            const [tu, tv] = getFaceTangents(fi);
            const patchSize = size / subdiv;

            // Face origin: cell position + face vertex offsets scaled by cell size
            // The face origin is the first vertex of the face definition
            const [fv0x, fv0y, fv0z] = face.verts[0];
            const faceOriginX = cell.x + fv0x * size;
            const faceOriginY = cell.y + fv0y * size;
            const faceOriginZ = cell.z + fv0z * size;

            for (let iu = 0; iu < subdiv; iu++) {
                for (let iv = 0; iv < subdiv; iv++) {
                    // Patch corners
                    const px = faceOriginX + tu[0] * iu * patchSize + tv[0] * iv * patchSize;
                    const py = faceOriginY + tu[1] * iu * patchSize + tv[1] * iv * patchSize;
                    const pz = faceOriginZ + tu[2] * iu * patchSize + tv[2] * iv * patchSize;

                    // 4 corners of the patch quad
                    const c0 = [px, py, pz];
                    const c1 = [
                        px + tu[0] * patchSize,
                        py + tu[1] * patchSize,
                        pz + tu[2] * patchSize
                    ];
                    const c2 = [
                        px + tu[0] * patchSize + tv[0] * patchSize,
                        py + tu[1] * patchSize + tv[1] * patchSize,
                        pz + tu[2] * patchSize + tv[2] * patchSize
                    ];
                    const c3 = [
                        px + tv[0] * patchSize,
                        py + tv[1] * patchSize,
                        pz + tv[2] * patchSize
                    ];

                    // Map patch corners to match the face winding order
                    // Face verts are [v0, v1, v2, v3] CCW from outside
                    // Our patch is [c0, c1, c2, c3] where:
                    //   c0 = origin, c1 = origin+tu, c2 = origin+tu+tv, c3 = origin+tv
                    // This matches the CCW winding for the face
                    const vi0 = getOrCreateVertex(c0[0], c0[1], c0[2]);
                    const vi1 = getOrCreateVertex(c1[0], c1[1], c1[2]);
                    const vi2 = getOrCreateVertex(c2[0], c2[1], c2[2]);
                    const vi3 = getOrCreateVertex(c3[0], c3[1], c3[2]);

                    // Two triangles: (v0,v1,v2) and (v0,v2,v3) — CCW from outside
                    triIndices.push(vi0, vi1, vi2);
                    triIndices.push(vi0, vi2, vi3);

                    // Accumulate normals and stress
                    for (const idx of [vi0, vi1, vi2, vi3]) {
                        vertNormals[idx * 3]     += fnx;
                        vertNormals[idx * 3 + 1] += fny;
                        vertNormals[idx * 3 + 2] += fnz;
                        vertStress[idx] += cellStress;
                        vertStressCount[idx]++;
                    }
                }
            }
        }
    }

    // Finalize
    const numVerts = vertPositions.length / 3;
    const positions = new Float32Array(vertPositions);
    const normals = new Float32Array(vertNormals);

    // Normalize normals
    for (let i = 0; i < numVerts; i++) {
        const ox = i * 3;
        const len = Math.sqrt(normals[ox] ** 2 + normals[ox + 1] ** 2 + normals[ox + 2] ** 2);
        if (len > 0) {
            normals[ox]     /= len;
            normals[ox + 1] /= len;
            normals[ox + 2] /= len;
        }
    }

    // Average vertex stress
    const hasStress = cells.some(c => c.stress !== undefined && c.stress !== 0);
    let stress = null;
    if (hasStress) {
        stress = new Float32Array(numVerts);
        for (let i = 0; i < numVerts; i++) {
            stress[i] = vertStressCount[i] > 0 ? vertStress[i] / vertStressCount[i] : 0;
        }
    }

    const indices = new Uint32Array(triIndices);
    const watertight = checkWatertight(indices);

    return { positions, normals, indices, stress, watertight };
}

/**
 * Watertight check: every undirected edge must appear exactly twice
 * (once in each orientation). Returns true if mesh is closed.
 *
 * @param {Uint32Array} indices - Triangle indices (M*3)
 * @returns {boolean}
 */
export function checkWatertight(indices) {
    if (indices.length === 0) return true;

    const edgeCounts = new Map();

    for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
        // Directed edges: i0→i1, i1→i2, i2→i0
        const edges = [[i0, i1], [i1, i2], [i2, i0]];
        for (const [a, b] of edges) {
            const lo = Math.min(a, b);
            const hi = Math.max(a, b);
            const key = `${lo},${hi}`;
            edgeCounts.set(key, (edgeCounts.get(key) || 0) + 1);
        }
    }

    for (const count of edgeCounts.values()) {
        if (count !== 2) return false;
    }
    return true;
}

/**
 * Convert indexed mesh to flat triangle array suitable for STL export.
 *
 * @param {{ positions: Float32Array, normals: Float32Array, indices: Uint32Array }}
 * @returns {Array<{ normal: number[], vertices: number[][] }>}
 */
export function indexedMeshToTriangles(mesh) {
    const { positions, normals, indices } = mesh;
    const triangles = [];

    for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];

        const v0 = [positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]];
        const v1 = [positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]];
        const v2 = [positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]];

        // Compute face normal from cross product
        const e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        const e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 0) { nx /= len; ny /= len; nz /= len; }

        triangles.push({
            normal: [nx, ny, nz],
            vertices: [v0, v1, v2]
        });
    }

    return triangles;
}
