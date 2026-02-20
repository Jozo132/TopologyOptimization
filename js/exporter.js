// Model exporter for STL and JSON
// Uses AMR boundary-face meshing or Marching Cubes for watertight STL export
import { DENSITY_THRESHOLD } from './constants.js';
import { generateUniformSurfaceMesh, generateAMRSurfaceMesh, indexedMeshToTriangles } from './amr-surface-mesh.js';
import { marchingCubes } from './marching-cubes.js';

export class ModelExporter {
    constructor() {}

    exportSTL(optimizedModel, filename, options = {}) {
        const { densities, nx, ny, nz } = optimizedModel;
        const threshold = options.threshold !== undefined ? options.threshold : DENSITY_THRESHOLD;
        const amrCells = optimizedModel.amrCells || options.amrCells || null;

        // Generate STL using AMR surface mesh when AMR cells are available,
        // otherwise fall back to uniform grid extraction
        const stl = this.generateSTL(densities, nx, ny, nz, { threshold, amrCells });

        // Create blob and download
        const blob = new Blob([stl], { type: 'application/octet-stream' });
        this.downloadBlob(blob, filename);

        console.log('STL exported:', filename);
    }

    exportJSON(optimizedModel, filename) {
        // Export full optimization data as JSON
        const data = {
            densities: Array.from(optimizedModel.densities),
            dimensions: {
                nx: optimizedModel.nx,
                ny: optimizedModel.ny,
                nz: optimizedModel.nz
            },
            results: {
                finalCompliance: optimizedModel.finalCompliance,
                iterations: optimizedModel.iterations,
                volumeFraction: optimizedModel.volumeFraction
            },
            timestamp: new Date().toISOString()
        };
        
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        this.downloadBlob(blob, filename);
        
        console.log('JSON exported:', filename);
    }

    /**
     * Generate a binary STL from the density field.
     * When smooth=true, uses Marching Cubes for smooth isosurface geometry.
     * Otherwise uses AMR boundary-face meshing for watertight box-face STL.
     */
    generateSTL(densities, nx, ny, nz, options = {}) {
        const threshold = options.threshold !== undefined ? options.threshold : DENSITY_THRESHOLD;
        const amrCells = options.amrCells || null;
        const smooth = options.smooth !== undefined ? options.smooth : false;

        if (smooth && densities) {
            return this._generateSmoothSTL(densities, nx, ny, nz, threshold);
        }

        let mesh;
        if (amrCells && amrCells.length > 0) {
            // Use AMR-aware surface mesh generation for multi-resolution grids
            mesh = generateAMRSurfaceMesh({ cells: amrCells, threshold });
        } else {
            // Uniform grid boundary-face extraction
            mesh = generateUniformSurfaceMesh({ densities, nx, ny, nz, threshold });
        }
        const triangles = indexedMeshToTriangles(mesh);

        return this._buildSTLBuffer(triangles);
    }

    /**
     * Generate a smooth STL using Marching Cubes isosurface extraction.
     */
    _generateSmoothSTL(densities, nx, ny, nz, threshold) {
        const mc = marchingCubes({ densities, nx, ny, nz, threshold });
        const { positions, normals, indices } = mc;

        // Build triangle array from MC output
        const triangles = [];
        for (let i = 0; i < indices.length; i += 3) {
            const i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
            // Compute face normal from vertices for STL (flat normal per triangle)
            const v0 = [positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]];
            const v1 = [positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]];
            const v2 = [positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]];
            const e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            const e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let nx_ = e1[1] * e2[2] - e1[2] * e2[1];
            let ny_ = e1[2] * e2[0] - e1[0] * e2[2];
            let nz_ = e1[0] * e2[1] - e1[1] * e2[0];
            const len = Math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_);
            if (len > 0) { nx_ /= len; ny_ /= len; nz_ /= len; }
            triangles.push({ normal: [nx_, ny_, nz_], vertices: [v0, v1, v2] });
        }

        return this._buildSTLBuffer(triangles);
    }

    /**
     * Build a binary STL buffer from an array of {normal, vertices} triangles.
     */
    _buildSTLBuffer(triangles) {
        const buffer = new ArrayBuffer(84 + triangles.length * 50);
        const view = new DataView(buffer);
        
        // Header (80 bytes)
        const header = 'Topology Optimization Result';
        for (let i = 0; i < 80; i++) {
            view.setUint8(i, i < header.length ? header.charCodeAt(i) : 0);
        }
        
        // Number of triangles
        view.setUint32(80, triangles.length, true);
        
        // Write triangles
        let offset = 84;
        triangles.forEach(triangle => {
            // Normal vector
            view.setFloat32(offset, triangle.normal[0], true);
            view.setFloat32(offset + 4, triangle.normal[1], true);
            view.setFloat32(offset + 8, triangle.normal[2], true);
            offset += 12;
            
            // Vertices
            for (let i = 0; i < 3; i++) {
                view.setFloat32(offset, triangle.vertices[i][0], true);
                view.setFloat32(offset + 4, triangle.vertices[i][1], true);
                view.setFloat32(offset + 8, triangle.vertices[i][2], true);
                offset += 12;
            }
            
            // Attribute byte count
            view.setUint16(offset, 0, true);
            offset += 2;
        });
        
        return buffer;
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}
