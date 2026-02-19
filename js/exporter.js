// Model exporter for STL and JSON
// Uses AMR boundary-face meshing for watertight STL export
import { DENSITY_THRESHOLD } from './constants.js';
import { generateUniformSurfaceMesh, indexedMeshToTriangles } from './amr-surface-mesh.js';

export class ModelExporter {
    constructor() {}

    exportSTL(optimizedModel, filename) {
        const { densities, nx, ny, nz } = optimizedModel;
        
        // Generate STL from density field using AMR boundary-face meshing
        const stl = this.generateSTL(densities, nx, ny, nz);
        
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
     * Generate a binary STL from the density field using AMR boundary-face meshing.
     * Produces a watertight closed surface by extracting only boundary faces
     * (between active and inactive voxels) with vertex deduplication.
     */
    generateSTL(densities, nx, ny, nz) {
        // Generate watertight surface mesh via AMR boundary-face extraction
        const mesh = generateUniformSurfaceMesh({ densities, nx, ny, nz });
        const triangles = indexedMeshToTriangles(mesh);
        
        // Create binary STL
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
