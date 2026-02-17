// Model exporter for STL and JSON
export class ModelExporter {
    constructor() {}

    exportSTL(optimizedModel, filename) {
        const { densities, nx, ny, nz } = optimizedModel;
        
        // Generate STL from density field
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

    generateSTL(densities, nx, ny, nz) {
        // Generate binary STL file
        const threshold = 0.3; // Density threshold for solid elements
        const triangles = [];
        
        // Convert density field to mesh (marching cubes simplified)
        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    const index = x + y * nx + z * nx * ny;
                    if (densities[index] > threshold) {
                        // Add cube faces as triangles
                        this.addCubeTriangles(triangles, x, y, z);
                    }
                }
            }
        }
        
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

    addCubeTriangles(triangles, x, y, z) {
        const size = 1;
        
        // Define 8 vertices of the cube
        const v = [
            [x, y, z],
            [x + size, y, z],
            [x + size, y + size, z],
            [x, y + size, z],
            [x, y, z + size],
            [x + size, y, z + size],
            [x + size, y + size, z + size],
            [x, y + size, z + size]
        ];
        
        // Define 6 faces (each with 2 triangles)
        const faces = [
            // Front face (z = 0)
            { vertices: [v[0], v[1], v[2]], normal: [0, 0, -1] },
            { vertices: [v[0], v[2], v[3]], normal: [0, 0, -1] },
            // Back face (z = size)
            { vertices: [v[5], v[4], v[7]], normal: [0, 0, 1] },
            { vertices: [v[5], v[7], v[6]], normal: [0, 0, 1] },
            // Left face (x = 0)
            { vertices: [v[4], v[0], v[3]], normal: [-1, 0, 0] },
            { vertices: [v[4], v[3], v[7]], normal: [-1, 0, 0] },
            // Right face (x = size)
            { vertices: [v[1], v[5], v[6]], normal: [1, 0, 0] },
            { vertices: [v[1], v[6], v[2]], normal: [1, 0, 0] },
            // Bottom face (y = 0)
            { vertices: [v[4], v[5], v[1]], normal: [0, -1, 0] },
            { vertices: [v[4], v[1], v[0]], normal: [0, -1, 0] },
            // Top face (y = size)
            { vertices: [v[3], v[2], v[6]], normal: [0, 1, 0] },
            { vertices: [v[3], v[6], v[7]], normal: [0, 1, 0] }
        ];
        
        faces.forEach(face => triangles.push(face));
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
