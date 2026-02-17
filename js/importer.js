// Model importer for STL files and template generation
export class ModelImporter {
    constructor() {
        this.reader = new FileReader();
        this.resolution = 20;
    }

    async importSTL(file, resolution) {
        this.resolution = resolution || 20;
        return new Promise((resolve, reject) => {
            this.reader.onload = (e) => {
                try {
                    const arrayBuffer = e.target.result;
                    const model = this.parseSTL(arrayBuffer);
                    resolve(model);
                } catch (error) {
                    reject(error);
                }
            };
            
            this.reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };
            
            this.reader.readAsArrayBuffer(file);
        });
    }

    parseSTL(arrayBuffer) {
        // Simple STL parser - converts to voxel grid
        const dataView = new DataView(arrayBuffer);
        
        // Check if binary or ASCII
        const isBinary = this.isBinarySTL(arrayBuffer);
        
        if (isBinary) {
            return this.parseBinarySTL(dataView);
        } else {
            return this.parseASCIISTL(arrayBuffer);
        }
    }

    isBinarySTL(arrayBuffer) {
        // Binary STL files start with 80-byte header, followed by triangle count
        if (arrayBuffer.byteLength < 84) return false;
        
        const text = new TextDecoder().decode(arrayBuffer.slice(0, 5));
        return !text.toLowerCase().startsWith('solid');
    }

    parseBinarySTL(dataView) {
        // Skip 80-byte header
        const triangleCount = dataView.getUint32(80, true);
        
        const vertices = [];
        let offset = 84;
        
        for (let i = 0; i < triangleCount; i++) {
            // Skip normal (3 floats)
            offset += 12;
            
            // Read 3 vertices (9 floats)
            for (let j = 0; j < 3; j++) {
                const x = dataView.getFloat32(offset, true);
                const y = dataView.getFloat32(offset + 4, true);
                const z = dataView.getFloat32(offset + 8, true);
                vertices.push({ x, y, z });
                offset += 12;
            }
            
            // Skip attribute byte count
            offset += 2;
        }
        
        // Convert vertices to voxel grid
        return this.voxelizeVertices(vertices);
    }

    parseASCIISTL(arrayBuffer) {
        const text = new TextDecoder().decode(arrayBuffer);
        const lines = text.split('\n');
        
        const vertices = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('vertex')) {
                const parts = line.split(/\s+/);
                vertices.push({
                    x: parseFloat(parts[1]),
                    y: parseFloat(parts[2]),
                    z: parseFloat(parts[3])
                });
            }
        }
        
        return this.voxelizeVertices(vertices);
    }

    voxelizeVertices(vertices, resolution = null) {
        // Find bounding box
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        vertices.forEach(v => {
            minX = Math.min(minX, v.x);
            minY = Math.min(minY, v.y);
            minZ = Math.min(minZ, v.z);
            maxX = Math.max(maxX, v.x);
            maxY = Math.max(maxY, v.y);
            maxZ = Math.max(maxZ, v.z);
        });
        
        // Create voxel grid using granule density parameter
        // Use provided resolution or fall back to instance resolution
        const res = resolution !== null ? resolution : (this.resolution || 20);
        const sizeX = maxX - minX || 1;
        const sizeY = maxY - minY || 1;
        const sizeZ = maxZ - minZ || 1;
        
        const nx = Math.min(res, Math.ceil(sizeX / Math.max(sizeX, sizeY, sizeZ) * res));
        const ny = Math.min(res, Math.ceil(sizeY / Math.max(sizeX, sizeY, sizeZ) * res));
        const nz = Math.min(res, Math.ceil(sizeZ / Math.max(sizeX, sizeY, sizeZ) * res));
        
        // Initialize voxel grid with all solid
        const elements = new Float32Array(nx * ny * nz).fill(1);
        
        return {
            nx,
            ny,
            nz,
            elements,
            bounds: { minX, minY, minZ, maxX, maxY, maxZ },
            originalVertices: vertices  // Store vertices for re-voxelization
        };
    }

    createTemplate(type, granuleDensity = 20) {
        switch (type) {
            case 'beam':
                return this.createBeamTemplate(granuleDensity);
            case 'bridge':
                return this.createBridgeTemplate(granuleDensity);
            case 'cube':
                return this.createCubeTemplate(granuleDensity);
            default:
                throw new Error('Unknown template type');
        }
    }

    createBeamTemplate(granuleDensity = 20) {
        // Cantilever beam: scaled based on granuleDensity
        // Original: 30x10x10 at granuleDensity=20
        const scale = granuleDensity / 20;
        const nx = Math.max(5, Math.round(30 * scale));
        const ny = Math.max(3, Math.round(10 * scale));
        const nz = Math.max(3, Math.round(10 * scale));
        const elements = new Float32Array(nx * ny * nz).fill(1);
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'beam',
            templateScale: { baseNx: 30, baseNy: 10, baseNz: 10, baseGranuleDensity: 20 }
        };
    }

    createBridgeTemplate(granuleDensity = 20) {
        // Bridge: scaled based on granuleDensity
        // Original: 40x15x8 at granuleDensity=20
        const scale = granuleDensity / 20;
        const nx = Math.max(5, Math.round(40 * scale));
        const ny = Math.max(3, Math.round(15 * scale));
        const nz = Math.max(3, Math.round(8 * scale));
        const elements = new Float32Array(nx * ny * nz).fill(1);
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'bridge',
            templateScale: { baseNx: 40, baseNy: 15, baseNz: 8, baseGranuleDensity: 20 }
        };
    }

    createCubeTemplate(granuleDensity = 20) {
        // Cube test: scaled based on granuleDensity
        // Original: 5x5x5 at granuleDensity=20
        // Keep it simple for cube test - direct scaling
        const scale = granuleDensity / 20;
        const nx = Math.max(3, Math.round(5 * scale));
        const ny = Math.max(3, Math.round(5 * scale));
        const nz = Math.max(3, Math.round(5 * scale));
        const elements = new Float32Array(nx * ny * nz).fill(1);
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'cube',
            templateScale: { baseNx: 5, baseNy: 5, baseNz: 5, baseGranuleDensity: 20 },
            // Predefined boundary conditions for cube test
            forcePosition: 'top-center',  // Force at top center
            constraintPositions: 'bottom-corners'  // Constraints at bottom 4 corners
        };
    }
}
