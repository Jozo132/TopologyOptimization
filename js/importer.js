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
        
        let result;
        if (isBinary) {
            result = this.parseBinarySTL(dataView);
        } else {
            result = this.parseASCIISTL(arrayBuffer);
        }

        // Fallback: if no triangles were found, try the other format
        if (result.originalVertices && result.originalVertices.length === 0) {
            if (isBinary) {
                result = this.parseASCIISTL(arrayBuffer);
            } else {
                result = this.parseBinarySTL(dataView);
            }
        }

        return result;
    }

    isBinarySTL(arrayBuffer) {
        // Binary STL files start with 80-byte header, followed by triangle count
        if (arrayBuffer.byteLength < 84) return false;
        
        // Read the expected triangle count from the header
        const dataView = new DataView(arrayBuffer);
        const triangleCount = dataView.getUint32(80, true);
        
        // Expected binary size: 80 (header) + 4 (count) + 50 per triangle
        const expectedSize = 84 + triangleCount * 50;
        
        // If the file size matches the expected binary layout, it is binary
        if (arrayBuffer.byteLength === expectedSize) return true;
        
        // Check whether the header starts with "solid" (ASCII STL marker).
        // Many binary exporters also write "solid" into the 80-byte header, so
        // we additionally look for ASCII keywords ("facet", "vertex") in the
        // first portion of the file to confirm it really is ASCII.
        const headerText = new TextDecoder().decode(arrayBuffer.slice(0, 5));
        if (headerText.toLowerCase().startsWith('solid')) {
            const probeSize = Math.min(1000, arrayBuffer.byteLength);
            const probe = new TextDecoder().decode(arrayBuffer.slice(0, probeSize));
            if (probe.includes('facet') && probe.includes('vertex')) {
                return false; // Looks like genuine ASCII STL
            }
            return true; // Starts with "solid" but no ASCII keywords → binary
        }
        
        return true; // Does not start with "solid" → binary
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
        
        // Physical dimensions in mm (STL coordinates are treated as mm)
        const sizeX = maxX - minX || 1;
        const sizeY = maxY - minY || 1;
        const sizeZ = maxZ - minZ || 1;

        // Use provided resolution or fall back to instance resolution.
        // The resolution parameter defines the voxel size: 1 voxel = 1mm at
        // base density 20.  Higher density gives smaller voxels so sub-1mm
        // features can be captured.
        const res = resolution !== null ? resolution : (this.resolution || 20);

        // Voxel size in mm – at base density 20 the longest axis gets 20 voxels.
        // Increasing density shrinks the voxel size proportionally so that thin
        // features (even sub-1mm) produce enough voxels to be resolved.
        const maxDim = Math.max(sizeX, sizeY, sizeZ);
        const voxelSize = maxDim / res;

        const nx = Math.max(1, Math.ceil(sizeX / voxelSize));
        const ny = Math.max(1, Math.ceil(sizeY / voxelSize));
        const nz = Math.max(1, Math.ceil(sizeZ / voxelSize));
        
        // Initialize voxel grid with all solid
        const elements = new Float32Array(nx * ny * nz).fill(1);
        
        return {
            nx,
            ny,
            nz,
            elements,
            voxelSize,
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
        const baseSize = 5;
        const scaledSize = Math.max(3, Math.round(baseSize * scale));
        const nx = scaledSize;
        const ny = scaledSize;
        const nz = scaledSize;
        const elements = new Float32Array(nx * ny * nz).fill(1);
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'cube',
            templateScale: { baseNx: baseSize, baseNy: baseSize, baseNz: baseSize, baseGranuleDensity: 20 },
            // Predefined boundary conditions for cube test
            forcePosition: 'top-center',  // Force at top center
            constraintPositions: 'bottom-corners'  // Constraints at bottom 4 corners
        };
    }
}
