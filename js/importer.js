// Model importer for STL, STEP (AP203/AP214), DXF and SVG files and template generation
import { STEPParser } from './step-parser.js';

export class ModelImporter {
    constructor() {
        this.reader = new FileReader();
        this.resolution = 20;
        this.voxelSizeMM = 1; // default voxel size in mm
    }

    /**
     * Import a file (STL or STEP) based on its extension.
     * @param {File} file
     * @param {number|null} resolution
     * @returns {Promise<object>}
     */
    async importFile(file, resolution) {
        const name = (file.name || '').toLowerCase();
        if (name.endsWith('.stp') || name.endsWith('.step')) {
            return this.importSTEP(file, resolution);
        }
        if (name.endsWith('.dxf')) {
            return this.importDXF(file, resolution);
        }
        if (name.endsWith('.svg')) {
            return this.importSVG(file, resolution);
        }
        return this.importSTL(file, resolution);
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

    /**
     * Import a STEP file (AP203 or AP214).
     * @param {File} file
     * @param {number|null} resolution
     * @returns {Promise<object>}
     */
    async importSTEP(file, resolution) {
        this.resolution = resolution || 20;
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const text = e.target.result;
                    const model = this.parseSTEP(text);
                    resolve(model);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => {
                reject(new Error('Failed to read STEP file'));
            };
            reader.readAsText(file);
        });
    }

    /**
     * Parse STEP file text and return a voxelized model.
     * @param {string} text - STEP file content
     * @returns {object} Voxelized model
     */
    parseSTEP(text) {
        const parser = new STEPParser();
        const { vertices, protocol } = parser.parse(text);

        if (!vertices || vertices.length < 3) {
            throw new Error('No geometry found in STEP file');
        }

        const model = this.voxelizeVertices(vertices);
        model.sourceFormat = 'STEP';
        model.protocol = protocol;
        return model;
    }

    /**
     * Transform vertices by applying scale and rotation.
     * @param {Array} vertices - Array of {x, y, z} objects
     * @param {number} scale - Uniform scale factor
     * @param {number} rotX - Rotation around X axis in degrees
     * @param {number} rotY - Rotation around Y axis in degrees
     * @param {number} rotZ - Rotation around Z axis in degrees
     * @returns {Array} Transformed vertices
     */
    transformVertices(vertices, scale = 1, rotX = 0, rotY = 0, rotZ = 0) {
        if (scale === 1 && rotX === 0 && rotY === 0 && rotZ === 0) {
            return vertices;
        }

        const degToRad = Math.PI / 180;
        const rx = rotX * degToRad;
        const ry = rotY * degToRad;
        const rz = rotZ * degToRad;

        const cosX = Math.cos(rx), sinX = Math.sin(rx);
        const cosY = Math.cos(ry), sinY = Math.sin(ry);
        const cosZ = Math.cos(rz), sinZ = Math.sin(rz);

        return vertices.map(v => {
            let x = v.x * scale;
            let y = v.y * scale;
            let z = v.z * scale;

            // Rotate around X axis
            if (rotX !== 0) {
                const y1 = y * cosX - z * sinX;
                const z1 = y * sinX + z * cosX;
                y = y1;
                z = z1;
            }

            // Rotate around Y axis
            if (rotY !== 0) {
                const x1 = x * cosY + z * sinY;
                const z1 = -x * sinY + z * cosY;
                x = x1;
                z = z1;
            }

            // Rotate around Z axis
            if (rotZ !== 0) {
                const x1 = x * cosZ - y * sinZ;
                const y1 = x * sinZ + y * cosZ;
                x = x1;
                y = y1;
            }

            return { x, y, z };
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
        if (result && result.originalVertices && result.originalVertices.length === 0) {
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
            const probe = new TextDecoder().decode(arrayBuffer.slice(0, probeSize)).toLowerCase();
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

    /**
     * Import a DXF file (2D profile).
     * @param {File} file
     * @param {number|null} resolution
     * @returns {Promise<object>}
     */
    async importDXF(file, resolution) {
        this.resolution = resolution || 20;
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const text = e.target.result;
                    const model = this.parseDXF(text);
                    resolve(model);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => {
                reject(new Error('Failed to read DXF file'));
            };
            reader.readAsText(file);
        });
    }

    /**
     * Import an SVG file (2D profile).
     * @param {File} file
     * @param {number|null} resolution
     * @returns {Promise<object>}
     */
    async importSVG(file, resolution) {
        this.resolution = resolution || 20;
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const text = e.target.result;
                    const model = this.parseSVG(text);
                    resolve(model);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => {
                reject(new Error('Failed to read SVG file'));
            };
            reader.readAsText(file);
        });
    }

    /**
     * Parse DXF file text and return a 2D voxelized model.
     * Supports LINE, LWPOLYLINE, POLYLINE, and CIRCLE entities.
     * @param {string} text - DXF file content
     * @returns {object} Voxelized model (nz=1)
     */
    parseDXF(text) {
        const lines = text.split(/\r?\n/);
        const polygons = [];
        let currentPolygon = null;
        let i = 0;

        // Helper to read group code / value pairs
        const readPair = () => {
            if (i + 1 >= lines.length) return null;
            const code = parseInt(lines[i].trim(), 10);
            const value = lines[i + 1].trim();
            i += 2;
            return { code, value };
        };

        // Scan for ENTITIES section
        while (i < lines.length) {
            const line = lines[i].trim();
            if (line === 'ENTITIES') { i++; break; }
            i++;
        }

        // Parse entities
        while (i < lines.length) {
            const pair = readPair();
            if (!pair) break;

            // End of ENTITIES section
            if (pair.code === 0 && pair.value === 'ENDSEC') break;

            if (pair.code === 0 && pair.value === 'LINE') {
                // Read a LINE entity (two endpoints)
                let x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                while (i < lines.length) {
                    const p = readPair();
                    if (!p) break;
                    if (p.code === 0) { i -= 2; break; } // next entity
                    if (p.code === 10) x1 = parseFloat(p.value);
                    if (p.code === 20) y1 = parseFloat(p.value);
                    if (p.code === 11) x2 = parseFloat(p.value);
                    if (p.code === 21) y2 = parseFloat(p.value);
                }
                polygons.push([{ x: x1, y: y1 }, { x: x2, y: y2 }]);
            } else if (pair.code === 0 && pair.value === 'LWPOLYLINE') {
                // Lightweight polyline
                const points = [];
                let closed = false;
                let cx = 0, cy = 0;
                let hasX = false;
                while (i < lines.length) {
                    const p = readPair();
                    if (!p) break;
                    if (p.code === 0) { i -= 2; break; }
                    if (p.code === 70) closed = (parseInt(p.value, 10) & 1) !== 0;
                    if (p.code === 10) {
                        if (hasX) points.push({ x: cx, y: cy });
                        cx = parseFloat(p.value);
                        hasX = true;
                    }
                    if (p.code === 20) {
                        cy = parseFloat(p.value);
                    }
                }
                if (hasX) points.push({ x: cx, y: cy });
                if (closed && points.length > 2) {
                    polygons.push(points);
                } else if (points.length >= 2) {
                    polygons.push(points);
                }
            } else if (pair.code === 0 && pair.value === 'CIRCLE') {
                // Approximate circle as polygon
                let cx = 0, cy = 0, r = 1;
                while (i < lines.length) {
                    const p = readPair();
                    if (!p) break;
                    if (p.code === 0) { i -= 2; break; }
                    if (p.code === 10) cx = parseFloat(p.value);
                    if (p.code === 20) cy = parseFloat(p.value);
                    if (p.code === 40) r = parseFloat(p.value);
                }
                const segs = 32;
                const circle = [];
                for (let s = 0; s < segs; s++) {
                    const angle = (2 * Math.PI * s) / segs;
                    circle.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
                }
                polygons.push(circle);
            } else if (pair.code === 0 && (pair.value === 'POLYLINE' || pair.value === 'SPLINE')) {
                // Skip entities we don't fully support - consume until next entity
                while (i < lines.length) {
                    const p = readPair();
                    if (!p) break;
                    if (p.code === 0) { i -= 2; break; }
                }
            }
        }

        if (polygons.length === 0) {
            throw new Error('No supported 2D geometry found in DXF file');
        }

        const model = this.voxelize2DPolygons(polygons);
        model.sourceFormat = 'DXF';
        return model;
    }

    /**
     * Parse SVG file text and return a 2D voxelized model.
     * Supports rect, circle, ellipse, polygon, polyline, line, and path elements.
     * @param {string} text - SVG file content
     * @returns {object} Voxelized model (nz=1)
     */
    parseSVG(text) {
        const polygons = [];

        // Helper: extract all attribute values from an element match
        const getAttr = (tag, name) => {
            const match = tag.match(new RegExp(`${name}\\s*=\\s*["']([^"']*)["']`));
            return match ? match[1] : null;
        };

        // Parse <rect>
        const rects = text.matchAll(/<rect\s[^>]*\/?>/gi);
        for (const m of rects) {
            const tag = m[0];
            const x = parseFloat(getAttr(tag, 'x') || '0');
            const y = parseFloat(getAttr(tag, 'y') || '0');
            const w = parseFloat(getAttr(tag, 'width') || '0');
            const h = parseFloat(getAttr(tag, 'height') || '0');
            if (w > 0 && h > 0) {
                polygons.push([
                    { x, y }, { x: x + w, y }, { x: x + w, y: y + h }, { x, y: y + h }
                ]);
            }
        }

        // Parse <circle>
        const circles = text.matchAll(/<circle\s[^>]*\/?>/gi);
        for (const m of circles) {
            const tag = m[0];
            const cx = parseFloat(getAttr(tag, 'cx') || '0');
            const cy = parseFloat(getAttr(tag, 'cy') || '0');
            const r = parseFloat(getAttr(tag, 'r') || '0');
            if (r > 0) {
                const segs = 32;
                const circle = [];
                for (let s = 0; s < segs; s++) {
                    const angle = (2 * Math.PI * s) / segs;
                    circle.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
                }
                polygons.push(circle);
            }
        }

        // Parse <ellipse>
        const ellipses = text.matchAll(/<ellipse\s[^>]*\/?>/gi);
        for (const m of ellipses) {
            const tag = m[0];
            const cx = parseFloat(getAttr(tag, 'cx') || '0');
            const cy = parseFloat(getAttr(tag, 'cy') || '0');
            const rx = parseFloat(getAttr(tag, 'rx') || '0');
            const ry = parseFloat(getAttr(tag, 'ry') || '0');
            if (rx > 0 && ry > 0) {
                const segs = 32;
                const ellipse = [];
                for (let s = 0; s < segs; s++) {
                    const angle = (2 * Math.PI * s) / segs;
                    ellipse.push({ x: cx + rx * Math.cos(angle), y: cy + ry * Math.sin(angle) });
                }
                polygons.push(ellipse);
            }
        }

        // Parse <polygon> and <polyline>
        const polys = text.matchAll(/<(polygon|polyline)\s[^>]*\/?>/gi);
        for (const m of polys) {
            const tag = m[0];
            const pointsStr = getAttr(tag, 'points');
            if (pointsStr) {
                const nums = pointsStr.trim().split(/[\s,]+/).map(Number);
                const pts = [];
                for (let j = 0; j + 1 < nums.length; j += 2) {
                    pts.push({ x: nums[j], y: nums[j + 1] });
                }
                if (pts.length >= 2) polygons.push(pts);
            }
        }

        // Parse <line>
        const svgLines = text.matchAll(/<line\s[^>]*\/?>/gi);
        for (const m of svgLines) {
            const tag = m[0];
            const x1 = parseFloat(getAttr(tag, 'x1') || '0');
            const y1 = parseFloat(getAttr(tag, 'y1') || '0');
            const x2 = parseFloat(getAttr(tag, 'x2') || '0');
            const y2 = parseFloat(getAttr(tag, 'y2') || '0');
            polygons.push([{ x: x1, y: y1 }, { x: x2, y: y2 }]);
        }

        // Parse <path> – support M, L, H, V, Z commands (lines only, no curves)
        const paths = text.matchAll(/<path\s[^>]*\/?>/gi);
        for (const m of paths) {
            const tag = m[0];
            const d = getAttr(tag, 'd');
            if (d) {
                const pts = this._parseSVGPathData(d);
                if (pts.length >= 2) polygons.push(pts);
            }
        }

        if (polygons.length === 0) {
            throw new Error('No supported 2D geometry found in SVG file');
        }

        const model = this.voxelize2DPolygons(polygons);
        model.sourceFormat = 'SVG';
        return model;
    }

    /**
     * Parse SVG path data string (d attribute) into polygon points.
     * Supports M, L, H, V, Z commands (absolute and relative).
     * Curves (C, S, Q, T, A) are approximated by their endpoints.
     * @param {string} d - SVG path data
     * @returns {Array<{x: number, y: number}>}
     */
    _parseSVGPathData(d) {
        const points = [];
        // Tokenize: split into commands and numbers
        const tokens = d.match(/[a-zA-Z]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g);
        if (!tokens) return points;

        let cx = 0, cy = 0;
        let startX = 0, startY = 0;
        let cmd = '';
        let ti = 0;

        const nextNum = () => {
            while (ti < tokens.length && /^[a-zA-Z]$/.test(tokens[ti])) ti++;
            if (ti < tokens.length) return parseFloat(tokens[ti++]);
            return 0;
        };

        while (ti < tokens.length) {
            const token = tokens[ti];
            if (/^[a-zA-Z]$/.test(token)) {
                cmd = token;
                ti++;
            }

            switch (cmd) {
                case 'M': cx = nextNum(); cy = nextNum(); startX = cx; startY = cy; points.push({ x: cx, y: cy }); cmd = 'L'; break;
                case 'm': cx += nextNum(); cy += nextNum(); startX = cx; startY = cy; points.push({ x: cx, y: cy }); cmd = 'l'; break;
                case 'L': cx = nextNum(); cy = nextNum(); points.push({ x: cx, y: cy }); break;
                case 'l': cx += nextNum(); cy += nextNum(); points.push({ x: cx, y: cy }); break;
                case 'H': cx = nextNum(); points.push({ x: cx, y: cy }); break;
                case 'h': cx += nextNum(); points.push({ x: cx, y: cy }); break;
                case 'V': cy = nextNum(); points.push({ x: cx, y: cy }); break;
                case 'v': cy += nextNum(); points.push({ x: cx, y: cy }); break;
                case 'Z': case 'z': cx = startX; cy = startY; ti++; break;
                case 'C': { // Cubic bezier – take endpoint
                    nextNum(); nextNum(); nextNum(); nextNum();
                    cx = nextNum(); cy = nextNum();
                    points.push({ x: cx, y: cy }); break;
                }
                case 'c': {
                    nextNum(); nextNum(); nextNum(); nextNum();
                    cx += nextNum(); cy += nextNum();
                    points.push({ x: cx, y: cy }); break;
                }
                case 'S': case 's': {
                    const rel = cmd === 's';
                    nextNum(); nextNum();
                    if (rel) { cx += nextNum(); cy += nextNum(); }
                    else { cx = nextNum(); cy = nextNum(); }
                    points.push({ x: cx, y: cy }); break;
                }
                case 'Q': case 'q': {
                    const rel = cmd === 'q';
                    nextNum(); nextNum();
                    if (rel) { cx += nextNum(); cy += nextNum(); }
                    else { cx = nextNum(); cy = nextNum(); }
                    points.push({ x: cx, y: cy }); break;
                }
                case 'T': cx = nextNum(); cy = nextNum(); points.push({ x: cx, y: cy }); break;
                case 't': cx += nextNum(); cy += nextNum(); points.push({ x: cx, y: cy }); break;
                case 'A': case 'a': {
                    const rel = cmd === 'a';
                    nextNum(); nextNum(); nextNum(); nextNum(); nextNum();
                    if (rel) { cx += nextNum(); cy += nextNum(); }
                    else { cx = nextNum(); cy = nextNum(); }
                    points.push({ x: cx, y: cy }); break;
                }
                default: ti++; break; // Unknown command, skip
            }
        }
        return points;
    }

    /**
     * Voxelize 2D polygon outlines into a flat grid (nz=1).
     * Uses point-in-polygon (ray casting) to determine filled voxels.
     * @param {Array<Array<{x: number, y: number}>>} polygons - Array of polygon point arrays
     * @param {number|null} voxelSizeMM - Voxel size in mm
     * @returns {object} Voxelized model with nz=1
     */
    voxelize2DPolygons(polygons, voxelSizeMM = null) {
        // Compute bounding box across all polygons
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const poly of polygons) {
            for (const p of poly) {
                if (p.x < minX) minX = p.x;
                if (p.y < minY) minY = p.y;
                if (p.x > maxX) maxX = p.x;
                if (p.y > maxY) maxY = p.y;
            }
        }

        const sizeX = maxX - minX || 1;
        const sizeY = maxY - minY || 1;
        const maxDim = Math.max(sizeX, sizeY);

        let voxelSize;
        if (voxelSizeMM !== null && voxelSizeMM > 0) {
            voxelSize = voxelSizeMM;
        } else {
            const res = this.resolution || 20;
            voxelSize = maxDim / res;
        }

        const nx = Math.max(1, Math.ceil(sizeX / voxelSize));
        const ny = Math.max(1, Math.ceil(sizeY / voxelSize));
        const nz = 1;

        const elements = new Float32Array(nx * ny);

        // Find closed polygons (3+ points) for fill testing
        const closedPolygons = polygons.filter(p => p.length >= 3);

        if (closedPolygons.length > 0) {
            // Fill voxels using point-in-polygon test (ray casting)
            for (let iy = 0; iy < ny; iy++) {
                const cy = minY + (iy + 0.5) * voxelSize;
                for (let ix = 0; ix < nx; ix++) {
                    const cx = minX + (ix + 0.5) * voxelSize;

                    // Test point against each polygon
                    for (const poly of closedPolygons) {
                        if (this._pointInPolygon(cx, cy, poly)) {
                            elements[ix + iy * nx] = 1;
                            break;
                        }
                    }
                }
            }
        } else {
            // No closed polygons – fill entire bounding box
            elements.fill(1);
        }

        // Build triangle vertices for the 2D shape (extruded to a thin slab for rendering)
        const vertices = this._polygonsToTriangles(polygons, 0, voxelSize);

        return {
            nx,
            ny,
            nz,
            elements,
            voxelSize,
            meshType: 'box',
            bounds: { minX, minY, minZ: 0, maxX, maxY, maxZ: voxelSize },
            originalVertices: vertices,
            is2D: true
        };
    }

    /**
     * Point-in-polygon test using ray casting (Jordan curve theorem).
     * @param {number} px - Test point X
     * @param {number} py - Test point Y
     * @param {Array<{x: number, y: number}>} polygon - Polygon vertices
     * @returns {boolean}
     */
    _pointInPolygon(px, py, polygon) {
        let inside = false;
        const n = polygon.length;
        for (let i = 0, j = n - 1; i < n; j = i++) {
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            if (((yi > py) !== (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }

    /**
     * Convert 2D polygons to triangle vertices for 3D rendering.
     * Creates a thin slab from z=0 to z=thickness.
     * @param {Array<Array<{x: number, y: number}>>} polygons
     * @param {number} zBase - Base Z coordinate
     * @param {number} thickness - Slab thickness
     * @returns {Array<{x: number, y: number, z: number}>}
     */
    _polygonsToTriangles(polygons, zBase, thickness) {
        const verts = [];
        const z0 = zBase;
        const z1 = zBase + thickness;

        for (const poly of polygons) {
            if (poly.length < 2) continue;

            // Create side walls between consecutive points
            for (let i = 0; i < poly.length; i++) {
                const p0 = poly[i];
                const p1 = poly[(i + 1) % poly.length];
                // Two triangles for each edge forming a quad wall
                verts.push(
                    { x: p0.x, y: p0.y, z: z0 },
                    { x: p1.x, y: p1.y, z: z0 },
                    { x: p1.x, y: p1.y, z: z1 },
                    { x: p0.x, y: p0.y, z: z0 },
                    { x: p1.x, y: p1.y, z: z1 },
                    { x: p0.x, y: p0.y, z: z1 }
                );
            }

            // Simple fan triangulation for top and bottom faces (closed polygons only)
            if (poly.length >= 3) {
                for (let i = 1; i + 1 < poly.length; i++) {
                    // Bottom face
                    verts.push(
                        { x: poly[0].x, y: poly[0].y, z: z0 },
                        { x: poly[i].x, y: poly[i].y, z: z0 },
                        { x: poly[i + 1].x, y: poly[i + 1].y, z: z0 }
                    );
                    // Top face
                    verts.push(
                        { x: poly[0].x, y: poly[0].y, z: z1 },
                        { x: poly[i + 1].x, y: poly[i + 1].y, z: z1 },
                        { x: poly[i].x, y: poly[i].y, z: z1 }
                    );
                }
            }
        }
        return verts;
    }

    voxelizeVertices(vertices, resolution = null, voxelSizeMM = null) {
        // Find bounding box
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        
        for (let i = 0, len = vertices.length; i < len; i++) {
            const v = vertices[i];
            if (v.x < minX) minX = v.x;
            if (v.y < minY) minY = v.y;
            if (v.z < minZ) minZ = v.z;
            if (v.x > maxX) maxX = v.x;
            if (v.y > maxY) maxY = v.y;
            if (v.z > maxZ) maxZ = v.z;
        }
        
        // Physical dimensions in mm (STL coordinates are treated as mm)
        const sizeX = maxX - minX || 1;
        const sizeY = maxY - minY || 1;
        const sizeZ = maxZ - minZ || 1;

        let voxelSize;
        const maxDim = Math.max(sizeX, sizeY, sizeZ);

        if (voxelSizeMM !== null && voxelSizeMM > 0) {
            // New mm-based voxel size: directly use the specified size
            voxelSize = voxelSizeMM;
        } else {
            // Legacy resolution-based: number of voxels on longest axis
            const res = resolution !== null ? resolution : (this.resolution || 20);
            voxelSize = maxDim / res;
        }

        const nx = Math.max(1, Math.ceil(sizeX / voxelSize));
        const ny = Math.max(1, Math.ceil(sizeY / voxelSize));
        const nz = Math.max(1, Math.ceil(sizeZ / voxelSize));
        
        // Build triangle list from vertices (every 3 consecutive vertices form a triangle)
        const numTriangles = Math.floor(vertices.length / 3);
        
        // If we have valid triangles, use ray-casting to determine interior voxels;
        // otherwise fall back to filling the entire bounding box.
        const elements = new Float32Array(nx * ny * nz);
        
        if (numTriangles > 0) {
            // Build a spatial grid index: bucket triangles by their X-Y bounding box
            // so each (ix, iy) column only tests triangles that overlap with it.
            const grid = new Array(nx * ny);
            for (let i = 0; i < nx * ny; i++) grid[i] = [];

            for (let t = 0; t < numTriangles; t++) {
                const v0 = vertices[t * 3];
                const v1 = vertices[t * 3 + 1];
                const v2 = vertices[t * 3 + 2];

                // Triangle X-Y bounding box mapped to grid cells
                const tMinX = Math.min(v0.x, v1.x, v2.x);
                const tMaxX = Math.max(v0.x, v1.x, v2.x);
                const tMinY = Math.min(v0.y, v1.y, v2.y);
                const tMaxY = Math.max(v0.y, v1.y, v2.y);

                const ixStart = Math.max(0, Math.floor((tMinX - minX) / voxelSize));
                const ixEnd   = Math.min(nx - 1, Math.floor((tMaxX - minX) / voxelSize));
                const iyStart = Math.max(0, Math.floor((tMinY - minY) / voxelSize));
                const iyEnd   = Math.min(ny - 1, Math.floor((tMaxY - minY) / voxelSize));

                for (let ix = ixStart; ix <= ixEnd; ix++) {
                    for (let iy = iyStart; iy <= iyEnd; iy++) {
                        grid[ix + iy * nx].push(t);
                    }
                }
            }

            // Ray-casting voxelization: for each (ix, iy) column cast a ray along Z,
            // find intersections with overlapping triangles, then fill voxels between
            // intersection pairs (inside the mesh via the Jordan curve theorem).
            for (let ix = 0; ix < nx; ix++) {
                const cx = minX + (ix + 0.5) * voxelSize;
                for (let iy = 0; iy < ny; iy++) {
                    const cy = minY + (iy + 0.5) * voxelSize;
                    const bucket = grid[ix + iy * nx];
                    if (bucket.length === 0) continue;
                    
                    // Collect Z-axis intersections with overlapping triangles
                    const intersections = [];
                    for (let b = 0, bLen = bucket.length; b < bLen; b++) {
                        const t = bucket[b];
                        const v0 = vertices[t * 3];
                        const v1 = vertices[t * 3 + 1];
                        const v2 = vertices[t * 3 + 2];
                        const zHit = this._rayTriangleIntersectZ(cx, cy, v0, v1, v2);
                        if (zHit !== null) {
                            intersections.push(zHit);
                        }
                    }
                    
                    if (intersections.length < 2) continue;
                    
                    intersections.sort((a, b) => a - b);
                    
                    // Deduplicate intersections caused by rays hitting shared edges/vertices
                    const DEDUP_FACTOR = 1e-6;
                    const eps = voxelSize * DEDUP_FACTOR;
                    const unique = [intersections[0]];
                    for (let i = 1; i < intersections.length; i++) {
                        if (intersections[i] - unique[unique.length - 1] > eps) {
                            unique.push(intersections[i]);
                        }
                    }
                    // Need an even number of crossings; drop the last unpaired one
                    const crossings = unique.length % 2 === 0 ? unique : unique.slice(0, -1);
                    if (crossings.length < 2) continue;
                    
                    // Fill voxels between pairs of intersections (inside the surface)
                    for (let p = 0; p + 1 < crossings.length; p += 2) {
                        const zStart = crossings[p];
                        const zEnd = crossings[p + 1];
                        
                        const izStart = Math.max(0, Math.floor((zStart - minZ) / voxelSize));
                        const izEnd = Math.min(nz - 1, Math.floor((zEnd - minZ) / voxelSize));
                        
                        for (let iz = izStart; iz <= izEnd; iz++) {
                            const voxelCenterZ = minZ + (iz + 0.5) * voxelSize;
                            if (voxelCenterZ >= zStart && voxelCenterZ <= zEnd) {
                                elements[ix + iy * nx + iz * nx * ny] = 1;
                            }
                        }
                    }
                }
            }
        } else {
            elements.fill(1);
        }
        
        return {
            nx,
            ny,
            nz,
            elements,
            voxelSize,
            meshType: 'box',
            bounds: { minX, minY, minZ, maxX, maxY, maxZ },
            originalVertices: vertices  // Store vertices for re-voxelization
        };
    }

    /**
     * Möller–Trumbore ray-triangle intersection for a Z-axis ray at (rx, ry).
     * Returns the Z coordinate of the intersection, or null if the ray misses.
     */
    _rayTriangleIntersectZ(rx, ry, v0, v1, v2) {
        // Ray: origin = (rx, ry, 0), direction = (0, 0, 1)
        const edge1x = v1.x - v0.x;
        const edge1y = v1.y - v0.y;
        const edge1z = v1.z - v0.z;
        const edge2x = v2.x - v0.x;
        const edge2y = v2.y - v0.y;
        const edge2z = v2.z - v0.z;

        // h = cross(direction, edge2) = cross((0,0,1), edge2)
        const hx = -edge2y;
        const hy = edge2x;
        // hz = 0

        // a = dot(edge1, h)
        const a = edge1x * hx + edge1y * hy;

        const RAY_PARALLEL_EPS = 1e-10;
        if (a > -RAY_PARALLEL_EPS && a < RAY_PARALLEL_EPS) return null; // Ray parallel to triangle

        const f = 1.0 / a;
        const sx = rx - v0.x;
        const sy = ry - v0.y;

        // u = f * dot(s, h)
        const u = f * (sx * hx + sy * hy);
        if (u < 0.0 || u > 1.0) return null;

        // q = cross(s, edge1);  s = (sx, sy, -v0.z) but sz not needed for u
        // For the cross product we need the full s vector
        const sz = -v0.z; // ray origin z is 0
        const qx = sy * edge1z - sz * edge1y;
        const qy = sz * edge1x - sx * edge1z;
        const qz = sx * edge1y - sy * edge1x;

        // v = f * dot(direction, q) = f * qz  (direction = (0,0,1))
        const v = f * qz;
        if (v < 0.0 || u + v > 1.0) return null;

        // t = f * dot(edge2, q)
        const t = f * (edge2x * qx + edge2y * qy + edge2z * qz);

        return t; // Z coordinate of intersection (ray origin z=0, direction z=1, so hit z = t)
    }

    /**
     * Blended curvature meshing for STEP files.
     * Instead of binary voxelization, this method creates a uniform hex grid
     * (nx × ny × nz) where element densities along curved boundaries are
     * blended (0–1) based on surface proximity and local curvature.  Interior
     * voxels keep density 1.0, exterior voxels 0.0, and boundary voxels
     * receive a smoothly varying density that encodes curvature information.
     * Because the grid structure is identical to voxelizeVertices, existing
     * solvers work unmodified while benefiting from smoother boundary
     * representation.
     *
     * @param {Array} vertices - Triangle vertices from STEP parser
     * @param {number|null} resolution - Grid resolution (voxels along longest axis)
     * @param {number|null} voxelSizeMM - Voxel size in mm (overrides resolution)
     * @returns {object} Model with meshType='blended-curvature'
     */
    blendedCurvatureMesh(vertices, resolution = null, voxelSizeMM = null) {
        // Find bounding box
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0, len = vertices.length; i < len; i++) {
            const v = vertices[i];
            if (v.x < minX) minX = v.x;
            if (v.y < minY) minY = v.y;
            if (v.z < minZ) minZ = v.z;
            if (v.x > maxX) maxX = v.x;
            if (v.y > maxY) maxY = v.y;
            if (v.z > maxZ) maxZ = v.z;
        }

        const sizeX = maxX - minX || 1;
        const sizeY = maxY - minY || 1;
        const sizeZ = maxZ - minZ || 1;

        let voxelSize;
        const maxDim = Math.max(sizeX, sizeY, sizeZ);

        if (voxelSizeMM !== null && voxelSizeMM > 0) {
            voxelSize = voxelSizeMM;
        } else {
            const res = resolution !== null ? resolution : (this.resolution || 20);
            voxelSize = maxDim / res;
        }

        const nx = Math.max(1, Math.ceil(sizeX / voxelSize));
        const ny = Math.max(1, Math.ceil(sizeY / voxelSize));
        const nz = Math.max(1, Math.ceil(sizeZ / voxelSize));

        const numTriangles = Math.floor(vertices.length / 3);
        const elements = new Float32Array(nx * ny * nz);

        if (numTriangles === 0) {
            elements.fill(1);
            return {
                nx, ny, nz, elements, voxelSize,
                meshType: 'blended-curvature',
                bounds: { minX, minY, minZ, maxX, maxY, maxZ },
                originalVertices: vertices
            };
        }

        // Compute per-triangle curvature estimate from face normal variation
        const triNormals = new Array(numTriangles);
        for (let t = 0; t < numTriangles; t++) {
            const v0 = vertices[t * 3];
            const v1 = vertices[t * 3 + 1];
            const v2 = vertices[t * 3 + 2];
            const e1x = v1.x - v0.x, e1y = v1.y - v0.y, e1z = v1.z - v0.z;
            const e2x = v2.x - v0.x, e2y = v2.y - v0.y, e2z = v2.z - v0.z;
            let nx_ = e1y * e2z - e1z * e2y;
            let ny_ = e1z * e2x - e1x * e2z;
            let nz_ = e1x * e2y - e1y * e2x;
            const len = Math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) || 1;
            triNormals[t] = { x: nx_ / len, y: ny_ / len, z: nz_ / len };
        }

        // Build spatial grid index for triangles (same as voxelizeVertices)
        const grid = new Array(nx * ny);
        for (let i = 0; i < nx * ny; i++) grid[i] = [];

        for (let t = 0; t < numTriangles; t++) {
            const v0 = vertices[t * 3];
            const v1 = vertices[t * 3 + 1];
            const v2 = vertices[t * 3 + 2];
            const tMinX = Math.min(v0.x, v1.x, v2.x);
            const tMaxX = Math.max(v0.x, v1.x, v2.x);
            const tMinY = Math.min(v0.y, v1.y, v2.y);
            const tMaxY = Math.max(v0.y, v1.y, v2.y);
            const ixStart = Math.max(0, Math.floor((tMinX - minX) / voxelSize));
            const ixEnd = Math.min(nx - 1, Math.floor((tMaxX - minX) / voxelSize));
            const iyStart = Math.max(0, Math.floor((tMinY - minY) / voxelSize));
            const iyEnd = Math.min(ny - 1, Math.floor((tMaxY - minY) / voxelSize));

            for (let ix = ixStart; ix <= ixEnd; ix++) {
                for (let iy = iyStart; iy <= iyEnd; iy++) {
                    grid[ix + iy * nx].push(t);
                }
            }
        }

        // Phase 1: Binary inside/outside via ray-casting (same as voxelizeVertices)
        const insideFlag = new Uint8Array(nx * ny * nz);

        for (let ix = 0; ix < nx; ix++) {
            const cx = minX + (ix + 0.5) * voxelSize;
            for (let iy = 0; iy < ny; iy++) {
                const cy = minY + (iy + 0.5) * voxelSize;
                const bucket = grid[ix + iy * nx];
                if (bucket.length === 0) continue;

                const intersections = [];
                for (let b = 0; b < bucket.length; b++) {
                    const t = bucket[b];
                    const v0 = vertices[t * 3];
                    const v1 = vertices[t * 3 + 1];
                    const v2 = vertices[t * 3 + 2];
                    const zHit = this._rayTriangleIntersectZ(cx, cy, v0, v1, v2);
                    if (zHit !== null) intersections.push(zHit);
                }
                if (intersections.length < 2) continue;
                intersections.sort((a, b) => a - b);

                const eps = voxelSize * 1e-6;
                const unique = [intersections[0]];
                for (let i = 1; i < intersections.length; i++) {
                    if (intersections[i] - unique[unique.length - 1] > eps) {
                        unique.push(intersections[i]);
                    }
                }
                const crossings = unique.length % 2 === 0 ? unique : unique.slice(0, -1);
                if (crossings.length < 2) continue;

                for (let p = 0; p + 1 < crossings.length; p += 2) {
                    const zStart = crossings[p];
                    const zEnd = crossings[p + 1];
                    const izStart = Math.max(0, Math.floor((zStart - minZ) / voxelSize));
                    const izEnd = Math.min(nz - 1, Math.floor((zEnd - minZ) / voxelSize));

                    for (let iz = izStart; iz <= izEnd; iz++) {
                        const voxelCenterZ = minZ + (iz + 0.5) * voxelSize;
                        if (voxelCenterZ >= zStart && voxelCenterZ <= zEnd) {
                            insideFlag[ix + iy * nx + iz * nx * ny] = 1;
                        }
                    }
                }
            }
        }

        // Phase 2: Compute curvature-blended densities for boundary voxels
        // Interior voxels get density 1.0; boundary voxels (adjacent to an
        // outside voxel) get a blended value based on local surface curvature.
        const blendRadius = voxelSize * 1.5;
        const blendRadiusSq = blendRadius * blendRadius;

        for (let iz = 0; iz < nz; iz++) {
            for (let iy = 0; iy < ny; iy++) {
                for (let ix = 0; ix < nx; ix++) {
                    const idx = ix + iy * nx + iz * nx * ny;
                    if (!insideFlag[idx]) {
                        elements[idx] = 0;
                        continue;
                    }

                    // Check if this is a boundary voxel (any neighbour is outside)
                    let isBoundary = false;
                    for (let dz = -1; dz <= 1 && !isBoundary; dz++) {
                        for (let dy = -1; dy <= 1 && !isBoundary; dy++) {
                            for (let dx = -1; dx <= 1 && !isBoundary; dx++) {
                                if (dx === 0 && dy === 0 && dz === 0) continue;
                                const bx = ix + dx, by = iy + dy, bz = iz + dz;
                                if (bx < 0 || bx >= nx || by < 0 || by >= ny || bz < 0 || bz >= nz) {
                                    isBoundary = true;
                                } else if (!insideFlag[bx + by * nx + bz * nx * ny]) {
                                    isBoundary = true;
                                }
                            }
                        }
                    }

                    if (!isBoundary) {
                        elements[idx] = 1.0;
                        continue;
                    }

                    // Boundary voxel: blend density based on curvature
                    const cx = minX + (ix + 0.5) * voxelSize;
                    const cy = minY + (iy + 0.5) * voxelSize;
                    const cz = minZ + (iz + 0.5) * voxelSize;

                    // Find nearby triangles and compute average curvature
                    const bucket = grid[ix + iy * nx] || [];
                    let curvatureSum = 0;
                    let curvatureCount = 0;

                    for (let b = 0; b < bucket.length; b++) {
                        const t = bucket[b];
                        const v0 = vertices[t * 3];
                        const v1 = vertices[t * 3 + 1];
                        const v2 = vertices[t * 3 + 2];

                        // Triangle centroid
                        const tcx = (v0.x + v1.x + v2.x) / 3;
                        const tcy = (v0.y + v1.y + v2.y) / 3;
                        const tcz = (v0.z + v1.z + v2.z) / 3;

                        const distSq = (cx - tcx) ** 2 + (cy - tcy) ** 2 + (cz - tcz) ** 2;
                        if (distSq > blendRadiusSq) continue;

                        // Estimate curvature: compare this triangle's normal with
                        // normals of nearby triangles in the same bucket
                        const n1 = triNormals[t];
                        let maxAngle = 0;
                        for (let b2 = 0; b2 < bucket.length; b2++) {
                            if (b2 === b) continue;
                            const n2 = triNormals[bucket[b2]];
                            const dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
                            const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
                            if (angle > maxAngle) maxAngle = angle;
                        }

                        curvatureSum += maxAngle;
                        curvatureCount++;
                    }

                    if (curvatureCount > 0) {
                        // Blend: high curvature → density closer to surface fraction
                        // low curvature → density closer to 1.0 (nearly flat boundary)
                        const avgCurvature = curvatureSum / curvatureCount;
                        // Normalize: curvature 0 → blend=1.0, curvature π → blend=0.5
                        const blend = 1.0 - 0.5 * Math.min(1, avgCurvature / Math.PI);
                        elements[idx] = Math.max(0.1, blend);
                    } else {
                        elements[idx] = 0.8; // boundary without nearby triangles
                    }
                }
            }
        }

        return {
            nx, ny, nz, elements, voxelSize,
            meshType: 'blended-curvature',
            bounds: { minX, minY, minZ, maxX, maxY, maxZ },
            originalVertices: vertices
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

    /**
     * Return the longest base dimension for a template type.
     * Used externally to convert voxel size in mm to a resolution value.
     */
    static getTemplateMaxDim(type) {
        const dims = { beam: 30, bridge: 40, cube: 50 };
        return dims[type] || 20;
    }

    /**
     * Generate 12 triangles (36 vertices) for a solid box from (0,0,0) to (lx,ly,lz).
     * Vertices are returned as an array of {x,y,z} objects matching the format used
     * by STL originalVertices so the reference mesh overlay works correctly.
     */
    _generateBoxTriangles(lx, ly, lz) {
        const v = (x, y, z) => ({ x, y, z });
        const faces = [
            [v(0, 0, 0), v(lx, 0, 0), v(lx, 0, lz), v(0, 0, lz)],    // -Y (y=0 plane)
            [v(0, ly, 0), v(lx, ly, 0), v(lx, ly, lz), v(0, ly, lz)],  // +Y (y=ly plane)
            [v(0, 0, 0), v(lx, 0, 0), v(lx, ly, 0), v(0, ly, 0)],      // -Z (z=0 plane)
            [v(0, 0, lz), v(lx, 0, lz), v(lx, ly, lz), v(0, ly, lz)],  // +Z (z=lz plane)
            [v(0, 0, 0), v(0, 0, lz), v(0, ly, lz), v(0, ly, 0)],      // -X (x=0 plane)
            [v(lx, 0, 0), v(lx, ly, 0), v(lx, ly, lz), v(lx, 0, lz)],  // +X (x=lx plane)
        ];
        const verts = [];
        for (const [a, b, c, d] of faces) {
            verts.push(a, b, c, a, c, d);
        }
        return verts;
    }

    createBeamTemplate(resolution = 30) {
        // Cantilever beam: 30×10×10 mm, resolution = voxels along max axis (30mm)
        const baseNx = 30, baseNy = 10, baseNz = 10;
        const scale = resolution / baseNx;
        const nx = Math.max(5, Math.round(baseNx * scale));
        const ny = Math.max(3, Math.round(baseNy * scale));
        const nz = Math.max(3, Math.round(baseNz * scale));
        const elements = new Float32Array(nx * ny * nz).fill(1);
        const voxelSize = baseNx / nx;
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'beam',
            templateScale: { baseNx, baseNy, baseNz },
            voxelSize,
            bounds: { minX: 0, maxX: baseNx, minY: 0, maxY: baseNy, minZ: 0, maxZ: baseNz },
            originalVertices: this._generateBoxTriangles(baseNx, baseNy, baseNz),
            // Predefined boundary conditions for cantilever beam
            forcePosition: 'right',           // Force at right end
            forceDirection: 'down',            // Force pointing down
            constraintPositions: 'left'        // Fixed left end
        };
    }

    createBridgeTemplate(resolution = 40) {
        // Bridge: 40×15×8 mm, resolution = voxels along max axis (40mm)
        const baseNx = 40, baseNy = 15, baseNz = 8;
        const scale = resolution / baseNx;
        const nx = Math.max(5, Math.round(baseNx * scale));
        const ny = Math.max(3, Math.round(baseNy * scale));
        const nz = Math.max(3, Math.round(baseNz * scale));
        const elements = new Float32Array(nx * ny * nz).fill(1);
        const voxelSize = baseNx / nx;
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'bridge',
            templateScale: { baseNx, baseNy, baseNz },
            voxelSize,
            bounds: { minX: 0, maxX: baseNx, minY: 0, maxY: baseNy, minZ: 0, maxZ: baseNz },
            originalVertices: this._generateBoxTriangles(baseNx, baseNy, baseNz),
            // Predefined boundary conditions for bridge
            forcePosition: 'top-center',       // Force at top center
            forceDirection: 'down',             // Force pointing down
            constraintPositions: 'both-ends'    // Both ends fixed
        };
    }

    createCubeTemplate(resolution = 50) {
        // Cube: 50×50×50 mm, resolution = voxels along max axis (50mm)
        const baseSize = 50;
        const scale = resolution / baseSize;
        const scaledSize = Math.max(3, Math.round(baseSize * scale));
        const nx = scaledSize;
        const ny = scaledSize;
        const nz = scaledSize;
        const elements = new Float32Array(nx * ny * nz).fill(1);
        const voxelSize = baseSize / nx;
        
        return {
            nx,
            ny,
            nz,
            elements,
            type: 'cube',
            templateScale: { baseNx: baseSize, baseNy: baseSize, baseNz: baseSize },
            voxelSize,
            bounds: { minX: 0, maxX: baseSize, minY: 0, maxY: baseSize, minZ: 0, maxZ: baseSize },
            originalVertices: this._generateBoxTriangles(baseSize, baseSize, baseSize),
            // Predefined boundary conditions for cube test
            forcePosition: 'top-center',  // Force at top center
            forceDirection: 'down',       // Force pointing down
            constraintPositions: 'bottom-corners'  // Constraints at bottom 4 corners
        };
    }
}
