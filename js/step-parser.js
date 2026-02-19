// STEP file parser for AP203 and AP214 protocols (ISO 10303-21)
// Extracts B-Rep geometry and tessellates surfaces into triangle meshes

export class STEPParser {
    constructor() {
        this.entities = {};    // Map of entity ID → parsed entity
        this.rawEntities = {}; // Map of entity ID → raw text
    }

    /**
     * Parse a STEP file string and return an array of {x, y, z} vertices
     * (every 3 consecutive vertices form a triangle).
     * @param {string} text - The full STEP file content
     * @returns {{ vertices: Array<{x:number,y:number,z:number}>, protocol: string }}
     */
    parse(text) {
        this.entities = {};
        this.rawEntities = {};

        const protocol = this._detectProtocol(text);

        this._parseDataSection(text);

        const vertices = this._extractTriangles();

        return { vertices, protocol };
    }

    // ─── Protocol detection ────────────────────────────────────────

    _detectProtocol(text) {
        const upper = text.toUpperCase();
        if (upper.includes('AP214')) return 'AP214';
        if (upper.includes('AP203')) return 'AP203';
        // Check for common full schema identifiers
        if (upper.includes('AUTOMOTIVE_DESIGN')) return 'AP214';
        if (upper.includes('CONFIG_CONTROL_DESIGN')) return 'AP203';
        return 'UNKNOWN';
    }

    // ─── DATA section parsing ──────────────────────────────────────

    _parseDataSection(text) {
        // Extract the DATA section between DATA; and ENDSEC;
        const dataMatch = text.match(/DATA\s*;([\s\S]*?)ENDSEC\s*;/i);
        if (!dataMatch) {
            throw new Error('No DATA section found in STEP file');
        }
        const dataBlock = dataMatch[1];

        // Parse entity lines: #id = TYPE_NAME(...);
        // Entities can span multiple lines, so we join and split on semicolons
        const normalized = dataBlock.replace(/\r\n/g, '\n');

        // Split on semicolons to get individual entity statements
        const statements = normalized.split(';');

        for (const stmt of statements) {
            const trimmed = stmt.trim();
            if (!trimmed) continue;

            // Match: #id = TYPE_NAME(...)
            const match = trimmed.match(/^#(\d+)\s*=\s*(\w+)\s*\(([\s\S]*)\)\s*$/);
            if (match) {
                const id = parseInt(match[1], 10);
                const type = match[2].toUpperCase();
                const args = match[3];
                this.rawEntities[id] = { type, args };
            }
        }
    }

    /**
     * Resolve an entity by ID (with caching).
     */
    _resolve(id) {
        if (typeof id !== 'number') return id;
        if (this.entities[id]) return this.entities[id];

        const raw = this.rawEntities[id];
        if (!raw) return null;

        const parsed = this._parseEntity(id, raw.type, raw.args);
        this.entities[id] = parsed;
        return parsed;
    }

    /**
     * Parse a single entity based on its type.
     */
    _parseEntity(id, type, argsStr) {
        const args = this._splitArgs(argsStr);

        switch (type) {
            case 'CARTESIAN_POINT':
                return this._parseCartesianPoint(id, args);
            case 'DIRECTION':
                return this._parseDirection(id, args);
            case 'VECTOR':
                return this._parseVector(id, args);
            case 'AXIS2_PLACEMENT_3D':
                return this._parseAxis2Placement3D(id, args);
            case 'AXIS1_PLACEMENT':
                return this._parseAxis1Placement(id, args);
            case 'PLANE':
                return this._parsePlane(id, args);
            case 'CYLINDRICAL_SURFACE':
                return this._parseCylindricalSurface(id, args);
            case 'CONICAL_SURFACE':
                return this._parseConicalSurface(id, args);
            case 'SPHERICAL_SURFACE':
                return this._parseSphericalSurface(id, args);
            case 'TOROIDAL_SURFACE':
                return this._parseToroidalSurface(id, args);
            case 'B_SPLINE_SURFACE_WITH_KNOTS':
                return this._parseBSplineSurface(id, args);
            case 'LINE':
                return this._parseLine(id, args);
            case 'CIRCLE':
                return this._parseCircle(id, args);
            case 'ELLIPSE':
                return this._parseEllipse(id, args);
            case 'VERTEX_POINT':
                return this._parseVertexPoint(id, args);
            case 'EDGE_CURVE':
                return this._parseEdgeCurve(id, args);
            case 'ORIENTED_EDGE':
                return this._parseOrientedEdge(id, args);
            case 'EDGE_LOOP':
                return this._parseEdgeLoop(id, args);
            case 'FACE_OUTER_BOUND':
            case 'FACE_BOUND':
                return this._parseFaceBound(id, type, args);
            case 'ADVANCED_FACE':
                return this._parseAdvancedFace(id, args);
            case 'CLOSED_SHELL':
            case 'OPEN_SHELL':
                return this._parseShell(id, type, args);
            case 'MANIFOLD_SOLID_BREP':
            case 'BREP_WITH_VOIDS':
                return this._parseBRep(id, type, args);
            default:
                return { id, type, args };
        }
    }

    // ─── Argument splitting ────────────────────────────────────────

    /**
     * Split a STEP argument string respecting parentheses nesting and quotes.
     */
    _splitArgs(str) {
        const result = [];
        let depth = 0;
        let current = '';
        let inString = false;

        for (let i = 0; i < str.length; i++) {
            const ch = str[i];
            if (ch === "'" && !inString) {
                inString = true;
                current += ch;
            } else if (ch === "'" && inString) {
                inString = false;
                current += ch;
            } else if (inString) {
                current += ch;
            } else if (ch === '(') {
                depth++;
                current += ch;
            } else if (ch === ')') {
                depth--;
                current += ch;
            } else if (ch === ',' && depth === 0) {
                result.push(current.trim());
                current = '';
            } else {
                current += ch;
            }
        }
        if (current.trim()) result.push(current.trim());
        return result;
    }

    /**
     * Parse an entity reference like #123 to a number.
     */
    _parseRef(str) {
        if (typeof str === 'number') return str;
        const s = str.trim();
        if (s.startsWith('#')) return parseInt(s.substring(1), 10);
        if (s === '$' || s === '*' || s === '') return null;
        return s;
    }

    /**
     * Parse a coordinate list like (1.0,2.0,3.0)
     */
    _parseCoordList(str) {
        const s = str.trim();
        // Remove outer parens
        const inner = s.startsWith('(') ? s.slice(1, -1) : s;
        return inner.split(',').map(v => parseFloat(v.trim()));
    }

    /**
     * Parse a list of references like (#1,#2,#3) or (1,2,3)
     */
    _parseRefList(str) {
        const s = str.trim();
        if (s === '$' || s === '' || s === '*') return [];
        const inner = s.startsWith('(') ? s.slice(1, -1) : s;
        if (!inner.trim()) return [];
        return inner.split(',').map(v => this._parseRef(v.trim()));
    }

    // ─── Entity parsers ────────────────────────────────────────────

    _parseCartesianPoint(id, args) {
        // CARTESIAN_POINT('label', (x, y, z))
        const coords = this._parseCoordList(args[1]);
        return {
            id, type: 'CARTESIAN_POINT',
            x: coords[0] || 0,
            y: coords[1] || 0,
            z: coords[2] || 0
        };
    }

    _parseDirection(id, args) {
        // DIRECTION('label', (dx, dy, dz))
        const coords = this._parseCoordList(args[1]);
        return {
            id, type: 'DIRECTION',
            x: coords[0] || 0,
            y: coords[1] || 0,
            z: coords[2] || 0
        };
    }

    _parseVector(id, args) {
        // VECTOR('label', #direction, magnitude)
        return {
            id, type: 'VECTOR',
            directionRef: this._parseRef(args[1]),
            magnitude: parseFloat(args[2])
        };
    }

    _parseAxis2Placement3D(id, args) {
        // AXIS2_PLACEMENT_3D('label', #origin, #z_dir, #x_dir)
        return {
            id, type: 'AXIS2_PLACEMENT_3D',
            originRef: this._parseRef(args[1]),
            zDirRef: this._parseRef(args[2]),
            xDirRef: args[3] ? this._parseRef(args[3]) : null
        };
    }

    _parseAxis1Placement(id, args) {
        return {
            id, type: 'AXIS1_PLACEMENT',
            originRef: this._parseRef(args[1]),
            dirRef: this._parseRef(args[2])
        };
    }

    _parsePlane(id, args) {
        // PLANE('label', #axis2_placement)
        return {
            id, type: 'PLANE',
            placementRef: this._parseRef(args[1])
        };
    }

    _parseCylindricalSurface(id, args) {
        return {
            id, type: 'CYLINDRICAL_SURFACE',
            placementRef: this._parseRef(args[1]),
            radius: parseFloat(args[2])
        };
    }

    _parseConicalSurface(id, args) {
        return {
            id, type: 'CONICAL_SURFACE',
            placementRef: this._parseRef(args[1]),
            radius: parseFloat(args[2]),
            semiAngle: parseFloat(args[3])
        };
    }

    _parseSphericalSurface(id, args) {
        return {
            id, type: 'SPHERICAL_SURFACE',
            placementRef: this._parseRef(args[1]),
            radius: parseFloat(args[2])
        };
    }

    _parseToroidalSurface(id, args) {
        return {
            id, type: 'TOROIDAL_SURFACE',
            placementRef: this._parseRef(args[1]),
            majorRadius: parseFloat(args[2]),
            minorRadius: parseFloat(args[3])
        };
    }

    _parseBSplineSurface(id, args) {
        return {
            id, type: 'B_SPLINE_SURFACE_WITH_KNOTS',
            raw: args
        };
    }

    _parseLine(id, args) {
        return {
            id, type: 'LINE',
            pointRef: this._parseRef(args[1]),
            vectorRef: this._parseRef(args[2])
        };
    }

    _parseCircle(id, args) {
        return {
            id, type: 'CIRCLE',
            placementRef: this._parseRef(args[1]),
            radius: parseFloat(args[2])
        };
    }

    _parseEllipse(id, args) {
        return {
            id, type: 'ELLIPSE',
            placementRef: this._parseRef(args[1]),
            semiAxis1: parseFloat(args[2]),
            semiAxis2: parseFloat(args[3])
        };
    }

    _parseVertexPoint(id, args) {
        return {
            id, type: 'VERTEX_POINT',
            pointRef: this._parseRef(args[1])
        };
    }

    _parseEdgeCurve(id, args) {
        return {
            id, type: 'EDGE_CURVE',
            startRef: this._parseRef(args[1]),
            endRef: this._parseRef(args[2]),
            curveRef: this._parseRef(args[3]),
            sameSense: args[4] ? args[4].trim() === '.T.' : true
        };
    }

    _parseOrientedEdge(id, args) {
        return {
            id, type: 'ORIENTED_EDGE',
            edgeRef: this._parseRef(args[3]),
            orientation: args[4] ? args[4].trim() === '.T.' : true
        };
    }

    _parseEdgeLoop(id, args) {
        return {
            id, type: 'EDGE_LOOP',
            edgeRefs: this._parseRefList(args[1])
        };
    }

    _parseFaceBound(id, type, args) {
        return {
            id, type,
            loopRef: this._parseRef(args[1]),
            orientation: args[2] ? args[2].trim() === '.T.' : true
        };
    }

    _parseAdvancedFace(id, args) {
        return {
            id, type: 'ADVANCED_FACE',
            boundRefs: this._parseRefList(args[1]),
            surfaceRef: this._parseRef(args[2]),
            sameSense: args[3] ? args[3].trim() === '.T.' : true
        };
    }

    _parseShell(id, type, args) {
        return {
            id, type,
            faceRefs: this._parseRefList(args[1])
        };
    }

    _parseBRep(id, type, args) {
        return {
            id, type,
            shellRef: this._parseRef(args[1])
        };
    }

    // ─── Geometry extraction ───────────────────────────────────────

    /**
     * Get the coordinate frame from an AXIS2_PLACEMENT_3D.
     * Returns { origin, zAxis, xAxis, yAxis }.
     */
    _getPlacement(placementRef) {
        const placement = this._resolve(placementRef);
        if (!placement || placement.type !== 'AXIS2_PLACEMENT_3D') {
            return { origin: { x: 0, y: 0, z: 0 }, zAxis: { x: 0, y: 0, z: 1 }, xAxis: { x: 1, y: 0, z: 0 }, yAxis: { x: 0, y: 1, z: 0 } };
        }

        const origin = this._resolve(placement.originRef) || { x: 0, y: 0, z: 0 };
        let zAxis = placement.zDirRef ? this._resolve(placement.zDirRef) : null;
        let xAxis = placement.xDirRef ? this._resolve(placement.xDirRef) : null;

        if (!zAxis) zAxis = { x: 0, y: 0, z: 1 };
        if (!xAxis) xAxis = { x: 1, y: 0, z: 0 };

        // Normalize
        zAxis = this._normalize(zAxis);
        xAxis = this._normalize(xAxis);

        // Ensure orthogonality: yAxis = zAxis × xAxis
        const yAxis = this._cross(zAxis, xAxis);

        return { origin, zAxis, xAxis, yAxis };
    }

    /**
     * Get the position of a vertex.
     */
    _getVertexPos(vertexRef) {
        const vertex = this._resolve(vertexRef);
        if (!vertex) return null;
        if (vertex.type === 'VERTEX_POINT') {
            const point = this._resolve(vertex.pointRef);
            return point ? { x: point.x, y: point.y, z: point.z } : null;
        }
        if (vertex.type === 'CARTESIAN_POINT') {
            return { x: vertex.x, y: vertex.y, z: vertex.z };
        }
        return null;
    }

    // ─── Triangle extraction ───────────────────────────────────────

    _extractTriangles() {
        const vertices = [];

        // Find all CLOSED_SHELL and OPEN_SHELL entities
        const shells = [];
        for (const id in this.rawEntities) {
            const raw = this.rawEntities[id];
            if (raw.type === 'CLOSED_SHELL' || raw.type === 'OPEN_SHELL') {
                shells.push(parseInt(id, 10));
            }
        }

        // If no shells found, look for ADVANCED_FACE directly
        if (shells.length === 0) {
            for (const id in this.rawEntities) {
                const raw = this.rawEntities[id];
                if (raw.type === 'ADVANCED_FACE') {
                    this._tessellateFace(parseInt(id, 10), vertices);
                }
            }
        } else {
            for (const shellId of shells) {
                const shell = this._resolve(shellId);
                if (shell && shell.faceRefs) {
                    for (const faceRef of shell.faceRefs) {
                        this._tessellateFace(faceRef, vertices);
                    }
                }
            }
        }

        return vertices;
    }

    /**
     * Tessellate an ADVANCED_FACE into triangles.
     */
    _tessellateFace(faceRef, vertices) {
        const face = this._resolve(faceRef);
        if (!face || face.type !== 'ADVANCED_FACE') return;

        const surface = this._resolve(face.surfaceRef);
        if (!surface) return;

        // Get the outer loop vertices
        const loopVertices = [];
        for (const boundRef of face.boundRefs) {
            const bound = this._resolve(boundRef);
            if (!bound) continue;
            const loopVerts = this._getLoopVertices(bound.loopRef, bound.orientation);
            if (bound.type === 'FACE_OUTER_BOUND') {
                loopVertices.unshift(loopVerts); // Outer bound first
            } else {
                loopVertices.push(loopVerts);
            }
        }

        if (loopVertices.length === 0) return;

        const outerLoop = loopVertices[0];
        if (!outerLoop || outerLoop.length < 3) return;

        switch (surface.type) {
            case 'PLANE':
                this._tessellatePlanarFace(outerLoop, vertices);
                break;
            case 'CYLINDRICAL_SURFACE':
                this._tessellateCylindricalFace(surface, outerLoop, vertices);
                break;
            case 'CONICAL_SURFACE':
                this._tessellateConicalFace(surface, outerLoop, vertices);
                break;
            case 'SPHERICAL_SURFACE':
                this._tessellateSphericalFace(surface, outerLoop, vertices);
                break;
            case 'TOROIDAL_SURFACE':
                this._tessellateToroidalFace(surface, outerLoop, vertices);
                break;
            default:
                // For unsupported surfaces (B-spline etc.), use planar approximation
                this._tessellatePlanarFace(outerLoop, vertices);
                break;
        }
    }

    /**
     * Get the ordered vertices of an edge loop.
     */
    _getLoopVertices(loopRef, orientation) {
        const loop = this._resolve(loopRef);
        if (!loop || loop.type !== 'EDGE_LOOP') return [];

        const vertices = [];
        for (const oeRef of loop.edgeRefs) {
            const oe = this._resolve(oeRef);
            if (!oe || oe.type !== 'ORIENTED_EDGE') continue;

            const edge = this._resolve(oe.edgeRef);
            if (!edge || edge.type !== 'EDGE_CURVE') continue;

            const curve = this._resolve(edge.curveRef);
            const startPos = this._getVertexPos(edge.startRef);
            const endPos = this._getVertexPos(edge.endRef);

            if (!startPos || !endPos) continue;

            const forward = oe.orientation === edge.sameSense;

            if (curve && curve.type === 'CIRCLE') {
                // Tessellate circular edge
                const arcPts = this._tessellateCircularEdge(curve, startPos, endPos, forward);
                for (const pt of arcPts) {
                    vertices.push(pt);
                }
            } else if (curve && curve.type === 'ELLIPSE') {
                const arcPts = this._tessellateEllipticalEdge(curve, startPos, endPos, forward);
                for (const pt of arcPts) {
                    vertices.push(pt);
                }
            } else {
                // Linear edge or unknown curve - just use start point
                vertices.push(forward ? startPos : endPos);
            }
        }

        if (!orientation) {
            vertices.reverse();
        }

        return vertices;
    }

    /**
     * Tessellate a circular arc edge into intermediate points.
     */
    _tessellateCircularEdge(circle, start, end, forward) {
        const placement = this._getPlacement(circle.placementRef);
        const { origin, xAxis, yAxis, zAxis } = placement;
        const r = circle.radius;

        // Project start and end onto the circle's local frame
        const startAngle = this._pointToAngle(start, origin, xAxis, yAxis);
        let endAngle = this._pointToAngle(end, origin, xAxis, yAxis);

        // Determine arc sweep
        let sweep = endAngle - startAngle;
        if (forward) {
            if (sweep <= 0) sweep += 2 * Math.PI;
        } else {
            if (sweep >= 0) sweep -= 2 * Math.PI;
        }

        // Full circle check
        const dist = this._distance(start, end);
        if (dist < r * 1e-6) {
            sweep = forward ? 2 * Math.PI : -2 * Math.PI;
        }

        const numSegments = Math.max(8, Math.ceil(Math.abs(sweep) / (Math.PI / 8)));
        const points = [];

        for (let i = 0; i <= numSegments; i++) {
            // Skip the last point (it's the start of the next edge)
            if (i === numSegments) continue;
            const t = i / numSegments;
            const angle = startAngle + sweep * t;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            points.push({
                x: origin.x + r * (cos * xAxis.x + sin * yAxis.x),
                y: origin.y + r * (cos * xAxis.y + sin * yAxis.y),
                z: origin.z + r * (cos * xAxis.z + sin * yAxis.z)
            });
        }

        return points;
    }

    /**
     * Tessellate an elliptical arc edge.
     */
    _tessellateEllipticalEdge(ellipse, start, end, forward) {
        const placement = this._getPlacement(ellipse.placementRef);
        const { origin, xAxis, yAxis } = placement;
        const a = ellipse.semiAxis1;
        const b = ellipse.semiAxis2;

        const startAngle = this._pointToAngle(start, origin, xAxis, yAxis);
        let endAngle = this._pointToAngle(end, origin, xAxis, yAxis);

        let sweep = endAngle - startAngle;
        if (forward) {
            if (sweep <= 0) sweep += 2 * Math.PI;
        } else {
            if (sweep >= 0) sweep -= 2 * Math.PI;
        }

        const dist = this._distance(start, end);
        if (dist < Math.max(a, b) * 1e-6) {
            sweep = forward ? 2 * Math.PI : -2 * Math.PI;
        }

        const numSegments = Math.max(8, Math.ceil(Math.abs(sweep) / (Math.PI / 8)));
        const points = [];

        for (let i = 0; i < numSegments; i++) {
            const t = i / numSegments;
            const angle = startAngle + sweep * t;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            points.push({
                x: origin.x + a * cos * xAxis.x + b * sin * yAxis.x,
                y: origin.y + a * cos * xAxis.y + b * sin * yAxis.y,
                z: origin.z + a * cos * xAxis.z + b * sin * yAxis.z
            });
        }

        return points;
    }

    // ─── Tessellation helpers ──────────────────────────────────────

    /**
     * Tessellate a planar face (fan triangulation from first vertex).
     */
    _tessellatePlanarFace(loopVertices, outVertices) {
        if (loopVertices.length < 3) return;

        // Simple ear-clipping / fan triangulation
        const v0 = loopVertices[0];
        for (let i = 1; i < loopVertices.length - 1; i++) {
            outVertices.push(
                { x: v0.x, y: v0.y, z: v0.z },
                { x: loopVertices[i].x, y: loopVertices[i].y, z: loopVertices[i].z },
                { x: loopVertices[i + 1].x, y: loopVertices[i + 1].y, z: loopVertices[i + 1].z }
            );
        }
    }

    /**
     * Tessellate a cylindrical face.
     * Projects loop vertices onto the cylinder's parametric space and generates
     * quad strips (split into triangles).
     */
    _tessellateCylindricalFace(surface, loopVertices, outVertices) {
        if (loopVertices.length < 3) return;

        const placement = this._getPlacement(surface.placementRef);
        const { origin, xAxis, yAxis, zAxis } = placement;
        const r = surface.radius;

        // Project loop vertices to (angle, height) in cylinder's parametric space
        const params = loopVertices.map(v => {
            const dx = v.x - origin.x;
            const dy = v.y - origin.y;
            const dz = v.z - origin.z;
            const h = dx * zAxis.x + dy * zAxis.y + dz * zAxis.z;
            const px = dx * xAxis.x + dy * xAxis.y + dz * xAxis.z;
            const py = dx * yAxis.x + dy * yAxis.y + dz * yAxis.z;
            const angle = Math.atan2(py, px);
            return { angle, h };
        });

        // Find height and angle range
        let minH = Infinity, maxH = -Infinity;
        let minA = Infinity, maxA = -Infinity;
        for (const p of params) {
            if (p.h < minH) minH = p.h;
            if (p.h > maxH) maxH = p.h;
            if (p.angle < minA) minA = p.angle;
            if (p.angle > maxA) maxA = p.angle;
        }

        // Decide if it's a full circle or partial arc
        const angleDiff = maxA - minA;
        const isFull = angleDiff > 1.8 * Math.PI;
        const sweepStart = isFull ? -Math.PI : minA;
        const sweepEnd = isFull ? Math.PI : maxA;
        const sweepAngle = sweepEnd - sweepStart;

        const numAngleSteps = Math.max(8, Math.ceil(Math.abs(sweepAngle) / (Math.PI / 8)));
        const numHeightSteps = Math.max(1, Math.ceil((maxH - minH) / (r * 0.5)));

        for (let i = 0; i < numAngleSteps; i++) {
            const a0 = sweepStart + (sweepAngle * i) / numAngleSteps;
            const a1 = sweepStart + (sweepAngle * (i + 1)) / numAngleSteps;
            for (let j = 0; j < numHeightSteps; j++) {
                const h0 = minH + ((maxH - minH) * j) / numHeightSteps;
                const h1 = minH + ((maxH - minH) * (j + 1)) / numHeightSteps;

                const p00 = this._cylPoint(origin, xAxis, yAxis, zAxis, r, a0, h0);
                const p10 = this._cylPoint(origin, xAxis, yAxis, zAxis, r, a1, h0);
                const p01 = this._cylPoint(origin, xAxis, yAxis, zAxis, r, a0, h1);
                const p11 = this._cylPoint(origin, xAxis, yAxis, zAxis, r, a1, h1);

                // Two triangles per quad
                outVertices.push(p00, p10, p11);
                outVertices.push(p00, p11, p01);
            }
        }
    }

    _cylPoint(origin, xAxis, yAxis, zAxis, r, angle, height) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return {
            x: origin.x + r * (cos * xAxis.x + sin * yAxis.x) + height * zAxis.x,
            y: origin.y + r * (cos * xAxis.y + sin * yAxis.y) + height * zAxis.y,
            z: origin.z + r * (cos * xAxis.z + sin * yAxis.z) + height * zAxis.z
        };
    }

    /**
     * Tessellate a conical face.
     */
    _tessellateConicalFace(surface, loopVertices, outVertices) {
        if (loopVertices.length < 3) return;

        const placement = this._getPlacement(surface.placementRef);
        const { origin, xAxis, yAxis, zAxis } = placement;
        const r = surface.radius;
        const semiAngle = surface.semiAngle * Math.PI / 180;

        const params = loopVertices.map(v => {
            const dx = v.x - origin.x;
            const dy = v.y - origin.y;
            const dz = v.z - origin.z;
            const h = dx * zAxis.x + dy * zAxis.y + dz * zAxis.z;
            const px = dx * xAxis.x + dy * xAxis.y + dz * xAxis.z;
            const py = dx * yAxis.x + dy * yAxis.y + dz * yAxis.z;
            return { angle: Math.atan2(py, px), h };
        });

        let minH = Infinity, maxH = -Infinity;
        let minA = Infinity, maxA = -Infinity;
        for (const p of params) {
            if (p.h < minH) minH = p.h;
            if (p.h > maxH) maxH = p.h;
            if (p.angle < minA) minA = p.angle;
            if (p.angle > maxA) maxA = p.angle;
        }

        const angleDiff = maxA - minA;
        const isFull = angleDiff > 1.8 * Math.PI;
        const sweepStart = isFull ? -Math.PI : minA;
        const sweepEnd = isFull ? Math.PI : maxA;
        const sweepAngle = sweepEnd - sweepStart;

        const numAngleSteps = Math.max(8, Math.ceil(Math.abs(sweepAngle) / (Math.PI / 8)));
        const numHeightSteps = Math.max(1, Math.ceil((maxH - minH) / (r * 0.5 || 1)));

        for (let i = 0; i < numAngleSteps; i++) {
            const a0 = sweepStart + (sweepAngle * i) / numAngleSteps;
            const a1 = sweepStart + (sweepAngle * (i + 1)) / numAngleSteps;
            for (let j = 0; j < numHeightSteps; j++) {
                const h0 = minH + ((maxH - minH) * j) / numHeightSteps;
                const h1 = minH + ((maxH - minH) * (j + 1)) / numHeightSteps;

                const coneR0 = r + h0 * Math.tan(semiAngle);
                const coneR1 = r + h1 * Math.tan(semiAngle);

                const p00 = this._cylPoint(origin, xAxis, yAxis, zAxis, coneR0, a0, h0);
                const p10 = this._cylPoint(origin, xAxis, yAxis, zAxis, coneR0, a1, h0);
                const p01 = this._cylPoint(origin, xAxis, yAxis, zAxis, coneR1, a0, h1);
                const p11 = this._cylPoint(origin, xAxis, yAxis, zAxis, coneR1, a1, h1);

                outVertices.push(p00, p10, p11);
                outVertices.push(p00, p11, p01);
            }
        }
    }

    /**
     * Tessellate a spherical face.
     */
    _tessellateSphericalFace(surface, loopVertices, outVertices) {
        if (loopVertices.length < 3) return;

        const placement = this._getPlacement(surface.placementRef);
        const { origin, xAxis, yAxis, zAxis } = placement;
        const r = surface.radius;

        // Project vertices to (theta, phi) on sphere
        const params = loopVertices.map(v => {
            const dx = v.x - origin.x;
            const dy = v.y - origin.y;
            const dz = v.z - origin.z;
            const rr = Math.sqrt(dx * dx + dy * dy + dz * dz) || r;
            const h = dx * zAxis.x + dy * zAxis.y + dz * zAxis.z;
            const phi = Math.acos(Math.max(-1, Math.min(1, h / rr)));
            const px = dx * xAxis.x + dy * xAxis.y + dz * xAxis.z;
            const py = dx * yAxis.x + dy * yAxis.y + dz * yAxis.z;
            const theta = Math.atan2(py, px);
            return { theta, phi };
        });

        let minTheta = Infinity, maxTheta = -Infinity;
        let minPhi = Infinity, maxPhi = -Infinity;
        for (const p of params) {
            if (p.theta < minTheta) minTheta = p.theta;
            if (p.theta > maxTheta) maxTheta = p.theta;
            if (p.phi < minPhi) minPhi = p.phi;
            if (p.phi > maxPhi) maxPhi = p.phi;
        }

        const thetaDiff = maxTheta - minTheta;
        const isFullTheta = thetaDiff > 1.8 * Math.PI;
        const thetaStart = isFullTheta ? -Math.PI : minTheta;
        const thetaEnd = isFullTheta ? Math.PI : maxTheta;
        const thetaSweep = thetaEnd - thetaStart;

        const nTheta = Math.max(8, Math.ceil(Math.abs(thetaSweep) / (Math.PI / 8)));
        const nPhi = Math.max(4, Math.ceil((maxPhi - minPhi) / (Math.PI / 8)));

        for (let i = 0; i < nTheta; i++) {
            const t0 = thetaStart + (thetaSweep * i) / nTheta;
            const t1 = thetaStart + (thetaSweep * (i + 1)) / nTheta;
            for (let j = 0; j < nPhi; j++) {
                const p0 = minPhi + ((maxPhi - minPhi) * j) / nPhi;
                const p1 = minPhi + ((maxPhi - minPhi) * (j + 1)) / nPhi;

                const v00 = this._spherePoint(origin, xAxis, yAxis, zAxis, r, t0, p0);
                const v10 = this._spherePoint(origin, xAxis, yAxis, zAxis, r, t1, p0);
                const v01 = this._spherePoint(origin, xAxis, yAxis, zAxis, r, t0, p1);
                const v11 = this._spherePoint(origin, xAxis, yAxis, zAxis, r, t1, p1);

                outVertices.push(v00, v10, v11);
                outVertices.push(v00, v11, v01);
            }
        }
    }

    _spherePoint(origin, xAxis, yAxis, zAxis, r, theta, phi) {
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);
        const cosTheta = Math.cos(theta);
        const sinTheta = Math.sin(theta);
        const lx = r * sinPhi * cosTheta;
        const ly = r * sinPhi * sinTheta;
        const lz = r * cosPhi;
        return {
            x: origin.x + lx * xAxis.x + ly * yAxis.x + lz * zAxis.x,
            y: origin.y + lx * xAxis.y + ly * yAxis.y + lz * zAxis.y,
            z: origin.z + lx * xAxis.z + ly * yAxis.z + lz * zAxis.z
        };
    }

    /**
     * Tessellate a toroidal face.
     */
    _tessellateToroidalFace(surface, loopVertices, outVertices) {
        if (loopVertices.length < 3) return;

        const placement = this._getPlacement(surface.placementRef);
        const { origin, xAxis, yAxis, zAxis } = placement;
        const R = surface.majorRadius;
        const rr = surface.minorRadius;

        // For toroidal faces, use a default full tessellation approach
        const nMajor = 16;
        const nMinor = 8;

        for (let i = 0; i < nMajor; i++) {
            const u0 = (2 * Math.PI * i) / nMajor;
            const u1 = (2 * Math.PI * (i + 1)) / nMajor;
            for (let j = 0; j < nMinor; j++) {
                const v0 = (2 * Math.PI * j) / nMinor;
                const v1 = (2 * Math.PI * (j + 1)) / nMinor;

                const p00 = this._torusPoint(origin, xAxis, yAxis, zAxis, R, rr, u0, v0);
                const p10 = this._torusPoint(origin, xAxis, yAxis, zAxis, R, rr, u1, v0);
                const p01 = this._torusPoint(origin, xAxis, yAxis, zAxis, R, rr, u0, v1);
                const p11 = this._torusPoint(origin, xAxis, yAxis, zAxis, R, rr, u1, v1);

                outVertices.push(p00, p10, p11);
                outVertices.push(p00, p11, p01);
            }
        }
    }

    _torusPoint(origin, xAxis, yAxis, zAxis, R, r, u, v) {
        const cosU = Math.cos(u), sinU = Math.sin(u);
        const cosV = Math.cos(v), sinV = Math.sin(v);
        const lx = (R + r * cosV) * cosU;
        const ly = (R + r * cosV) * sinU;
        const lz = r * sinV;
        return {
            x: origin.x + lx * xAxis.x + ly * yAxis.x + lz * zAxis.x,
            y: origin.y + lx * xAxis.y + ly * yAxis.y + lz * zAxis.y,
            z: origin.z + lx * xAxis.z + ly * yAxis.z + lz * zAxis.z
        };
    }

    // ─── Vector math ───────────────────────────────────────────────

    _normalize(v) {
        const len = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len < 1e-15) return { x: 0, y: 0, z: 1 };
        return { x: v.x / len, y: v.y / len, z: v.z / len };
    }

    _cross(a, b) {
        return {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x
        };
    }

    _dot(a, b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    _distance(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = a.z - b.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    _pointToAngle(point, origin, xAxis, yAxis) {
        const dx = point.x - origin.x;
        const dy = point.y - origin.y;
        const dz = point.z - origin.z;
        const px = dx * xAxis.x + dy * xAxis.y + dz * xAxis.z;
        const py = dx * yAxis.x + dy * yAxis.y + dz * yAxis.z;
        return Math.atan2(py, px);
    }

    /**
     * Check if the given text content looks like a valid STEP file.
     * @param {string} text
     * @returns {boolean}
     */
    static isSTEP(text) {
        const upper = text.toUpperCase();
        return upper.includes('ISO-10303-21') || (upper.includes('FILE_SCHEMA') && upper.includes('DATA'));
    }
}
