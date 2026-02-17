// 3D Viewer using WebGL for professional CAD-like rendering
import { DENSITY_THRESHOLD } from './constants.js';

// ─── Color Constants ────────────────────────────────────────────────────────

const DENSITY_COLOR_GREEN = 0.298;           // Fixed green channel for density coloring
const DEFAULT_MESH_COLOR = [0.29, 0.565, 0.886]; // Default blue (no density data)
const GRID_COLOR = [0.25, 0.25, 0.35];      // Ground grid line color
const DEFAULT_TRIANGLE_DENSITY = 0.5;
const WIREFRAME_EDGE_COLOR = 0.2;
const EDGE_COLOR_COMPONENT_COUNT = 18;

// ─── WebGL Shader Sources ───────────────────────────────────────────────────

const MESH_VERTEX_SHADER = `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    attribute vec3 aColor;

    uniform mat4 uProjection;
    uniform mat4 uModelView;
    uniform mat3 uNormalMatrix;

    varying vec3 vNormal;
    varying vec3 vColor;
    varying vec3 vPosition;

    void main() {
        vec4 mvPosition = uModelView * vec4(aPosition, 1.0);
        gl_Position = uProjection * mvPosition;
        vNormal = normalize(uNormalMatrix * aNormal);
        vColor = aColor;
        vPosition = mvPosition.xyz;
    }
`;

const MESH_FRAGMENT_SHADER = `
    precision mediump float;

    varying vec3 vNormal;
    varying vec3 vColor;
    varying vec3 vPosition;

    uniform vec3 uLightDir;
    uniform float uAmbient;

    void main() {
        vec3 normal = normalize(vNormal);
        float diffuse = abs(dot(normal, uLightDir));
        float light = uAmbient + (1.0 - uAmbient) * diffuse;
        vec3 color = vColor * light;
        gl_FragColor = vec4(color, 1.0);
    }
`;

const LINE_VERTEX_SHADER = `
    attribute vec3 aPosition;
    attribute vec3 aColor;

    uniform mat4 uProjection;
    uniform mat4 uModelView;

    varying vec3 vColor;

    void main() {
        gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
        vColor = aColor;
    }
`;

const LINE_FRAGMENT_SHADER = `
    precision mediump float;
    varying vec3 vColor;
    uniform float uAlpha;

    void main() {
        gl_FragColor = vec4(vColor, uAlpha);
    }
`;

const OVERLAY_VERTEX_SHADER = `
    attribute vec3 aPosition;

    uniform mat4 uProjection;
    uniform mat4 uModelView;

    void main() {
        gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
    }
`;

const OVERLAY_FRAGMENT_SHADER = `
    precision mediump float;
    uniform vec4 uColor;

    void main() {
        gl_FragColor = uColor;
    }
`;

// ─── Math Utilities ─────────────────────────────────────────────────────────

function mat4Perspective(fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0
    ]);
}

function mat4Identity() {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
}

function mat4Multiply(a, b) {
    const r = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            r[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] +
                           a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
        }
    }
    return r;
}

function mat4RotateX(m, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const r = mat4Identity();
    r[5] = c; r[6] = s; r[9] = -s; r[10] = c;
    return mat4Multiply(m, r);
}

function mat4RotateY(m, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const r = mat4Identity();
    r[0] = c; r[2] = -s; r[8] = s; r[10] = c;
    return mat4Multiply(m, r);
}

function mat4Translate(m, v) {
    const r = mat4Identity();
    r[12] = v[0]; r[13] = v[1]; r[14] = v[2];
    return mat4Multiply(m, r);
}

function mat3NormalFromMat4(m) {
    const a00 = m[0], a01 = m[1], a02 = m[2];
    const a10 = m[4], a11 = m[5], a12 = m[6];
    const a20 = m[8], a21 = m[9], a22 = m[10];
    const det = a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20);
    if (Math.abs(det) < 1e-10) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);
    const id = 1.0 / det;
    return new Float32Array([
        (a11*a22-a21*a12)*id, (a02*a21-a01*a22)*id, (a01*a12-a02*a11)*id,
        (a12*a20-a10*a22)*id, (a00*a22-a02*a20)*id, (a02*a10-a00*a12)*id,
        (a10*a21-a11*a20)*id, (a01*a20-a00*a21)*id, (a00*a11-a01*a10)*id
    ]);
}

function vec3Normalize(v) {
    const len = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    return len > 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0,0,0];
}

// ─── Viewer3D Class ─────────────────────────────────────────────────────────

export class Viewer3D {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = null;
        this.canvas = null;
        this.gl = null;
        this.overlayCanvas = null;
        this.ctx = null;
        this.model = null;
        this.wireframe = false;

        // Camera (orbit) settings
        this.rotation = { x: 0.5, y: 0.5 };
        this.pan = { x: 0, y: 0 };
        this.zoom = 1;
        this.isDragging = false;
        this.isPanning = false;
        this.lastMousePos = { x: 0, y: 0 };

        this.densities = null;
        this.meshData = null;

        // Strain range filter (0..1 normalized)
        this.strainMin = 0;
        this.strainMax = 1;

        // Paint mode for face selection
        this.paintMode = null;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this.forceDirection = 'down';
        this.forceMagnitude = 1000;
        this.isPainting = false;

        // Cached boundary faces for picking
        this._boundaryFaces = [];

        // WebGL resources
        this._meshProgram = null;
        this._lineProgram = null;
        this._overlayProgram = null;

        // Cached GPU buffers
        this._meshBuffers = null;
        this._edgeBuffers = null;
        this._gridBuffers = null;

        this._needsRebuild = true;
        this._lastMeshData = null;
        this._lastDensities = null;
    }

    async init() {
        this.container = document.getElementById(this.containerId);

        // Create WebGL canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.container.appendChild(this.canvas);

        // Create 2D overlay canvas for text, paint indicators, axes
        this.overlayCanvas = document.createElement('canvas');
        this.overlayCanvas.width = this.canvas.width;
        this.overlayCanvas.height = this.canvas.height;
        this.overlayCanvas.style.width = '100%';
        this.overlayCanvas.style.height = '100%';
        this.overlayCanvas.style.position = 'absolute';
        this.overlayCanvas.style.top = '0';
        this.overlayCanvas.style.left = '0';
        this.overlayCanvas.style.pointerEvents = 'none';
        this.container.appendChild(this.overlayCanvas);
        this.ctx = this.overlayCanvas.getContext('2d');

        const gl = this.canvas.getContext('webgl', {
            antialias: true,
            alpha: false,
            depth: true,
            stencil: false,
            preserveDrawingBuffer: false
        });

        if (!gl) {
            console.error('WebGL not available');
            return;
        }

        this.gl = gl;
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);
        gl.frontFace(gl.CCW);
        gl.clearColor(0.102, 0.102, 0.18, 1.0);

        // Compile shaders
        this._meshProgram = this._createProgram(gl, MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER);
        this._lineProgram = this._createProgram(gl, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);
        this._overlayProgram = this._createProgram(gl, OVERLAY_VERTEX_SHADER, OVERLAY_FRAGMENT_SHADER);

        this.setupControls();
        window.addEventListener('resize', () => this.onWindowResize());
        this.draw();
        console.log('WebGL Viewer initialized');
    }

    // ─── Shader compilation ─────────────────────────────────────────────────

    _compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    _createProgram(gl, vsSrc, fsSrc) {
        const vs = this._compileShader(gl, gl.VERTEX_SHADER, vsSrc);
        const fs = this._compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(prog));
            return null;
        }
        return prog;
    }

    // ─── Controls ───────────────────────────────────────────────────────────

    setupControls() {
        const canvas = this.canvas;

        canvas.addEventListener('mousedown', (e) => {
            if (this.paintMode && this.model) {
                this.isPainting = true;
                this.handlePaintClick(e);
                return;
            }
            if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                this.isPanning = true;
            } else if (e.button === 0) {
                this.isDragging = true;
            }
            this.lastMousePos = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener('mousemove', (e) => {
            if (this.isPainting && this.paintMode && this.model) {
                this.handlePaintClick(e);
                return;
            }
            const dx = e.clientX - this.lastMousePos.x;
            const dy = e.clientY - this.lastMousePos.y;

            if (this.isPanning) {
                const panScale = 0.003 / this.zoom;
                this.pan.x += dx * panScale;
                this.pan.y -= dy * panScale;
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.draw();
            } else if (this.isDragging) {
                this.rotation.y -= dx * 0.01;
                this.rotation.x -= dy * 0.01;
                this.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation.x));
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.draw();
            }
        });

        canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.isPanning = false;
            this.isPainting = false;
        });

        canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.isPanning = false;
            this.isPainting = false;
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom *= (1 - e.deltaY * 0.001);
            this.zoom = Math.max(0.1, Math.min(10, this.zoom));
            this.draw();
        });

        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    setPaintMode(mode) {
        this.paintMode = mode;
        this.canvas.style.cursor = mode ? 'crosshair' : 'grab';
    }

    handlePaintClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        if (!this.model || this._boundaryFaces.length === 0) return;
        const { nx, ny, nz } = this.model;
        const { width, height } = this.canvas;

        const aspect = width / height;
        const proj = mat4Perspective(Math.PI / 4, aspect, 0.1, 1000);
        const mv = this._buildModelView(nx, ny, nz);

        let bestDist = Infinity;
        let bestFace = null;
        let bestZ = Infinity;

        for (const bf of this._boundaryFaces) {
            const screenVerts = bf.projVerts.map(pv =>
                this._projectToScreen(pv, mv, proj, width, height)
            );

            const cx = (screenVerts[0].x + screenVerts[1].x + screenVerts[2].x + screenVerts[3].x) / 4;
            const cy = (screenVerts[0].y + screenVerts[1].y + screenVerts[2].y + screenVerts[3].y) / 4;
            const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);

            const maxR = Math.max(
                ...screenVerts.map(v => Math.sqrt((v.x - cx) ** 2 + (v.y - cy) ** 2))
            );
            if (dist < maxR * 1.2) {
                const avgZ = (screenVerts[0].z + screenVerts[1].z + screenVerts[2].z + screenVerts[3].z) / 4;
                if (avgZ < bestZ || (Math.abs(avgZ - bestZ) < 0.001 && dist < bestDist)) {
                    bestDist = dist;
                    bestFace = bf;
                    bestZ = avgZ;
                }
            }
        }

        if (bestFace) {
            const key = bestFace.key;
            const targetSet = this.paintMode === 'constraint' ? this.paintedConstraintFaces : this.paintedForceFaces;
            if (e.shiftKey) {
                targetSet.delete(key);
            } else {
                targetSet.add(key);
            }
            this._needsRebuild = true;
            this.draw();
        }
    }

    _projectToScreen(worldPos, mv, proj, w, h) {
        const x = worldPos.x !== undefined ? worldPos.x : worldPos[0];
        const y = worldPos.y !== undefined ? worldPos.y : worldPos[1];
        const z = worldPos.z !== undefined ? worldPos.z : worldPos[2];

        const ex = mv[0]*x + mv[4]*y + mv[8]*z + mv[12];
        const ey = mv[1]*x + mv[5]*y + mv[9]*z + mv[13];
        const ez = mv[2]*x + mv[6]*y + mv[10]*z + mv[14];
        const ew = mv[3]*x + mv[7]*y + mv[11]*z + mv[15];

        const cx2 = proj[0]*ex + proj[4]*ey + proj[8]*ez + proj[12]*ew;
        const cy2 = proj[1]*ex + proj[5]*ey + proj[9]*ez + proj[13]*ew;
        const cz2 = proj[2]*ex + proj[6]*ey + proj[10]*ez + proj[14]*ew;
        const cw2 = proj[3]*ex + proj[7]*ey + proj[11]*ez + proj[15]*ew;

        const ndcx = cx2 / cw2;
        const ndcy = cy2 / cw2;
        const ndcz = cz2 / cw2;

        return {
            x: (ndcx * 0.5 + 0.5) * w,
            y: (1.0 - (ndcy * 0.5 + 0.5)) * h,
            z: ndcz
        };
    }

    onWindowResize() {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.overlayCanvas.width = this.canvas.width;
        this.overlayCanvas.height = this.canvas.height;
        if (this.gl) {
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }
        this.draw();
    }

    setModel(model) {
        this.model = model;
        this.densities = null;
        this.meshData = null;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this._needsRebuild = true;
        this.draw();
    }

    // ─── Main Draw ──────────────────────────────────────────────────────────

    draw() {
        if (!this.gl) return;

        const gl = this.gl;
        const { width, height } = this.canvas;

        gl.viewport(0, 0, width, height);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        // Clear overlay
        this.ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        if (!this.model) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '20px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Load a model to view', width / 2, height / 2);
            return;
        }

        const { nx, ny, nz } = this.model;
        const aspect = width / height;

        const projection = mat4Perspective(Math.PI / 4, aspect, 0.1, 1000);
        const modelView = this._buildModelView(nx, ny, nz);
        const normalMatrix = mat3NormalFromMat4(modelView);

        if (this._needsRebuild || this._lastMeshData !== this.meshData || this._lastDensities !== this.densities) {
            this._lastMeshData = this.meshData;
            this._lastDensities = this.densities;
            this._rebuildBuffers(nx, ny, nz);
            this._needsRebuild = false;
        }

        // Draw ground grid
        this._drawGrid(gl, projection, modelView);

        // Draw mesh
        if (this._meshBuffers && this._meshBuffers.count > 0) {
            this._drawMesh(gl, projection, modelView, normalMatrix);
        }

        // Draw edges
        if (this._edgeBuffers && this._edgeBuffers.count > 0 && this.wireframe) {
            this._drawEdges(gl, projection, modelView);
        }

        // Draw overlays
        this._drawOverlays(gl, projection, modelView, nx, ny, nz);

        // Draw 2D overlay
        this._drawAxesOverlay();

        if (this.paintMode) {
            this.ctx.fillStyle = this.paintMode === 'constraint' ? 'rgba(0,200,100,0.8)' : 'rgba(255,100,50,0.8)';
            this.ctx.font = '14px Arial';
            this.ctx.textAlign = 'left';
            const label = this.paintMode === 'constraint'
                ? '🖌 Painting Constraints (Shift+click to remove)'
                : '🖌 Painting Forces (Shift+click to remove)';
            this.ctx.fillText(label, 10, height - 10);
        }
    }

    _buildModelView(nx, ny, nz) {
        const maxDim = Math.max(nx, ny, nz);
        const dist = maxDim * 1.8 / this.zoom;

        let mv = mat4Identity();
        mv = mat4Translate(mv, [this.pan.x * maxDim, this.pan.y * maxDim, -dist]);
        mv = mat4RotateX(mv, this.rotation.x);
        mv = mat4RotateY(mv, this.rotation.y);
        mv = mat4Translate(mv, [-nx / 2, -ny / 2, -nz / 2]);

        return mv;
    }

    // ─── Buffer building ────────────────────────────────────────────────────

    _rebuildBuffers(nx, ny, nz) {
        if (this.meshData && this.meshData.length > 0) {
            this._buildTriangleMeshBuffers(nx, ny, nz);
        } else {
            this._buildVoxelBuffers(nx, ny, nz);
        }
        this._buildGridBuffers(nx, ny, nz);
    }

    /**
     * Build GPU buffers from optimizer mesh data with strain-aware closed-surface regeneration.
     * When strain filter is active, elements outside the range become void and new boundary
     * faces are generated to seal the surface properly.
     */
    _buildTriangleMeshBuffers(nx, ny, nz) {
        const gl = this.gl;

        const hasStrainFilter = this.strainMin > 0 || this.strainMax < 1;
        if (!hasStrainFilter) {
            this._buildDirectTriangleMeshBuffers(gl);
            return;
        }

        // Build per-element visibility map and metadata from mesh triangles
        const visibleElements = new Uint8Array(nx * ny * nz);
        const elementDensity = new Float32Array(nx * ny * nz);

        for (const tri of this.meshData) {
            const v = tri.vertices[0];
            let ex = Math.min(Math.max(Math.floor(v[0]), 0), nx - 1);
            let ey = Math.min(Math.max(Math.floor(v[1]), 0), ny - 1);
            let ez = Math.min(Math.max(Math.floor(v[2]), 0), nz - 1);

            const idx = ex + ey * nx + ez * nx * ny;
            elementDensity[idx] = tri.density;
            const strain = tri.strain !== undefined ? tri.strain : 0;

            if (hasStrainFilter) {
                if (strain >= this.strainMin && strain <= this.strainMax) {
                    visibleElements[idx] = 1;
                }
            } else {
                if (tri.density > DENSITY_THRESHOLD) {
                    visibleElements[idx] = 1;
                }
            }
        }

        // Generate closed mesh from visible elements
        this._generateClosedMeshBuffers(gl, nx, ny, nz, visibleElements, elementDensity, true);
    }

    _buildDirectTriangleMeshBuffers(gl) {
        const positions = [];
        const normals = [];
        const colors = [];
        const edgePositions = [];
        const edgeColors = [];

        this._boundaryFaces = [];

        for (const tri of this.meshData) {
            const density = tri.density !== undefined ? tri.density : DEFAULT_TRIANGLE_DENSITY;
            const n = tri.normal || [0, 0, 1];
            const r = density;
            const g = DENSITY_COLOR_GREEN;
            const b = 1 - density;

            positions.push(...tri.vertices[0], ...tri.vertices[1], ...tri.vertices[2]);
            normals.push(...n, ...n, ...n);
            colors.push(r, g, b, r, g, b, r, g, b);

            edgePositions.push(
                ...tri.vertices[0], ...tri.vertices[1],
                ...tri.vertices[1], ...tri.vertices[2],
                ...tri.vertices[2], ...tri.vertices[0]
            );
            edgeColors.push(...Array(EDGE_COLOR_COMPONENT_COUNT).fill(WIREFRAME_EDGE_COLOR));
        }

        this._uploadMeshBuffers(gl, positions, normals, colors);
        this._uploadEdgeBuffers(gl, edgePositions, edgeColors);
    }

    _buildVoxelBuffers(nx, ny, nz) {
        const gl = this.gl;
        const { elements } = this.model;

        const occupied = new Uint8Array(nx * ny * nz);
        const densityMap = new Float32Array(nx * ny * nz);

        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    const index = x + y * nx + z * nx * ny;
                    const density = this.densities ? this.densities[index] : elements[index];
                    densityMap[index] = density;
                    if (density > DENSITY_THRESHOLD) {
                        occupied[index] = 1;
                    }
                }
            }
        }

        this._generateClosedMeshBuffers(gl, nx, ny, nz, occupied, densityMap, !!this.densities);
    }

    /**
     * Core mesh generation: creates a watertight closed surface from a visibility grid.
     * Every boundary face (between visible and non-visible voxels) is emitted.
     */
    _generateClosedMeshBuffers(gl, nx, ny, nz, visibleElements, densityMap, hasDensityColors) {
        const positions = [];
        const normals = [];
        const colors = [];
        const edgePositions = [];
        const edgeColors = [];

        const isVisible = (x, y, z) => {
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return false;
            return visibleElements[x + y * nx + z * nx * ny] === 1;
        };

        // Face definitions: [normal, neighbor direction, 4 vertex offsets (CCW from outside)]
        const faceDefinitions = [
            { normal: [-1, 0, 0], dir: [-1, 0, 0], verts: [[0,0,0],[0,0,1],[0,1,1],[0,1,0]] },
            { normal: [1, 0, 0],  dir: [1, 0, 0],  verts: [[1,0,1],[1,0,0],[1,1,0],[1,1,1]] },
            { normal: [0, -1, 0], dir: [0, -1, 0], verts: [[0,0,0],[1,0,0],[1,0,1],[0,0,1]] },
            { normal: [0, 1, 0],  dir: [0, 1, 0],  verts: [[0,1,0],[0,1,1],[1,1,1],[1,1,0]] },
            { normal: [0, 0, -1], dir: [0, 0, -1], verts: [[0,0,0],[0,1,0],[1,1,0],[1,0,0]] },
            { normal: [0, 0, 1],  dir: [0, 0, 1],  verts: [[0,0,1],[1,0,1],[1,1,1],[0,1,1]] },
        ];

        this._boundaryFaces = [];

        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    if (!isVisible(x, y, z)) continue;
                    const idx = x + y * nx + z * nx * ny;
                    const density = densityMap[idx];

                    for (let fi = 0; fi < 6; fi++) {
                        const face = faceDefinitions[fi];
                        const [dx, dy, dz] = face.dir;
                        if (isVisible(x + dx, y + dy, z + dz)) continue;

                        // Compute color
                        let r, g, b;
                        if (hasDensityColors) {
                            r = density;
                            g = DENSITY_COLOR_GREEN;
                            b = 1 - density;
                        } else {
                            r = DEFAULT_MESH_COLOR[0];
                            g = DEFAULT_MESH_COLOR[1];
                            b = DEFAULT_MESH_COLOR[2];
                        }

                        const verts = face.verts.map(([vx, vy, vz]) => [x + vx, y + vy, z + vz]);
                        const n = face.normal;

                        // Two triangles per quad (CCW)
                        const triIndices = [[0, 1, 2], [0, 2, 3]];
                        for (const [i0, i1, i2] of triIndices) {
                            positions.push(...verts[i0], ...verts[i1], ...verts[i2]);
                            normals.push(...n, ...n, ...n);
                            colors.push(r, g, b, r, g, b, r, g, b);
                        }

                        // Edge lines for wireframe
                        for (let i = 0; i < 4; i++) {
                            const j = (i + 1) % 4;
                            edgePositions.push(...verts[i], ...verts[j]);
                            edgeColors.push(0.2, 0.2, 0.2, 0.2, 0.2, 0.2);
                        }

                        // Cache boundary face for paint picking
                        const key = `${x},${y},${z},${fi}`;
                        this._boundaryFaces.push({
                            key,
                            projVerts: verts.map(v => ({ x: v[0], y: v[1], z: v[2] })),
                            avgDepth: 0
                        });
                    }
                }
            }
        }

        this._uploadMeshBuffers(gl, positions, normals, colors);
        this._uploadEdgeBuffers(gl, edgePositions, edgeColors);
    }

    _uploadMeshBuffers(gl, positions, normals, colors) {
        if (this._meshBuffers) {
            gl.deleteBuffer(this._meshBuffers.position);
            gl.deleteBuffer(this._meshBuffers.normal);
            gl.deleteBuffer(this._meshBuffers.color);
        }
        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        const normBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
        const colBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

        this._meshBuffers = {
            position: posBuf,
            normal: normBuf,
            color: colBuf,
            count: positions.length / 3
        };
    }

    _uploadEdgeBuffers(gl, positions, colors) {
        if (this._edgeBuffers) {
            gl.deleteBuffer(this._edgeBuffers.position);
            gl.deleteBuffer(this._edgeBuffers.color);
        }
        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        const colBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

        this._edgeBuffers = {
            position: posBuf,
            color: colBuf,
            count: positions.length / 3
        };
    }

    _buildGridBuffers(nx, ny, nz) {
        const gl = this.gl;
        const positions = [];
        const colors = [];

        const gridSize = Math.max(nx, nz);

        for (let i = 0; i <= gridSize; i++) {
            positions.push(i, 0, 0, i, 0, gridSize);
            colors.push(...GRID_COLOR, ...GRID_COLOR);
            positions.push(0, 0, i, gridSize, 0, i);
            colors.push(...GRID_COLOR, ...GRID_COLOR);
        }

        if (this._gridBuffers) {
            gl.deleteBuffer(this._gridBuffers.position);
            gl.deleteBuffer(this._gridBuffers.color);
        }
        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        const colBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

        this._gridBuffers = {
            position: posBuf,
            color: colBuf,
            count: positions.length / 3
        };
    }

    // ─── Draw calls ─────────────────────────────────────────────────────────

    _drawMesh(gl, projection, modelView, normalMatrix) {
        const prog = this._meshProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uNormalMatrix'), false, normalMatrix);

        const lightDir = vec3Normalize([0.3, 0.5, 0.8]);
        gl.uniform3fv(gl.getUniformLocation(prog, 'uLightDir'), lightDir);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAmbient'), 0.4);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const normLoc = gl.getAttribLocation(prog, 'aNormal');
        const colLoc = gl.getAttribLocation(prog, 'aColor');

        gl.bindBuffer(gl.ARRAY_BUFFER, this._meshBuffers.position);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._meshBuffers.normal);
        gl.enableVertexAttribArray(normLoc);
        gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._meshBuffers.color);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, this._meshBuffers.count);

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(normLoc);
        gl.disableVertexAttribArray(colLoc);
    }

    _drawEdges(gl, projection, modelView) {
        const prog = this._lineProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.5);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const colLoc = gl.getAttribLocation(prog, 'aColor');

        gl.bindBuffer(gl.ARRAY_BUFFER, this._edgeBuffers.position);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._edgeBuffers.color);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.LINES, 0, this._edgeBuffers.count);

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(colLoc);
    }

    _drawGrid(gl, projection, modelView) {
        if (!this._gridBuffers || this._gridBuffers.count === 0) return;

        const prog = this._lineProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.3);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const colLoc = gl.getAttribLocation(prog, 'aColor');

        gl.bindBuffer(gl.ARRAY_BUFFER, this._gridBuffers.position);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._gridBuffers.color);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.LINES, 0, this._gridBuffers.count);

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(colLoc);
    }

    _drawOverlays(gl, projection, modelView, nx, ny, nz) {
        const hasCon = this.paintedConstraintFaces.size > 0;
        const hasForce = this.paintedForceFaces.size > 0;
        if (!hasCon && !hasForce) return;

        const prog = this._overlayProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.disable(gl.CULL_FACE);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');

        if (hasCon) {
            const positions = [];
            for (const bf of this._boundaryFaces) {
                if (!this.paintedConstraintFaces.has(bf.key)) continue;
                const v = bf.projVerts;
                positions.push(v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z, v[2].x, v[2].y, v[2].z);
                positions.push(v[0].x, v[0].y, v[0].z, v[2].x, v[2].y, v[2].z, v[3].x, v[3].y, v[3].z);
            }
            if (positions.length > 0) {
                const buf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
                gl.enableVertexAttribArray(posLoc);
                gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
                gl.uniform4fv(gl.getUniformLocation(prog, 'uColor'), [0, 0.78, 0.39, 0.45]);
                gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
                gl.disableVertexAttribArray(posLoc);
                gl.deleteBuffer(buf);
            }
        }

        if (hasForce) {
            const positions = [];
            for (const bf of this._boundaryFaces) {
                if (!this.paintedForceFaces.has(bf.key)) continue;
                const v = bf.projVerts;
                positions.push(v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z, v[2].x, v[2].y, v[2].z);
                positions.push(v[0].x, v[0].y, v[0].z, v[2].x, v[2].y, v[2].z, v[3].x, v[3].y, v[3].z);
            }
            if (positions.length > 0) {
                const buf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
                gl.enableVertexAttribArray(posLoc);
                gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
                gl.uniform4fv(gl.getUniformLocation(prog, 'uColor'), [1.0, 0.39, 0.2, 0.4]);
                gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
                gl.disableVertexAttribArray(posLoc);
                gl.deleteBuffer(buf);
            }

            this._drawForceArrows2D(nx, ny, nz, projection, modelView);
        }

        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
    }

    _drawForceArrows2D(nx, ny, nz, projection, modelView) {
        const { width, height } = this.overlayCanvas;

        let arrowDir;
        switch (this.forceDirection) {
            case 'down':  arrowDir = [0, -1, 0]; break;
            case 'up':    arrowDir = [0, 1, 0]; break;
            case 'left':  arrowDir = [-1, 0, 0]; break;
            case 'right': arrowDir = [1, 0, 0]; break;
            default:      arrowDir = [0, -1, 0];
        }
        const arrowLen = 1.5;

        for (const bf of this._boundaryFaces) {
            if (!this.paintedForceFaces.has(bf.key)) continue;

            const parts = bf.key.split(',');
            const vx = parseInt(parts[0], 10) + 0.5;
            const vy = parseInt(parts[1], 10) + 0.5;
            const vz = parseInt(parts[2], 10) + 0.5;

            const startScreen = this._projectToScreen(
                { x: vx, y: vy, z: vz }, modelView, projection, width, height
            );
            const endScreen = this._projectToScreen(
                { x: vx + arrowDir[0] * arrowLen, y: vy + arrowDir[1] * arrowLen, z: vz + arrowDir[2] * arrowLen },
                modelView, projection, width, height
            );

            this.ctx.strokeStyle = 'rgba(255, 60, 30, 0.9)';
            this.ctx.lineWidth = 2.5;
            this.ctx.beginPath();
            this.ctx.moveTo(startScreen.x, startScreen.y);
            this.ctx.lineTo(endScreen.x, endScreen.y);
            this.ctx.stroke();

            const angle = Math.atan2(endScreen.y - startScreen.y, endScreen.x - startScreen.x);
            const headLen = 8;
            this.ctx.beginPath();
            this.ctx.moveTo(endScreen.x, endScreen.y);
            this.ctx.lineTo(endScreen.x - headLen * Math.cos(angle - 0.4), endScreen.y - headLen * Math.sin(angle - 0.4));
            this.ctx.moveTo(endScreen.x, endScreen.y);
            this.ctx.lineTo(endScreen.x - headLen * Math.cos(angle + 0.4), endScreen.y - headLen * Math.sin(angle + 0.4));
            this.ctx.stroke();
            this.ctx.lineWidth = 1;
        }
    }

    // ─── 2D Overlay (axes) ──────────────────────────────────────────────────

    _drawAxesOverlay() {
        const axisOriginX = 50;
        const axisOriginY = this.overlayCanvas.height - 50;
        const axisLen = 35;

        const axes = [
            { dir: [1, 0, 0], color: '#ff0000', label: 'X' },
            { dir: [0, 1, 0], color: '#00ff00', label: 'Y' },
            { dir: [0, 0, 1], color: '#0088ff', label: 'Z' }
        ];

        for (const axis of axes) {
            const [dx, dy, dz] = axis.dir;
            const projected = this._projectDir(dx, dy, dz);

            const ex = axisOriginX + projected.x * axisLen;
            const ey = axisOriginY - projected.y * axisLen;

            this.ctx.strokeStyle = axis.color;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(axisOriginX, axisOriginY);
            this.ctx.lineTo(ex, ey);
            this.ctx.stroke();

            this.ctx.fillStyle = axis.color;
            this.ctx.font = 'bold 12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(axis.label, ex + (projected.x * 10), ey - (projected.y * 10));
        }
        this.ctx.lineWidth = 1;
    }

    _projectDir(x, y, z) {
        const cosY = Math.cos(this.rotation.y);
        const sinY = Math.sin(this.rotation.y);
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;

        const cosX = Math.cos(this.rotation.x);
        const sinX = Math.sin(this.rotation.x);
        const y1 = y * cosX - z1 * sinX;

        return { x: x1, y: y1 };
    }

    // ─── Public API ─────────────────────────────────────────────────────────

    updateMesh(meshData) {
        this.meshData = meshData;
        this._needsRebuild = true;
        this.draw();
    }

    updateDensities(densities) {
        this.densities = densities;
        this._needsRebuild = true;
        this.draw();
    }

    setStrainRange(min, max) {
        this.strainMin = min;
        this.strainMax = max;
        this._needsRebuild = true;
        this.draw();
    }

    toggleWireframe() {
        this.wireframe = !this.wireframe;
        this.draw();
    }

    resetCamera() {
        this.rotation = { x: 0.5, y: 0.5 };
        this.pan = { x: 0, y: 0 };
        this.zoom = 1;
        this.draw();
    }

    clear() {
        this.model = null;
        this.densities = null;
        this.meshData = null;
        this.strainMin = 0;
        this.strainMax = 1;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this.paintMode = null;
        this._boundaryFaces = [];
        this._needsRebuild = true;
        this.draw();
    }

    // Keep for compatibility
    project3D(x, y, z, nx, ny, nz) {
        x = x - nx / 2;
        y = y - ny / 2;
        z = z - nz / 2;

        const cosY = Math.cos(this.rotation.y);
        const sinY = Math.sin(this.rotation.y);
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;

        const cosX = Math.cos(this.rotation.x);
        const sinX = Math.sin(this.rotation.x);
        const y1 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;

        return { x: x1, y: y1, z: z2 };
    }

    getDepth(x, y, z) {
        const projected = this.project3D(x, y, z, 1, 1, 1);
        return projected.z;
    }
}
