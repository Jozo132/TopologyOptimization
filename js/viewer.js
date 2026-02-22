// 3D Viewer using WebGL for professional CAD-like rendering
import { DENSITY_THRESHOLD } from './constants.js';
import { generateUniformSurfaceMesh, generateAMRSurfaceMesh } from './amr-surface-mesh.js';

// ─── Color Constants ────────────────────────────────────────────────────────

const DENSITY_COLOR_GREEN = 0.298;           // Fixed green channel for density coloring
const DEFAULT_MESH_COLOR = [0.29, 0.565, 0.886]; // Default blue (no density data)
const GRID_COLOR = [0.25, 0.25, 0.35];      // Ground grid line color
const DEFAULT_TRIANGLE_DENSITY = 0.5;
const WIREFRAME_EDGE_COLOR = 0.2;
const EDGE_COLOR_COMPONENT_COUNT = 18;
const BRUSH_DEPTH_TOLERANCE = 0.1;           // Depth tolerance for 3D brush face selection

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
    varying vec3 vWorldPos;

    void main() {
        vec4 mvPosition = uModelView * vec4(aPosition, 1.0);
        gl_Position = uProjection * mvPosition;
        vNormal = normalize(uNormalMatrix * aNormal);
        vColor = aColor;
        vPosition = mvPosition.xyz;
        vWorldPos = aPosition;
    }
`;

const MESH_FRAGMENT_SHADER = `
    precision mediump float;

    varying vec3 vNormal;
    varying vec3 vColor;
    varying vec3 vPosition;
    varying vec3 vWorldPos;

    uniform vec3 uLightDir;
    uniform float uAmbient;
    uniform float uAlpha;
    uniform float uClipEnabled;
    uniform vec3 uClipNormal;
    uniform float uClipOffset;

    void main() {
        if (uClipEnabled > 0.5 && dot(uClipNormal, vWorldPos) - uClipOffset > 0.0) discard;
        vec3 normal = normalize(vNormal);
        float diffuse = abs(dot(normal, uLightDir));
        float light = uAmbient + (1.0 - uAmbient) * diffuse;
        vec3 color = vColor * light;
        float alpha = uAlpha > 0.0 ? uAlpha : 1.0;
        gl_FragColor = vec4(color, alpha);
    }
`;

const LINE_VERTEX_SHADER = `
    attribute vec3 aPosition;
    attribute vec3 aColor;

    uniform mat4 uProjection;
    uniform mat4 uModelView;

    varying vec3 vColor;
    varying vec3 vWorldPos;

    void main() {
        gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
        vColor = aColor;
        vWorldPos = aPosition;
    }
`;

const LINE_FRAGMENT_SHADER = `
    precision mediump float;
    varying vec3 vColor;
    varying vec3 vWorldPos;
    uniform float uAlpha;
    uniform float uClipEnabled;
    uniform vec3 uClipNormal;
    uniform float uClipOffset;

    void main() {
        if (uClipEnabled > 0.5 && dot(uClipNormal, vWorldPos) - uClipOffset > 0.0) discard;
        gl_FragColor = vec4(vColor, uAlpha);
    }
`;

const OVERLAY_VERTEX_SHADER = `
    attribute vec3 aPosition;

    uniform mat4 uProjection;
    uniform mat4 uModelView;

    varying vec3 vWorldPos;

    void main() {
        gl_Position = uProjection * uModelView * vec4(aPosition, 1.0);
        vWorldPos = aPosition;
    }
`;

const OVERLAY_FRAGMENT_SHADER = `
    precision mediump float;
    varying vec3 vWorldPos;
    uniform vec4 uColor;
    uniform float uClipEnabled;
    uniform vec3 uClipNormal;
    uniform float uClipOffset;

    void main() {
        if (uClipEnabled > 0.5 && dot(uClipNormal, vWorldPos) - uClipOffset > 0.0) discard;
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

        // View mode: 'auto' uses mesh when available, 'voxel' forces voxel display
        this.viewMode = 'auto';
        // Whether the mesh (and edges) should be drawn
        this.meshVisible = true;

        // Camera (orbit) settings
        this.rotation = { x: 0.5, y: 0.5 };
        this.pan = { x: 0, y: 0 };
        this.zoom = 1;
        this.isDragging = false;
        this.isPanning = false;
        this.lastMousePos = { x: 0, y: 0 };

        this.densities = null;
        this.meshData = null;
        this.amrCells = null;

        // Strain range filter (0..1 normalized)
        this.strainMin = 0;
        this.strainMax = 1;
        this.maxStress = 0; // Max stress value (for MPa scale)

        // Yield strength visualization
        this.yieldStrength = 0; // Yield strength in MPa (0 = not set)
        this._yieldNormalized = 0; // Yield threshold normalized to [0..1] stress range

        // Density threshold for visibility (user-adjustable)
        this.densityThreshold = DENSITY_THRESHOLD;

        // Cached visibility array (shared between buffer build and section drawing)
        this._cachedVisible = null;
        this._cachedDensityMap = null;
        this._cachedStressMap = null;
        this._volumetricStressMap = null;

        // Hover stress tooltip
        this._hoverStressValue = null; // Stress value at hovered position
        this._hoverScreenPos = null;   // Screen position for tooltip

        // Paint mode for face selection
        this.paintMode = null;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this.paintedKeepFaces = new Set();
        this.forceDirection = 'down';
        this.forceVector = null; // Custom force vector [fx, fy, fz], overrides forceDirection
        this.forceMagnitude = 1000;
        this.forceType = 'total'; // 'total' or 'pressure'
        this.isPainting = false;
        this._paintErasing = false;

        // Angle tolerance surface selection
        this.angleTolerance = 45; // degrees
        this.useAngleSelection = false;
        this._seedFaceKey = null; // seed face for angle-based flood-fill
        this._seedFaceNormal = null;

        // Selection groups: each group has { id, name, type: 'force'|'constraint', faces: Set, params: {} }
        this.selectionGroups = [];
        this._nextGroupId = 1;
        this.activeGroupId = null;

        // Selection shapes: each shape has { id, shapeType, position, size, rotation }
        this.selectionShapes = [];
        this._nextShapeId = 1;
        this._shapeHighlightFaces = new Set();

        // 3D brush
        this.brushSize = 1;
        this._hoverFaces = [];
        this._hoverMousePos = null;

        // Cached boundary faces for picking
        this._boundaryFaces = [];
        this._boundaryFaceMap = {}; // key → face object for quick lookup

        // WebGL resources
        this._meshProgram = null;
        this._lineProgram = null;
        this._overlayProgram = null;

        // Section (clipping) plane
        this.sectionEnabled = false;
        this.sectionNormal = [1, 0, 0];
        this.sectionOffset = 0;
        this._sectionInitialized = false;
        this._isSectionDragging = false;
        this._sectionPlaneBuffer = null;
        this._sectionCapBuffer = null;
        this._sectionCapEdgeBuffer = null;
        this.onSectionChange = null; // callback when section plane changes

        // Stress bar drag handles
        this._stressBarRect = null;
        this._stressHandleDrag = null; // 'min' | 'max' | null
        this.onStrainRangeChange = null; // callback when strain range changes via handles

        // Stress bar label (changes in fatigue mode)
        this.stressBarLabel = 'Stress (N/mm² = MPa)';
        this.stressBarUnit = 'MPa';

        // Displacement visualization
        this.displacementData = null;  // { U: Float32Array, nx, ny, nz }
        this.showDisplacement = false;
        this.displacementScale = 1;

        // Cached GPU buffers
        this._meshBuffers = null;
        this._edgeBuffers = null;
        this._gridBuffers = null;

        // Reference model: original high-res mesh from STL/STEP import
        this.referenceVertices = null; // flat array of {x,y,z} triangle vertices
        this.showReference = true;     // toggle for reference model visibility
        this._referenceBuffers = null; // GPU buffers for reference mesh

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
                // Paint mode controls: MMB rotates, SHIFT+MMB pans
                if (e.button === 1 && e.shiftKey) {
                    this.isPanning = true;
                } else if (e.button === 1) {
                    this.isDragging = true;
                } else if (e.button === 0) {
                    // Left click: paint
                    this.isPainting = true;
                    this._paintErasing = false;
                    this.handlePaintClick(e);
                } else if (e.button === 2) {
                    // Right click: erase
                    this.isPainting = true;
                    this._paintErasing = true;
                    this.handlePaintClick(e);
                }
            } else {
                // Check for stress bar handle hit
                const handleHit = this._hitTestStressHandle(e);
                if (e.button === 0 && handleHit) {
                    this._stressHandleDrag = handleHit;
                } else if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                    this.isPanning = true;
                } else if (e.button === 0 && e.altKey && this.sectionEnabled) {
                    this._isSectionDragging = true;
                } else if (e.button === 0) {
                    this.isDragging = true;
                }
            }
            this.lastMousePos = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener('mousemove', (e) => {
            if (this._stressHandleDrag) {
                this._updateStressHandleDrag(e);
                return;
            }

            if (this.isPainting && this.paintMode && this.model) {
                this.handlePaintClick(e);
                return;
            }

            // Hover highlight in paint mode
            if (this.paintMode && this.model && !this.isDragging && !this.isPanning) {
                this._updateHover(e);
            }

            // Stress hover tooltip (when mesh data available and not dragging)
            if (!this.paintMode && this.meshData && this.meshData.length > 0 && !this.isDragging && !this.isPanning) {
                this._updateStressHover(e);
            }

            const dx = e.clientX - this.lastMousePos.x;
            const dy = e.clientY - this.lastMousePos.y;

            if (this._isSectionDragging && this.sectionEnabled) {
                this._rotateSectionPlane(dx, dy);
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.draw();
                if (this.onSectionChange) this.onSectionChange();
            } else if (this.isPanning) {
                const panScale = 0.003 / this.zoom;
                this.pan.x += dx * panScale;
                this.pan.y -= dy * panScale;
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.draw();
            } else if (this.isDragging) {
                this.rotation.y += dx * 0.01;
                this.rotation.x += dy * 0.01;
                this.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation.x));
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.draw();
            }
        });

        canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.isPanning = false;
            this.isPainting = false;
            this._isSectionDragging = false;
            this._stressHandleDrag = null;
        });

        canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.isPanning = false;
            this.isPainting = false;
            this._isSectionDragging = false;
            this._stressHandleDrag = null;
            if (this._hoverFaces.length > 0) {
                this._hoverFaces = [];
                this.draw();
            }
            if (this._hoverStressValue !== null) {
                this._hoverStressValue = null;
                this._hoverScreenPos = null;
                this.draw();
            }
        });

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.altKey && this.sectionEnabled && this.model) {
                const maxDim = Math.max(this.model.nx, this.model.ny, this.model.nz);
                this.sectionOffset -= e.deltaY * 0.005 * maxDim;
                this.draw();
                if (this.onSectionChange) this.onSectionChange();
            } else {
                this.zoom *= (1 - e.deltaY * 0.001);
                this.zoom = Math.max(0.1, Math.min(10, this.zoom));
                this.draw();
            }
        });

        canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // ── Touch support for mobile ────────────────────────────────────────
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                const t = e.touches[0];
                if (this.paintMode && this.model) {
                    this.isPainting = true;
                    this._paintErasing = false;
                    this.handlePaintClick(t);
                } else {
                    this.isDragging = true;
                }
                this.lastMousePos = { x: t.clientX, y: t.clientY };
                this._lastTouchDist = null;
                this._lastTouchMid = null;
            } else if (e.touches.length === 2) {
                this.isDragging = false;
                this.isPainting = false;
                const [a, b] = [e.touches[0], e.touches[1]];
                this._lastTouchDist = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
                this._lastTouchMid = { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 };
            }
        }, { passive: false });

        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                const t = e.touches[0];
                if (this.isPainting && this.paintMode && this.model) {
                    this.handlePaintClick(t);
                    this.lastMousePos = { x: t.clientX, y: t.clientY };
                    return;
                }
                const dx = t.clientX - this.lastMousePos.x;
                const dy = t.clientY - this.lastMousePos.y;
                if (this.isDragging) {
                    this.rotation.y += dx * 0.01;
                    this.rotation.x += dy * 0.01;
                    this.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation.x));
                    this.draw();
                }
                this.lastMousePos = { x: t.clientX, y: t.clientY };
            } else if (e.touches.length === 2) {
                const [a, b] = [e.touches[0], e.touches[1]];
                const dist = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
                const mid = { x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 };
                if (this._lastTouchDist !== null) {
                    // Pinch → zoom
                    const scale = dist / this._lastTouchDist;
                    this.zoom *= scale;
                    this.zoom = Math.max(0.1, Math.min(10, this.zoom));
                    // Two-finger drag → pan
                    if (this._lastTouchMid !== null) {
                        const dx = mid.x - this._lastTouchMid.x;
                        const dy = mid.y - this._lastTouchMid.y;
                        const panScale = 0.003 / this.zoom;
                        this.pan.x += dx * panScale;
                        this.pan.y -= dy * panScale;
                    }
                    this.draw();
                }
                this._lastTouchDist = dist;
                this._lastTouchMid = mid;
            }
        }, { passive: false });

        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (e.touches.length === 0) {
                this.isDragging = false;
                this.isPanning = false;
                this.isPainting = false;
                this._lastTouchDist = null;
                this._lastTouchMid = null;
            } else if (e.touches.length === 1) {
                this._lastTouchDist = null;
                this._lastTouchMid = null;
                const t = e.touches[0];
                this.lastMousePos = { x: t.clientX, y: t.clientY };
                if (!this.paintMode) this.isDragging = true;
            }
        }, { passive: false });
    }

    setPaintMode(mode) {
        this.paintMode = mode;
        this.canvas.style.cursor = mode ? 'crosshair' : 'grab';
        if (!mode) {
            this._hoverFaces = [];
            this.draw();
        }
    }

    _rotateSectionPlane(dx, dy) {
        const speed = 0.01;
        const [ox, oy, oz] = this.sectionNormal;

        // Rotate around Y axis (horizontal drag)
        const cosY = Math.cos(dx * speed);
        const sinY = Math.sin(dx * speed);
        const nx = ox * cosY + oz * sinY;
        const nz1 = -ox * sinY + oz * cosY;

        // Rotate around X axis (vertical drag), applied after Y rotation
        const cosX = Math.cos(dy * speed);
        const sinX = Math.sin(dy * speed);
        const ny = oy * cosX - nz1 * sinX;
        const nz = oy * sinX + nz1 * cosX;

        this.sectionNormal = vec3Normalize([nx, ny, nz]);
    }

    _setClipUniforms(gl, prog) {
        const loc = (name) => gl.getUniformLocation(prog, name);
        gl.uniform1f(loc('uClipEnabled'), this.sectionEnabled ? 1.0 : 0.0);
        if (this.sectionEnabled) {
            gl.uniform3fv(loc('uClipNormal'), this.sectionNormal);
            gl.uniform1f(loc('uClipOffset'), this.sectionOffset);
        }
    }

    handlePaintClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        if (!this.model || this._boundaryFaces.length === 0) return;

        const facesInBrush = this._findFacesInBrush(mx, my);

        // Determine target: active group or flat set
        const activeGroup = this.getActiveGroup();
        let targetSet;
        if (activeGroup) {
            targetSet = activeGroup.faces;
        } else if (this.paintMode === 'constraint') {
            targetSet = this.paintedConstraintFaces;
        } else if (this.paintMode === 'keep') {
            targetSet = this.paintedKeepFaces;
        } else {
            targetSet = this.paintedForceFaces;
        }

        if (this.useAngleSelection && facesInBrush.length > 0 && !this._paintErasing) {
            // Angle-based selection: use first hit as seed, flood-fill
            const seedFace = facesInBrush[0];
            this._seedFaceKey = seedFace.key;
            this._seedFaceNormal = seedFace.normal;
            const selected = this.selectConnectedByAngle(seedFace.key, this.angleTolerance);
            targetSet.clear();
            for (const key of selected) {
                targetSet.add(key);
            }
        } else {
            for (const face of facesInBrush) {
                if (this._paintErasing) {
                    targetSet.delete(face.key);
                } else {
                    targetSet.add(face.key);
                }
            }
        }

        if (activeGroup) {
            this._syncGroupsToFaceSets();
        }

        if (facesInBrush.length > 0) {
            this._needsRebuild = true;
            this.draw();
        }
    }

    _updateHover(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        if (!this.model || this._boundaryFaces.length === 0) {
            if (this._hoverFaces.length > 0) {
                this._hoverFaces = [];
                this.draw();
            }
            return;
        }

        this._hoverFaces = this._findFacesInBrush(mx, my);
        this.draw();
    }

    /**
     * Update stress hover tooltip by picking the closest mesh triangle under cursor.
     */
    _updateStressHover(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        if (!this.model || !this._boundaryFaces || this._boundaryFaces.length === 0) {
            if (this._hoverStressValue !== null) {
                this._hoverStressValue = null;
                this._hoverScreenPos = null;
                this.draw();
            }
            return;
        }

        const { nx, ny, nz } = this.model;
        const { width, height } = this.canvas;
        const aspect = width / height;
        const proj = mat4Perspective(Math.PI / 4, aspect, 0.1, 1000);
        const mv = this._buildModelView(nx, ny, nz);

        // Find closest rendered boundary face center to cursor (normalized stress 0..1)
        let bestZ = Infinity;
        let closestStress = null;

        const stressMap = this._cachedStressMap || null;
        for (let i = 0; i < this._boundaryFaces.length; i++) {
            const face = this._boundaryFaces[i];
            const v = face.projVerts;
            const cx3d = (v[0].x + v[1].x + v[2].x + v[3].x) * 0.25;
            const cy3d = (v[0].y + v[1].y + v[2].y + v[3].y) * 0.25;
            const cz3d = (v[0].z + v[1].z + v[2].z + v[3].z) * 0.25;
            const sp = this._projectToScreen({ x: cx3d, y: cy3d, z: cz3d }, mv, proj, width, height);
            const dist = Math.sqrt((mx - sp.x) ** 2 + (my - sp.y) ** 2);
            if (dist < 20 && sp.z < bestZ) {
                bestZ = sp.z;
                if (stressMap && face.voxel) {
                    const [vx, vy, vz] = face.voxel;
                    const idx = vx + vy * nx + vz * nx * ny;
                    closestStress = stressMap[idx] || 0;
                } else {
                    closestStress = 0;
                }
            }
        }

        if (closestStress !== null) {
            this._hoverStressValue = closestStress;
            this._hoverScreenPos = { x: mx, y: my };
        } else {
            this._hoverStressValue = null;
            this._hoverScreenPos = null;
        }
        this.draw();
    }

    /**
     * Select connected faces by angle tolerance from a seed face.
     * Uses flood-fill: starting from seedKey, expand to adjacent boundary faces
     * whose normals are within angleTolerance degrees of the seed normal.
     */
    selectConnectedByAngle(seedKey, angleTolerance) {
        const seedFace = this._boundaryFaceMap[seedKey];
        if (!seedFace || !seedFace.normal) return new Set();

        const toleranceRad = (angleTolerance * Math.PI) / 180;
        const cosTolerance = Math.cos(toleranceRad);
        const seedNormal = seedFace.normal;

        const selected = new Set();
        const visited = new Set();
        const queue = [seedKey];

        while (queue.length > 0) {
            const currentKey = queue.shift();
            if (visited.has(currentKey)) continue;
            visited.add(currentKey);

            const face = this._boundaryFaceMap[currentKey];
            if (!face || !face.normal) continue;

            // Check angle between face normal and seed normal
            const dot = seedNormal[0] * face.normal[0] + seedNormal[1] * face.normal[1] + seedNormal[2] * face.normal[2];
            if (dot < cosTolerance) continue;

            selected.add(currentKey);

            // Find adjacent boundary faces (share an edge = same voxel axis-aligned neighbor)
            const [x, y, z] = face.voxel;
            const fi = face.faceIndex;

            // Adjacent faces: same face index on neighboring voxels, or adjacent face indices on same voxel
            const neighbors = this._getAdjacentFaceKeys(x, y, z, fi);
            for (const nk of neighbors) {
                if (!visited.has(nk)) {
                    queue.push(nk);
                }
            }
        }

        return selected;
    }

    /**
     * Get keys of adjacent boundary faces that share an edge with face (x,y,z,fi).
     */
    _getAdjacentFaceKeys(x, y, z, fi) {
        const keys = [];
        // For each face, determine which neighbor voxels share edges
        // fi: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
        const adjacencyMap = {
            0: [  // -X face: neighbors along Y and Z edges
                [x,y-1,z,0], [x,y+1,z,0], [x,y,z-1,0], [x,y,z+1,0],
                [x,y,z,2], [x,y,z,3], [x,y,z,4], [x,y,z,5]
            ],
            1: [  // +X face
                [x,y-1,z,1], [x,y+1,z,1], [x,y,z-1,1], [x,y,z+1,1],
                [x,y,z,2], [x,y,z,3], [x,y,z,4], [x,y,z,5]
            ],
            2: [  // -Y face
                [x-1,y,z,2], [x+1,y,z,2], [x,y,z-1,2], [x,y,z+1,2],
                [x,y,z,0], [x,y,z,1], [x,y,z,4], [x,y,z,5]
            ],
            3: [  // +Y face
                [x-1,y,z,3], [x+1,y,z,3], [x,y,z-1,3], [x,y,z+1,3],
                [x,y,z,0], [x,y,z,1], [x,y,z,4], [x,y,z,5]
            ],
            4: [  // -Z face
                [x-1,y,z,4], [x+1,y,z,4], [x,y-1,z,4], [x,y+1,z,4],
                [x,y,z,0], [x,y,z,1], [x,y,z,2], [x,y,z,3]
            ],
            5: [  // +Z face
                [x-1,y,z,5], [x+1,y,z,5], [x,y-1,z,5], [x,y+1,z,5],
                [x,y,z,0], [x,y,z,1], [x,y,z,2], [x,y,z,3]
            ]
        };

        for (const [nx, ny, nz, nfi] of adjacencyMap[fi]) {
            const key = `${nx},${ny},${nz},${nfi}`;
            if (this._boundaryFaceMap[key]) {
                keys.push(key);
            }
        }
        return keys;
    }

    /**
     * Update angle-based selection: re-run flood-fill from the stored seed face
     * with the current angle tolerance and update the active group or paint set.
     */
    updateAngleSelection() {
        if (!this._seedFaceKey || !this.useAngleSelection) return;

        const selected = this.selectConnectedByAngle(this._seedFaceKey, this.angleTolerance);

        if (this.activeGroupId !== null) {
            const group = this.selectionGroups.find(g => g.id === this.activeGroupId);
            if (group) {
                group.faces = selected;
                this._syncGroupsToFaceSets();
            }
        } else {
            const targetSet = this.paintMode === 'constraint' ? this.paintedConstraintFaces : this.paintedForceFaces;
            targetSet.clear();
            for (const key of selected) {
                targetSet.add(key);
            }
        }

        this._needsRebuild = true;
        this.draw();
    }

    // ─── Selection Shape Management ─────────────────────────────────────────

    addSelectionShape(shapeType) {
        const id = this._nextShapeId++;
        const nx = this.model ? this.model.nx : 10;
        const ny = this.model ? this.model.ny : 10;
        const nz = this.model ? this.model.nz : 10;
        const defaultSize = Math.max(nx, ny, nz) / 4;
        const shape = {
            id,
            shapeType: shapeType || 'cube',
            position: [nx / 2, ny / 2, nz / 2],
            size: [defaultSize, defaultSize, defaultSize],
            rotation: [0, 0, 0]
        };
        this.selectionShapes.push(shape);
        this._computeShapeHighlights();
        this.draw();
        return shape;
    }

    removeSelectionShape(shapeId) {
        this.selectionShapes = this.selectionShapes.filter(s => s.id !== shapeId);
        this._computeShapeHighlights();
        this.draw();
    }

    updateSelectionShape(shapeId, props) {
        const shape = this.selectionShapes.find(s => s.id === shapeId);
        if (!shape) return;
        Object.assign(shape, props);
        this._computeShapeHighlights();
        this.draw();
    }

    _computeShapeHighlights() {
        this._shapeHighlightFaces.clear();
        if (this.selectionShapes.length === 0) return;
        for (const bf of this._boundaryFaces) {
            const v = bf.projVerts;
            const cx = (v[0].x + v[1].x + v[2].x + v[3].x) / 4;
            const cy = (v[0].y + v[1].y + v[2].y + v[3].y) / 4;
            const cz = (v[0].z + v[1].z + v[2].z + v[3].z) / 4;
            for (const shape of this.selectionShapes) {
                if (this._isFaceInShape(cx, cy, cz, shape)) {
                    this._shapeHighlightFaces.add(bf.key);
                    break;
                }
            }
        }
    }

    _isFaceInShape(px, py, pz, shape) {
        const { position, size, rotation, shapeType } = shape;
        // Translate to shape-local origin
        let x = px - position[0];
        let y = py - position[1];
        let z = pz - position[2];

        // Build rotation matrix R = Rz * Ry * Rx (XYZ Euler order)
        const cosX = Math.cos(rotation[0] * Math.PI / 180);
        const sinX = Math.sin(rotation[0] * Math.PI / 180);
        const cosY = Math.cos(rotation[1] * Math.PI / 180);
        const sinY = Math.sin(rotation[1] * Math.PI / 180);
        const cosZ = Math.cos(rotation[2] * Math.PI / 180);
        const sinZ = Math.sin(rotation[2] * Math.PI / 180);

        const R00 = cosY * cosZ;
        const R10 = cosY * sinZ;
        const R20 = -sinY;
        const R01 = sinX * sinY * cosZ - cosX * sinZ;
        const R11 = sinX * sinY * sinZ + cosX * cosZ;
        const R21 = sinX * cosY;
        const R02 = cosX * sinY * cosZ + sinX * sinZ;
        const R12 = cosX * sinY * sinZ - sinX * cosZ;
        const R22 = cosX * cosY;

        // Apply inverse rotation (R^T) to transform to local space
        const lx = R00 * x + R10 * y + R20 * z;
        const ly = R01 * x + R11 * y + R21 * z;
        const lz = R02 * x + R12 * y + R22 * z;

        // Scale to unit space (guard against zero-size components)
        const ux = size[0] > 0 ? lx / size[0] : (lx === 0 ? 0 : Infinity);
        const uy = size[1] > 0 ? ly / size[1] : (ly === 0 ? 0 : Infinity);
        const uz = size[2] > 0 ? lz / size[2] : (lz === 0 ? 0 : Infinity);

        switch (shapeType) {
            case 'cube':
                return Math.abs(ux) <= 1 && Math.abs(uy) <= 1 && Math.abs(uz) <= 1;
            case 'sphere':
                return ux * ux + uy * uy + uz * uz <= 1;
            case 'cylinder':
                // Y-axis cylinder: circular cross-section in XZ, extent along Y
                return ux * ux + uz * uz <= 1 && Math.abs(uy) <= 1;
            default:
                return false;
        }
    }

    applyShapeSelectionToGroup(groupId, shapeId) {
        const targetId = groupId !== undefined ? groupId : this.activeGroupId;
        const group = this.selectionGroups.find(g => g.id === targetId);
        if (!group) return;

        if (shapeId !== undefined) {
            // Apply only the faces from the specified shape
            const shape = this.selectionShapes.find(s => s.id === shapeId);
            if (!shape) return;
            for (const bf of this._boundaryFaces) {
                const v = bf.projVerts;
                const cx = (v[0].x + v[1].x + v[2].x + v[3].x) / 4;
                const cy = (v[0].y + v[1].y + v[2].y + v[3].y) / 4;
                const cz = (v[0].z + v[1].z + v[2].z + v[3].z) / 4;
                if (this._isFaceInShape(cx, cy, cz, shape)) {
                    group.faces.add(bf.key);
                }
            }
        } else {
            // Apply all shape highlights
            this._computeShapeHighlights();
            for (const key of this._shapeHighlightFaces) {
                group.faces.add(key);
            }
        }

        this._syncGroupsToFaceSets();
        this._needsRebuild = true;
        this.draw();
    }

    _generateShapeGeometry(shape) {
        const { shapeType, position, size, rotation } = shape;
        const unitPos = [];
        const unitNorm = [];

        if (shapeType === 'cube') {
            const faceData = [
                { n: [0, 0, 1],  v: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]] },
                { n: [0, 0, -1], v: [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]] },
                { n: [-1, 0, 0], v: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]] },
                { n: [1, 0, 0],  v: [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]] },
                { n: [0, 1, 0],  v: [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]] },
                { n: [0, -1, 0], v: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]] }
            ];
            for (const { n, v } of faceData) {
                const [v0, v1, v2, v3] = v;
                unitPos.push(...v0, ...v1, ...v2, ...v0, ...v2, ...v3);
                unitNorm.push(...n, ...n, ...n, ...n, ...n, ...n);
            }
        } else if (shapeType === 'sphere') {
            const segs = 16;
            for (let lat = 0; lat < segs; lat++) {
                const t0 = (lat / segs) * Math.PI;
                const t1 = ((lat + 1) / segs) * Math.PI;
                for (let lon = 0; lon < segs; lon++) {
                    const p0 = (lon / segs) * 2 * Math.PI;
                    const p1 = ((lon + 1) / segs) * 2 * Math.PI;
                    const v00 = [Math.sin(t0) * Math.cos(p0), Math.cos(t0), Math.sin(t0) * Math.sin(p0)];
                    const v01 = [Math.sin(t0) * Math.cos(p1), Math.cos(t0), Math.sin(t0) * Math.sin(p1)];
                    const v10 = [Math.sin(t1) * Math.cos(p0), Math.cos(t1), Math.sin(t1) * Math.sin(p0)];
                    const v11 = [Math.sin(t1) * Math.cos(p1), Math.cos(t1), Math.sin(t1) * Math.sin(p1)];
                    unitPos.push(...v00, ...v10, ...v11, ...v00, ...v11, ...v01);
                    unitNorm.push(...v00, ...v10, ...v11, ...v00, ...v11, ...v01);
                }
            }
        } else if (shapeType === 'cylinder') {
            const segs = 32;
            for (let i = 0; i < segs; i++) {
                const a0 = (i / segs) * 2 * Math.PI;
                const a1 = ((i + 1) / segs) * 2 * Math.PI;
                const x0 = Math.cos(a0), z0 = Math.sin(a0);
                const x1 = Math.cos(a1), z1 = Math.sin(a1);
                // Side
                unitPos.push(x0, -1, z0, x1, -1, z1, x1, 1, z1, x0, -1, z0, x1, 1, z1, x0, 1, z0);
                unitNorm.push(x0, 0, z0, x1, 0, z1, x1, 0, z1, x0, 0, z0, x1, 0, z1, x0, 0, z0);
                // Top cap
                unitPos.push(0, 1, 0, x0, 1, z0, x1, 1, z1);
                unitNorm.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
                // Bottom cap
                unitPos.push(0, -1, 0, x1, -1, z1, x0, -1, z0);
                unitNorm.push(0, -1, 0, 0, -1, 0, 0, -1, 0);
            }
        }

        // Build rotation matrix R = Rz * Ry * Rx
        const cosX = Math.cos(rotation[0] * Math.PI / 180);
        const sinX = Math.sin(rotation[0] * Math.PI / 180);
        const cosY = Math.cos(rotation[1] * Math.PI / 180);
        const sinY = Math.sin(rotation[1] * Math.PI / 180);
        const cosZ = Math.cos(rotation[2] * Math.PI / 180);
        const sinZ = Math.sin(rotation[2] * Math.PI / 180);

        const R00 = cosY * cosZ;
        const R10 = cosY * sinZ;
        const R20 = -sinY;
        const R01 = sinX * sinY * cosZ - cosX * sinZ;
        const R11 = sinX * sinY * sinZ + cosX * cosZ;
        const R21 = sinX * cosY;
        const R02 = cosX * sinY * cosZ + sinX * sinZ;
        const R12 = cosX * sinY * sinZ - sinX * cosZ;
        const R22 = cosX * cosY;

        const count = unitPos.length / 3;
        const worldPos = new Float32Array(unitPos.length);
        const worldNorm = new Float32Array(unitNorm.length);

        for (let i = 0; i < unitPos.length; i += 3) {
            // Scale then rotate then translate
            const sx = unitPos[i] * size[0];
            const sy = unitPos[i + 1] * size[1];
            const sz = unitPos[i + 2] * size[2];
            worldPos[i]     = R00 * sx + R01 * sy + R02 * sz + position[0];
            worldPos[i + 1] = R10 * sx + R11 * sy + R12 * sz + position[1];
            worldPos[i + 2] = R20 * sx + R21 * sy + R22 * sz + position[2];

            // Normals: only rotate (no scale, no translate)
            const nx = unitNorm[i], ny = unitNorm[i + 1], nz = unitNorm[i + 2];
            worldNorm[i]     = R00 * nx + R01 * ny + R02 * nz;
            worldNorm[i + 1] = R10 * nx + R11 * ny + R12 * nz;
            worldNorm[i + 2] = R20 * nx + R21 * ny + R22 * nz;
        }

        return { positions: worldPos, normals: worldNorm, count };
    }

    _drawSelectionShapes(gl, projection, modelView, normalMatrix) {
        if (this.selectionShapes.length === 0) return;

        const prog = this._meshProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uNormalMatrix'), false, normalMatrix);

        const lightDir = vec3Normalize([0.3, 0.5, 0.8]);
        gl.uniform3fv(gl.getUniformLocation(prog, 'uLightDir'), lightDir);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAmbient'), 0.5);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.3);
        gl.uniform1f(gl.getUniformLocation(prog, 'uClipEnabled'), 0.0);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const normLoc = gl.getAttribLocation(prog, 'aNormal');
        const colLoc = gl.getAttribLocation(prog, 'aColor');

        const shapeColors = { cube: [0.2, 0.7, 1.0], sphere: [1.0, 0.7, 0.2], cylinder: [0.2, 1.0, 0.5] };

        const posBuf = gl.createBuffer();
        const normBuf = gl.createBuffer();
        const colBuf = gl.createBuffer();

        for (const shape of this.selectionShapes) {
            const { positions, normals, count } = this._generateShapeGeometry(shape);
            const color = shapeColors[shape.shapeType] || [0.5, 0.5, 1.0];
            const colors = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                colors[i * 3] = color[0]; colors[i * 3 + 1] = color[1]; colors[i * 3 + 2] = color[2];
            }

            gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
            gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

            gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
            gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(normLoc);
            gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

            gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
            gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(colLoc);
            gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

            gl.drawArrays(gl.TRIANGLES, 0, count);
        }

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(normLoc);
        gl.disableVertexAttribArray(colLoc);

        gl.deleteBuffer(posBuf);
        gl.deleteBuffer(normBuf);
        gl.deleteBuffer(colBuf);

        gl.depthMask(true);
        gl.enable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.0);
    }

    // ─── Selection Group Management ─────────────────────────────────────────

    addSelectionGroup(type, name) {
        const id = this._nextGroupId++;
        const group = {
            id,
            name: name || `${type} ${id}`,
            type, // 'force', 'constraint', or 'keep'
            faces: new Set(),
            params: type === 'force'
                ? { direction: 'down', vector: null, magnitude: 1000, forceType: 'total', dofs: 'all' }
                : { dofs: 'all' }
        };
        this.selectionGroups.push(group);
        this.activeGroupId = group.id;
        return group;
    }

    removeSelectionGroup(groupId) {
        this.selectionGroups = this.selectionGroups.filter(g => g.id !== groupId);
        if (this.activeGroupId === groupId) {
            this.activeGroupId = this.selectionGroups.length > 0 ? this.selectionGroups[0].id : null;
        }
        this._syncGroupsToFaceSets();
        this._needsRebuild = true;
        this.draw();
    }

    setActiveGroup(groupId) {
        this.activeGroupId = groupId;
    }

    getActiveGroup() {
        return this.selectionGroups.find(g => g.id === this.activeGroupId) || null;
    }

    /**
     * Sync selection groups back to the flat paintedConstraintFaces/paintedForceFaces sets
     * for rendering compatibility.
     */
    _syncGroupsToFaceSets() {
        this.paintedConstraintFaces.clear();
        this.paintedForceFaces.clear();
        this.paintedKeepFaces.clear();
        for (const group of this.selectionGroups) {
            let targetSet;
            if (group.type === 'constraint') targetSet = this.paintedConstraintFaces;
            else if (group.type === 'force') targetSet = this.paintedForceFaces;
            else targetSet = this.paintedKeepFaces;
            for (const key of group.faces) {
                targetSet.add(key);
            }
        }
    }

    _findFacesInBrush(mx, my) {
        if (!this.model || this._boundaryFaces.length === 0) return [];

        const { nx, ny, nz } = this.model;
        const { width, height } = this.canvas;
        const aspect = width / height;
        const proj = mat4Perspective(Math.PI / 4, aspect, 0.1, 1000);
        const mv = this._buildModelView(nx, ny, nz);

        // Step 1: Find the closest face to cursor (depth-prioritized)
        let bestDist = Infinity;
        let centerFace = null;
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
                    centerFace = bf;
                    bestZ = avgZ;
                }
            }
        }

        if (!centerFace) return [];
        if (this.brushSize <= 1) return [centerFace];

        // Step 2: Find all faces within brushSize voxel units in 3D from center face
        const cv = centerFace.projVerts;
        const centerX = (cv[0].x + cv[1].x + cv[2].x + cv[3].x) / 4;
        const centerY = (cv[0].y + cv[1].y + cv[2].y + cv[3].y) / 4;
        const centerZ = (cv[0].z + cv[1].z + cv[2].z + cv[3].z) / 4;

        const brushRadiusSq = this.brushSize * this.brushSize;
        const result = [];

        for (const bf of this._boundaryFaces) {
            const v = bf.projVerts;
            const fx = (v[0].x + v[1].x + v[2].x + v[3].x) / 4;
            const fy = (v[0].y + v[1].y + v[2].y + v[3].y) / 4;
            const fz = (v[0].z + v[1].z + v[2].z + v[3].z) / 4;

            const dx = fx - centerX;
            const dy = fy - centerY;
            const dz = fz - centerZ;
            const distSq = dx * dx + dy * dy + dz * dz;

            if (distSq <= brushRadiusSq) {
                // Depth check: only include faces on the same side (not through the model)
                const sv = bf.projVerts.map(pv =>
                    this._projectToScreen(pv, mv, proj, width, height)
                );
                const faceZ = (sv[0].z + sv[1].z + sv[2].z + sv[3].z) / 4;
                if (Math.abs(faceZ - bestZ) < BRUSH_DEPTH_TOLERANCE) {
                    result.push(bf);
                }
            }
        }

        return result;
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
        this.amrCells = null;
        this._volumetricStressMap = null;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this._needsRebuild = true;

        // Store original mesh vertices as reference model if available
        if (model && model.originalVertices && model.originalVertices.length >= 3) {
            this.setReferenceModel(model.originalVertices, model.bounds);
        } else {
            this.referenceVertices = null;
            this._referenceBuffers = null;
        }

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
        if (this.meshVisible && this._meshBuffers && this._meshBuffers.count > 0) {
            if (this.sectionEnabled) gl.disable(gl.CULL_FACE);
            this._drawMesh(gl, projection, modelView, normalMatrix);
            if (this.sectionEnabled) gl.enable(gl.CULL_FACE);
        }

        // Draw edges: always for AMR triangle mesh (to show block boundaries), wireframe-only otherwise
        if (this.meshVisible && this._edgeBuffers && this._edgeBuffers.count > 0 && (this.wireframe || this.meshData)) {
            this._drawEdges(gl, projection, modelView);
        }

        // Draw reference model (original high-res STL/STEP mesh) as semi-transparent overlay
        if (!this.sectionEnabled && this.showReference && this._referenceBuffers && this._referenceBuffers.count > 0) {
            this._drawReference(gl, projection, modelView, normalMatrix);
        }

        // Draw selection shapes (semi-transparent 3D primitives)
        this._drawSelectionShapes(gl, projection, modelView, normalMatrix);

        // Draw section plane visualization and cross-section cap
        if (this.sectionEnabled) {
            this._drawSectionCap(gl, projection, modelView, normalMatrix, nx, ny, nz);
            this._drawSectionPlane(gl, projection, modelView, nx, ny, nz);
        }

        // Draw overlays
        this._drawOverlays(gl, projection, modelView, nx, ny, nz);

        // Draw hover highlight for brush preview
        if (this.paintMode && this._hoverFaces.length > 0) {
            this._drawHoverHighlight(gl, projection, modelView);
        }

        // Draw 2D overlay
        this._drawAxesOverlay();

        // Draw vertical stress scale bar when mesh data is available
        if (this.meshData && this.meshData.length > 0) {
            this._drawStressScale(width, height);
        }

        // Draw stress hover tooltip
        if (this._hoverStressValue !== null && this._hoverScreenPos) {
            this._drawStressTooltip();
        }

        if (this.paintMode) {
            const paintColors = { constraint: 'rgba(0,200,100,0.8)', force: 'rgba(255,100,50,0.8)', keep: 'rgba(50,100,255,0.8)' };
            this.ctx.fillStyle = paintColors[this.paintMode] || 'rgba(255,100,50,0.8)';
            this.ctx.font = '14px Arial';
            this.ctx.textAlign = 'left';
            const paintLabels = {
                constraint: '🖌 Painting Constraints (Right-click to remove)',
                force: '🖌 Painting Forces (Right-click to remove)',
                keep: '🔒 Painting Keep Region (Right-click to remove)'
            };
            const label = paintLabels[this.paintMode] || paintLabels.force;
            this.ctx.fillText(label, 10, height - 10);
        }

        if (this.sectionEnabled) {
            this.ctx.fillStyle = 'rgba(100,180,255,0.9)';
            this.ctx.font = '13px Arial';
            this.ctx.textAlign = 'left';
            this.ctx.fillText('✂ Section View  (Alt+Drag: rotate, Alt+Scroll: move)', 10, 20);
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
        // Build voxel density/stress maps shared by all render paths.
        const total = nx * ny * nz;
        const { elements } = this.model;
        const densityMap = new Float32Array(total);
        for (let i = 0; i < total; i++) {
            densityMap[i] = this.densities ? this.densities[i] : elements[i];
        }

        // Stress map in normalized range [0..1], default 0 when unavailable.
        const stressSum = new Float32Array(total);
        const stressCount = new Uint16Array(total);
        if (this.meshData && this.meshData.length > 0) {
            for (const tri of this.meshData) {
                const v = tri.vertices[0];
                const ex = Math.min(Math.max(Math.floor(v[0]), 0), nx - 1);
                const ey = Math.min(Math.max(Math.floor(v[1]), 0), ny - 1);
                const ez = Math.min(Math.max(Math.floor(v[2]), 0), nz - 1);
                const idx = ex + ey * nx + ez * nx * ny;

                if (tri.density !== undefined) densityMap[idx] = tri.density;

                const stress = tri.strain !== undefined ? tri.strain : 0;
                stressSum[idx] += stress;
                stressCount[idx] += 1;
            }
        }
        let stressMap = new Float32Array(total);
        if (this._volumetricStressMap && this._volumetricStressMap.length === total) {
            stressMap = this._volumetricStressMap;
        } else {
            for (let i = 0; i < total; i++) {
                stressMap[i] = stressCount[i] > 0 ? (stressSum[i] / stressCount[i]) : 0;
            }
        }

        // Build visibility mask (shared with section cap/plane)
        const visible = new Uint8Array(total);
        const hasStrainFilter = this.strainMin > 0 || this.strainMax < 1;

        for (let i = 0; i < total; i++) {
            if (densityMap[i] <= this.densityThreshold) continue;
            if (hasStrainFilter) {
                const stress = stressMap[i]; // 0 when null/unavailable
                if (stress < this.strainMin || stress > this.strainMax) continue;
            }
            visible[i] = 1;
        }

        // Cache for section cap/plane to reuse
        this._cachedVisible = visible;
        this._cachedDensityMap = densityMap;
        this._cachedStressMap = stressMap;

        // Unified rendering path: AMR boundary-face surface mesh for all modes.
        this._generateClosedMeshBuffers(this.gl, nx, ny, nz, visible, densityMap, stressMap);
        this._buildGridBuffers(nx, ny, nz);
        this._buildReferenceBuffers();
        // Update shape highlight faces after boundary faces are rebuilt
        this._computeShapeHighlights();
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
        const elementStress = new Float32Array(nx * ny * nz);

        for (const tri of this.meshData) {
            const v = tri.vertices[0];
            let ex = Math.min(Math.max(Math.floor(v[0]), 0), nx - 1);
            let ey = Math.min(Math.max(Math.floor(v[1]), 0), ny - 1);
            let ez = Math.min(Math.max(Math.floor(v[2]), 0), nz - 1);

            const idx = ex + ey * nx + ez * nx * ny;
            elementDensity[idx] = tri.density;
            const strain = tri.strain !== undefined ? tri.strain : 0;
            elementStress[idx] = strain;

            if (hasStrainFilter) {
                if (strain >= this.strainMin && strain <= this.strainMax) {
                    visibleElements[idx] = 1;
                }
            } else {
                if (tri.density > this.densityThreshold) {
                    visibleElements[idx] = 1;
                }
            }
        }

        // Generate closed mesh from visible elements
        this._generateClosedMeshBuffers(gl, nx, ny, nz, visibleElements, elementDensity, elementStress);
    }

    _buildDirectTriangleMeshBuffers(gl, applyStrainFilter = false) {
        const positions = [];
        const normals = [];
        const colors = [];
        const edgePositions = [];
        const edgeColors = [];

        this._boundaryFaces = [];
        this._boundaryFaceMap = {};
        // For merged AMR blocks (blockSize > 1) emit the 4 perimeter edges of the quad.
        // For individual-element triangles (blockSize === 1) emit no edges (no visible grid clutter).
        for (let ti = 0; ti < this.meshData.length; ti++) {
            const tri = this.meshData[ti];
            const density = tri.density !== undefined ? tri.density : DEFAULT_TRIANGLE_DENSITY;
            if (density <= this.densityThreshold) continue;

            if (applyStrainFilter) {
                const strain = tri.strain !== undefined ? tri.strain : 0;
                if (strain < this.strainMin || strain > this.strainMax) continue;
            }

            const n = tri.normal || [0, 0, 1];
            const r = density;
            const g = DENSITY_COLOR_GREEN;
            const b = 1 - density;

            // Flip winding for worker triangles so outward faces are front faces (CCW)
            positions.push(...tri.vertices[0], ...tri.vertices[2], ...tri.vertices[1]);
            normals.push(...n, ...n, ...n);
            colors.push(r, g, b, r, g, b, r, g, b);

            // Emit block boundary edges only for merged AMR quads.
            // Each face is emitted as two consecutive triangles: (v0,v1,v2) and (v0,v2,v3).
            // Perimeter edges are v0→v1, v1→v2 on the first triangle, and v2→v3, v3→v0
            // on the second (i.e. v2→v3 and v3→v0 where v3 = second tri's v2).
            const blockSize = tri.blockSize !== undefined ? tri.blockSize : 1;
            if (blockSize > 1) {
                const next = this.meshData[ti + 1];
                if (next && ti % 2 === 0) {
                    // v0, v1, v2 from first tri; v3 = next.vertices[2] (4th corner)
                    const v0 = tri.vertices[0], v1 = tri.vertices[1], v2 = tri.vertices[2];
                    const v3 = next.vertices[2];
                    // 4 perimeter edges: v0→v1, v1→v2, v2→v3, v3→v0
                    edgePositions.push(...v0, ...v1, ...v1, ...v2, ...v2, ...v3, ...v3, ...v0);
                    edgeColors.push(...Array(4 * 2 * 3).fill(WIREFRAME_EDGE_COLOR)); // 4 edges × 2 vertices × 3 RGB
                }
            }
        }

        this._uploadMeshBuffers(gl, positions, normals, colors);
        this._uploadEdgeBuffers(gl, edgePositions, edgeColors);
    }

    _buildVoxelBuffers(nx, ny, nz) {
        const gl = this.gl;
        const { elements } = this.model;

        const densityMap = new Float32Array(nx * ny * nz);
        for (let i = 0; i < nx * ny * nz; i++) {
            densityMap[i] = this.densities ? this.densities[i] : elements[i];
        }
        const stressMap = this._cachedStressMap || new Float32Array(nx * ny * nz);

        // Use AMR surface mesh module for boundary-face extraction
        const mesh = generateUniformSurfaceMesh({
            densities: densityMap,
            nx, ny, nz,
            threshold: this.densityThreshold
        });

        this._buildBuffersFromIndexedMesh(gl, mesh, nx, ny, nz, densityMap, stressMap);
    }

    /**
     * Core mesh generation: creates a watertight closed surface from a visibility grid.
     * Delegates boundary-face extraction to the AMR surface mesh module.
     */
    _generateClosedMeshBuffers(gl, nx, ny, nz, visibleElements, densityMap, stressMap) {
        let mesh;

        if (this.amrCells && this.amrCells.length > 0) {
            const hasStrainFilter = this.strainMin > 0 || this.strainMax < 1;
            const filteredCells = [];
            for (const cell of this.amrCells) {
                const d = cell.density !== undefined ? cell.density : 0;
                if (d <= this.densityThreshold) continue;
                const s = cell.stress !== undefined ? cell.stress : 0;
                if (hasStrainFilter && (s < this.strainMin || s > this.strainMax)) continue;
                filteredCells.push(cell);
            }

            mesh = generateAMRSurfaceMesh({
                cells: filteredCells,
                threshold: this.densityThreshold
            });
        } else {
            // Uniform fallback with pre-built visibility mask
            mesh = generateUniformSurfaceMesh({
                densities: densityMap,
                nx, ny, nz,
                visible: visibleElements
            });
        }

        this._buildBuffersFromIndexedMesh(gl, mesh, nx, ny, nz, densityMap, stressMap);
    }

    /**
     * Convert an indexed surface mesh (from the AMR surface mesh module) into
     * flat WebGL buffers with per-face colors, edge lines, and picking cache.
     */
    _buildBuffersFromIndexedMesh(gl, mesh, nx, ny, nz, densityMap, stressMap) {
        const { positions: meshPositions, indices } = mesh;

        const positions = [];
        const normals = [];
        const colors = [];
        const edgePositions = [];
        const edgeColors = [];

        this._boundaryFaces = [];
        this._boundaryFaceMap = {};

        // Face normals for face-index lookup
        const FACE_NORMALS = [
            [-1, 0, 0], [1, 0, 0],   // -X, +X
            [0, -1, 0], [0, 1, 0],   // -Y, +Y
            [0, 0, -1], [0, 0, 1]    // -Z, +Z
        ];

        // Displacement lookup helper.
        // The AMR/uniform surface mesh generators place all vertices at integer voxel-grid
        // coordinates (they are always at voxel corners = FEA node positions), so
        // Math.round() is an exact lookup with no interpolation error.
        const dispEnabled = this.showDisplacement && this.displacementData && this.displacementData.U;
        const dispU = dispEnabled ? this.displacementData.U : null;
        const dispScale = this.displacementScale || 1;
        const dispNny = ny + 1, dispNnz = nz + 1;

        const _applyDisp = dispEnabled ? (px, py, pz) => {
            const ix = Math.min(Math.max(Math.round(px), 0), nx);
            const iy = Math.min(Math.max(Math.round(py), 0), ny);
            const iz = Math.min(Math.max(Math.round(pz), 0), nz);
            const nodeIdx = ix * dispNny * dispNnz + iy * dispNnz + iz;
            return [
                px + dispU[3 * nodeIdx] * dispScale,
                py + dispU[3 * nodeIdx + 1] * dispScale,
                pz + dispU[3 * nodeIdx + 2] * dispScale
            ];
        } : (px, py, pz) => [px, py, pz];

        // Process triangle pairs (each quad = 2 triangles = 6 indices)
        for (let qi = 0; qi < indices.length; qi += 6) {
            // Quad indices: tri1 = (i0, i1, i2), tri2 = (i0, i2, i3)
            const a0 = indices[qi], a1 = indices[qi + 1], a2 = indices[qi + 2];
            const a5 = indices[qi + 5]; // Fourth corner of quad

            // Quad corners (with optional displacement applied)
            const v0 = _applyDisp(meshPositions[a0 * 3], meshPositions[a0 * 3 + 1], meshPositions[a0 * 3 + 2]);
            const v1 = _applyDisp(meshPositions[a1 * 3], meshPositions[a1 * 3 + 1], meshPositions[a1 * 3 + 2]);
            const v2 = _applyDisp(meshPositions[a2 * 3], meshPositions[a2 * 3 + 1], meshPositions[a2 * 3 + 2]);
            const v3 = _applyDisp(meshPositions[a5 * 3], meshPositions[a5 * 3 + 1], meshPositions[a5 * 3 + 2]);

            // Face normal from cross product (flat shading)
            const e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
            const e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];
            let fnx = e1y * e2z - e1z * e2y;
            let fny = e1z * e2x - e1x * e2z;
            let fnz = e1x * e2y - e1y * e2x;
            const flen = Math.sqrt(fnx * fnx + fny * fny + fnz * fnz);
            if (flen > 0) { fnx /= flen; fny /= flen; fnz /= flen; }

            // Find which voxel this face belongs to using un-displaced positions:
            // Face center minus half-normal step lands inside the active voxel
            const cx = (meshPositions[a0*3] + meshPositions[a1*3] + meshPositions[a2*3] + meshPositions[a5*3]) * 0.25;
            const cy = (meshPositions[a0*3+1] + meshPositions[a1*3+1] + meshPositions[a2*3+1] + meshPositions[a5*3+1]) * 0.25;
            const cz = (meshPositions[a0*3+2] + meshPositions[a1*3+2] + meshPositions[a2*3+2] + meshPositions[a5*3+2]) * 0.25;
            const vx = Math.min(Math.max(Math.floor(cx - fnx * 0.5), 0), nx - 1);
            const vy = Math.min(Math.max(Math.floor(cy - fny * 0.5), 0), ny - 1);
            const vz = Math.min(Math.max(Math.floor(cz - fnz * 0.5), 0), nz - 1);
            const cellIdx = vx + vy * nx + vz * nx * ny;
            const stressRaw = stressMap ? stressMap[cellIdx] : 0;
            const stress = Math.max(0, Math.min(1, stressRaw || 0));

            // Per-face color from closest voxel stress (0 when unavailable)
            // Below yield: navy-blue → red gradient; above yield: bright purple
            let r, g, b;
            const yieldNorm = this._yieldNormalized;
            if (yieldNorm > 0 && yieldNorm < 1 && stress > yieldNorm) {
                r = 0.75;
                g = 0;
                b = 1.0;
            } else {
                const t = yieldNorm > 0 && yieldNorm < 1 ? stress / yieldNorm : stress;
                r = t;
                g = DENSITY_COLOR_GREEN;
                b = 0.5 * (1 - t);
            }

            // Two triangles for this quad
            positions.push(...v0, ...v1, ...v2, ...v0, ...v2, ...v3);
            normals.push(fnx, fny, fnz, fnx, fny, fnz, fnx, fny, fnz,
                         fnx, fny, fnz, fnx, fny, fnz, fnx, fny, fnz);
            colors.push(r, g, b, r, g, b, r, g, b,
                        r, g, b, r, g, b, r, g, b);

            // Edge lines (4 edges of the quad)
            edgePositions.push(...v0, ...v1, ...v1, ...v2, ...v2, ...v3, ...v3, ...v0);
            edgeColors.push(0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                            0.2, 0.2, 0.2, 0.2, 0.2, 0.2);

            // Determine face index from normal for picking
            let fi = 0;
            let bestDot = -2;
            for (let f = 0; f < 6; f++) {
                const d = fnx * FACE_NORMALS[f][0] + fny * FACE_NORMALS[f][1] + fnz * FACE_NORMALS[f][2];
                if (d > bestDot) { bestDot = d; fi = f; }
            }

            // Cache boundary face for paint picking
            const key = `${vx},${vy},${vz},${fi}`;
            const faceObj = {
                key,
                projVerts: [v0, v1, v2, v3].map(v => ({ x: v[0], y: v[1], z: v[2] })),
                normal: [fnx, fny, fnz],
                voxel: [vx, vy, vz],
                faceIndex: fi,
                avgDepth: 0
            };
            this._boundaryFaces.push(faceObj);
            this._boundaryFaceMap[key] = faceObj;
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

    /**
     * Build GPU buffers for the original high-res reference mesh.
     * The reference mesh is the imported STL/STEP triangle surface,
     * rendered as a semi-transparent overlay.
     */
    _buildReferenceBuffers() {
        const gl = this.gl;
        if (this._referenceBuffers) {
            gl.deleteBuffer(this._referenceBuffers.position);
            gl.deleteBuffer(this._referenceBuffers.normal);
            gl.deleteBuffer(this._referenceBuffers.color);
            this._referenceBuffers = null;
        }

        if (!this.referenceVertices || this.referenceVertices.length < 3) return;

        const verts = this.referenceVertices;
        const model = this.model;
        if (!model) return;

        const bounds = this._referenceBounds || model.bounds;
        if (!bounds) return;

        const { nx, ny, nz } = model;
        const voxelSize = model.voxelSize || 1;

        // Transform original mesh coordinates to voxel grid coordinates
        const positions = [];
        const normals = [];
        const colors = [];
        const refColor = [0.5, 0.7, 0.9]; // Light blue-grey reference color

        const numTris = Math.floor(verts.length / 3);
        for (let t = 0; t < numTris; t++) {
            const v0 = verts[t * 3];
            const v1 = verts[t * 3 + 1];
            const v2 = verts[t * 3 + 2];

            // Transform to voxel space: (v - min) / voxelSize
            const p0 = [(v0.x - bounds.minX) / voxelSize, (v0.y - bounds.minY) / voxelSize, (v0.z - bounds.minZ) / voxelSize];
            const p1 = [(v1.x - bounds.minX) / voxelSize, (v1.y - bounds.minY) / voxelSize, (v1.z - bounds.minZ) / voxelSize];
            const p2 = [(v2.x - bounds.minX) / voxelSize, (v2.y - bounds.minY) / voxelSize, (v2.z - bounds.minZ) / voxelSize];

            // Compute face normal
            const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            const e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let fnx = e1[1] * e2[2] - e1[2] * e2[1];
            let fny = e1[2] * e2[0] - e1[0] * e2[2];
            let fnz = e1[0] * e2[1] - e1[1] * e2[0];
            const len = Math.sqrt(fnx * fnx + fny * fny + fnz * fnz);
            if (len > 0) { fnx /= len; fny /= len; fnz /= len; }

            positions.push(...p0, ...p1, ...p2);
            normals.push(fnx, fny, fnz, fnx, fny, fnz, fnx, fny, fnz);
            colors.push(...refColor, ...refColor, ...refColor);
        }

        if (positions.length === 0) return;

        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        const normBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
        const colBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

        this._referenceBuffers = {
            position: posBuf,
            normal: normBuf,
            color: colBuf,
            count: positions.length / 3
        };
    }

    /**
     * Draw the reference mesh as a semi-transparent overlay.
     */
    _drawReference(gl, projection, modelView, normalMatrix) {
        if (!this._referenceBuffers || this._referenceBuffers.count === 0) return;

        const prog = this._meshProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uNormalMatrix'), false, normalMatrix);

        const lightDir = vec3Normalize([0.3, 0.5, 0.8]);
        gl.uniform3fv(gl.getUniformLocation(prog, 'uLightDir'), lightDir);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAmbient'), 0.5);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.25);
        this._setClipUniforms(gl, prog);

        // Enable blending for transparency
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const normLoc = gl.getAttribLocation(prog, 'aNormal');
        const colLoc = gl.getAttribLocation(prog, 'aColor');

        gl.bindBuffer(gl.ARRAY_BUFFER, this._referenceBuffers.position);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._referenceBuffers.normal);
        gl.enableVertexAttribArray(normLoc);
        gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._referenceBuffers.color);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, this._referenceBuffers.count);

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(normLoc);
        gl.disableVertexAttribArray(colLoc);

        // Restore state
        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.0);
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
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.0);
        this._setClipUniforms(gl, prog);

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
        this._setClipUniforms(gl, prog);

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
        // Don't clip the ground grid
        gl.uniform1f(gl.getUniformLocation(prog, 'uClipEnabled'), 0.0);

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
        const hasKeep = this.paintedKeepFaces.size > 0;
        const hasShapeHL = this._shapeHighlightFaces.size > 0;
        if (!hasCon && !hasForce && !hasKeep && !hasShapeHL) return;

        const prog = this._overlayProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        this._setClipUniforms(gl, prog);

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

        if (hasKeep) {
            const positions = [];
            for (const bf of this._boundaryFaces) {
                if (!this.paintedKeepFaces.has(bf.key)) continue;
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
                gl.uniform4fv(gl.getUniformLocation(prog, 'uColor'), [0.2, 0.4, 1.0, 0.45]);
                gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
                gl.disableVertexAttribArray(posLoc);
                gl.deleteBuffer(buf);
            }
        }

        if (hasShapeHL) {
            const positions = [];
            for (const bf of this._boundaryFaces) {
                if (!this._shapeHighlightFaces.has(bf.key)) continue;
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
                gl.uniform4fv(gl.getUniformLocation(prog, 'uColor'), [1.0, 0.85, 0.1, 0.55]);
                gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
                gl.disableVertexAttribArray(posLoc);
                gl.deleteBuffer(buf);
            }
        }

        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
    }

    _drawSectionPlane(gl, projection, modelView, nx, ny, nz) {
        const n = this.sectionNormal;
        const d = this.sectionOffset;

        // Intersect section plane with model bounding box so border is perfectly aligned.
        const boxCorners = [
            [0, 0, 0], [nx, 0, 0], [0, ny, 0], [nx, ny, 0],
            [0, 0, nz], [nx, 0, nz], [0, ny, nz], [nx, ny, nz]
        ];
        const boxEdges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];

        const dists = new Float64Array(8);
        for (let i = 0; i < 8; i++) {
            const p = boxCorners[i];
            dists[i] = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - d;
        }

        const pts = [];
        const addPointUnique = (p) => {
            const eps = 1e-6;
            for (const q of pts) {
                if (Math.abs(p[0] - q[0]) < eps && Math.abs(p[1] - q[1]) < eps && Math.abs(p[2] - q[2]) < eps) return;
            }
            pts.push(p);
        };

        for (const [a, b] of boxEdges) {
            const da = dists[a];
            const db = dists[b];
            const pa = boxCorners[a];
            const pb = boxCorners[b];

            if (Math.abs(da) < 1e-9) addPointUnique([pa[0], pa[1], pa[2]]);
            if (Math.abs(db) < 1e-9) addPointUnique([pb[0], pb[1], pb[2]]);
            if ((da > 0) !== (db > 0)) {
                const t = da / (da - db);
                addPointUnique([
                    pa[0] + t * (pb[0] - pa[0]),
                    pa[1] + t * (pb[1] - pa[1]),
                    pa[2] + t * (pb[2] - pa[2])
                ]);
            }
        }

        if (pts.length < 3) return;

        const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
        const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
        const cz = pts.reduce((s, p) => s + p[2], 0) / pts.length;

        const up = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
        const t1 = vec3Normalize([
            up[1] * n[2] - up[2] * n[1],
            up[2] * n[0] - up[0] * n[2],
            up[0] * n[1] - up[1] * n[0]
        ]);
        const t2 = [
            n[1] * t1[2] - n[2] * t1[1],
            n[2] * t1[0] - n[0] * t1[2],
            n[0] * t1[1] - n[1] * t1[0]
        ];

        pts.sort((a, b) => {
            const da = [a[0] - cx, a[1] - cy, a[2] - cz];
            const db = [b[0] - cx, b[1] - cy, b[2] - cz];
            const angleA = Math.atan2(da[0]*t2[0]+da[1]*t2[1]+da[2]*t2[2], da[0]*t1[0]+da[1]*t1[1]+da[2]*t1[2]);
            const angleB = Math.atan2(db[0]*t2[0]+db[1]*t2[1]+db[2]*t2[2], db[0]*t1[0]+db[1]*t1[1]+db[2]*t1[2]);
            return angleA - angleB;
        });

        const positions = [];
        for (let i = 0; i < pts.length; i++) {
            const j = (i + 1) % pts.length;
            positions.push(...pts[i], ...pts[j]);
        }
        const lineColors = [];
        for (let i = 0; i < positions.length / 3; i++) {
            lineColors.push(0.45, 0.65, 1.0);
        }

        if (positions.length === 0) return;

        const prog = this._lineProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAlpha'), 0.9);
        // Do not clip the plane outline itself
        gl.uniform1f(gl.getUniformLocation(prog, 'uClipEnabled'), 0.0);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.disable(gl.CULL_FACE);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const colLoc = gl.getAttribLocation(prog, 'aColor');
        if (!this._sectionPlaneBuffer) {
            this._sectionPlaneBuffer = { position: gl.createBuffer(), color: gl.createBuffer() };
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this._sectionPlaneBuffer.position);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this._sectionPlaneBuffer.color);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineColors), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.LINES, 0, positions.length / 3);

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(colLoc);

        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
    }

    /**
     * Build and draw cross-section cap: solid fill at the clipping plane intersection.
     * For each solid voxel that the plane passes through, computes the plane–cube
     * intersection polygon and renders it as filled triangles.
     */
    _drawSectionCap(gl, projection, modelView, normalMatrix, nx, ny, nz) {
        const n = this.sectionNormal;
        const d = this.sectionOffset;
        const planeEps = 1e-7;

        const total = nx * ny * nz;
        const visible = this._cachedVisible || new Uint8Array(total);
        const stressMap = this._cachedStressMap || new Float32Array(total);

        const corners = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]];
        const edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];

        // Displacement lookup (mirrors _buildBuffersFromIndexedMesh logic)
        const dispEnabled = this.showDisplacement && this.displacementData && this.displacementData.U;
        const dispU = dispEnabled ? this.displacementData.U : null;
        const dispScale = this.displacementScale || 1;
        const dispNny = ny + 1, dispNnz = nz + 1;
        const _applyDisp = dispEnabled ? (px, py, pz) => {
            const ix = Math.min(Math.max(Math.round(px), 0), nx);
            const iy = Math.min(Math.max(Math.round(py), 0), ny);
            const iz = Math.min(Math.max(Math.round(pz), 0), nz);
            const nodeIdx = ix * dispNny * dispNnz + iy * dispNnz + iz;
            return [
                px + dispU[3 * nodeIdx] * dispScale,
                py + dispU[3 * nodeIdx + 1] * dispScale,
                pz + dispU[3 * nodeIdx + 2] * dispScale
            ];
        } : null;

        const positions = [];
        const normals = [];
        const colors = [];
        const capNormal = [-n[0], -n[1], -n[2]];

        const appendIntersectionPolygon = (pts, stress) => {
            if (pts.length < 3) return;

            const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
            const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
            const cz = pts.reduce((s, p) => s + p[2], 0) / pts.length;

            const up = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
            const t1 = vec3Normalize([
                up[1] * n[2] - up[2] * n[1],
                up[2] * n[0] - up[0] * n[2],
                up[0] * n[1] - up[1] * n[0]
            ]);
            const t2 = [
                capNormal[1] * t1[2] - capNormal[2] * t1[1],
                capNormal[2] * t1[0] - capNormal[0] * t1[2],
                capNormal[0] * t1[1] - capNormal[1] * t1[0]
            ];

            pts.sort((a, b) => {
                const da = [a[0] - cx, a[1] - cy, a[2] - cz];
                const db = [b[0] - cx, b[1] - cy, b[2] - cz];
                const angleA = Math.atan2(da[0]*t2[0]+da[1]*t2[1]+da[2]*t2[2], da[0]*t1[0]+da[1]*t1[1]+da[2]*t1[2]);
                const angleB = Math.atan2(db[0]*t2[0]+db[1]*t2[1]+db[2]*t2[2], db[0]*t1[0]+db[1]*t1[1]+db[2]*t1[2]);
                return angleA - angleB;
            });

            // Color based on stress and yield threshold
            const s = Math.max(0, Math.min(1, stress || 0));
            let cr, cg, cb;
            const yieldNorm = this._yieldNormalized;
            if (yieldNorm > 0 && yieldNorm < 1 && s > yieldNorm) {
                // Plastic zone: interpolate orange→red above yield
                const t = Math.min(1, (s - yieldNorm) / (1 - yieldNorm));
                cr = 1.0;
                cg = 0.6 * (1 - t);
                cb = 0;
            } else {
                cr = s;
                cg = DENSITY_COLOR_GREEN;
                cb = 1 - s;
            }

            // Apply displacement to sorted polygon points before rendering
            if (_applyDisp) {
                for (let i = 0; i < pts.length; i++) {
                    pts[i] = _applyDisp(pts[i][0], pts[i][1], pts[i][2]);
                }
            }

            // Recompute cap normal from displaced points for correct shading
            const capN = _applyDisp ? (() => {
                if (pts.length < 3) return capNormal;
                const e1 = [pts[1][0]-pts[0][0], pts[1][1]-pts[0][1], pts[1][2]-pts[0][2]];
                const e2 = [pts[2][0]-pts[0][0], pts[2][1]-pts[0][1], pts[2][2]-pts[0][2]];
                const nx2 = e1[1]*e2[2]-e1[2]*e2[1], ny2 = e1[2]*e2[0]-e1[0]*e2[2], nz2 = e1[0]*e2[1]-e1[1]*e2[0];
                const len = Math.sqrt(nx2*nx2+ny2*ny2+nz2*nz2);
                return len > 0 ? [nx2/len, ny2/len, nz2/len] : capNormal;
            })() : capNormal;

            for (let i = 1; i < pts.length - 1; i++) {
                positions.push(...pts[0], ...pts[i], ...pts[i + 1]);
                normals.push(...capN, ...capN, ...capN);
                colors.push(cr, cg, cb, cr, cg, cb, cr, cg, cb);
            }
        };

        if (this.amrCells && this.amrCells.length > 0) {
            const hasStrainFilter = this.strainMin > 0 || this.strainMax < 1;
            for (const cell of this.amrCells) {
                const density = cell.density !== undefined ? cell.density : 0;
                if (density <= this.densityThreshold) continue;

                const stress = cell.stress !== undefined ? cell.stress : 0;
                if (hasStrainFilter && (stress < this.strainMin || stress > this.strainMax)) continue;

                const size = Math.max(1e-6, cell.size || 1);
                const bx0 = cell.x;
                const by0 = cell.y;
                const bz0 = cell.z;
                const bx1 = bx0 + size;
                const by1 = by0 + size;
                const bz1 = bz0 + size;

                const cellCorners = [
                    [bx0, by0, bz0], [bx1, by0, bz0], [bx0, by1, bz0], [bx1, by1, bz0],
                    [bx0, by0, bz1], [bx1, by0, bz1], [bx0, by1, bz1], [bx1, by1, bz1]
                ];

                const dists = new Float64Array(8);
                let hasStrictPos = false, hasStrictNeg = false;
                let hasZero = false;
                for (let c = 0; c < 8; c++) {
                    const p = cellCorners[c];
                    dists[c] = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - d;
                    if (dists[c] > planeEps) hasStrictPos = true;
                    else if (dists[c] < -planeEps) hasStrictNeg = true;
                    else hasZero = true;
                }

                // Plane must strictly pass through cell volume to create a cap.
                // If it only touches a face/edge/corner (coplanar tangent), skip
                // to avoid double-drawing existing surface polygons.
                if (!(hasStrictPos && hasStrictNeg)) continue;

                const pts = [];
                for (const [a, b] of edges) {
                    if ((dists[a] > planeEps) !== (dists[b] > planeEps)) {
                        const t = dists[a] / (dists[a] - dists[b]);
                        const pa = cellCorners[a];
                        const pb = cellCorners[b];
                        pts.push([
                            pa[0] + t * (pb[0] - pa[0]),
                            pa[1] + t * (pb[1] - pa[1]),
                            pa[2] + t * (pb[2] - pa[2])
                        ]);
                    }
                }

                appendIntersectionPolygon(pts, stress);
            }
        } else {
            for (let x = 0; x < nx; x++) {
                for (let y = 0; y < ny; y++) {
                    for (let z = 0; z < nz; z++) {
                        const idx = x + y * nx + z * nx * ny;
                        if (!visible[idx]) continue;

                        const dists = new Float64Array(8);
                        let hasStrictPos = false, hasStrictNeg = false;
                        let hasZero = false;
                        for (let c = 0; c < 8; c++) {
                            dists[c] = n[0] * (x + corners[c][0]) + n[1] * (y + corners[c][1]) + n[2] * (z + corners[c][2]) - d;
                            if (dists[c] > planeEps) hasStrictPos = true;
                            else if (dists[c] < -planeEps) hasStrictNeg = true;
                            else hasZero = true;
                        }

                        if (!(hasStrictPos && hasStrictNeg)) continue;

                        const pts = [];
                        for (const [a, b] of edges) {
                            if ((dists[a] > planeEps) !== (dists[b] > planeEps)) {
                                const t = dists[a] / (dists[a] - dists[b]);
                                pts.push([
                                    x + corners[a][0] + t * (corners[b][0] - corners[a][0]),
                                    y + corners[a][1] + t * (corners[b][1] - corners[a][1]),
                                    z + corners[a][2] + t * (corners[b][2] - corners[a][2])
                                ]);
                            }
                        }

                        appendIntersectionPolygon(pts, stressMap[idx] || 0);
                    }
                }
            }
        }

        if (positions.length === 0) return;

        if (!this._sectionCapBuffer) {
            this._sectionCapBuffer = { position: gl.createBuffer(), normal: gl.createBuffer(), color: gl.createBuffer() };
        }
        const scb = this._sectionCapBuffer;
        gl.bindBuffer(gl.ARRAY_BUFFER, scb.position);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, scb.normal);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, scb.color);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);

        const prog = this._meshProgram;
        gl.useProgram(prog);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uNormalMatrix'), false, normalMatrix);
        const lightDir = vec3Normalize([0.3, 0.5, 0.8]);
        gl.uniform3fv(gl.getUniformLocation(prog, 'uLightDir'), lightDir);
        gl.uniform1f(gl.getUniformLocation(prog, 'uAmbient'), 0.4);
        gl.uniform1f(gl.getUniformLocation(prog, 'uClipEnabled'), 0.0);

        gl.disable(gl.CULL_FACE);
        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const normLoc = gl.getAttribLocation(prog, 'aNormal');
        const colLoc = gl.getAttribLocation(prog, 'aColor');

        gl.bindBuffer(gl.ARRAY_BUFFER, scb.position);
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, scb.normal);
        gl.enableVertexAttribArray(normLoc);
        gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, scb.color);
        gl.enableVertexAttribArray(colLoc);
        gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);

        gl.disableVertexAttribArray(posLoc);
        gl.disableVertexAttribArray(normLoc);
        gl.disableVertexAttribArray(colLoc);

        gl.enable(gl.CULL_FACE);
    }

    _drawHoverHighlight(gl, projection, modelView) {
        if (this._hoverFaces.length === 0) return;

        const prog = this._overlayProgram;
        gl.useProgram(prog);

        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProjection'), false, projection);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uModelView'), false, modelView);
        this._setClipUniforms(gl, prog);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.disable(gl.CULL_FACE);

        const posLoc = gl.getAttribLocation(prog, 'aPosition');
        const positions = [];

        for (const bf of this._hoverFaces) {
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

            // White semi-transparent highlight
            gl.uniform4fv(gl.getUniformLocation(prog, 'uColor'), [1.0, 1.0, 1.0, 0.3]);
            gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);

            gl.disableVertexAttribArray(posLoc);
            gl.deleteBuffer(buf);
        }

        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
    }

    _drawForceArrows2D(nx, ny, nz, projection, modelView) {
        const { width, height } = this.overlayCanvas;

        let arrowDir;
        if (this.forceVector) {
            arrowDir = vec3Normalize(this.forceVector);
        } else {
            switch (this.forceDirection) {
                case 'down':  arrowDir = [0, -1, 0]; break;
                case 'up':    arrowDir = [0, 1, 0]; break;
                case 'left':  arrowDir = [-1, 0, 0]; break;
                case 'right': arrowDir = [1, 0, 0]; break;
                default:      arrowDir = [0, -1, 0];
            }
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

    // ─── Vertical Stress Scale Bar ────────────────────────────────────────────

    _drawStressScale(viewWidth, viewHeight) {
        const ctx = this.ctx;
        const barWidth = 20;
        const barHeight = Math.min(200, viewHeight * 0.5);
        const barX = viewWidth - 60;
        const barY = (viewHeight - barHeight) / 2;
        const maxStress = this.maxStress || 1;

        // Store bar geometry for hit testing on drag handles
        this._stressBarRect = { x: barX, y: barY, width: barWidth, height: barHeight };

        // Draw gradient bar using mode-specific color palette
        // When yield strength is set and in stress mode, above-yield region uses bright purple
        const yieldNorm = this._yieldNormalized;
        const barMode = this._colorMode || 'stress';
        for (let i = 0; i < barHeight; i++) {
            const t = 1 - i / barHeight; // 0 at bottom, 1 at top
            let r, g, b;
            if (barMode === 'stress') {
                // Stress mode: preserve original yield-aware gradient
                if (yieldNorm > 0 && yieldNorm < 1 && t > yieldNorm) {
                    r = 0.75;
                    g = 0;
                    b = 1.0;
                } else {
                    const tn = yieldNorm > 0 && yieldNorm < 1 ? t / yieldNorm : t;
                    r = tn;
                    g = DENSITY_COLOR_GREEN;
                    b = 0.5 * (1 - tn);
                }
            } else {
                // Use mode-specific color palette for the bar
                const c = this._getColorForMode(t, barMode);
                r = c.r; g = c.g; b = c.b;
            }
            ctx.fillStyle = `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
            ctx.fillRect(barX, barY + i, barWidth, 1);
        }

        // Draw border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = 1;
        ctx.strokeRect(barX, barY, barWidth, barHeight);

        // Draw strain range filter handles and shading
        const minY = barY + barHeight - this.strainMin * barHeight;
        const maxY = barY + barHeight - this.strainMax * barHeight;
        const handleW = 10;
        const handleH = 8;

        // Shade out-of-range regions
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        if (this.strainMin > 0) {
            ctx.fillRect(barX, minY, barWidth, barY + barHeight - minY);
        }
        if (this.strainMax < 1) {
            ctx.fillRect(barX, barY, barWidth, maxY - barY);
        }

        // Draw handle lines
        if (this.strainMin > 0 || this.strainMax < 1) {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(barX - 5, minY);
            ctx.lineTo(barX + barWidth + 5, minY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(barX - 5, maxY);
            ctx.lineTo(barX + barWidth + 5, maxY);
            ctx.stroke();
        }

        // Draw draggable handle triangles (left side of bar)
        const drawHandle = (y, isActive) => {
            ctx.fillStyle = isActive ? '#ffffff' : 'rgba(255, 255, 255, 0.85)';
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.lineWidth = 1;
            // Left triangle handle pointing right
            ctx.beginPath();
            ctx.moveTo(barX - handleW, y - handleH);
            ctx.lineTo(barX, y);
            ctx.lineTo(barX - handleW, y + handleH);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            // Right triangle handle pointing left
            ctx.beginPath();
            ctx.moveTo(barX + barWidth + handleW, y - handleH);
            ctx.lineTo(barX + barWidth, y);
            ctx.lineTo(barX + barWidth + handleW, y + handleH);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        };

        drawHandle(minY, this._stressHandleDrag === 'min');
        drawHandle(maxY, this._stressHandleDrag === 'max');

        // Draw tick labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = '10px Arial';
        ctx.textAlign = 'right';
        const tickCount = 5;
        for (let i = 0; i <= tickCount; i++) {
            const t = i / tickCount;
            const y = barY + barHeight - t * barHeight;
            const stressVal = t * maxStress;
            // Tick mark
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(barX - 3, y);
            ctx.lineTo(barX, y);
            ctx.stroke();
            // Label
            ctx.fillText(stressVal.toFixed(1), barX - 5, y + 3);
        }

        // Draw title
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 11px Arial';
        ctx.textAlign = 'center';
        ctx.translate(barX + barWidth + 16, barY + barHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(this.stressBarLabel, 0, 0);
        ctx.restore();

        // Draw yield strength threshold line on the stress bar
        if (yieldNorm > 0 && yieldNorm < 1) {
            const yieldY = barY + barHeight - yieldNorm * barHeight;
            ctx.strokeStyle = '#ffcc00';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(barX - 6, yieldY);
            ctx.lineTo(barX + barWidth + 6, yieldY);
            ctx.stroke();
            ctx.setLineDash([]);
            // Label
            ctx.fillStyle = '#ffcc00';
            ctx.font = 'bold 9px Arial';
            ctx.textAlign = 'left';
            ctx.fillText('⚠ Yield', barX + barWidth + 8, yieldY + 3);
            ctx.font = '8px Arial';
            ctx.fillText(`${this.yieldStrength.toFixed(0)} MPa`, barX + barWidth + 8, yieldY + 13);
        }

        // Draw hover indicator on the scale bar
        if (this._hoverStressValue !== null) {
            const hoverY = barY + barHeight - this._hoverStressValue * barHeight;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(barX - 8, hoverY);
            ctx.lineTo(barX + barWidth + 8, hoverY);
            ctx.stroke();
            // Draw small triangle indicator
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.moveTo(barX - 8, hoverY);
            ctx.lineTo(barX - 3, hoverY - 4);
            ctx.lineTo(barX - 3, hoverY + 4);
            ctx.closePath();
            ctx.fill();
        }
    }

    _drawStressTooltip() {
        if (this._hoverStressValue === null || !this._hoverScreenPos) return;
        const ctx = this.ctx;
        const maxStress = this.maxStress || 1;
        const stressVal = this._hoverStressValue * maxStress;

        // Show yield status in tooltip when yield strength is set
        const yieldNorm = this._yieldNormalized;
        const aboveYield = yieldNorm > 0 && yieldNorm < 1 && this._hoverStressValue > yieldNorm;
        const yieldTag = aboveYield ? ' ⚠ PLASTIC' : (yieldNorm > 0 ? ' (elastic)' : '');
        const text = `${stressVal.toFixed(2)} ${this.stressBarUnit}${yieldTag}`;
        ctx.font = 'bold 12px Arial';
        const metrics = ctx.measureText(text);
        const padding = 6;
        const tooltipW = metrics.width + padding * 2;
        const tooltipH = 20;
        const tx = this._hoverScreenPos.x + 15;
        const ty = this._hoverScreenPos.y - 10;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
        ctx.fillRect(tx, ty - tooltipH / 2, tooltipW, tooltipH);

        // Text
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'left';
        ctx.fillText(text, tx + padding, ty + 4);
    }

    /** Test if mouse position hits a stress bar drag handle. Returns 'min', 'max', or null. */
    _hitTestStressHandle(e) {
        if (!this._stressBarRect || !this.meshData || this.meshData.length === 0) return null;
        const bar = this._stressBarRect;
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const handleW = 10;
        const handleH = 8;

        const minY = bar.y + bar.height - this.strainMin * bar.height;
        const maxY = bar.y + bar.height - this.strainMax * bar.height;

        // Hit zone extends across handle triangles on both sides of bar
        const hitLeft = bar.x - handleW - 2;
        const hitRight = bar.x + bar.width + handleW + 2;

        if (mx >= hitLeft && mx <= hitRight) {
            if (Math.abs(my - minY) <= handleH) return 'min';
            if (Math.abs(my - maxY) <= handleH) return 'max';
        }
        return null;
    }

    /** Update strain range when dragging a stress bar handle. */
    _updateStressHandleDrag(e) {
        if (!this._stressBarRect) return;
        const bar = this._stressBarRect;
        const rect = this.canvas.getBoundingClientRect();
        const my = e.clientY - rect.top;

        // Convert mouse Y to normalized strain value (0 at bottom, 1 at top)
        let val = 1 - (my - bar.y) / bar.height;
        val = Math.max(0, Math.min(1, val));

        if (this._stressHandleDrag === 'min') {
            this.strainMin = Math.min(val, this.strainMax - 0.01); // 1% minimum gap between handles
        } else if (this._stressHandleDrag === 'max') {
            this.strainMax = Math.max(val, this.strainMin + 0.01); // 1% minimum gap between handles
        }

        this._needsRebuild = true;
        this.draw();
        if (this.onStrainRangeChange) this.onStrainRangeChange(this.strainMin, this.strainMax);
    }

    // ─── 2D Overlay (axes) ──────────────────────────────────────────────────

    _drawAxesOverlay() {
        if (!this.model) return;

        const axisOriginX = 50;
        const axisOriginY = this.overlayCanvas.height - 50;
        const axisLen = 35;

        const { nx, ny, nz } = this.model;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const aspect = width / height;
        const projection = mat4Perspective(Math.PI / 4, aspect, 0.1, 1000);
        const modelView = this._buildModelView(nx, ny, nz);

        // Use the actual model center as axis origin so overlay orientation matches scene exactly.
        const center = { x: nx * 0.5, y: ny * 0.5, z: nz * 0.5 };
        const centerScreen = this._projectToScreen(center, modelView, projection, width, height);

        const axes = [
            { dir: [1, 0, 0], color: '#ff0000', label: 'X' },
            { dir: [0, 1, 0], color: '#00ff00', label: 'Y' },
            { dir: [0, 0, 1], color: '#0088ff', label: 'Z' }
        ];

        for (const axis of axes) {
            const [dx, dy, dz] = axis.dir;
            const tipWorld = { x: center.x + dx, y: center.y + dy, z: center.z + dz };
            const tipScreen = this._projectToScreen(tipWorld, modelView, projection, width, height);

            let vx = tipScreen.x - centerScreen.x;
            let vy = centerScreen.y - tipScreen.y; // invert screen Y to overlay Y-up
            const vlen = Math.hypot(vx, vy);
            if (vlen < 1e-6) continue;
            vx /= vlen;
            vy /= vlen;

            const ex = axisOriginX + vx * axisLen;
            const ey = axisOriginY - vy * axisLen;

            this.ctx.strokeStyle = axis.color;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(axisOriginX, axisOriginY);
            this.ctx.lineTo(ex, ey);
            this.ctx.stroke();

            this.ctx.fillStyle = axis.color;
            this.ctx.font = 'bold 12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(axis.label, ex + (vx * 10), ey - (vy * 10));
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

    updateMesh(meshData, maxStress) {
        this.meshData = meshData;
        this.amrCells = (meshData && meshData.amrCells && meshData.amrCells.length > 0) ? meshData.amrCells : null;
        if (maxStress !== undefined) this.maxStress = maxStress;
        this._updateYieldNormalized();
        this._needsRebuild = true;
        this.draw();
    }

    setVolumetricStressData(volumetricData) {
        if (!volumetricData || !this.model) {
            this._volumetricStressMap = null;
            this._needsRebuild = true;
            this.draw();
            return;
        }

        const { nx, ny, nz } = this.model;
        if (volumetricData.nx !== nx || volumetricData.ny !== ny || volumetricData.nz !== nz || !volumetricData.stress) {
            return;
        }

        const source = volumetricData.stress;
        const total = nx * ny * nz;
        if (source.length !== total) {
            return;
        }

        let maxStress = volumetricData.maxStress || 0;
        if (!maxStress || !Number.isFinite(maxStress) || maxStress <= 0) {
            for (let i = 0; i < source.length; i++) {
                if (source[i] > maxStress) maxStress = source[i];
            }
        }

        const normalized = new Float32Array(total);
        if (maxStress > 0) {
            const invMax = 1 / maxStress;
            for (let i = 0; i < total; i++) {
                normalized[i] = Math.max(0, Math.min(1, source[i] * invMax));
            }
        }

        this._volumetricStressMap = normalized;
        this.maxStress = maxStress;
        this._updateYieldNormalized();
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

    setDensityThreshold(value) {
        let clamped = Math.max(0, Math.min(1, value));
        // Keep the top endpoint usable: threshold 1.0 would otherwise hide
        // everything with strict `density > threshold` comparisons.
        if (clamped >= 1) clamped = 0.99;
        this.densityThreshold = clamped;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Set the displacement vector data from a FEA solve.
     * @param {{ U: Float32Array, nx: number, ny: number, nz: number } | null} data
     */
    setDisplacementData(data) {
        this.displacementData = data;
        if (!data) this.showDisplacement = false;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Toggle whether the deformed shape is shown using the stored displacement data.
     */
    toggleDisplacement() {
        if (!this.displacementData) return;
        this.showDisplacement = !this.showDisplacement;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Set the displacement magnification scale factor.
     * @param {number} scale
     */
    setDisplacementScale(scale) {
        this.displacementScale = Math.max(0, scale);
        if (this.showDisplacement) {
            this._needsRebuild = true;
            this.draw();
        }
    }

    /**
     * Update the stress scale bar label and unit (e.g. for fatigue mode).
     * @param {string} label - Text for the rotated bar title
     * @param {string} unit  - Unit shown in hover tooltip
     */
    setStressBarLabel(label, unit) {
        this.stressBarLabel = label || 'Stress (N/mm² = MPa)';
        this.stressBarUnit = unit || 'MPa';
    }

    /**
     * Set the material yield strength (MPa). When set, voxels exceeding yield
     * are highlighted and the stress bar shows a yield threshold line.
     * @param {number} yieldMPa - Yield strength in MPa (0 to disable)
     */
    setYieldStrength(yieldMPa) {
        this.yieldStrength = Math.max(0, yieldMPa || 0);
        this._updateYieldNormalized();
        this._needsRebuild = true;
        this.draw();
    }

    /** Recompute normalized yield threshold from current maxStress. */
    _updateYieldNormalized() {
        if (this.yieldStrength > 0 && this.maxStress > 0) {
            this._yieldNormalized = Math.min(1, this.yieldStrength / this.maxStress);
        } else {
            this._yieldNormalized = 0;
        }
    }

    toggleWireframe() {
        this.wireframe = !this.wireframe;
        this._needsRebuild = true;
        this.draw();
    }

    setViewMode(mode) {
        if (mode !== 'auto' && mode !== 'voxel') return;
        this.viewMode = mode;
        this._needsRebuild = true;
        this.draw();
    }

    toggleMeshVisibility() {
        this.meshVisible = !this.meshVisible;
        this.draw();
    }

    // ── Advanced analysis visualization ──────────────────────────────────────

    /**
     * Set the damage/phase-field data for fracture visualization.
     * Values in [0,1]: 0 = intact, 1 = fully damaged/cracked.
     * @param {Float32Array|number[]|null} damageField - Per-element damage values
     */
    setDamageField(damageField) {
        this._damageField = damageField || null;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Set the eroded/deleted element set for element erosion visualization.
     * Eroded elements are hidden (not rendered).
     * @param {Set<number>|null} erodedElements - Set of eroded element indices
     */
    setErodedElements(erodedElements) {
        this._erodedElements = erodedElements || null;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Set the color mode for rendering.
     * @param {'stress'|'strain'|'damage'|'displacement'|'density'|'triaxiality'|'plasticStrain'} mode
     */
    setColorMode(mode) {
        const validModes = ['stress', 'strain', 'damage', 'displacement', 'density', 'triaxiality', 'plasticStrain'];
        if (!validModes.includes(mode)) return;
        this._colorMode = mode;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Get the current color mode.
     * @returns {string}
     */
    getColorMode() {
        return this._colorMode || 'stress';
    }

    /**
     * Set per-element scalar field for custom coloring (triaxiality, plastic strain, etc.).
     * @param {string} fieldName - Name of the field ('triaxiality', 'plasticStrain', etc.)
     * @param {Float32Array|number[]|null} values - Per-element values
     * @param {number} [minVal=0] - Min value for color mapping
     * @param {number} [maxVal=1] - Max value for color mapping
     */
    setScalarField(fieldName, values, minVal = 0, maxVal = 1) {
        if (!this._scalarFields) this._scalarFields = {};
        this._scalarFields[fieldName] = values ? { values, min: minVal, max: maxVal } : null;
        this._needsRebuild = true;
        this.draw();
    }

    /**
     * Get the color for a given element based on the current color mode.
     * Used internally by _buildBuffersFromIndexedMesh for per-face coloring.
     * @param {number} cellIdx - Element index
     * @param {number} stress - Normalized stress value [0,1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _getElementColor(cellIdx, stress) {
        const mode = this._colorMode || 'stress';

        if (mode === 'damage' && this._damageField) {
            const d = Math.max(0, Math.min(1, this._damageField[cellIdx] || 0));
            // Intact (0): blue, Damaged (1): bright red
            return { r: d, g: 0.1 * (1 - d), b: 0.8 * (1 - d) };
        }

        if (mode === 'strain') {
            // Strain energy density: "hot" colormap — black → red → yellow → white
            return this._hotColor(stress);
        }

        if (mode === 'density') {
            // Color by material density: dark purple (0) → teal (0.5) → bright green (1)
            const dMap = this._cachedDensityMap;
            if (dMap && dMap[cellIdx] !== undefined) {
                const d = Math.max(0, Math.min(1, dMap[cellIdx]));
                return this._densityColor(d);
            }
            return { r: 0.5, g: 0.5, b: 0.5 };
        }

        if (mode === 'displacement' && this.displacementData) {
            const U = this.displacementData.U;
            if (U) {
                const ux = U[cellIdx * 3] || 0;
                const uy = U[cellIdx * 3 + 1] || 0;
                const uz = U[cellIdx * 3 + 2] || 0;
                const mag = Math.sqrt(ux * ux + uy * uy + uz * uz);
                const maxMag = this._maxDisplacementMag || 1;
                const t = Math.min(1, mag / maxMag);
                // Cool-warm diverging: dark blue → light blue → white → orange → dark red
                return this._coolWarmColor(t);
            }
        }

        if (mode === 'triaxiality') {
            const field = this._scalarFields && this._scalarFields[mode];
            if (field && field.values) {
                const raw = field.values[cellIdx] || 0;
                const range = field.max - field.min || 1;
                const t = Math.max(0, Math.min(1, (raw - field.min) / range));
                // Diverging: compression (teal) → neutral (white) → tension (magenta)
                return this._triaxialityColor(t);
            }
        }

        if (mode === 'plasticStrain') {
            const field = this._scalarFields && this._scalarFields[mode];
            if (field && field.values) {
                const raw = field.values[cellIdx] || 0;
                const range = field.max - field.min || 1;
                const t = Math.max(0, Math.min(1, (raw - field.min) / range));
                // Sequential purple: black → purple → magenta → pink
                return this._plasticStrainColor(t);
            }
        }

        // Default: stress coloring (existing behavior)
        const yieldNorm = this._yieldNormalized;
        if (yieldNorm > 0 && yieldNorm < 1 && stress > yieldNorm) {
            return { r: 0.75, g: 0, b: 1.0 };
        }
        const t = yieldNorm > 0 && yieldNorm < 1 ? stress / yieldNorm : stress;
        return { r: t, g: DENSITY_COLOR_GREEN, b: 0.5 * (1 - t) };
    }

    /**
     * Standard heatmap: blue → cyan → green → yellow → red.
     * @param {number} t - Value in [0, 1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _heatmapColor(t) {
        let r, g, b;
        if (t < 0.25) {
            const s = t / 0.25;
            r = 0; g = s; b = 1;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0; g = 1; b = 1 - s;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = s; g = 1; b = 0;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1; g = 1 - s; b = 0;
        }
        return { r, g, b };
    }

    /**
     * "Hot" colormap: black → red → yellow → white. Used for strain energy.
     * @param {number} t - Value in [0, 1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _hotColor(t) {
        let r, g, b;
        if (t < 0.33) {
            const s = t / 0.33;
            r = s; g = 0; b = 0;
        } else if (t < 0.66) {
            const s = (t - 0.33) / 0.33;
            r = 1; g = s; b = 0;
        } else {
            const s = (t - 0.66) / (1.0 - 0.66);
            r = 1; g = 1; b = s;
        }
        return { r, g, b };
    }

    /**
     * Cool-warm diverging: dark blue → light blue → white → orange → dark red.
     * Used for displacement magnitude.
     * @param {number} t - Value in [0, 1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _coolWarmColor(t) {
        let r, g, b;
        if (t < 0.25) {
            const s = t / 0.25;
            r = 0.1 * s; g = 0.1 + 0.4 * s; b = 0.5 + 0.5 * s;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0.1 + 0.9 * s; g = 0.5 + 0.5 * s; b = 1;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = 1; g = 1 - 0.35 * s; b = 1 - 0.8 * s;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1 - 0.3 * s; g = 0.65 - 0.45 * s; b = 0.2 - 0.15 * s;
        }
        return { r, g, b };
    }

    /**
     * Density colormap: dark purple → teal → bright green.
     * @param {number} t - Value in [0, 1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _densityColor(t) {
        let r, g, b;
        if (t < 0.5) {
            const s = t / 0.5;
            r = 0.3 * (1 - s); g = 0.1 + 0.5 * s; b = 0.5 + 0.1 * s;
        } else {
            const s = (t - 0.5) / 0.5;
            r = 0.1 * s; g = 0.6 + 0.4 * s; b = 0.6 - 0.4 * s;
        }
        return { r, g, b };
    }

    /**
     * Triaxiality diverging colormap: teal (compression) → white → magenta (tension).
     * @param {number} t - Value in [0, 1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _triaxialityColor(t) {
        let r, g, b;
        if (t < 0.5) {
            const s = t / 0.5;
            r = s; g = 0.6 + 0.4 * s; b = 0.7 + 0.3 * s;
        } else {
            const s = (t - 0.5) / 0.5;
            r = 1 - 0.2 * s; g = 1 - 0.8 * s; b = 1 - 0.2 * s;
        }
        return { r, g, b };
    }

    /**
     * Plastic strain sequential: black → purple → magenta → pink.
     * @param {number} t - Value in [0, 1]
     * @returns {{ r: number, g: number, b: number }}
     */
    _plasticStrainColor(t) {
        let r, g, b;
        if (t < 0.33) {
            const s = t / 0.33;
            r = 0.3 * s; g = 0; b = 0.4 * s;
        } else if (t < 0.66) {
            const s = (t - 0.33) / 0.33;
            r = 0.3 + 0.5 * s; g = 0; b = 0.4 + 0.3 * s;
        } else {
            const s = (t - 0.66) / (1.0 - 0.66);
            r = 0.8 + 0.2 * s; g = 0.3 * s; b = 0.7 + 0.3 * s;
        }
        return { r, g, b };
    }

    /**
     * Get the color for a given normalized value based on the specified color mode.
     * Used by the stress bar and element coloring to ensure visual consistency.
     * @param {number} t - Normalized value in [0, 1]
     * @param {string} [mode] - Color mode (defaults to current mode)
     * @returns {{ r: number, g: number, b: number }}
     */
    _getColorForMode(t, mode) {
        mode = mode || this._colorMode || 'stress';
        switch (mode) {
            case 'strain': return this._hotColor(t);
            case 'displacement': return this._coolWarmColor(t);
            case 'density': return this._densityColor(t);
            case 'triaxiality': return this._triaxialityColor(t);
            case 'plasticStrain': return this._plasticStrainColor(t);
            case 'damage': {
                const d = t;
                return { r: d, g: 0.1 * (1 - d), b: 0.8 * (1 - d) };
            }
            default: return this._heatmapColor(t); // stress uses standard heatmap in bar
        }
    }

    /**
     * Set the reference model from original STL/STEP vertices.
     * The reference mesh is rendered as a semi-transparent overlay.
     * @param {Array} vertices - Array of {x,y,z} objects (every 3 = 1 triangle)
     * @param {object} [bounds] - Bounding box {minX,minY,minZ,maxX,maxY,maxZ}
     */
    setReferenceModel(vertices, bounds) {
        this.referenceVertices = vertices;
        this._referenceBounds = bounds || null;
        this._needsRebuild = true;
    }

    /**
     * Toggle visibility of the reference model overlay.
     */
    toggleReference() {
        this.showReference = !this.showReference;
        this.draw();
    }

    toggleSection() {
        this.sectionEnabled = !this.sectionEnabled;
        if (this.sectionEnabled && this.model && !this._sectionInitialized) {
            this.sectionNormal = [1, 0, 0];
            this.sectionOffset = this.model.nx / 2;
            this._sectionInitialized = true;
        }
        this.draw();
    }

    resetCamera() {
        this.rotation = { x: 0.5, y: 0.5 };
        this.pan = { x: 0, y: 0 };
        this.zoom = 1;
        this.sectionEnabled = false;
        this.draw();
    }

    clear() {
        this.model = null;
        this.densities = null;
        this.meshData = null;
        this.amrCells = null;
        this.strainMin = 0;
        this.strainMax = 1;
        this.maxStress = 0;
        this.densityThreshold = DENSITY_THRESHOLD;
        this._cachedVisible = null;
        this._cachedDensityMap = null;
        this._cachedStressMap = null;
        this._volumetricStressMap = null;
        this._hoverStressValue = null;
        this._hoverScreenPos = null;
        this.viewMode = 'auto';
        this.meshVisible = true;
        this.showReference = true;
        this.referenceVertices = null;
        this._referenceBounds = null;
        this._referenceBuffers = null;
        this.sectionEnabled = false;
        this.sectionNormal = [1, 0, 0];
        this.sectionOffset = 0;
        this._sectionInitialized = false;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this.paintMode = null;
        this._boundaryFaces = [];
        this._boundaryFaceMap = {};
        this._hoverFaces = [];
        this._seedFaceKey = null;
        this._seedFaceNormal = null;
        this.selectionGroups = [];
        this._nextGroupId = 1;
        this.activeGroupId = null;
        this.selectionShapes = [];
        this._nextShapeId = 1;
        this._shapeHighlightFaces = new Set();
        this.displacementData = null;
        this.showDisplacement = false;
        this.stressBarLabel = 'Stress (N/mm² = MPa)';
        this.stressBarUnit = 'MPa';
        this.yieldStrength = 0;
        this._yieldNormalized = 0;
        this._damageField = null;
        this._erodedElements = null;
        this._colorMode = 'stress';
        this._scalarFields = null;
        this._maxDisplacementMag = 0;
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
