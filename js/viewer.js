// 3D Viewer using Canvas 2D (no external dependencies)
import { DENSITY_THRESHOLD } from './constants.js';

export class Viewer3D {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = null;
        this.canvas = null;
        this.ctx = null;
        this.model = null;
        this.wireframe = false;
        
        // Camera settings
        this.rotation = { x: 0.5, y: 0.5 };
        this.zoom = 1;
        this.isDragging = false;
        this.lastMousePos = { x: 0, y: 0 };
        
        this.densities = null;
        this.meshData = null; // Triangle mesh data from optimizer

        // Paint mode for face selection
        this.paintMode = null; // null | 'constraint' | 'force'
        this.paintedConstraintFaces = new Set(); // "x,y,z,face" keys
        this.paintedForceFaces = new Set();
        this.forceDirection = 'down';
        this.forceMagnitude = 1000;
        this.isPainting = false;

        // Cached boundary faces for picking
        this._boundaryFaces = [];
    }

    async init() {
        this.container = document.getElementById(this.containerId);
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.container.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        // Setup mouse controls
        this.setupControls();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Draw initial state
        this.draw();
        
        console.log('Viewer initialized');
    }

    setupControls() {
        this.canvas.addEventListener('mousedown', (e) => {
            if (this.paintMode && this.model) {
                this.isPainting = true;
                this.handlePaintClick(e);
                return;
            }
            this.isDragging = true;
            this.lastMousePos = { x: e.clientX, y: e.clientY };
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isPainting && this.paintMode && this.model) {
                this.handlePaintClick(e);
                return;
            }
            if (this.isDragging) {
                const dx = e.clientX - this.lastMousePos.x;
                const dy = e.clientY - this.lastMousePos.y;
                
                this.rotation.y += dx * 0.01;
                this.rotation.x += dy * 0.01;
                
                // Clamp x rotation
                this.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.rotation.x));
                
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.draw();
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.isPainting = false;
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.isPainting = false;
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom *= (1 - e.deltaY * 0.001);
            this.zoom = Math.max(0.1, Math.min(5, this.zoom));
            this.draw();
        });
    }

    /** Set the paint mode: null (orbit), 'constraint', or 'force' */
    setPaintMode(mode) {
        this.paintMode = mode;
        this.canvas.style.cursor = mode ? 'crosshair' : 'grab';
    }

    /** Handle a paint click/drag on the canvas */
    handlePaintClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        // Find closest boundary face to click position
        const { width, height } = this.canvas;
        const { nx, ny, nz } = this.model;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;

        let bestDist = Infinity;
        let bestFace = null;

        for (const bf of this._boundaryFaces) {
            const screenVerts = bf.projVerts.map(pv => ({
                x: centerX + pv.x * scale,
                y: centerY - pv.y * scale
            }));
            // Test if point is inside the quad (approximate with centroid distance)
            const cx = (screenVerts[0].x + screenVerts[1].x + screenVerts[2].x + screenVerts[3].x) / 4;
            const cy = (screenVerts[0].y + screenVerts[1].y + screenVerts[2].y + screenVerts[3].y) / 4;
            const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);

            // Verify point is roughly within the quad bounds
            const maxR = Math.max(
                ...screenVerts.map(v => Math.sqrt((v.x - cx) ** 2 + (v.y - cy) ** 2))
            );
            if (dist < maxR * 1.2) {
                // Prefer the closest face to the camera (highest avgDepth)
                if (bestFace === null || bf.avgDepth > bestFace.avgDepth) {
                    bestDist = dist;
                    bestFace = bf;
                } else if (Math.abs(bf.avgDepth - bestFace.avgDepth) < 0.01 && dist < bestDist) {
                    bestDist = dist;
                    bestFace = bf;
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
            this.draw();
        }
    }

    onWindowResize() {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.draw();
    }

    setModel(model) {
        this.model = model;
        this.densities = null;
        this.meshData = null;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this.draw();
    }

    draw() {
        if (!this.ctx) return;
        
        const { width, height } = this.canvas;
        
        // Clear canvas
        this.ctx.fillStyle = '#1a1a2e';
        this.ctx.fillRect(0, 0, width, height);
        
        if (!this.model) {
            // Draw placeholder text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '20px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Load a model to view', width / 2, height / 2);
            return;
        }
        
        const { nx, ny, nz } = this.model;
        
        if (this.meshData && this.meshData.length > 0) {
            this.drawTriangleMesh(nx, ny, nz);
        } else {
            this.drawVoxels(nx, ny, nz);
        }

        // Draw constraint and force overlays on top
        this.drawConstraintOverlay(nx, ny, nz);
        this.drawForceArrows(nx, ny, nz);
        
        // Draw axes
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;
        this.drawAxes(centerX, centerY, scale);

        // Draw paint mode indicator
        if (this.paintMode) {
            this.ctx.fillStyle = this.paintMode === 'constraint' ? 'rgba(0,200,100,0.8)' : 'rgba(255,100,50,0.8)';
            this.ctx.font = '14px Arial';
            this.ctx.textAlign = 'left';
            const label = this.paintMode === 'constraint' ? '🖌 Painting Constraints (Shift+click to remove)' : '🖌 Painting Forces (Shift+click to remove)';
            this.ctx.fillText(label, 10, height - 10);
        }
    }

    drawTriangleMesh(nx, ny, nz) {
        const { width, height } = this.canvas;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;

        // Project and sort triangles by depth (painter's algorithm)
        const projectedTriangles = this.meshData.map(tri => {
            const projVerts = tri.vertices.map(v =>
                this.project3D(v[0], v[1], v[2], nx, ny, nz)
            );
            const avgDepth = (projVerts[0].z + projVerts[1].z + projVerts[2].z) / 3;
            return { projVerts, density: tri.density, normal: tri.normal, avgDepth };
        });

        projectedTriangles.sort((a, b) => a.avgDepth - b.avgDepth);

        projectedTriangles.forEach(tri => {
            const screenVerts = tri.projVerts.map(pv => ({
                x: centerX + pv.x * scale,
                y: centerY - pv.y * scale
            }));

            // Color based on density
            let r, g, b;
            r = Math.floor(tri.density * 255);
            g = 76;
            b = Math.floor((1 - tri.density) * 255);

            // Apply lighting from face normal dot product with light direction
            const lightDir = { x: 0.3, y: 0.5, z: 0.8 };
            const mag = Math.sqrt(lightDir.x ** 2 + lightDir.y ** 2 + lightDir.z ** 2);
            const dot = Math.abs(
                (tri.normal[0] * lightDir.x + tri.normal[1] * lightDir.y + tri.normal[2] * lightDir.z) / mag
            );
            const lightFactor = 0.4 + 0.6 * dot;
            r = Math.floor(r * lightFactor);
            g = Math.floor(g * lightFactor);
            b = Math.floor(b * lightFactor);

            this.ctx.beginPath();
            this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
            this.ctx.lineTo(screenVerts[1].x, screenVerts[1].y);
            this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
            this.ctx.closePath();

            if (this.wireframe) {
                this.ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                this.ctx.stroke();
            } else {
                this.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                this.ctx.fill();
            }
        });
    }

    drawVoxels(nx, ny, nz) {
        const { width, height } = this.canvas;
        const { elements } = this.model;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;

        // Build a lookup set for occupied voxels to cull interior faces
        const occupied = new Uint8Array(nx * ny * nz);
        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    const index = x + y * nx + z * nx * ny;
                    const density = this.densities ? this.densities[index] : elements[index];
                    if (density > DENSITY_THRESHOLD) {
                        occupied[index] = 1;
                    }
                }
            }
        }

        const isOccupied = (x, y, z) => {
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return false;
            return occupied[x + y * nx + z * nx * ny] === 1;
        };

        // Collect all visible faces as triangles
        const faces = [];
        // Define 6 face directions with their vertex offsets and normals
        const faceDefinitions = [
            { dir: [0, 0, -1], verts: [[0,0,0],[1,0,0],[1,1,0],[0,1,0]] },  // Front (z-)
            { dir: [0, 0, 1],  verts: [[1,0,1],[0,0,1],[0,1,1],[1,1,1]] },  // Back (z+)
            { dir: [-1, 0, 0], verts: [[0,0,1],[0,0,0],[0,1,0],[0,1,1]] },  // Left (x-)
            { dir: [1, 0, 0],  verts: [[1,0,0],[1,0,1],[1,1,1],[1,1,0]] },  // Right (x+)
            { dir: [0, -1, 0], verts: [[0,0,1],[1,0,1],[1,0,0],[0,0,0]] },  // Bottom (y-)
            { dir: [0, 1, 0],  verts: [[0,1,0],[1,1,0],[1,1,1],[0,1,1]] },  // Top (y+)
        ];

        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    if (!isOccupied(x, y, z)) continue;
                    const index = x + y * nx + z * nx * ny;
                    const density = this.densities ? this.densities[index] : elements[index];

                    for (let fi = 0; fi < faceDefinitions.length; fi++) {
                        const face = faceDefinitions[fi];
                        const [dx, dy, dz] = face.dir;
                        // Skip face if neighbour in that direction is also occupied
                        if (isOccupied(x + dx, y + dy, z + dz)) continue;

                        const verts = face.verts.map(([vx, vy, vz]) =>
                            this.project3D(x + vx, y + vy, z + vz, nx, ny, nz)
                        );
                        const avgDepth = (verts[0].z + verts[1].z + verts[2].z + verts[3].z) / 4;
                        const key = `${x},${y},${z},${fi}`;

                        faces.push({ verts, density, normal: face.dir, avgDepth, key, projVerts: verts });
                    }
                }
            }
        }

        // Sort by depth (painter's algorithm — farthest first)
        faces.sort((a, b) => a.avgDepth - b.avgDepth);

        // Cache boundary faces for paint picking
        this._boundaryFaces = faces;

        const lightDir = { x: 0.3, y: 0.5, z: 0.8 };
        const lightMag = Math.sqrt(lightDir.x ** 2 + lightDir.y ** 2 + lightDir.z ** 2);

        faces.forEach(face => {
            const screenVerts = face.verts.map(pv => ({
                x: centerX + pv.x * scale,
                y: centerY - pv.y * scale
            }));

            // Color based on density
            let r, g, b;
            if (this.densities) {
                r = Math.floor(face.density * 255);
                g = 76;
                b = Math.floor((1 - face.density) * 255);
            } else {
                r = 74;
                g = 144;
                b = 226;
            }

            // Apply per-face lighting from normal
            const dot = Math.abs(
                (face.normal[0] * lightDir.x + face.normal[1] * lightDir.y + face.normal[2] * lightDir.z) / lightMag
            );
            const lightFactor = 0.4 + 0.6 * dot;
            r = Math.floor(r * lightFactor);
            g = Math.floor(g * lightFactor);
            b = Math.floor(b * lightFactor);

            // Draw quad as two triangles
            if (this.wireframe) {
                this.ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                this.ctx.beginPath();
                this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
                this.ctx.lineTo(screenVerts[1].x, screenVerts[1].y);
                this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
                this.ctx.closePath();
                this.ctx.stroke();
                this.ctx.beginPath();
                this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
                this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
                this.ctx.lineTo(screenVerts[3].x, screenVerts[3].y);
                this.ctx.closePath();
                this.ctx.stroke();
            } else {
                this.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                this.ctx.beginPath();
                this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
                this.ctx.lineTo(screenVerts[1].x, screenVerts[1].y);
                this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
                this.ctx.closePath();
                this.ctx.fill();
                this.ctx.beginPath();
                this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
                this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
                this.ctx.lineTo(screenVerts[3].x, screenVerts[3].y);
                this.ctx.closePath();
                this.ctx.fill();
            }
        });
    }

    /** Draw semi-transparent green overlay on constraint-painted faces */
    drawConstraintOverlay(nx, ny, nz) {
        if (this.paintedConstraintFaces.size === 0) return;
        const { width, height } = this.canvas;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;

        for (const bf of this._boundaryFaces) {
            if (!this.paintedConstraintFaces.has(bf.key)) continue;
            const screenVerts = bf.projVerts.map(pv => ({
                x: centerX + pv.x * scale,
                y: centerY - pv.y * scale
            }));

            this.ctx.fillStyle = 'rgba(0, 200, 100, 0.45)';
            this.ctx.beginPath();
            this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
            this.ctx.lineTo(screenVerts[1].x, screenVerts[1].y);
            this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
            this.ctx.lineTo(screenVerts[3].x, screenVerts[3].y);
            this.ctx.closePath();
            this.ctx.fill();

            // Draw hatching lines for fixed constraint visualization
            this.ctx.strokeStyle = 'rgba(0, 150, 70, 0.7)';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
            this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
            this.ctx.moveTo(screenVerts[1].x, screenVerts[1].y);
            this.ctx.lineTo(screenVerts[3].x, screenVerts[3].y);
            this.ctx.stroke();
        }
    }

    /** Draw force arrows on painted force faces */
    drawForceArrows(nx, ny, nz) {
        if (this.paintedForceFaces.size === 0) return;
        const { width, height } = this.canvas;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;

        // Determine arrow direction in 3D based on forceDirection
        let arrowDir;
        switch (this.forceDirection) {
            case 'down':  arrowDir = [0, -1, 0]; break;
            case 'up':    arrowDir = [0, 1, 0]; break;
            case 'left':  arrowDir = [-1, 0, 0]; break;
            case 'right': arrowDir = [1, 0, 0]; break;
            default:      arrowDir = [0, -1, 0];
        }

        const arrowLen = 1.5; // length in world units

        for (const bf of this._boundaryFaces) {
            if (!this.paintedForceFaces.has(bf.key)) continue;
            const screenVerts = bf.projVerts.map(pv => ({
                x: centerX + pv.x * scale,
                y: centerY - pv.y * scale
            }));

            // Highlight the face in orange
            this.ctx.fillStyle = 'rgba(255, 100, 50, 0.4)';
            this.ctx.beginPath();
            this.ctx.moveTo(screenVerts[0].x, screenVerts[0].y);
            this.ctx.lineTo(screenVerts[1].x, screenVerts[1].y);
            this.ctx.lineTo(screenVerts[2].x, screenVerts[2].y);
            this.ctx.lineTo(screenVerts[3].x, screenVerts[3].y);
            this.ctx.closePath();
            this.ctx.fill();

            // Parse face key to get voxel position: "x,y,z,fi"
            const parts = bf.key.split(',');
            const vx = parseInt(parts[0], 10) + 0.5;
            const vy = parseInt(parts[1], 10) + 0.5;
            const vz = parseInt(parts[2], 10) + 0.5;

            // Arrow start: face center in 3D
            const startProj = this.project3D(vx, vy, vz, nx, ny, nz);
            const endProj = this.project3D(
                vx + arrowDir[0] * arrowLen,
                vy + arrowDir[1] * arrowLen,
                vz + arrowDir[2] * arrowLen,
                nx, ny, nz
            );

            const sx = centerX + startProj.x * scale;
            const sy = centerY - startProj.y * scale;
            const ex = centerX + endProj.x * scale;
            const ey = centerY - endProj.y * scale;

            // Draw arrow line
            this.ctx.strokeStyle = 'rgba(255, 60, 30, 0.9)';
            this.ctx.lineWidth = 2.5;
            this.ctx.beginPath();
            this.ctx.moveTo(sx, sy);
            this.ctx.lineTo(ex, ey);
            this.ctx.stroke();

            // Draw arrowhead
            const angle = Math.atan2(ey - sy, ex - sx);
            const headLen = 8;
            this.ctx.beginPath();
            this.ctx.moveTo(ex, ey);
            this.ctx.lineTo(ex - headLen * Math.cos(angle - 0.4), ey - headLen * Math.sin(angle - 0.4));
            this.ctx.moveTo(ex, ey);
            this.ctx.lineTo(ex - headLen * Math.cos(angle + 0.4), ey - headLen * Math.sin(angle + 0.4));
            this.ctx.stroke();
            this.ctx.lineWidth = 1;
        }
    }

    project3D(x, y, z, nx, ny, nz) {
        // Center the coordinates
        x = x - nx / 2;
        y = y - ny / 2;
        z = z - nz / 2;
        
        // Rotate around Y axis
        const cosY = Math.cos(this.rotation.y);
        const sinY = Math.sin(this.rotation.y);
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;
        
        // Rotate around X axis
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

    drawAxes(centerX, centerY, scale) {
        const axisLength = scale * 2;
        
        // X axis (red)
        const xEnd = this.project3D(axisLength / scale, 0, 0, 1, 1, 1);
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY);
        this.ctx.lineTo(centerX + xEnd.x * scale, centerY - xEnd.y * scale);
        this.ctx.stroke();
        
        // Y axis (green)
        const yEnd = this.project3D(0, axisLength / scale, 0, 1, 1, 1);
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY);
        this.ctx.lineTo(centerX + yEnd.x * scale, centerY - yEnd.y * scale);
        this.ctx.stroke();
        
        // Z axis (blue)
        const zEnd = this.project3D(0, 0, axisLength / scale, 1, 1, 1);
        this.ctx.strokeStyle = '#0000ff';
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, centerY);
        this.ctx.lineTo(centerX + zEnd.x * scale, centerY - zEnd.y * scale);
        this.ctx.stroke();
        
        this.ctx.lineWidth = 1;
    }

    updateMesh(meshData) {
        this.meshData = meshData;
        this.draw();
    }

    updateDensities(densities) {
        this.densities = densities;
        this.draw();
    }

    toggleWireframe() {
        this.wireframe = !this.wireframe;
        this.draw();
    }

    resetCamera() {
        this.rotation = { x: 0.5, y: 0.5 };
        this.zoom = 1;
        this.draw();
    }

    clear() {
        this.model = null;
        this.densities = null;
        this.meshData = null;
        this.paintedConstraintFaces = new Set();
        this.paintedForceFaces = new Set();
        this.paintMode = null;
        this._boundaryFaces = [];
        this.draw();
    }
}
