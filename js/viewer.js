// 3D Viewer using Canvas 2D (no external dependencies)
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
            this.isDragging = true;
            this.lastMousePos = { x: e.clientX, y: e.clientY };
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
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
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom *= (1 - e.deltaY * 0.001);
            this.zoom = Math.max(0.1, Math.min(5, this.zoom));
            this.draw();
        });
    }

    onWindowResize() {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.draw();
    }

    setModel(model) {
        this.model = model;
        this.densities = null;
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
        
        const { nx, ny, nz, elements } = this.model;
        
        // Project 3D voxels to 2D
        const voxels = [];
        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    const index = x + y * nx + z * nx * ny;
                    const density = this.densities ? this.densities[index] : elements[index];
                    
                    if (density > 0.3) { // Threshold for visibility
                        voxels.push({ x, y, z, density });
                    }
                }
            }
        }
        
        // Sort by depth (painter's algorithm)
        voxels.sort((a, b) => {
            const depthA = this.getDepth(a.x, a.y, a.z);
            const depthB = this.getDepth(b.x, b.y, b.z);
            return depthA - depthB;
        });
        
        // Draw voxels
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) / Math.max(nx, ny, nz) * 0.7 * this.zoom;
        
        voxels.forEach(voxel => {
            const projected = this.project3D(voxel.x, voxel.y, voxel.z, nx, ny, nz);
            const screenX = centerX + projected.x * scale;
            const screenY = centerY - projected.y * scale;
            const size = scale * 0.8;
            
            // Color based on density
            let r, g, b;
            if (this.densities) {
                // Gradient from blue (low density) to red (high density)
                r = Math.floor(voxel.density * 255);
                g = Math.floor(76);
                b = Math.floor((1 - voxel.density) * 255);
            } else {
                // Default blue
                r = 74;
                g = 144;
                b = 226;
            }
            
            // Apply lighting based on depth
            const lightFactor = 0.5 + 0.5 * (projected.z + 1) / 2;
            r = Math.floor(r * lightFactor);
            g = Math.floor(g * lightFactor);
            b = Math.floor(b * lightFactor);
            
            this.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            
            if (this.wireframe) {
                this.ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                this.ctx.strokeRect(screenX - size / 2, screenY - size / 2, size, size);
            } else {
                this.ctx.fillRect(screenX - size / 2, screenY - size / 2, size, size);
            }
        });
        
        // Draw axes
        this.drawAxes(centerX, centerY, scale);
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
        this.draw();
    }
}
