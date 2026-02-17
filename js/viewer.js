// 3D Viewer using Three.js
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

export class Viewer3D {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.mesh = null;
        this.wireframe = false;
        this.initialCameraPosition = null;
    }

    async init() {
        this.container = document.getElementById(this.containerId);
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(30, 30, 30);
        this.initialCameraPosition = this.camera.position.clone();
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(10, 10, 10);
        this.scene.add(directionalLight1);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-10, -10, -10);
        this.scene.add(directionalLight2);
        
        // Grid
        const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x333333);
        this.scene.add(gridHelper);
        
        // Axes
        const axesHelper = new THREE.AxesHelper(15);
        this.scene.add(axesHelper);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start animation loop
        this.animate();
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    setModel(model) {
        // Remove existing mesh
        if (this.mesh) {
            this.scene.remove(this.mesh);
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
        }
        
        // Create geometry from voxel grid
        const geometry = this.createVoxelGeometry(model);
        
        // Create material
        const material = new THREE.MeshStandardMaterial({
            color: 0x4a90e2,
            metalness: 0.3,
            roughness: 0.7,
            wireframe: this.wireframe
        });
        
        this.mesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.mesh);
        
        // Center the model
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        this.mesh.position.sub(center);
        
        // Reset camera to view the model
        this.resetCamera();
    }

    createVoxelGeometry(model) {
        const geometry = new THREE.BufferGeometry();
        const vertices = [];
        const indices = [];
        const normals = [];
        
        const { nx, ny, nz, elements } = model;
        const voxelSize = 1;
        
        // Create voxels for solid elements
        let vertexIndex = 0;
        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                for (let z = 0; z < nz; z++) {
                    const index = x + y * nx + z * nx * ny;
                    if (elements[index] > 0.5) { // Only show solid elements
                        this.addVoxel(
                            vertices, indices, normals,
                            x * voxelSize, y * voxelSize, z * voxelSize,
                            voxelSize, vertexIndex
                        );
                        vertexIndex += 24; // 24 vertices per cube (4 per face * 6 faces)
                    }
                }
            }
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        geometry.setIndex(indices);
        
        return geometry;
    }

    addVoxel(vertices, indices, normals, x, y, z, size, startIndex) {
        const half = size / 2;
        
        // Define the 8 corners of the cube
        const corners = [
            [x - half, y - half, z - half],
            [x + half, y - half, z - half],
            [x + half, y + half, z - half],
            [x - half, y + half, z - half],
            [x - half, y - half, z + half],
            [x + half, y - half, z + half],
            [x + half, y + half, z + half],
            [x - half, y + half, z + half]
        ];
        
        // Define the 6 faces (each face has 4 vertices)
        const faces = [
            [0, 1, 2, 3], // Front
            [5, 4, 7, 6], // Back
            [4, 0, 3, 7], // Left
            [1, 5, 6, 2], // Right
            [3, 2, 6, 7], // Top
            [4, 5, 1, 0]  // Bottom
        ];
        
        const faceNormals = [
            [0, 0, -1],  // Front
            [0, 0, 1],   // Back
            [-1, 0, 0],  // Left
            [1, 0, 0],   // Right
            [0, 1, 0],   // Top
            [0, -1, 0]   // Bottom
        ];
        
        faces.forEach((face, faceIndex) => {
            const [i0, i1, i2, i3] = face;
            const normal = faceNormals[faceIndex];
            
            // Add 4 vertices for this face
            [i0, i1, i2, i3].forEach(i => {
                vertices.push(...corners[i]);
                normals.push(...normal);
            });
            
            // Add 2 triangles (6 indices) for this face
            const base = startIndex + faceIndex * 4;
            indices.push(base, base + 1, base + 2);
            indices.push(base, base + 2, base + 3);
        });
    }

    updateDensities(densities) {
        if (!this.mesh) return;
        
        // Update mesh based on density values
        // This is simplified - in production, you'd use instanced rendering
        const geometry = this.mesh.geometry;
        const positions = geometry.attributes.position.array;
        const colors = new Float32Array(positions.length);
        
        // Color based on density (blue = low, red = high)
        for (let i = 0; i < densities.length; i++) {
            const density = densities[i];
            const colorIndex = i * 3;
            
            // Gradient from blue (low) to red (high)
            colors[colorIndex] = density; // R
            colors[colorIndex + 1] = 0.3; // G
            colors[colorIndex + 2] = 1 - density; // B
        }
        
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        this.mesh.material.vertexColors = true;
        this.mesh.material.needsUpdate = true;
    }

    toggleWireframe() {
        if (this.mesh) {
            this.wireframe = !this.wireframe;
            this.mesh.material.wireframe = this.wireframe;
        }
    }

    resetCamera() {
        if (this.mesh) {
            // Calculate bounding sphere
            this.mesh.geometry.computeBoundingSphere();
            const radius = this.mesh.geometry.boundingSphere.radius;
            
            // Position camera to view the entire model
            const distance = radius * 2.5;
            this.camera.position.set(distance, distance, distance);
            this.camera.lookAt(0, 0, 0);
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }

    clear() {
        if (this.mesh) {
            this.scene.remove(this.mesh);
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
            this.mesh = null;
        }
    }
}
