// WebGPU Compute Module for GPU-accelerated matrix operations
// Provides GPU acceleration for the CG solver's matrix-vector multiply
// Falls back gracefully when WebGPU is not available

export class GPUCompute {
    constructor() {
        this.device = null;
        this.available = false;
        this._initPromise = null;
    }

    /**
     * Initialize WebGPU device and check availability
     * @returns {Promise<boolean>} Whether GPU compute is available
     */
    async init() {
        if (this._initPromise) return this._initPromise;
        
        this._initPromise = this._doInit();
        return this._initPromise;
    }

    async _doInit() {
        try {
            if (typeof navigator === 'undefined' || !navigator.gpu) {
                console.log('WebGPU not supported in this environment');
                return false;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.log('No WebGPU adapter available');
                return false;
            }

            this.device = await adapter.requestDevice();
            this.available = true;
            console.log('WebGPU compute initialized successfully');
            return true;
        } catch (error) {
            console.log('WebGPU initialization failed:', error.message);
            this.available = false;
            return false;
        }
    }

    /**
     * Check if GPU compute is available
     */
    isAvailable() {
        return this.available && this.device !== null;
    }

    /**
     * GPU-accelerated dense matrix-vector multiply: result = A * x
     * Useful for the WASM CG solver path where dense matrices are used
     * @param {Float64Array} A - Dense matrix (n x n, row-major)
     * @param {Float64Array} x - Input vector (n)
     * @param {number} n - Dimension
     * @returns {Promise<Float64Array>} Result vector
     */
    async matVecMul(A, x, n) {
        if (!this.isAvailable()) {
            throw new Error('GPU compute not available');
        }

        const device = this.device;

        // Use Float32 for GPU (WebGPU doesn't support f64 in compute shaders universally)
        const A32 = new Float32Array(A.length);
        const x32 = new Float32Array(x.length);
        for (let i = 0; i < A.length; i++) A32[i] = A[i];
        for (let i = 0; i < x.length; i++) x32[i] = x[i];

        // Create GPU buffers
        const matrixBuffer = device.createBuffer({
            size: A32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const vectorBuffer = device.createBuffer({
            size: x32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const resultBuffer = device.createBuffer({
            size: n * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readBuffer = device.createBuffer({
            size: n * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const uniformBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Write data
        device.queue.writeBuffer(matrixBuffer, 0, A32);
        device.queue.writeBuffer(vectorBuffer, 0, x32);
        device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([n]));

        // Shader for mat-vec multiply
        const shaderModule = device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> matrix: array<f32>;
                @group(0) @binding(1) var<storage, read> vec: array<f32>;
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;
                @group(0) @binding(3) var<uniform> params: u32;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let row = id.x;
                    let n = params;
                    if (row >= n) { return; }
                    
                    var sum: f32 = 0.0;
                    for (var j: u32 = 0u; j < n; j = j + 1u) {
                        sum = sum + matrix[row * n + j] * vec[j];
                    }
                    result[row] = sum;
                }
            `
        });

        // Create pipeline
        const pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' }
        });

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: matrixBuffer } },
                { binding: 1, resource: { buffer: vectorBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: uniformBuffer } },
            ]
        });

        // Dispatch compute
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(n / 64));
        passEncoder.end();

        // Copy result to readable buffer
        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, n * 4);
        device.queue.submit([commandEncoder.finish()]);

        // Read back
        await readBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(readBuffer.getMappedRange());
        const result64 = new Float64Array(n);
        for (let i = 0; i < n; i++) result64[i] = resultData[i];
        readBuffer.unmap();

        // Cleanup
        matrixBuffer.destroy();
        vectorBuffer.destroy();
        resultBuffer.destroy();
        readBuffer.destroy();
        uniformBuffer.destroy();

        return result64;
    }

    /**
     * GPU-accelerated element energy computation
     * Computes u^T * KE * u for all elements in parallel
     * @param {Float32Array} U - Global displacement vector
     * @param {Float32Array} KEflat - Flattened element stiffness matrix
     * @param {Int32Array} edofs - Element DOF indices (nel * edofSize)
     * @param {number} nel - Number of elements
     * @param {number} edofSize - DOFs per element (8 for 2D, 24 for 3D)
     * @returns {Promise<Float32Array>} Element energies
     */
    async computeElementEnergies(U, KEflat, edofs, nel, edofSize) {
        if (!this.isAvailable()) {
            throw new Error('GPU compute not available');
        }

        const device = this.device;

        // Create buffers
        const uBuffer = device.createBuffer({
            size: U.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const keBuffer = device.createBuffer({
            size: KEflat.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const edofBuffer = device.createBuffer({
            size: edofs.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const resultBuffer = device.createBuffer({
            size: nel * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readBuffer = device.createBuffer({
            size: nel * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const uniformBuffer = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(uBuffer, 0, U);
        device.queue.writeBuffer(keBuffer, 0, KEflat);
        device.queue.writeBuffer(edofBuffer, 0, edofs);
        device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([nel, edofSize]));

        const shaderModule = device.createShaderModule({
            code: `
                struct Params {
                    nel: u32,
                    edofSize: u32,
                }

                @group(0) @binding(0) var<storage, read> U: array<f32>;
                @group(0) @binding(1) var<storage, read> KE: array<f32>;
                @group(0) @binding(2) var<storage, read> edofs: array<i32>;
                @group(0) @binding(3) var<storage, read_write> energies: array<f32>;
                @group(0) @binding(4) var<uniform> params: Params;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let el = id.x;
                    if (el >= params.nel) { return; }
                    
                    let edofSize = params.edofSize;
                    let edofOffset = el * edofSize;
                    
                    var energy: f32 = 0.0;
                    for (var i: u32 = 0u; i < edofSize; i = i + 1u) {
                        let dofI = edofs[edofOffset + i];
                        if (dofI < 0) { continue; }
                        let ui = U[dofI];
                        
                        for (var j: u32 = 0u; j < edofSize; j = j + 1u) {
                            let dofJ = edofs[edofOffset + j];
                            if (dofJ < 0) { continue; }
                            let uj = U[dofJ];
                            energy = energy + ui * KE[i * edofSize + j] * uj;
                        }
                    }
                    energies[el] = energy;
                }
            `
        });

        const pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' }
        });

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uBuffer } },
                { binding: 1, resource: { buffer: keBuffer } },
                { binding: 2, resource: { buffer: edofBuffer } },
                { binding: 3, resource: { buffer: resultBuffer } },
                { binding: 4, resource: { buffer: uniformBuffer } },
            ]
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(nel / 64));
        passEncoder.end();

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, nel * 4);
        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();

        // Cleanup
        uBuffer.destroy();
        keBuffer.destroy();
        edofBuffer.destroy();
        resultBuffer.destroy();
        readBuffer.destroy();
        uniformBuffer.destroy();

        return result;
    }

    /**
     * Destroy GPU resources
     */
    destroy() {
        if (this.device) {
            this.device.destroy();
            this.device = null;
        }
        this.available = false;
    }
}
