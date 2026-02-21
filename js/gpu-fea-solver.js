// GPU-accelerated FEA solver using WebGPU compute shaders
// Implements a complete Jacobi-preconditioned Conjugate Gradient (CG) solver
// that keeps all data on GPU during CG iterations - no readbacks until convergence.
//
// Cross-compatible: works in browser (navigator.gpu) and Node.js (via 'webgpu' / dawn.node)
//
// Key GPU operations:
//   applyA (EbE mat-vec): each element gathers 24 DOFs, does 24x24 mat-vec, scatter-adds to global vector
//   dotProduct: workgroup-level tree reduction, then CPU sums partials
//   axpy/update: combined vector updates
//   Jacobi preconditioner: diagonal scaling

/**
 * Obtain a GPU instance that works in both browser and Node.js.
 * Browser  → navigator.gpu  (native)
 * Node.js  → 'webgpu' npm package (Dawn bindings)
 *
 * The result is cached so that only ONE Dawn instance exists per process
 * (multiple Dawn instances on the same D3D12 device can crash).
 * @returns {Promise<GPU|null>}
 */
let _cachedGPU = null;
let _gpuProbed = false;
export async function _getGPU() {
    if (_cachedGPU) return _cachedGPU;
    if (_gpuProbed) return null;
    _gpuProbed = true;

    // 1. Browser path
    if (typeof navigator !== 'undefined' && navigator.gpu) {
        _cachedGPU = navigator.gpu;
        return _cachedGPU;
    }

    // 2. Node.js path – try to load the 'webgpu' package (dawn.node)
    //    Dawn's create() can crash the process on systems without a GPU,
    //    so we probe in a subprocess first.
    try {
        const mod = await import('webgpu');
        if (typeof mod.create !== 'function') return null;

        // Probe Dawn in a subprocess to avoid crashing the main process
        // (Dawn's create() can abort with a native assertion failure when no GPU is present)
        // Uses execFileSync (no shell) to avoid Windows cmd.exe single-quote issues.
        const { execFileSync } = await import('child_process');
        const probeScript = [
            'import { create, globals } from "webgpu";',
            'Object.assign(globalThis, globals);',
            'const g = create([]);',
            'const a = await g.requestAdapter();',
            'process.exit(a ? 0 : 1);',
        ].join(' ');
        try {
            execFileSync(process.execPath, ['--input-type=module', '-e', probeScript],
                { timeout: 8000, stdio: 'ignore' }
            );
        } catch (_probeErr) {
            // Subprocess crashed or returned non-zero → Dawn not usable
            return null;
        }

        // Probe succeeded – safe to create in main process
        const { create, globals } = mod;
        if (typeof GPUBufferUsage === 'undefined') Object.assign(globalThis, globals);
        _cachedGPU = create([]);
        return _cachedGPU;
    } catch (_) {
        // Package not installed or Dawn init failed – GPU won't be available
    }
    return null;
}

export class GPUFEASolver {
    constructor() {
        this.device = null;
        this.available = false;
        this._initPromise = null;
        this._buffers = null;
        this._pipelines = null;
        this._dotBindGroups = null;
        this._nel = 0;
        this._ndof = 0;
        this._fixedMaskF32 = null;
        this._WG_SIZE = 64;
        this._DOT_WG_SIZE = 64;
    }

    /**
     * Initialize WebGPU device and check availability.
     * @returns {Promise<boolean>} Whether GPU compute is available
     */
    async init() {
        if (this._initPromise) return this._initPromise;
        this._initPromise = this._doInit();
        return this._initPromise;
    }

    async _doInit() {
        try {
            const gpu = await _getGPU();
            if (!gpu) {
                console.log('WebGPU not supported in this environment');
                return false;
            }
            const adapter = await gpu.requestAdapter();
            if (!adapter) {
                console.log('No WebGPU adapter available');
                return false;
            }
            // Request higher buffer limits if the adapter supports them
            const adapterLimits = adapter.limits || {};
            const maxBuf = adapterLimits.maxStorageBufferBindingSize || 134217728;
            this.device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: maxBuf,
                    maxBufferSize: adapterLimits.maxBufferSize || 268435456,
                },
            });
            this.available = true;
            console.log('GPUFEASolver: WebGPU initialized successfully');
            return true;
        } catch (error) {
            console.log('GPUFEASolver: WebGPU initialization failed:', error.message);
            this.available = false;
            return false;
        }
    }

    /** Check if GPU FEA solver is available. */
    isAvailable() {
        return this.available && this.device !== null;
    }

    /**
     * Upload mesh data and prepare GPU buffers and pipelines for the CG solve.
     * Call once per problem (or when mesh / densities change).
     *
     * @param {Object} params
     * @param {Float32Array} params.KEflat    - Flattened 24x24 element stiffness matrix (576 floats)
     * @param {Int32Array}   params.edofArray - Element DOF indices (nel * 24)
     * @param {Float32Array} params.densities - Element densities (nel)
     * @param {Float32Array} params.F         - Global force vector (ndof)
     * @param {Uint8Array}   params.fixedMask - 1 if DOF is fixed, 0 otherwise (ndof)
     * @param {number}       params.nel       - Number of elements
     * @param {number}       params.ndof      - Total number of DOFs
     * @param {number}       params.E0        - Young's modulus of solid
     * @param {number}       params.Emin      - Minimum Young's modulus (ersatz)
     * @param {number}       params.penal     - SIMP penalization exponent
     */
    setup(params) {
        if (!this.isAvailable()) {
            throw new Error('GPUFEASolver: GPU not available');
        }
        const { KEflat, edofArray, densities, fixedMask, F,
                nel, ndof, E0, Emin, penal } = params;

        this._nel = nel;
        this._ndof = ndof;
        this._destroyBuffers();

        // Precompute element Young's moduli on CPU
        const dE = E0 - Emin;
        const E_vals = new Float32Array(nel);
        for (let e = 0; e < nel; e++) {
            E_vals[e] = Emin + Math.pow(densities[e], penal) * dE;
        }

        // Precompute Jacobi inverse-diagonal on CPU
        const diag = new Float32Array(ndof);
        for (let e = 0; e < nel; e++) {
            const E = E_vals[e];
            const eOff = e * 24;
            for (let i = 0; i < 24; i++) {
                diag[edofArray[eOff + i]] += E * KEflat[i * 24 + i];
            }
        }
        const invDiag = new Float32Array(ndof);
        for (let i = 0; i < ndof; i++) {
            if (!fixedMask[i] && diag[i] > 1e-20) {
                invDiag[i] = 1.0 / diag[i];
            }
        }

        // Convert fixedMask to f32 for GPU shaders
        this._fixedMaskF32 = new Float32Array(ndof);
        for (let i = 0; i < ndof; i++) {
            this._fixedMaskF32[i] = fixedMask[i] ? 1.0 : 0.0;
        }

        this._createBuffers(KEflat, edofArray, E_vals, invDiag, F, nel, ndof);
        this._createPipelines();
    }

    /** Create all GPU buffers. */
    _createBuffers(KEflat, edofArray, E_vals, invDiag, F, nel, ndof) {
        const device = this.device;
        const dotNumGroups = Math.ceil(ndof / this._DOT_WG_SIZE);

        // Max workgroups per dimension (WebGPU spec limit)
        const MAX_WG_DIM = 65535;
        // For 2D dispatch of applyALocal (1 workgroup = 1 element)
        this._dispatchX_nel = Math.min(nel, MAX_WG_DIM);
        this._dispatchY_nel = Math.ceil(nel / this._dispatchX_nel);

        const upload = (data, usage) => {
            const buf = device.createBuffer({ size: data.byteLength, usage });
            device.queue.writeBuffer(buf, 0, data);
            return buf;
        };
        const SRD = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const SRW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

        this._buffers = {
            // Read-only mesh data
            KE:        upload(KEflat, SRD),
            edofs:     upload(edofArray, SRD),
            E_vals:    upload(E_vals, SRD),
            invDiag:   upload(invDiag, SRD),
            fixedMask: upload(this._fixedMaskF32, SRD),
            F:         upload(F, SRD),

            // CG vectors (persistent on GPU across iterations)
            x:  device.createBuffer({ size: ndof * 4, usage: SRW }),
            r:  device.createBuffer({ size: ndof * 4, usage: SRW }),
            z:  device.createBuffer({ size: ndof * 4, usage: SRW }),
            p:  device.createBuffer({ size: ndof * 4, usage: SRW }),
            Ap: device.createBuffer({ size: ndof * 4, usage: SRW }),

            // Per-element contributions for applyA (nel * 24 floats)
            elemContribs: device.createBuffer({
                size: nel * 24 * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),

            // Atomic scatter buffer for applyA
            Ap_atomic: device.createBuffer({ size: ndof * 4, usage: SRW }),

            // Dot-product partial sums
            // Dawn / some drivers require storage-buffer sizes >= 4-byte aligned and > 0
            partials: device.createBuffer({
                size: Math.max(dotNumGroups * 4, 16),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            }),
            partialsRead: device.createBuffer({
                size: Math.max(dotNumGroups * 4, 16),
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            }),

            // Uniforms: [nel, ndof, dispatchX_nel, 0]
            uniforms: device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),

            // Scalar uniform for alpha / beta
            scalarUniform: device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),

            // Readback buffer for final solution
            readback: device.createBuffer({
                size: ndof * 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            }),
        };

        device.queue.writeBuffer(
            this._buffers.uniforms, 0,
            new Uint32Array([nel, ndof, this._dispatchX_nel, 0])
        );
    }

    /** Create all compute pipelines and bind groups. */
    _createPipelines() {
        const device = this.device;
        const b = this._buffers;

        // ──── applyA pass 1: local element mat-vec ────
        // Each workgroup = 1 element (24 threads). Gathers 24 DOFs, computes
        // KE * p_local, stores E * (KE * p_local) in elemContribs[e*24+i].
        const applyALocalShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> KE: array<f32>;
            @group(0) @binding(1) var<storage, read> edofs: array<i32>;
            @group(0) @binding(2) var<storage, read> E_vals: array<f32>;
            @group(0) @binding(3) var<storage, read> pVec: array<f32>;
            @group(0) @binding(4) var<storage, read_write> elemContribs: array<f32>;
            @group(0) @binding(5) var<uniform> params: vec4<u32>;
            var<workgroup> lp: array<f32, 24>;
            @compute @workgroup_size(24)
            fn main(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_id) lid: vec3<u32>) {
                let e = wid.x + wid.y * params.z;
                if (e >= params.x) { return; }
                let i = lid.x;
                let off = e * 24u;
                lp[i] = pVec[edofs[off + i]];
                workgroupBarrier();
                var s: f32 = 0.0;
                for (var j: u32 = 0u; j < 24u; j++) { s += KE[i * 24u + j] * lp[j]; }
                elemContribs[off + i] = E_vals[e] * s;
            }
        ` });
        const applyALocalPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: applyALocalShader, entryPoint: 'main' },
        });
        const applyALocalBG = device.createBindGroup({
            layout: applyALocalPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.KE } },
                { binding: 1, resource: { buffer: b.edofs } },
                { binding: 2, resource: { buffer: b.E_vals } },
                { binding: 3, resource: { buffer: b.p } },
                { binding: 4, resource: { buffer: b.elemContribs } },
                { binding: 5, resource: { buffer: b.uniforms } },
            ],
        });
        const applyALocalBGx = device.createBindGroup({
            layout: applyALocalPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.KE } },
                { binding: 1, resource: { buffer: b.edofs } },
                { binding: 2, resource: { buffer: b.E_vals } },
                { binding: 3, resource: { buffer: b.x } },
                { binding: 4, resource: { buffer: b.elemContribs } },
                { binding: 5, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── applyA pass 2: scatter-add via CAS-based atomic f32 add ────
        const applyAScatterShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> edofs: array<i32>;
            @group(0) @binding(1) var<storage, read> elemContribs: array<f32>;
            @group(0) @binding(2) var<storage, read_write> ApAtomic: array<atomic<u32>>;
            @group(0) @binding(3) var<uniform> params: vec4<u32>;
            fn atomicAddF32(p: ptr<storage, atomic<u32>, read_write>, v: f32) {
                var o = atomicLoad(p);
                loop {
                    let r = atomicCompareExchangeWeak(p, o, bitcast<u32>(bitcast<f32>(o) + v));
                    if (r.exchanged) { break; }
                    o = r.old_value;
                }
            }
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x;
                if (idx >= params.x * 24u) { return; }
                atomicAddF32(&ApAtomic[u32(edofs[idx])], elemContribs[idx]);
            }
        ` });
        const applyAScatterPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: applyAScatterShader, entryPoint: 'main' },
        });
        const applyAScatterBG = device.createBindGroup({
            layout: applyAScatterPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.edofs } },
                { binding: 1, resource: { buffer: b.elemContribs } },
                { binding: 2, resource: { buffer: b.Ap_atomic } },
                { binding: 3, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── applyA pass 3: copy atomic u32 -> f32 Ap, zero fixed DOFs ────
        const copyAtomicShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> ApAtomic: array<u32>;
            @group(0) @binding(1) var<storage, read_write> ApVec: array<f32>;
            @group(0) @binding(2) var<storage, read> fixedMask: array<f32>;
            @group(0) @binding(3) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= params.y) { return; }
                if (fixedMask[i] > 0.5) { ApVec[i] = 0.0; }
                else { ApVec[i] = bitcast<f32>(ApAtomic[i]); }
            }
        ` });
        const copyAtomicPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: copyAtomicShader, entryPoint: 'main' },
        });
        const copyAtomicBG = device.createBindGroup({
            layout: copyAtomicPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.Ap_atomic } },
                { binding: 1, resource: { buffer: b.Ap } },
                { binding: 2, resource: { buffer: b.fixedMask } },
                { binding: 3, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── Zero atomic buffer ────
        const zeroShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read_write> buf: array<atomic<u32>>;
            @group(0) @binding(1) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= params.y) { return; }
                atomicStore(&buf[gid.x], 0u);
            }
        ` });
        const zeroPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: zeroShader, entryPoint: 'main' },
        });
        const zeroBG = device.createBindGroup({
            layout: zeroPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.Ap_atomic } },
                { binding: 1, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── dotProduct: workgroup tree reduction ────
        // Using wg_size 64 (not 256) keeps workgroup-memory pressure
        // low and avoids native crashes observed on some Dawn/D3D12 drivers when
        // workgroup shared memory is large.
        const dotShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> vecA: array<f32>;
            @group(0) @binding(1) var<storage, read> vecB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> partials: array<f32>;
            @group(0) @binding(3) var<uniform> params: vec4<u32>;
            var<workgroup> wg_scratch: array<f32, 64>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>,
                    @builtin(local_invocation_id)  lid: vec3<u32>,
                    @builtin(workgroup_id)         wid: vec3<u32>) {
                var v: f32 = 0.0;
                if (gid.x < params.y) { v = vecA[gid.x] * vecB[gid.x]; }
                wg_scratch[lid.x] = v;
                workgroupBarrier();
                for (var s: u32 = 32u; s > 0u; s >>= 1u) {
                    if (lid.x < s) { wg_scratch[lid.x] += wg_scratch[lid.x + s]; }
                    workgroupBarrier();
                }
                if (lid.x == 0u) { partials[wid.x] = wg_scratch[0]; }
            }
        ` });
        const dotPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: dotShader, entryPoint: 'main' },
        });

        // ──── Jacobi preconditioner: z = invDiag * r ────
        const jacobiShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> invDiag: array<f32>;
            @group(0) @binding(1) var<storage, read> rVec: array<f32>;
            @group(0) @binding(2) var<storage, read_write> zVec: array<f32>;
            @group(0) @binding(3) var<storage, read> fixedMask: array<f32>;
            @group(0) @binding(4) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= params.y) { return; }
                if (fixedMask[i] > 0.5) { zVec[i] = 0.0; }
                else { zVec[i] = invDiag[i] * rVec[i]; }
            }
        ` });
        const jacobiPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: jacobiShader, entryPoint: 'main' },
        });
        const jacobiBG = device.createBindGroup({
            layout: jacobiPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.invDiag } },
                { binding: 1, resource: { buffer: b.r } },
                { binding: 2, resource: { buffer: b.z } },
                { binding: 3, resource: { buffer: b.fixedMask } },
                { binding: 4, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── updateXR: x += alpha*p, r -= alpha*Ap ────
        const updateXRShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read_write> xVec: array<f32>;
            @group(0) @binding(1) var<storage, read_write> rVec: array<f32>;
            @group(0) @binding(2) var<storage, read> pVec: array<f32>;
            @group(0) @binding(3) var<storage, read> ApVec: array<f32>;
            @group(0) @binding(4) var<uniform> alpha: f32;
            @group(0) @binding(5) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= params.y) { return; }
                xVec[i] += alpha * pVec[i];
                rVec[i] -= alpha * ApVec[i];
            }
        ` });
        const updateXRPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: updateXRShader, entryPoint: 'main' },
        });
        const updateXRBG = device.createBindGroup({
            layout: updateXRPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.x } },
                { binding: 1, resource: { buffer: b.r } },
                { binding: 2, resource: { buffer: b.p } },
                { binding: 3, resource: { buffer: b.Ap } },
                { binding: 4, resource: { buffer: b.scalarUniform } },
                { binding: 5, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── updateP: p = z + beta*p ────
        const updatePShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read_write> pVec: array<f32>;
            @group(0) @binding(1) var<storage, read> zVec: array<f32>;
            @group(0) @binding(2) var<uniform> beta: f32;
            @group(0) @binding(3) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= params.y) { return; }
                pVec[i] = zVec[i] + beta * pVec[i];
            }
        ` });
        const updatePPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: updatePShader, entryPoint: 'main' },
        });
        const updatePBG = device.createBindGroup({
            layout: updatePPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.p } },
                { binding: 1, resource: { buffer: b.z } },
                { binding: 2, resource: { buffer: b.scalarUniform } },
                { binding: 3, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── initResidual: r = F - Ap, zero fixed DOFs ────
        const initResShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> F: array<f32>;
            @group(0) @binding(1) var<storage, read> ApVec: array<f32>;
            @group(0) @binding(2) var<storage, read_write> rVec: array<f32>;
            @group(0) @binding(3) var<storage, read> fixedMask: array<f32>;
            @group(0) @binding(4) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                if (i >= params.y) { return; }
                if (fixedMask[i] > 0.5) { rVec[i] = 0.0; }
                else { rVec[i] = F[i] - ApVec[i]; }
            }
        ` });
        const initResPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: initResShader, entryPoint: 'main' },
        });
        const initResBG = device.createBindGroup({
            layout: initResPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.F } },
                { binding: 1, resource: { buffer: b.Ap } },
                { binding: 2, resource: { buffer: b.r } },
                { binding: 3, resource: { buffer: b.fixedMask } },
                { binding: 4, resource: { buffer: b.uniforms } },
            ],
        });

        // ──── copyVec: dst = src ────
        const copyShader = device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var<storage, read> src: array<f32>;
            @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
            @group(0) @binding(2) var<uniform> params: vec4<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= params.y) { return; }
                dst[gid.x] = src[gid.x];
            }
        ` });
        const copyPipe = device.createComputePipeline({
            layout: 'auto',
            compute: { module: copyShader, entryPoint: 'main' },
        });
        const copyZtoPBG = device.createBindGroup({
            layout: copyPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: b.z } },
                { binding: 1, resource: { buffer: b.p } },
                { binding: 2, resource: { buffer: b.uniforms } },
            ],
        });

        // Store pipelines
        this._pipelines = {
            applyALocal:   { pipe: applyALocalPipe,   bg: applyALocalBG, bgX: applyALocalBGx },
            applyAScatter: { pipe: applyAScatterPipe,  bg: applyAScatterBG },
            copyAtomic:    { pipe: copyAtomicPipe,     bg: copyAtomicBG },
            zero:          { pipe: zeroPipe,           bg: zeroBG },
            dot:           { pipe: dotPipe,            layout: dotPipe.getBindGroupLayout(0) },
            jacobi:        { pipe: jacobiPipe,         bg: jacobiBG },
            updateXR:      { pipe: updateXRPipe,       bg: updateXRBG },
            updateP:       { pipe: updatePPipe,        bg: updatePBG },
            initResidual:  { pipe: initResPipe,        bg: initResBG },
            copyVec:       { pipe: copyPipe,           bgZtoP: copyZtoPBG },
        };

        // Pre-build dot-product bind groups for r*z, r*r, p*Ap
        const mkDotBG = (bufA, bufB) => device.createBindGroup({
            layout: this._pipelines.dot.layout,
            entries: [
                { binding: 0, resource: { buffer: bufA } },
                { binding: 1, resource: { buffer: bufB } },
                { binding: 2, resource: { buffer: b.partials } },
                { binding: 3, resource: { buffer: b.uniforms } },
            ],
        });
        this._dotBindGroups = {
            rz:  mkDotBG(b.r, b.z),
            rr:  mkDotBG(b.r, b.r),
            pAp: mkDotBG(b.p, b.Ap),
        };
    }

    /**
     * GPU dot product: dispatch reduction, read back partials, sum on CPU.
     * @param {GPUBindGroup} bg
     * @returns {Promise<number>}
     */
    async _gpuDot(bg) {
        const device = this.device;
        const ndof = this._ndof;
        const groups = Math.ceil(ndof / this._DOT_WG_SIZE);
        const b = this._buffers;
        // Dawn / some drivers require copy size >= 4-byte aligned and > 0
        const copyBytes = Math.max(groups * 4, 16);

        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this._pipelines.dot.pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(b.partials, 0, b.partialsRead, 0, copyBytes);
        device.queue.submit([enc.finish()]);

        // Read back (must copy via .slice(0) before unmap detaches the buffer)
        await b.partialsRead.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(b.partialsRead.getMappedRange().slice(0));
        b.partialsRead.unmap();

        let sum = 0;
        for (let i = 0; i < groups; i++) sum += data[i];
        return sum;
    }

    /**
     * Encode the full applyA (EbE mat-vec) into a command encoder.
     * Four compute passes: zero -> local mat-vec -> scatter-add -> copy/mask.
     * @param {GPUCommandEncoder} enc
     * @param {boolean} useX - true to multiply A*x, false for A*p
     */
    _encodeApplyA(enc, useX) {
        const nel = this._nel;
        const ndof = this._ndof;
        const pip = this._pipelines;
        const wg = this._WG_SIZE;

        // Zero the atomic buffer
        let pass = enc.beginComputePass();
        pass.setPipeline(pip.zero.pipe);
        pass.setBindGroup(0, pip.zero.bg);
        pass.dispatchWorkgroups(Math.ceil(ndof / wg));
        pass.end();

        // Local element mat-vec -> elemContribs (2D dispatch for nel > 65535)
        pass = enc.beginComputePass();
        pass.setPipeline(pip.applyALocal.pipe);
        pass.setBindGroup(0, useX ? pip.applyALocal.bgX : pip.applyALocal.bg);
        pass.dispatchWorkgroups(this._dispatchX_nel, this._dispatchY_nel, 1);
        pass.end();

        // Scatter-add element contributions using atomic CAS
        pass = enc.beginComputePass();
        pass.setPipeline(pip.applyAScatter.pipe);
        pass.setBindGroup(0, pip.applyAScatter.bg);
        pass.dispatchWorkgroups(Math.ceil(nel * 24 / wg));
        pass.end();

        // Copy atomic -> f32 Ap, zero fixed DOFs
        pass = enc.beginComputePass();
        pass.setPipeline(pip.copyAtomic.pipe);
        pass.setBindGroup(0, pip.copyAtomic.bg);
        pass.dispatchWorkgroups(Math.ceil(ndof / wg));
        pass.end();
    }

    /**
     * Solve Ku = F using Jacobi-preconditioned CG entirely on GPU.
     * Only dot-product partial sums are read back each iteration;
     * the final solution is read back after convergence.
     *
     * @param {Object}       [options]
     * @param {number}       [options.maxIterations=400] - Max CG iterations
     * @param {number}       [options.tolerance=1e-8]    - Relative tolerance
     * @param {Float32Array} [options.warmStart]         - Initial guess (ndof)
     * @returns {Promise<{U: Float32Array, iterations: number, converged: boolean}>}
     */
    async solve(options = {}) {
        if (!this.isAvailable() || !this._pipelines || !this._buffers) {
            throw new Error('GPUFEASolver: not initialized. Call init() and setup() first.');
        }

        const { maxIterations = 400, tolerance = 1e-8, warmStart = null } = options;
        const device = this.device;
        const ndof = this._ndof;
        const b = this._buffers;
        const pip = this._pipelines;
        const wg = this._WG_SIZE;
        const EPSILON = 1e-12;

        // Upload initial guess
        if (warmStart) {
            device.queue.writeBuffer(b.x, 0, warmStart);
        } else {
            device.queue.writeBuffer(b.x, 0, new Float32Array(ndof));
        }

        // Batch init: Ap = A*x → r = F-Ap → z = M⁻¹r → p = z
        // All four steps are encoded into a single command buffer.
        // The GPU queue serializes them; the subsequent _gpuDot mapAsync
        // provides the implicit sync before reading results.
        let enc = device.createCommandEncoder();
        this._encodeApplyA(enc, true);

        let pass = enc.beginComputePass();
        pass.setPipeline(pip.initResidual.pipe);
        pass.setBindGroup(0, pip.initResidual.bg);
        pass.dispatchWorkgroups(Math.ceil(ndof / wg));
        pass.end();

        pass = enc.beginComputePass();
        pass.setPipeline(pip.jacobi.pipe);
        pass.setBindGroup(0, pip.jacobi.bg);
        pass.dispatchWorkgroups(Math.ceil(ndof / wg));
        pass.end();

        pass = enc.beginComputePass();
        pass.setPipeline(pip.copyVec.pipe);
        pass.setBindGroup(0, pip.copyVec.bgZtoP);
        pass.dispatchWorkgroups(Math.ceil(ndof / wg));
        pass.end();

        device.queue.submit([enc.finish()]);

        // Compute initial rz and r0norm2 (mapAsync provides sync)
        let rz = await this._gpuDot(this._dotBindGroups.rz);
        const r0norm2 = await this._gpuDot(this._dotBindGroups.rr);
        const tolSq = tolerance * tolerance * Math.max(r0norm2, 1e-30);

        // ── CG iteration loop ──
        // Only dot-product readbacks require CPU-GPU sync (via mapAsync).
        // All other GPU dispatches are submitted without explicit waits;
        // the GPU queue serializes them and mapAsync provides the barrier.
        let iterations = 0;
        let converged = false;

        for (let iter = 0; iter < maxIterations; iter++) {
            // Convergence check: ||r||²
            const rnorm2 = await this._gpuDot(this._dotBindGroups.rr);
            if (rnorm2 < tolSq) { converged = true; break; }
            iterations = iter + 1;

            // Ap = A * p (4-pass applyA)
            enc = device.createCommandEncoder();
            this._encodeApplyA(enc, false);
            device.queue.submit([enc.finish()]);

            // alpha = rz / (p . Ap) — mapAsync syncs on prior applyA
            const pAp = await this._gpuDot(this._dotBindGroups.pAp);
            const alpha = rz / (pAp + EPSILON);

            // Batch: x += alpha*p; r -= alpha*Ap; then z = M⁻¹r
            device.queue.writeBuffer(b.scalarUniform, 0, new Float32Array([alpha]));
            enc = device.createCommandEncoder();
            pass = enc.beginComputePass();
            pass.setPipeline(pip.updateXR.pipe);
            pass.setBindGroup(0, pip.updateXR.bg);
            pass.dispatchWorkgroups(Math.ceil(ndof / wg));
            pass.end();
            pass = enc.beginComputePass();
            pass.setPipeline(pip.jacobi.pipe);
            pass.setBindGroup(0, pip.jacobi.bg);
            pass.dispatchWorkgroups(Math.ceil(ndof / wg));
            pass.end();
            device.queue.submit([enc.finish()]);

            // rz_new = r . z — mapAsync syncs on prior updateXR + jacobi
            const rz_new = await this._gpuDot(this._dotBindGroups.rz);
            const beta = rz_new / (rz + EPSILON);

            // p = z + beta*p (next iter's _gpuDot will sync on this)
            device.queue.writeBuffer(b.scalarUniform, 0, new Float32Array([beta]));
            enc = device.createCommandEncoder();
            pass = enc.beginComputePass();
            pass.setPipeline(pip.updateP.pipe);
            pass.setBindGroup(0, pip.updateP.bg);
            pass.dispatchWorkgroups(Math.ceil(ndof / wg));
            pass.end();
            device.queue.submit([enc.finish()]);

            rz = rz_new;
        }

        // ── Read back solution ──
        enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(b.x, 0, b.readback, 0, ndof * 4);
        device.queue.submit([enc.finish()]);

        await b.readback.mapAsync(GPUMapMode.READ);
        const U = new Float32Array(b.readback.getMappedRange().slice(0));
        b.readback.unmap();

        return { U, iterations, converged };
    }

    /** Destroy all GPU buffers. */
    _destroyBuffers() {
        if (this._buffers) {
            for (const key of Object.keys(this._buffers)) {
                const buf = this._buffers[key];
                if (buf && typeof buf.destroy === 'function') buf.destroy();
            }
            this._buffers = null;
        }
        this._pipelines = null;
        this._dotBindGroups = null;
    }

    /** Destroy GPU resources and release the device. */
    destroy() {
        this._destroyBuffers();
        if (this.device) {
            this.device.destroy();
            this.device = null;
        }
        this.available = false;
        this._initPromise = null;
    }
}
