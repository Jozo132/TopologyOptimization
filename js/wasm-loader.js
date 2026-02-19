// WASM loader for high-performance matrix operations
// Provides a JavaScript wrapper around the AssemblyScript WASM module

class WasmMatrixOps {
    constructor() {
        this.instance = null;
        this.memory = null;
        this.loaded = false;
    }

    async load() {
        try {
            // Resolve WASM path relative to the project root using import.meta.url.
            // This module lives in js/, so go up one level to reach wasm/.
            const baseUrl = new URL('..', import.meta.url).href;
            const response = await fetch(new URL('wasm/matrix-ops.wasm', baseUrl).href);
            const buffer = await response.arrayBuffer();
            const module = await WebAssembly.compile(buffer);
            
            this.instance = await WebAssembly.instantiate(module, {
                env: {
                    abort: () => console.error('WASM abort called'),
                    seed: () => Date.now()
                }
            });
            
            this.memory = this.instance.exports.memory;
            this.loaded = true;
            console.log('WASM matrix operations module loaded successfully');
            return true;
        } catch (error) {
            console.warn('Failed to load WASM module, falling back to pure JS:', error);
            this.loaded = false;
            return false;
        }
    }

    /**
     * Check if WASM is loaded and available
     */
    isAvailable() {
        return this.loaded && this.instance !== null;
    }

    /**
     * Copy Float64Array to WASM memory
     */
    _copyToWasm(arr) {
        const ptr = this.instance.exports.__new(arr.length, 4); // 4 = Float64Array ID
        const view = new Float64Array(this.memory.buffer, ptr, arr.length);
        view.set(arr);
        return ptr;
    }

    /**
     * Copy Int32Array to WASM memory
     */
    _copyInt32ToWasm(arr) {
        const ptr = this.instance.exports.__new(arr.length, 3); // 3 = Int32Array ID
        const view = new Int32Array(this.memory.buffer, ptr, arr.length);
        view.set(arr);
        return ptr;
    }

    /**
     * Read Float64Array from WASM memory
     */
    _readFromWasm(ptr, length) {
        const view = new Float64Array(this.memory.buffer, ptr, length);
        return Float64Array.from(view);
    }

    /**
     * Free WASM memory
     */
    _free(ptr) {
        if (this.instance.exports.__pin) {
            this.instance.exports.__unpin(ptr);
        }
    }

    /**
     * Matrix-vector multiplication: result = K * x
     */
    matVecMul(K, x, result, n) {
        if (!this.isAvailable()) {
            throw new Error('WASM not loaded');
        }

        const ptrK = this._copyToWasm(K);
        const ptrX = this._copyToWasm(x);
        const ptrResult = this._copyToWasm(result);

        this.instance.exports.matVecMul(ptrK, ptrX, ptrResult, n);

        const output = this._readFromWasm(ptrResult, n);
        result.set(output);

        this._free(ptrK);
        this._free(ptrX);
        this._free(ptrResult);
    }

    /**
     * Conjugate Gradient solver
     */
    conjugateGradient(K, F, U, n, maxIter, tolerance) {
        if (!this.isAvailable()) {
            throw new Error('WASM not loaded');
        }

        const ptrK = this._copyToWasm(K);
        const ptrF = this._copyToWasm(F);
        const ptrU = this._copyToWasm(U);

        const iterations = this.instance.exports.conjugateGradient(
            ptrK, ptrF, ptrU, n, maxIter, tolerance
        );

        const output = this._readFromWasm(ptrU, n);
        U.set(output);

        this._free(ptrK);
        this._free(ptrF);
        this._free(ptrU);

        return iterations;
    }

    /**
     * Apply density filter
     */
    applyDensityFilter(densities, filtered, nx, ny, nz, radius) {
        if (!this.isAvailable()) {
            throw new Error('WASM not loaded');
        }

        const ptrDensities = this._copyToWasm(densities);
        const ptrFiltered = this._copyToWasm(filtered);

        this.instance.exports.applyDensityFilter(
            ptrDensities, ptrFiltered, nx, ny, nz, radius
        );

        const output = this._readFromWasm(ptrFiltered, densities.length);
        filtered.set(output);

        this._free(ptrDensities);
        this._free(ptrFiltered);
    }

    /**
     * Compute element energies
     */
    computeElementEnergies(U, KE, energies, edofs, numElements, edofSize) {
        if (!this.isAvailable()) {
            throw new Error('WASM not loaded');
        }

        const ptrU = this._copyToWasm(U);
        const ptrKE = this._copyToWasm(KE);
        const ptrEnergies = this._copyToWasm(energies);
        const ptrEdofs = this._copyInt32ToWasm(edofs);

        this.instance.exports.computeElementEnergies(
            ptrU, ptrKE, ptrEnergies, ptrEdofs, numElements, edofSize
        );

        const output = this._readFromWasm(ptrEnergies, numElements);
        energies.set(output);

        this._free(ptrU);
        this._free(ptrKE);
        this._free(ptrEnergies);
        this._free(ptrEdofs);
    }

    /**
     * Full self-contained Element-By-Element Preconditioned Conjugate Gradient
     * (EbE-PCG) FEA solver.
     *
     * Performs the entire Jacobi-preconditioned CG solve in WASM, eliminating
     * per-iteration JS↔WASM boundary crossings.
     *
     * @param {Float64Array} densities - element densities (nel)
     * @param {Float64Array} KEflat - reference element stiffness (edofSize*edofSize)
     * @param {Int32Array} edofs - element DOF connectivity (nel*edofSize)
     * @param {Float64Array} F - global force vector (ndof), accepts Float32Array too
     * @param {Int32Array} freedofs - free DOF indices (nfree)
     * @param {number} nel - number of elements
     * @param {number} edofSize - DOFs per element (8 for 2D, 24 for 3D)
     * @param {number} ndof - total degrees of freedom
     * @param {number} Emin - minimum Young's modulus
     * @param {number} E0 - base Young's modulus
     * @param {number} penal - SIMP penalization exponent
     * @param {number} maxIter - maximum CG iterations
     * @param {number} tolerance - convergence tolerance
     * @returns {{ U: Float64Array, iterations: number, compliance: number }}
     */
    ebePCG(densities, KEflat, edofs, F, freedofs, nel, edofSize, ndof, Emin, E0, penal, maxIter, tolerance) {
        if (!this.isAvailable()) {
            throw new Error('WASM not loaded');
        }

        const mem = this.instance.exports.memory;
        const nfree = freedofs.length;
        const align8 = (v) => (v + 7) & ~7;

        // Convert F to Float64Array if necessary (the worker uses Float32Array for F)
        const F64 = F instanceof Float64Array ? F : Float64Array.from(F);

        // Calculate memory layout
        const densSize = nel * 8;
        const keSize = edofSize * edofSize * 8;
        const edofsSize = nel * edofSize * 4;
        const fSize = ndof * 8;
        const uSize = ndof * 8;
        const freedofsSize = nfree * 4;
        // Workspace: E_vals[nel] + active[nel] + diag[ndof] +
        //            Uf[nfree] + r[nfree] + z[nfree] + p[nfree] + Ap[nfree] +
        //            p_full[ndof] + Ap_full[ndof] + scratch[edofSize] + invDiag[nfree]
        const workSize = nel * 8 + nel * 4 + ndof * 8 +
                         5 * nfree * 8 + 2 * ndof * 8 + edofSize * 8 + nfree * 8;

        const totalBytes = densSize + keSize + edofsSize + fSize + uSize + freedofsSize + workSize + 128;
        const neededPages = Math.ceil(totalBytes / 65536) + 1;

        // Ensure enough WASM memory
        const currentBytes = mem.buffer.byteLength;
        if (currentBytes < totalBytes + 65536) {
            const additionalPages = Math.ceil((totalBytes + 65536 - currentBytes) / 65536);
            mem.grow(additionalPages);
        }
        const dataStart = mem.buffer.byteLength - totalBytes - 64;

        let offset = align8(dataStart);
        const densOff = offset; offset += densSize;
        const keOff = offset; offset += keSize;
        const edofsOff = offset; offset += edofsSize;
        offset = align8(offset);
        const fOff = offset; offset += fSize;
        const uOff = offset; offset += uSize;
        const freedofsOff = offset; offset += freedofsSize;
        offset = align8(offset);
        const workOff = offset;

        // Copy input data to WASM memory
        new Float64Array(mem.buffer, densOff, nel).set(densities);
        new Float64Array(mem.buffer, keOff, edofSize * edofSize).set(KEflat);
        new Int32Array(mem.buffer, edofsOff, nel * edofSize).set(edofs);
        new Float64Array(mem.buffer, fOff, ndof).set(F64);
        new Int32Array(mem.buffer, freedofsOff, nfree).set(freedofs);

        // Call WASM ebePCG
        const iterations = this.instance.exports.ebePCG(
            densOff, keOff, edofsOff, fOff, uOff, freedofsOff,
            nel, edofSize, ndof, nfree,
            Emin, E0, penal, maxIter, tolerance, workOff
        );

        // Read results
        const U = new Float64Array(ndof);
        U.set(new Float64Array(mem.buffer, uOff, ndof));

        // Compute compliance: c = F^T U
        let compliance = 0;
        for (let i = 0; i < ndof; i++) {
            compliance += F64[i] * U[i];
        }

        return { U, iterations, compliance };
    }
}

// Singleton instance
let wasmOpsInstance = null;

/**
 * Get or create the WASM operations instance
 */
export async function getWasmOps() {
    if (!wasmOpsInstance) {
        wasmOpsInstance = new WasmMatrixOps();
        await wasmOpsInstance.load();
    }
    return wasmOpsInstance;
}

/**
 * Check if WASM is available
 */
export function isWasmAvailable() {
    return wasmOpsInstance && wasmOpsInstance.isAvailable();
}
