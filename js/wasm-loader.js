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
