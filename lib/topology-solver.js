/**
 * topology-solver.js
 *
 * Reusable topology optimization library for Web (browser) and Node.js.
 *
 * Features:
 *  - Auto-detects environment (browser / Node.js) and picks the best execution path
 *  - Spawns optimization workers internally — no manual worker setup required
 *  - Solver selection via `config.solver`: 'auto' | '2d' | '3d' | 'mgpcg' | 'cg' | 'petsc'
 *  - Pause / resume mid-optimization without losing solver state
 *  - Dynamic parameter updates via updateConfig() consumed at next iteration boundary
 *  - Exposes GPU compute (WebGPU/WebGL) and WASM acceleration automatically
 *  - Debug tools included (see `TopologySolver.debug`)
 *
 * Usage (browser):
 *   import { TopologySolver } from './lib/topology-solver.js';
 *   const solver = new TopologySolver();
 *   const result = await solver.optimize(model, { solver: 'auto', ... }, onProgress);
 *   solver.pause();   // pause at next iteration boundary
 *   solver.resume();  // resume
 *   solver.updateConfig({ penaltyFactor: 5 }); // update on the fly
 *   solver.cancel();  // abort
 *
 * Usage (Node.js):
 *   import { TopologySolver } from './lib/topology-solver.js';
 *   // Same API — worker_threads are used automatically
 */

// ─── Environment detection ────────────────────────────────────────────────────

/** True when running inside a browser window */
const IS_BROWSER = typeof window !== 'undefined' && typeof window.document !== 'undefined';

/** True when running in Node.js (not a browser, but has process.versions.node) */
const IS_NODE = !IS_BROWSER && typeof process !== 'undefined' && !!process.versions?.node;

// ─── Worker factory ───────────────────────────────────────────────────────────

/**
 * Resolve the URL of a worker script relative to this library file.
 * Works for both browser (import.meta.url) and Node.js.
 * @param {string} filename - basename inside js/
 * @returns {URL}
 */
function _workerUrl(filename) {
    return new URL(`../js/${filename}`, import.meta.url);
}

/**
 * Normalised worker wrapper so the same interface works in browser and Node.js.
 *
 * Browser  → native Web Worker (type: 'module')
 * Node.js  → worker_threads.Worker  (parentPort ↔ Web-Worker message bridge is
 *             already installed by the shim at the top of each worker file)
 */
class WorkerWrapper {
    constructor(worker, isNode) {
        this._w = worker;
        this._isNode = isNode;
        this.onmessage = null;
        this.onerror = null;

        if (isNode) {
            // worker_threads delivers the raw message data (no .data wrapper)
            worker.on('message', (data) => {
                if (this.onmessage) this.onmessage({ data });
            });
            worker.on('error', (err) => {
                if (this.onerror) this.onerror(err);
            });
        } else {
            // Web Worker delivers an Event with a .data property
            worker.onmessage = (e) => {
                if (this.onmessage) this.onmessage(e);
            };
            worker.onerror = (e) => {
                if (this.onerror) this.onerror(e);
            };
        }
    }

    postMessage(data) {
        this._w.postMessage(data);
    }

    terminate() {
        this._w.terminate();
    }
}

/**
 * Create and return a WorkerWrapper for the given worker script URL.
 * @param {URL} url
 * @returns {Promise<WorkerWrapper>}
 */
async function _createWorker(url) {
    if (IS_NODE) {
        const { Worker } = await import('worker_threads');
        const { fileURLToPath } = await import('url');
        const w = new Worker(fileURLToPath(url));
        return new WorkerWrapper(w, true);
    } else {
        const w = new Worker(url, { type: 'module' });
        return new WorkerWrapper(w, false);
    }
}

// ─── GPU availability probe ───────────────────────────────────────────────────

/**
 * Probe WebGPU availability (non-blocking, best-effort).
 * Returns 'webgpu', 'webgl', or 'none'.
 * @returns {Promise<string>}
 */
async function _probeGPU() {
    if (IS_NODE) return 'none';
    try {
        if (navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) return 'webgpu';
        }
    } catch (_) { /* ignore */ }
    try {
        const canvas = document.createElement('canvas');
        if (canvas.getContext('webgl2') || canvas.getContext('webgl')) return 'webgl';
    } catch (_) { /* ignore */ }
    return 'none';
}

// ─── Main solver class ────────────────────────────────────────────────────────

export class TopologySolver {
    constructor() {
        /** @type {WorkerWrapper|null} */
        this._worker = null;
        this._paused = false;
        this._cancelReject = null;
        this.lastPausedData = null;
        this.lastVolumetricData = null;
        this.onPaused = null;
        this.onVolumetric = null;

        /**
         * Environment info populated after the first optimize() call.
         * @type {{env: string, gpu: string}}
         */
        this.environment = {
            env: IS_NODE ? 'node' : (IS_BROWSER ? 'browser' : 'unknown'),
            gpu: 'unknown',
        };
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Run topology optimization in a background worker.
     *
     * @param {object} model
     *   { nx, ny, nz, elements?: Float32Array|number[], type?: string }
     * @param {object} config
     *   All standard optimization parameters, plus:
     *   - solver: 'auto' | '2d' | '3d' | 'mgpcg' | 'cg' | 'petsc' | 'fea' | 'genetic'
     *       'auto'    – pick 2-D or 3-D based on model geometry (default)
     *       '2d'      – force 2-D SIMP solver
     *       '3d'      – force 3-D SIMP solver with default linear solver
     *       'mgpcg'   – 3-D solver, multigrid-preconditioned CG (fastest for 3-D)
     *       'cg'      – 3-D solver, plain Jacobi-CG
     *       'petsc'   – 3-D solver, PETSc-style KSP with BDDC domain decomposition
     *                   (supports config.petscPC: 'bddc' | 'mg' | 'jacobi')
     *       'fea'     – single FEA solve only (no optimization), returns stress data
     *       'genetic' – genetic algorithm optimization (tournament selection, crossover, mutation)
     *   Genetic algorithm specific parameters (when solver: 'genetic'):
     *   - populationSize: number of individuals (default 20)
     *   - eliteCount:     top individuals to keep (default 2)
     *   - mutationRate:   per-element mutation probability (default 0.02)
     *   - crossoverRate:  probability of crossover vs clone (default 0.8)
     *   - tournamentSize: tournament selection size (default 3)
     *   - volumePenalty:  weight for volume constraint penalty (default 2.0)
     * @param {function} [progressCallback]
     *   (iteration, compliance, meshData, timing, maxStress) => void
     * @returns {Promise<object>} Final result object from the worker
     */
    optimize(model, config = {}, progressCallback) {
        // Using a regular executor (not async) to avoid the "async executor" anti-pattern.
        // All async work is done in a self-invoking async IIFE that properly propagates errors.
        return new Promise((resolve, reject) => {
            this._cancelReject = reject;
            this._paused = false;

            (async () => {
            // Probe GPU in browser (fire-and-forget, purely informational)
            _probeGPU().then((gpu) => { this.environment.gpu = gpu; });

            // ── Solver selection ──────────────────────────────────────────
            const solverKey = (config.solver || 'auto').toLowerCase();
            let use3D;
            let linearSolver;
            let feaOnly = false;
            let geneticOptimizer = false;

            if (solverKey === '2d') {
                use3D = false;
            } else if (solverKey === '3d') {
                use3D = true;
            } else if (solverKey === 'mgpcg') {
                use3D = true;
                linearSolver = 'mgpcg';
            } else if (solverKey === 'cg') {
                use3D = true;
                linearSolver = 'cg';
            } else if (solverKey === 'petsc') {
                use3D = true;
                linearSolver = 'petsc';
            } else if (solverKey === 'fea') {
                feaOnly = true;
                use3D = model.type === 'cube' || (model.nz != null && model.nz > 1);
            } else if (solverKey === 'fatigue') {
                feaOnly = true;
                use3D = model.type === 'cube' || (model.nz != null && model.nz > 1);
            } else if (solverKey === 'genetic') {
                geneticOptimizer = true;
                use3D = model.type === 'cube' || (model.nz != null && model.nz > 1);
            } else {
                // 'auto': choose based on model geometry
                use3D = model.type === 'cube' || (model.nz != null && model.nz > 1);
            }

            const workerFilename = use3D ? 'optimizer-worker-3d.js' : 'optimizer-worker.js';

            // ── Create worker ─────────────────────────────────────────────
            let worker;
            try {
                worker = await _createWorker(_workerUrl(workerFilename));
            } catch (err) {
                reject(new Error(`TopologySolver: failed to create worker — ${err.message}`));
                return;
            }
            this._worker = worker;

            // ── Wire up messages ──────────────────────────────────────────
            worker.onmessage = (e) => {
                const msg = e.data;
                switch (msg.type) {
                    case 'progress':
                        if (progressCallback) {
                            progressCallback(msg.iteration, msg.compliance, msg.meshData, msg.timing, msg.maxStress, msg.volumetricData || null);
                        }
                        break;
                    case 'paused':
                        this.lastPausedData = msg;
                        this.lastVolumetricData = msg.volumetricData || this.lastVolumetricData;
                        if (typeof this.onPaused === 'function') {
                            this.onPaused(msg);
                        }
                        break;
                    case 'volumetric':
                        this.lastVolumetricData = msg.volumetricData || null;
                        if (typeof this.onVolumetric === 'function') {
                            this.onVolumetric(msg);
                        }
                        break;
                    case 'complete':
                        this.lastVolumetricData = msg.result?.volumetricData || null;
                        worker.terminate();
                        this._worker = null;
                        this._cancelReject = null;
                        resolve(msg.result);
                        break;
                    case 'cancelled':
                        worker.terminate();
                        this._worker = null;
                        this._cancelReject = null;
                        reject(new Error('Optimization cancelled'));
                        break;
                }
            };

            worker.onerror = (err) => {
                worker.terminate();
                this._worker = null;
                reject(new Error(err.message || String(err)));
            };

            // ── Build worker config ───────────────────────────────────────
            const workerConfig = { ...config };
            // Remove library-level key before sending to worker
            delete workerConfig.solver;
            if (linearSolver) workerConfig.linearSolver = linearSolver;
            if (feaOnly) workerConfig.feaOnly = true;
            if (solverKey === 'fatigue') workerConfig.fatigueMode = true;
            if (geneticOptimizer) workerConfig.optimizer = 'genetic';

            // ── Start optimization ────────────────────────────────────────
            worker.postMessage({
                type: 'start',
                model: {
                    nx: model.nx,
                    ny: model.ny,
                    nz: model.nz,
                    type: model.type,
                    meshType: model.meshType || 'box',
                    elements: model.elements ? Array.from(model.elements) : undefined,
                },
                config: workerConfig,
            });
            })().catch(reject);
        });
    }

    /**
     * Run a single Finite Element Analysis on the model without optimization.
     * Returns stress data for visualization and analysis.
     *
     * This is a convenience wrapper around optimize() with solver: 'fea'.
     *
     * @param {object} model
     *   { nx, ny, nz, elements?: Float32Array|number[], type?: string }
     * @param {object} config
     *   Same as optimize() config, but solver is forced to 'fea'.
     *   Key parameters: constraintPosition, forceDirection, forceMagnitude,
     *   youngsModulus, poissonsRatio, penaltyFactor
     * @param {function} [progressCallback]
     *   (iteration, compliance, meshData, timing, maxStress) => void
     * @returns {Promise<object>} Result with stress data:
     *   { densities, finalCompliance, meshData, maxStress, elementStress,
     *     feaOnly: true, nx, ny, nz, timing }
     */
    analyzeFEA(model, config = {}, progressCallback) {
        return this.optimize(model, { ...config, solver: 'fea' }, progressCallback);
    }

    /**
     * Pause the running optimization at the next iteration boundary.
     * The worker posts a 'paused' acknowledgement message when it has halted.
     * State (densities, solver internals) is fully preserved.
     */
    pause() {
        if (this._worker) {
            this._paused = true;
            this._worker.postMessage({ type: 'pause' });
        }
    }

    /**
     * Resume a paused optimization.
     */
    resume() {
        if (this._worker && this._paused) {
            this._paused = false;
            this._worker.postMessage({ type: 'resume' });
        }
    }

    /**
     * Update optimization parameters on the fly.
     * The patch is applied at the start of the next iteration, so the current
     * iteration (including its FEA solve) is unaffected.
     *
     * Supported patch fields:
     *   penaltyFactor, filterRadius, volumeFraction, maxIterations, linearSolver
     *
     * @param {object} patch - Partial config object
     */
    updateConfig(patch) {
        if (this._worker) {
            this._worker.postMessage({ type: 'updateConfig', config: patch });
        }
    }

    /**
     * Request the latest available volumetric stress snapshot from a running worker.
     */
    requestVolumetricSnapshot() {
        if (this._worker) {
            this._worker.postMessage({ type: 'requestVolumetric' });
        } else if (this.lastVolumetricData && typeof this.onVolumetric === 'function') {
            this.onVolumetric({ type: 'volumetric', reason: 'cached', volumetricData: this.lastVolumetricData });
        }
    }

    /**
     * Cancel a running (or paused) optimization.
     * The returned Promise from optimize() will reject with 'Optimization cancelled'.
     */
    cancel() {
        if (this._worker) {
            this._worker.postMessage({ type: 'cancel' });
            // Force-terminate after a short grace period in case the worker is busy
            setTimeout(() => {
                if (this._worker) {
                    this._worker.terminate();
                    this._worker = null;
                    if (this._cancelReject) {
                        this._cancelReject(new Error('Optimization cancelled'));
                        this._cancelReject = null;
                    }
                }
            }, 300);
        }
    }

    // ── Static debug / info tools ────────────────────────────────────────────

    /**
     * Debug and introspection utilities.
     */
    static get debug() {
        return {
            /**
             * Probe the GPU environment.
             * @returns {Promise<string>} 'webgpu' | 'webgl' | 'none'
             */
            probeGPU: _probeGPU,

            /**
             * Detect current runtime environment.
             * @returns {{ env: string, workerType: string }}
             */
            detectEnvironment() {
                return {
                    env: IS_NODE ? 'node' : (IS_BROWSER ? 'browser' : 'unknown'),
                    workerType: IS_NODE ? 'worker_threads' : 'WebWorker',
                };
            },

            /**
             * Resolve the absolute URL that would be used for a given solver.
             * @param {'2d'|'3d'} solverType
             * @returns {string}
             */
            workerUrl(solverType) {
                const filename = solverType === '2d' ? 'optimizer-worker.js' : 'optimizer-worker-3d.js';
                return _workerUrl(filename).href;
            },
        };
    }
}

export default TopologySolver;
