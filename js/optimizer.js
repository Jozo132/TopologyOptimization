// Topology Optimizer - delegates to Web Worker for non-blocking computation
export class TopologyOptimizer {
    constructor() {
        this.worker = null;
    }

    /**
     * Run optimization in a Web Worker.
     * @param {object} model - The model with nx, ny, nz, elements
     * @param {object} config - Optimization configuration
     * @param {function} progressCallback - Called with (iteration, compliance, meshData)
     * @returns {Promise<object>} The optimization result
     */
    optimize(model, config, progressCallback) {
        return new Promise((resolve, reject) => {
            this._cancelReject = reject;
            this.worker = new Worker('js/optimizer-worker.js');

            this.worker.onmessage = (e) => {
                const { type } = e.data;

                if (type === 'progress') {
                    if (progressCallback) {
                        progressCallback(e.data.iteration, e.data.compliance, e.data.meshData);
                    }
                } else if (type === 'complete') {
                    this.worker.terminate();
                    this.worker = null;
                    this._cancelReject = null;
                    resolve(e.data.result);
                } else if (type === 'cancelled') {
                    this.worker.terminate();
                    this.worker = null;
                    this._cancelReject = null;
                    reject(new Error('Optimization cancelled'));
                }
            };

            this.worker.onerror = (err) => {
                this.worker.terminate();
                this.worker = null;
                reject(new Error(err.message));
            };

            this.worker.postMessage({
                type: 'start',
                model: { nx: model.nx, ny: model.ny, nz: model.nz },
                config
            });
        });
    }

    /**
     * Cancel a running optimization by terminating the worker.
     */
    cancel() {
        if (this.worker) {
            this.worker.postMessage({ type: 'cancel' });
            // Force-terminate after a short grace period and trigger rejection
            setTimeout(() => {
                if (this.worker) {
                    this.worker.terminate();
                    this.worker = null;
                    // Trigger the pending reject via _cancelReject if set
                    if (this._cancelReject) {
                        this._cancelReject(new Error('Optimization cancelled'));
                        this._cancelReject = null;
                    }
                }
            }, 200);
        }
    }
}
