// Main application entry point
import { Viewer3D } from './viewer.js';
import { ModelImporter } from './importer.js';
import { TopologyOptimizer } from './optimizer.js';
import { ModelExporter } from './exporter.js';
import { WorkflowManager } from './workflow.js';

class TopologyApp {
    constructor() {
        this.viewer = null;
        this.importer = null;
        this.optimizer = null;
        this.exporter = null;
        this.workflow = null;
        
        this.currentModel = null;
        this.optimizedModel = null;
        this.config = {
            volumeFraction: 0.4,
            forceDirection: 'down',
            forceMagnitude: 1000,
            constraintPosition: 'left',
            maxIterations: 100,
            penaltyFactor: 3,
            filterRadius: 1.5,
            granuleDensity: 20,
            minCrossSection: 0,
            useAMR: true,
            minGranuleSize: 0.5,
            maxGranuleSize: 2
        };
        
        // Benchmark tracking
        this.benchmarkHistory = this.loadBenchmarkHistory();
    }

    async init() {
        console.log('Initializing Topology Optimization App...');
        
        // Initialize modules
        this.viewer = new Viewer3D('viewer3D');
        await this.viewer.init();
        
        this.importer = new ModelImporter();
        this.optimizer = new TopologyOptimizer();
        this.exporter = new ModelExporter();
        this.workflow = new WorkflowManager();
        this.workflow.init();
        
        // Clear paint mode when leaving step 2
        this.workflow.onStepChange = (step) => {
            if (step !== 2) {
                this.viewer.setPaintMode(null);
                document.getElementById('paintConstraint').classList.remove('active-tool');
                document.getElementById('paintForce').classList.remove('active-tool');
            }
        };
        
        // Setup event listeners
        this.setupEventListeners();
        
        console.log('App initialized successfully');
    }

    setupEventListeners() {
        // Step 1: Import
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) this.handleFileImport(file);
        });
        
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this.handleFileImport(file);
        });
        
        // Template buttons
        document.getElementById('useBeamTemplate').addEventListener('click', () => {
            this.loadTemplate('beam');
        });
        document.getElementById('useBridgeTemplate').addEventListener('click', () => {
            this.loadTemplate('bridge');
        });
        document.getElementById('useCubeTemplate').addEventListener('click', () => {
            this.loadTemplate('cube');
        });

        // Granule density slider
        document.getElementById('granuleDensity').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            this.config.granuleDensity = value;
            document.getElementById('granuleDensityValue').textContent = value;
            
            // Re-voxelize the current model if one exists
            if (this.currentModel) {
                let newModel = null;
                
                // Check if it's an STL model with original vertices
                if (this.currentModel.originalVertices) {
                    newModel = this.importer.voxelizeVertices(
                        this.currentModel.originalVertices,
                        value
                    );
                    // Preserve the model type if it was set
                    if (this.currentModel.type) {
                        newModel.type = this.currentModel.type;
                    }
                }
                // Check if it's a template with scaling info
                else if (this.currentModel.templateScale) {
                    newModel = this.importer.createTemplate(
                        this.currentModel.type,
                        value
                    );
                    // Preserve any custom boundary conditions
                    if (this.currentModel.forcePosition) {
                        newModel.forcePosition = this.currentModel.forcePosition;
                    }
                    if (this.currentModel.constraintPositions) {
                        newModel.constraintPositions = this.currentModel.constraintPositions;
                    }
                }
                
                if (newModel) {
                    this.currentModel = newModel;
                    
                    // Update display
                    const info = document.getElementById('modelInfo');
                    if (!info.classList.contains('hidden')) {
                        const modelName = newModel.type ? `${newModel.type} template` : 'Model';
                        const elementCount = newModel.nx * newModel.ny * newModel.nz;
                        info.innerHTML = `
                            <strong>${modelName} updated!</strong><br>
                            <strong>Elements:</strong> ${elementCount}<br>
                            <strong>Dimensions:</strong> ${newModel.nx} x ${newModel.ny} x ${newModel.nz}
                        `;
                    }
                    
                    // Update viewer with new voxel grid
                    this.viewer.setModel(newModel);
                }
            }
        });
        
        // Step 2: Assign
        document.getElementById('volumeFraction').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.config.volumeFraction = value;
            document.getElementById('volumeFractionValue').textContent = `${Math.round(value * 100)}%`;
        });
        
        document.getElementById('forceDirection').addEventListener('change', (e) => {
            this.config.forceDirection = e.target.value;
            this.viewer.forceDirection = e.target.value;
            this.viewer.draw();
        });
        
        document.getElementById('forceMagnitude').addEventListener('input', (e) => {
            this.config.forceMagnitude = parseFloat(e.target.value);
            this.viewer.forceMagnitude = parseFloat(e.target.value);
        });
        
        document.getElementById('constraintPosition').addEventListener('change', (e) => {
            this.config.constraintPosition = e.target.value;
        });

        // Paint toolbar
        document.getElementById('paintConstraint').addEventListener('click', () => {
            this.viewer.setPaintMode('constraint');
            document.getElementById('paintConstraint').classList.add('active-tool');
            document.getElementById('paintForce').classList.remove('active-tool');
        });
        document.getElementById('paintForce').addEventListener('click', () => {
            this.viewer.setPaintMode('force');
            document.getElementById('paintForce').classList.add('active-tool');
            document.getElementById('paintConstraint').classList.remove('active-tool');
        });
        document.getElementById('clearPaint').addEventListener('click', () => {
            this.viewer.setPaintMode(null);
            document.getElementById('paintConstraint').classList.remove('active-tool');
            document.getElementById('paintForce').classList.remove('active-tool');
        });
        
        // Step 3: Solve
        document.getElementById('maxIterations').addEventListener('input', (e) => {
            this.config.maxIterations = parseInt(e.target.value);
        });
        
        document.getElementById('penaltyFactor').addEventListener('input', (e) => {
            this.config.penaltyFactor = parseFloat(e.target.value);
        });
        
        document.getElementById('filterRadius').addEventListener('input', (e) => {
            this.config.filterRadius = parseFloat(e.target.value);
        });
        
        document.getElementById('minCrossSection').addEventListener('input', (e) => {
            this.config.minCrossSection = parseFloat(e.target.value);
        });
        
        // AMR controls
        document.getElementById('useAMR').addEventListener('change', (e) => {
            this.config.useAMR = e.target.checked;
            const amrControls = document.getElementById('amrControls');
            const amrControls2 = document.getElementById('amrControls2');
            if (amrControls && amrControls2) {
                amrControls.style.display = e.target.checked ? '' : 'none';
                amrControls2.style.display = e.target.checked ? '' : 'none';
            }
        });
        
        document.getElementById('minGranuleSize').addEventListener('input', (e) => {
            this.config.minGranuleSize = parseFloat(e.target.value);
        });
        
        document.getElementById('maxGranuleSize').addEventListener('input', (e) => {
            this.config.maxGranuleSize = parseFloat(e.target.value);
        });
        
        document.getElementById('runOptimization').addEventListener('click', () => {
            this.runOptimization();
        });
        
        document.getElementById('cancelOptimization').addEventListener('click', () => {
            this.cancelOptimization();
        });
        
        // Step 4: Export
        document.getElementById('downloadSTL').addEventListener('click', () => {
            this.exportSTL();
        });
        
        document.getElementById('downloadJSON').addEventListener('click', () => {
            this.exportJSON();
        });
        
        document.getElementById('resetApp').addEventListener('click', () => {
            this.reset();
        });
        
        // Viewer controls
        document.getElementById('toggleWireframe').addEventListener('click', () => {
            this.viewer.toggleWireframe();
        });
        
        document.getElementById('resetCamera').addEventListener('click', () => {
            this.viewer.resetCamera();
        });
    }

    async handleFileImport(file) {
        try {
            console.log('Importing file:', file.name);
            const model = await this.importer.importSTL(file, this.config.granuleDensity);
            this.currentModel = model;
            
            // Display model info
            const info = document.getElementById('modelInfo');
            info.classList.remove('hidden');
            info.innerHTML = `
                <strong>Model loaded:</strong> ${file.name}<br>
                <strong>Elements:</strong> ${model.nx * model.ny * model.nz}<br>
                <strong>Dimensions:</strong> ${model.nx} x ${model.ny} x ${model.nz}
            `;
            
            // Visualize
            this.viewer.setModel(model);
            
            // Enable and navigate to step 2
            this.workflow.enableStep(2);
            this.workflow.goToStep(2);
            
        } catch (error) {
            console.error('Import error:', error);
            alert('Failed to import file: ' + error.message);
        }
    }

    loadTemplate(type) {
        console.log('Loading template:', type);
        const model = this.importer.createTemplate(type, this.config.granuleDensity);
        this.currentModel = model;
        
        // For cube template, set specific boundary conditions
        if (type === 'cube' && model.forcePosition && model.constraintPositions) {
            this.config.forceDirection = 'down';  // Force pointing down at top center
            this.config.constraintPosition = 'bottom-corners';  // Special setting for cube
            document.getElementById('forceDirection').value = 'down';
            // Note: constraint dropdown doesn't have 'bottom-corners', it will use the config value
        }
        
        // Display model info
        const info = document.getElementById('modelInfo');
        info.classList.remove('hidden');
        info.innerHTML = `
            <strong>Template loaded:</strong> ${type}<br>
            <strong>Elements:</strong> ${model.nx * model.ny * model.nz}<br>
            <strong>Dimensions:</strong> ${model.nx} x ${model.ny} x ${model.nz}
        `;
        
        // Visualize
        this.viewer.setModel(model);
        
        // Enable and navigate to step 2
        this.workflow.enableStep(2);
        this.workflow.goToStep(2);
    }

    async runOptimization() {
        if (!this.currentModel) {
            alert('Please import a model first');
            return;
        }
        
        console.log('Starting optimization with config:', this.config);
        
        // Reset mesh data so the viewer shows the original model while solving
        this.viewer.meshData = null;
        this.viewer.densities = null;
        this.viewer.draw();
        
        // Show progress
        const progressContainer = document.getElementById('progressContainer');
        progressContainer.classList.remove('hidden');
        
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const complianceText = document.getElementById('complianceText');
        
        // Toggle buttons: hide Run, show Cancel
        const runButton = document.getElementById('runOptimization');
        const cancelButton = document.getElementById('cancelOptimization');
        runButton.classList.add('hidden');
        cancelButton.classList.remove('hidden');
        
        try {
            // Include painted constraint/force data in config
            const optimConfig = {
                ...this.config,
                paintedConstraints: Array.from(this.viewer.paintedConstraintFaces),
                paintedForces: Array.from(this.viewer.paintedForceFaces)
            };

            // Run optimization in worker
            const result = await this.optimizer.optimize(
                this.currentModel,
                optimConfig,
                (iteration, compliance, meshData, timing) => {
                    // Progress callback
                    const progress = (iteration / this.config.maxIterations) * 100;
                    progressFill.style.width = `${progress}%`;
                    progressText.textContent = `Iteration ${iteration} / ${this.config.maxIterations}`;
                    
                    let complianceInfo = `Compliance: ${compliance.toFixed(2)}`;
                    if (timing) {
                        const wasmBadge = timing.usingWasm ? ' 🚀 WASM' : ' JS';
                        complianceInfo += wasmBadge;
                        complianceInfo += `<br>Time/Iter: ${timing.iterationTime.toFixed(1)}ms`;
                        complianceInfo += ` | Avg: ${timing.avgIterationTime.toFixed(1)}ms`;
                        complianceInfo += `<br>Elapsed: ${(timing.elapsedTime / 1000).toFixed(1)}s`;
                    }
                    complianceText.innerHTML = complianceInfo;
                    
                    // Update visualization with triangle mesh
                    if (meshData) {
                        this.viewer.updateMesh(meshData);
                    }
                }
            );
            
            this.optimizedModel = result;
            
            // Update viewer with final mesh
            if (result.meshData) {
                this.viewer.updateMesh(result.meshData);
            }
            
            // Show results
            let resultsHTML = `
                <strong>Optimization Complete!</strong><br>
                Final Compliance: ${result.finalCompliance.toFixed(2)}<br>
                Iterations: ${result.iterations}<br>
                Volume Fraction: ${(result.volumeFraction * 100).toFixed(1)}%
            `;
            
            // Add AMR statistics if available
            if (result.amrStats) {
                resultsHTML += `<br><br><strong>AMR Statistics:</strong><br>`;
                resultsHTML += `Groups: ${result.amrStats.groupCount}<br>`;
                resultsHTML += `Size Range: ${result.amrStats.minGroupSize.toFixed(1)} - ${result.amrStats.maxGroupSize.toFixed(1)}<br>`;
                resultsHTML += `Avg Size: ${result.amrStats.avgGroupSize.toFixed(1)}<br>`;
                resultsHTML += `Refinements: ${result.amrStats.refinementCount}`;
            }
            
            if (result.timing) {
                const wasmBadge = result.timing.usingWasm ? '🚀 WASM' : 'JS';
                resultsHTML += `<br><br><strong>Performance (${wasmBadge}):</strong><br>`;
                resultsHTML += `Total Time: ${(result.timing.totalTime / 1000).toFixed(2)}s<br>`;
                resultsHTML += `Avg Time/Iteration: ${result.timing.avgIterationTime.toFixed(1)}ms<br>`;
                resultsHTML += `Throughput: ${(result.iterations / (result.timing.totalTime / 1000)).toFixed(2)} iter/s`;
            }
            
            document.getElementById('optimizationResults').innerHTML = resultsHTML;
            
            // Add to benchmark history
            if (result.timing) {
                this.addBenchmark(this.currentModel.type || 'custom', result);
            }
            
            // Enable export step and navigate to it
            this.workflow.enableStep(4);
            this.workflow.goToStep(4);
            
            console.log('Optimization completed successfully');
            
        } catch (error) {
            if (error.message === 'Optimization cancelled') {
                console.log('Optimization was cancelled by user');
                progressText.textContent = 'Cancelled';
                // Reset progress UI so user can restart
                progressFill.style.width = '0%';
                complianceText.textContent = '';
                progressContainer.classList.add('hidden');
            } else {
                console.error('Optimization error:', error);
                alert('Optimization failed: ' + error.message);
            }
        } finally {
            runButton.classList.remove('hidden');
            cancelButton.classList.add('hidden');
        }
    }

    cancelOptimization() {
        console.log('Cancelling optimization...');
        this.optimizer.cancel();
    }

    exportSTL() {
        if (!this.optimizedModel) {
            alert('Please run optimization first');
            return;
        }
        
        console.log('Exporting STL...');
        this.exporter.exportSTL(this.optimizedModel, 'optimized_model.stl');
    }

    exportJSON() {
        if (!this.optimizedModel) {
            alert('Please run optimization first');
            return;
        }
        
        console.log('Exporting JSON...');
        this.exporter.exportJSON(this.optimizedModel, 'optimization_data.json');
    }

    reset() {
        console.log('Resetting app...');
        
        // Cancel any running optimization
        this.optimizer.cancel();
        
        this.currentModel = null;
        this.optimizedModel = null;
        
        // Clear UI
        document.getElementById('modelInfo').classList.add('hidden');
        document.getElementById('progressContainer').classList.add('hidden');
        document.getElementById('fileInput').value = '';
        document.getElementById('runOptimization').classList.remove('hidden');
        document.getElementById('cancelOptimization').classList.add('hidden');
        
        // Reset viewer
        this.viewer.clear();
        
        // Reset workflow
        this.workflow.reset();
    }
    
    // Benchmark management methods
    loadBenchmarkHistory() {
        try {
            const stored = localStorage.getItem('topologyBenchmarkHistory');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.warn('Failed to load benchmark history:', e);
            return [];
        }
    }
    
    saveBenchmarkHistory() {
        try {
            localStorage.setItem('topologyBenchmarkHistory', JSON.stringify(this.benchmarkHistory));
        } catch (e) {
            console.warn('Failed to save benchmark history:', e);
        }
    }
    
    addBenchmark(modelType, result) {
        const benchmark = {
            timestamp: new Date().toISOString(),
            modelType: modelType,
            dimensions: `${result.nx}x${result.ny}x${result.nz}`,
            iterations: result.iterations,
            totalTime: result.timing.totalTime,
            avgIterationTime: result.timing.avgIterationTime,
            compliance: result.finalCompliance,
            volumeFraction: result.volumeFraction,
            usingWasm: result.timing.usingWasm || false
        };
        
        this.benchmarkHistory.push(benchmark);
        
        // Keep only last 10 benchmarks
        if (this.benchmarkHistory.length > 10) {
            this.benchmarkHistory = this.benchmarkHistory.slice(-10);
        }
        
        this.saveBenchmarkHistory();
        this.displayBenchmarkHistory();
    }
    
    displayBenchmarkHistory() {
        const benchmarkInfo = document.getElementById('benchmarkInfo');
        const benchmarkResults = document.getElementById('benchmarkResults');
        
        if (this.benchmarkHistory.length === 0) {
            benchmarkInfo.classList.add('hidden');
            return;
        }
        
        benchmarkInfo.classList.remove('hidden');
        
        // Find baseline (first cube test or first entry)
        const baseline = this.benchmarkHistory.find(b => b.modelType === 'cube') || this.benchmarkHistory[0];
        
        let html = '<table><thead><tr><th>Model</th><th>Engine</th><th>Avg Iter (ms)</th><th>Total (s)</th><th>vs Baseline</th></tr></thead><tbody>';
        
        // Show most recent benchmarks first
        const recent = this.benchmarkHistory.slice(-5).reverse();
        for (const bench of recent) {
            const isBaseline = bench === baseline;
            const improvement = baseline.avgIterationTime > 0 
                ? ((baseline.avgIterationTime - bench.avgIterationTime) / baseline.avgIterationTime * 100)
                : 0;
            const rowClass = isBaseline ? ' class="benchmark-baseline"' : '';
            const engineBadge = bench.usingWasm ? '🚀 WASM' : 'JS';
            
            let comparisonText = '';
            if (!isBaseline) {
                const compClass = improvement > 0 ? 'benchmark-improvement' : 'benchmark-regression';
                const sign = improvement > 0 ? '↓' : '↑';
                comparisonText = `<span class="${compClass}">${sign}${Math.abs(improvement).toFixed(1)}%</span>`;
            } else {
                comparisonText = '<strong>Baseline</strong>';
            }
            
            html += `<tr${rowClass}>
                <td>${bench.modelType} (${bench.dimensions})</td>
                <td>${engineBadge}</td>
                <td>${bench.avgIterationTime.toFixed(1)}</td>
                <td>${(bench.totalTime / 1000).toFixed(1)}</td>
                <td>${comparisonText}</td>
            </tr>`;
        }
        
        html += '</tbody></table>';
        benchmarkResults.innerHTML = html;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const app = new TopologyApp();
    await app.init();
    
    // Store globally for debugging
    window.topologyApp = app;
});
