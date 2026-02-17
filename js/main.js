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
            filterRadius: 1.5
        };
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
        
        // Step 2: Assign
        document.getElementById('volumeFraction').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.config.volumeFraction = value;
            document.getElementById('volumeFractionValue').textContent = `${Math.round(value * 100)}%`;
        });
        
        document.getElementById('forceDirection').addEventListener('change', (e) => {
            this.config.forceDirection = e.target.value;
        });
        
        document.getElementById('forceMagnitude').addEventListener('input', (e) => {
            this.config.forceMagnitude = parseFloat(e.target.value);
        });
        
        document.getElementById('constraintPosition').addEventListener('change', (e) => {
            this.config.constraintPosition = e.target.value;
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
            const model = await this.importer.importSTL(file);
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
            
            // Enable next step
            this.workflow.enableStep(2);
            
        } catch (error) {
            console.error('Import error:', error);
            alert('Failed to import file: ' + error.message);
        }
    }

    loadTemplate(type) {
        console.log('Loading template:', type);
        const model = this.importer.createTemplate(type);
        this.currentModel = model;
        
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
        
        // Enable next step
        this.workflow.enableStep(2);
        this.workflow.enableStep(3);
    }

    async runOptimization() {
        if (!this.currentModel) {
            alert('Please import a model first');
            return;
        }
        
        console.log('Starting optimization with config:', this.config);
        
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
            // Run optimization in worker
            const result = await this.optimizer.optimize(
                this.currentModel,
                this.config,
                (iteration, compliance, meshData) => {
                    // Progress callback
                    const progress = (iteration / this.config.maxIterations) * 100;
                    progressFill.style.width = `${progress}%`;
                    progressText.textContent = `Iteration ${iteration} / ${this.config.maxIterations}`;
                    complianceText.textContent = `Compliance: ${compliance.toFixed(2)}`;
                    
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
            document.getElementById('optimizationResults').innerHTML = `
                <strong>Optimization Complete!</strong><br>
                Final Compliance: ${result.finalCompliance.toFixed(2)}<br>
                Iterations: ${result.iterations}<br>
                Volume Fraction: ${(result.volumeFraction * 100).toFixed(1)}%
            `;
            
            // Enable export step
            this.workflow.enableStep(4);
            
            console.log('Optimization completed successfully');
            
        } catch (error) {
            if (error.message === 'Optimization cancelled') {
                console.log('Optimization was cancelled by user');
                progressText.textContent = 'Cancelled';
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
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const app = new TopologyApp();
    await app.init();
    
    // Store globally for debugging
    window.topologyApp = app;
});
