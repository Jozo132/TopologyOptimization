// Main application entry point
import { Viewer3D } from './viewer.js';
import { ModelImporter } from './importer.js';
import { TopologyOptimizer } from './optimizer.js';
import { ModelExporter } from './exporter.js';
import { WorkflowManager } from './workflow.js';

const MATERIAL_PRESETS = {
    plastic: { youngsModulus: 2.3, poissonsRatio: 0.35 },
    aluminum: { youngsModulus: 69, poissonsRatio: 0.33 },
    steel: { youngsModulus: 200, poissonsRatio: 0.30 }
};

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
            volumeFraction: 0.1,
            forceDirection: 'down',
            forceMagnitude: 1000,
            constraintPosition: 'left',
            maxIterations: 30,
            penaltyFactor: 3,
            filterRadius: 0.9,
            voxelSizeMM: 5,
            minCrossSection: 0,
            constrainToSolid: false,
            useAMR: true,
            amrInterval: 3,
            minGranuleSize: 0.02,
            maxGranuleSize: 2,
            youngsModulus: 2.3,
            poissonsRatio: 0.35,
            material: 'plastic',
            // Accuracy scheduling: adaptive CG tolerance (auto-enabled)
            // Penalization continuation: ramp penal from penalStart → penaltyFactor
            penalStart: 1.5,
            continuationIters: 20,
            // Heaviside projection with beta-continuation
            useProjection: true,
            betaMax: 64,
            betaInterval: 5
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

        // Apply transform button (scale + rotate)
        document.getElementById('applyTransform').addEventListener('click', () => {
            if (!this.currentModel || !this.currentModel.originalVertices) return;
            
            const scale = parseFloat(document.getElementById('modelScale').value) || 1;
            const rotX = parseFloat(document.getElementById('rotateX').value) || 0;
            const rotY = parseFloat(document.getElementById('rotateY').value) || 0;
            const rotZ = parseFloat(document.getElementById('rotateZ').value) || 0;
            
            const transformed = this.importer.transformVertices(
                this.currentModel.originalVertices,
                scale, rotX, rotY, rotZ
            );
            
            const voxelSizeMM = this.config.voxelSizeMM;
            const newModel = this.importer.voxelizeVertices(transformed, null, voxelSizeMM);
            // Store the raw (untransformed) vertices so transform can be re-applied
            newModel.originalVertices = transformed;
            if (this.currentModel.type) newModel.type = this.currentModel.type;
            
            this.currentModel = newModel;
            this.viewer.setModel(newModel);
            
            // Update info display
            const info = document.getElementById('modelInfo');
            if (!info.classList.contains('hidden')) {
                const elementCount = newModel.nx * newModel.ny * newModel.nz;
                let infoHTML = `<strong>Model transformed!</strong><br>`;
                if (newModel.bounds) {
                    const physX = (newModel.bounds.maxX - newModel.bounds.minX).toFixed(1);
                    const physY = (newModel.bounds.maxY - newModel.bounds.minY).toFixed(1);
                    const physZ = (newModel.bounds.maxZ - newModel.bounds.minZ).toFixed(1);
                    infoHTML += `<strong>Size:</strong> ${physX} × ${physY} × ${physZ} mm<br>`;
                }
                if (newModel.voxelSize) {
                    infoHTML += `<strong>Voxel size:</strong> ${newModel.voxelSize.toFixed(2)} mm<br>`;
                }
                infoHTML += `<strong>Elements:</strong> ${elementCount}<br>`;
                infoHTML += `<strong>Grid:</strong> ${newModel.nx} × ${newModel.ny} × ${newModel.nz}`;
                info.innerHTML = infoHTML;
            }
        });

        // Voxel size slider (mm-based, 10mm to 0.1mm)
        const voxelSizeSlider = document.getElementById('voxelSize');
        const voxelSizeInput = document.getElementById('voxelSizeInput');

        const handleVoxelSizeChange = (voxelSizeMM) => {
            this.config.voxelSizeMM = voxelSizeMM;
            voxelSizeSlider.value = voxelSizeMM;
            voxelSizeInput.value = voxelSizeMM;
            
            // Re-voxelize the current model if one exists
            if (this.currentModel) {
                let newModel = null;
                
                // Check if it's an STL model with original vertices
                if (this.currentModel.originalVertices) {
                    newModel = this.importer.voxelizeVertices(
                        this.currentModel.originalVertices,
                        null,
                        voxelSizeMM
                    );
                    // Preserve the model type if it was set
                    if (this.currentModel.type) {
                        newModel.type = this.currentModel.type;
                    }
                }
                // Check if it's a template with scaling info
                else if (this.currentModel.templateScale) {
                    // For templates, convert mm size to resolution based on template dimensions
                    const baseNx = this.currentModel.templateScale.baseNx;
                    const baseNy = this.currentModel.templateScale.baseNy;
                    const baseNz = this.currentModel.templateScale.baseNz;
                    const maxDim = Math.max(baseNx, baseNy, baseNz);
                    const resolution = Math.max(3, Math.round(maxDim / voxelSizeMM));
                    newModel = this.importer.createTemplate(
                        this.currentModel.type,
                        resolution
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
                        let infoHTML = `<strong>${modelName} updated!</strong><br>`;
                        if (newModel.bounds) {
                            const physX = (newModel.bounds.maxX - newModel.bounds.minX).toFixed(1);
                            const physY = (newModel.bounds.maxY - newModel.bounds.minY).toFixed(1);
                            const physZ = (newModel.bounds.maxZ - newModel.bounds.minZ).toFixed(1);
                            infoHTML += `<strong>Size:</strong> ${physX} × ${physY} × ${physZ} mm<br>`;
                        }
                        if (newModel.voxelSize) {
                            infoHTML += `<strong>Voxel size:</strong> ${newModel.voxelSize.toFixed(2)} mm<br>`;
                        }
                        infoHTML += `<strong>Elements:</strong> ${elementCount}<br>`;
                        infoHTML += `<strong>Grid:</strong> ${newModel.nx} × ${newModel.ny} × ${newModel.nz}`;
                        info.innerHTML = infoHTML;
                    }
                    
                    // Update viewer with new voxel grid
                    this.viewer.setModel(newModel);
                }
            }
        };

        voxelSizeSlider.addEventListener('input', (e) => {
            handleVoxelSizeChange(parseFloat(e.target.value));
        });

        voxelSizeInput.addEventListener('change', (e) => {
            let val = parseFloat(e.target.value);
            const min = parseFloat(voxelSizeInput.min);
            const max = parseFloat(voxelSizeInput.max);
            if (isNaN(val)) val = this.config.voxelSizeMM;
            val = Math.max(min, Math.min(max, val));
            handleVoxelSizeChange(val);
        });
        
        // Step 2: Assign - Material selection
        document.getElementById('materialSelect').addEventListener('change', (e) => {
            const material = e.target.value;
            this.config.material = material;
            if (MATERIAL_PRESETS[material]) {
                const preset = MATERIAL_PRESETS[material];
                this.config.youngsModulus = preset.youngsModulus;
                this.config.poissonsRatio = preset.poissonsRatio;
                document.getElementById('youngsModulus').value = preset.youngsModulus;
                document.getElementById('poissonsRatio').value = preset.poissonsRatio;
            }
        });

        document.getElementById('youngsModulus').addEventListener('input', (e) => {
            this.config.youngsModulus = parseFloat(e.target.value);
            document.getElementById('materialSelect').value = 'custom';
            this.config.material = 'custom';
        });

        document.getElementById('poissonsRatio').addEventListener('input', (e) => {
            this.config.poissonsRatio = parseFloat(e.target.value);
            document.getElementById('materialSelect').value = 'custom';
            this.config.material = 'custom';
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

        document.getElementById('constrainToSolid').addEventListener('change', (e) => {
            this.config.constrainToSolid = e.target.checked;
        });
        
        // AMR controls
        document.getElementById('useAMR').addEventListener('change', (e) => {
            this.config.useAMR = e.target.checked;
            const amrControls = document.getElementById('amrControls');
            const amrControls2 = document.getElementById('amrControls2');
            const amrControls3 = document.getElementById('amrControls3');
            if (amrControls && amrControls2) {
                amrControls.style.display = e.target.checked ? '' : 'none';
                amrControls2.style.display = e.target.checked ? '' : 'none';
            }
            if (amrControls3) {
                amrControls3.style.display = e.target.checked ? '' : 'none';
            }
        });
        
        document.getElementById('minGranuleSize').addEventListener('input', (e) => {
            this.config.minGranuleSize = parseFloat(e.target.value);
        });
        
        document.getElementById('maxGranuleSize').addEventListener('input', (e) => {
            this.config.maxGranuleSize = parseFloat(e.target.value);
        });
        
        document.getElementById('amrInterval').addEventListener('input', (e) => {
            this.config.amrInterval = parseInt(e.target.value);
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
        document.getElementById('toggleViewMode').addEventListener('click', () => {
            const newMode = this.viewer.viewMode === 'auto' ? 'voxel' : 'auto';
            this.viewer.setViewMode(newMode);
            const btn = document.getElementById('toggleViewMode');
            btn.title = newMode === 'voxel' ? 'Switch to Auto View' : 'Switch to Voxel View';
            btn.classList.toggle('active-tool', newMode === 'voxel');
        });

        document.getElementById('toggleMeshVisibility').addEventListener('click', () => {
            this.viewer.toggleMeshVisibility();
            const btn = document.getElementById('toggleMeshVisibility');
            btn.classList.toggle('active-tool', !this.viewer.meshVisible);
            btn.title = this.viewer.meshVisible ? 'Hide Mesh' : 'Show Mesh';
        });

        document.getElementById('toggleWireframe').addEventListener('click', () => {
            this.viewer.toggleWireframe();
        });
        
        document.getElementById('resetCamera').addEventListener('click', () => {
            this.viewer.resetCamera();
        });

        // Strain range slider
        const strainMinInput = document.getElementById('strainMin');
        const strainMaxInput = document.getElementById('strainMax');
        const strainMinValue = document.getElementById('strainMinValue');
        const strainMaxValue = document.getElementById('strainMaxValue');
        const strainFill = document.getElementById('strainSliderFill');

        const updateStrainSlider = () => {
            let min = parseFloat(strainMinInput.value);
            let max = parseFloat(strainMaxInput.value);
            // Ensure min <= max
            if (min > max) {
                if (this._lastStrainInput === 'min') {
                    max = min;
                    strainMaxInput.value = max;
                } else {
                    min = max;
                    strainMinInput.value = min;
                }
            }
            strainMinValue.textContent = `${Math.round(min * 100)}%`;
            strainMaxValue.textContent = `${Math.round(max * 100)}%`;
            // Update fill bar position
            strainFill.style.left = `${min * 100}%`;
            strainFill.style.width = `${(max - min) * 100}%`;
            this.viewer.setStrainRange(min, max);
        };

        strainMinInput.addEventListener('input', () => {
            this._lastStrainInput = 'min';
            updateStrainSlider();
        });
        strainMaxInput.addEventListener('input', () => {
            this._lastStrainInput = 'max';
            updateStrainSlider();
        });
    }

    async handleFileImport(file) {
        try {
            console.log('Importing file:', file.name);
            // Parse STL first, then voxelize with mm-based voxel size
            const model = await this.importer.importSTL(file, null);
            // Re-voxelize with the configured voxel size in mm
            const voxelSizeMM = this.config.voxelSizeMM;
            const revoxelized = this.importer.voxelizeVertices(model.originalVertices, null, voxelSizeMM);
            revoxelized.originalVertices = model.originalVertices;
            this.currentModel = revoxelized;
            
            // Show transform controls for imported STL models
            document.getElementById('transformControls').classList.remove('hidden');
            
            // Display model info
            const info = document.getElementById('modelInfo');
            info.classList.remove('hidden');
            const physX = revoxelized.bounds ? (revoxelized.bounds.maxX - revoxelized.bounds.minX).toFixed(1) : '?';
            const physY = revoxelized.bounds ? (revoxelized.bounds.maxY - revoxelized.bounds.minY).toFixed(1) : '?';
            const physZ = revoxelized.bounds ? (revoxelized.bounds.maxZ - revoxelized.bounds.minZ).toFixed(1) : '?';
            const voxelSizeStr = revoxelized.voxelSize ? revoxelized.voxelSize.toFixed(2) : '?';
            info.innerHTML = `
                <strong>Model loaded:</strong> ${file.name}<br>
                <strong>Size:</strong> ${physX} × ${physY} × ${physZ} mm<br>
                <strong>Voxel size:</strong> ${voxelSizeStr} mm<br>
                <strong>Elements:</strong> ${revoxelized.nx * revoxelized.ny * revoxelized.nz}<br>
                <strong>Grid:</strong> ${revoxelized.nx} × ${revoxelized.ny} × ${revoxelized.nz}
            `;
            
            // Visualize
            this.viewer.setModel(revoxelized);
            
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
        // Convert voxelSizeMM to resolution for templates
        const maxDim = ModelImporter.getTemplateMaxDim(type);
        const resolution = Math.max(3, Math.round(maxDim / this.config.voxelSizeMM));
        const model = this.importer.createTemplate(type, resolution);
        this.currentModel = model;
        
        // Hide transform controls for templates (no vertices to transform)
        document.getElementById('transformControls').classList.add('hidden');
        
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
                        // Show strain slider when mesh data is available
                        document.getElementById('strainSliderContainer').classList.remove('hidden');
                    }
                }
            );
            
            this.optimizedModel = result;
            
            // Update viewer with final mesh
            if (result.meshData) {
                this.viewer.updateMesh(result.meshData);
                document.getElementById('strainSliderContainer').classList.remove('hidden');
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
        document.getElementById('strainSliderContainer').classList.add('hidden');
        document.getElementById('transformControls').classList.add('hidden');
        
        // Reset transform inputs
        document.getElementById('modelScale').value = 1;
        document.getElementById('rotateX').value = 0;
        document.getElementById('rotateY').value = 0;
        document.getElementById('rotateZ').value = 0;
        
        // Reset strain slider values
        document.getElementById('strainMin').value = 0;
        document.getElementById('strainMax').value = 1;
        document.getElementById('strainMinValue').textContent = '0%';
        document.getElementById('strainMaxValue').textContent = '100%';
        document.getElementById('strainSliderFill').style.left = '0%';
        document.getElementById('strainSliderFill').style.width = '100%';
        
        // Reset viewer control button states
        document.getElementById('toggleViewMode').classList.remove('active-tool');
        document.getElementById('toggleMeshVisibility').classList.remove('active-tool');
        
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
            usingWasm: result.timing.usingWasm || false,
            // AMR statistics
            useAMR: result.amrStats !== null,
            amrGroups: result.amrStats ? result.amrStats.groupCount : null,
            amrRefinements: result.amrStats ? result.amrStats.refinementCount : null,
            amrSizeRange: result.amrStats ? 
                `${result.amrStats.minGroupSize.toFixed(1)}-${result.amrStats.maxGroupSize.toFixed(1)}` : null
        };
        
        this.benchmarkHistory.push(benchmark);
        
        // Keep only last 15 benchmarks to allow more comparisons
        if (this.benchmarkHistory.length > 15) {
            this.benchmarkHistory = this.benchmarkHistory.slice(-15);
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
        
        let html = '<table><thead><tr><th>Model</th><th>Engine</th><th>AMR</th><th>Avg Iter (ms)</th><th>Total (s)</th><th>vs Baseline</th></tr></thead><tbody>';
        
        // Show most recent benchmarks first
        const recent = this.benchmarkHistory.slice(-8).reverse();
        for (const bench of recent) {
            const isBaseline = bench === baseline;
            const improvement = baseline.avgIterationTime > 0 
                ? ((baseline.avgIterationTime - bench.avgIterationTime) / baseline.avgIterationTime * 100)
                : 0;
            const rowClass = isBaseline ? ' class="benchmark-baseline"' : '';
            const engineBadge = bench.usingWasm ? '🚀 WASM' : 'JS';
            const amrBadge = bench.useAMR ? `✓ (${bench.amrRefinements})` : '✗';
            
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
                <td>${amrBadge}</td>
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
