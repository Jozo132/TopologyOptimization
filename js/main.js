// Main application entry point
import { Viewer3D } from './viewer.js';
import { ModelImporter } from './importer.js';
import { TopologySolver } from '../lib/topology-solver.js';
import { ModelExporter } from './exporter.js';
import { WorkflowManager } from './workflow.js';

const MATERIAL_PRESETS = {
    // ── Rubber-like materials ──
    'natural-rubber':  { youngsModulus: 0.01, poissonsRatio: 0.499, yieldStrength: 0, nonlinearType: 'neo-hookean', category: 'rubber' },
    'silicone':        { youngsModulus: 0.005, poissonsRatio: 0.499, yieldStrength: 0, nonlinearType: 'mooney-rivlin', category: 'rubber' },
    'neoprene':        { youngsModulus: 0.007, poissonsRatio: 0.499, yieldStrength: 0, nonlinearType: 'neo-hookean', category: 'rubber' },
    'epdm':            { youngsModulus: 0.008, poissonsRatio: 0.499, yieldStrength: 0, nonlinearType: 'ogden', category: 'rubber' },
    // ── Hard plastics ──
    'abs':             { youngsModulus: 2.3, poissonsRatio: 0.35, yieldStrength: 40, nonlinearType: 'j2-plasticity', category: 'plastic' },
    'polycarbonate':   { youngsModulus: 2.4, poissonsRatio: 0.37, yieldStrength: 62, nonlinearType: 'j2-plasticity', category: 'plastic' },
    'nylon':           { youngsModulus: 2.7, poissonsRatio: 0.39, yieldStrength: 70, nonlinearType: 'j2-plasticity', category: 'plastic' },
    'pla':             { youngsModulus: 3.5, poissonsRatio: 0.36, yieldStrength: 60, nonlinearType: 'linear-elastic', category: 'plastic' },
    'peek':            { youngsModulus: 4.0, poissonsRatio: 0.38, yieldStrength: 100, nonlinearType: 'j2-plasticity', category: 'plastic' },
    // ── Metals ──
    'aluminum':        { youngsModulus: 69, poissonsRatio: 0.33, yieldStrength: 270, nonlinearType: 'j2-plasticity', category: 'metal' },
    'steel':           { youngsModulus: 200, poissonsRatio: 0.30, yieldStrength: 250, nonlinearType: 'j2-plasticity', category: 'metal' },
    'stainless-steel': { youngsModulus: 193, poissonsRatio: 0.29, yieldStrength: 205, nonlinearType: 'j2-plasticity', category: 'metal' },
    'titanium':        { youngsModulus: 116, poissonsRatio: 0.34, yieldStrength: 880, nonlinearType: 'j2-plasticity', category: 'metal' },
    'copper':          { youngsModulus: 117, poissonsRatio: 0.34, yieldStrength: 70, nonlinearType: 'j2-plasticity', category: 'metal' },
    'brass':           { youngsModulus: 100, poissonsRatio: 0.34, yieldStrength: 200, nonlinearType: 'j2-plasticity', category: 'metal' },
    'cast-iron':       { youngsModulus: 170, poissonsRatio: 0.26, yieldStrength: 130, nonlinearType: 'drucker-prager', category: 'metal' }
};

const DOF_DESCRIPTIONS = {
    'all': 'Fixed: X ✓ Y ✓ Z ✓ — Fully constrained',
    'xy':  'Fixed: X ✓ Y ✓ Z ✗ — Free to slide along Z',
    'xz':  'Fixed: X ✓ Y ✗ Z ✓ — Free to slide along Y',
    'yz':  'Fixed: X ✗ Y ✓ Z ✓ — Free to slide along X',
    'x':   'Fixed: X ✓ Y ✗ Z ✗ — Rail constraint (X axis)',
    'y':   'Fixed: X ✗ Y ✓ Z ✗ — Rail constraint (Y axis)',
    'z':   'Fixed: X ✗ Y ✗ Z ✓ — Rail constraint (Z axis)'
};

const FORCE_DIRECTION_VECTORS = {
    down: [0, -1, 0],
    up: [0, 1, 0],
    left: [-1, 0, 0],
    right: [1, 0, 0]
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
        this.pausedVolumetricData = null;
        this.finalVolumetricData = null;
        this.lastVolumetricData = null;
        this._optimizationPaused = false;
        this.config = {
            solver: 'auto',
            petscPC: 'bddc',
            volumeFraction: 0.7,
            forceDirection: 'down',
            forceVector: null, // Custom force vector [fx, fy, fz]
            forceMagnitude: 1000,
            forceType: 'total', // 'total' or 'pressure'
            constraintPosition: 'left',
            constraintDOFs: 'all',
            maxIterations: 30,
            penaltyFactor: 3,
            filterRadius: 0.9,
            voxelSizeMM: 5,
            minCrossSection: 0,
            constrainToSolid: false,
            preventVoids: false,
            manufacturingConstraint: false,
            manufacturingAngle: 90,
            manufacturingMinRadius: 0,
            manufacturingMaxDepth: 0,
            useGPU: false,
            feaSolverBackend: 'auto',
            volumetricOutputMode: 'on-stop',
            useAMR: false,
            amrInterval: 3,
            minGranuleSize: 0.02,
            maxGranuleSize: 2,
            youngsModulus: 2.3,
            poissonsRatio: 0.35,
            yieldStrength: 40,
            material: 'abs',
            // Accuracy scheduling: adaptive CG tolerance (auto-enabled)
            // Penalization continuation: ramp penal from penalStart → penaltyFactor
            penalStart: 1.5,
            continuationIters: 20,
            // Heaviside projection with beta-continuation
            useProjection: true,
            betaMax: 64,
            betaInterval: 5,
            // Solution type: 'topology' | 'fea' | 'nonlinear' | 'fatigue'
            solutionType: 'topology',
            // Linear FEA sub-type: 'static' | 'buckling' | 'shear'
            feaSubType: 'static',
            // Linear FEA buckling parameters
            feaBucklingModes: 3,
            feaBucklingImperfection: 0.01,
            // Viewer result color mode persistence
            resultColorMode: 'stress',
            // Nonlinear FEA parameters
            nonlinearLoadSteps: 10,
            nonlinearMaxNewtonIter: 20,
            nonlinearTolerance: 1e-6,
            // Nonlinear analysis sub-type and material model
            nonlinearAnalysisType: 'large-deformation',
            nonlinearMaterialModel: 'neo-hookean',
            // Fracture parameters
            fractureMethod: 'phase-field',
            fractureToughness: 2700,
            fractureLengthScale: 0.01,
            // Buckling parameters
            bucklingModes: 3,
            bucklingImperfection: 0.01,
            // Plasticity parameters
            hardeningModulus: 1000
        };
        
        // Benchmark tracking
        this.benchmarkHistory = this.loadBenchmarkHistory();
    }

    async init() {
        console.log('Initializing Topology Optimization App...');
        
        // Initialize modules
        this.viewer = new Viewer3D('viewer3D');
        await this.viewer.init();
        this.viewer.onSectionChange = () => this._updateSectionSliders();
        
        this.importer = new ModelImporter();
        this.optimizer = new TopologySolver();
        this.optimizer.onPaused = (msg) => {
            this.pausedVolumetricData = msg?.volumetricData || null;
            this._applyVolumetricSnapshot(this.pausedVolumetricData, 'pause');
        };
        this.optimizer.onVolumetric = (msg) => {
            const volumetricData = msg?.volumetricData || null;
            this._applyVolumetricSnapshot(volumetricData, msg?.reason || 'update');
        };
        this.exporter = new ModelExporter();
        this.workflow = new WorkflowManager();
        this.workflow.init();
        
        // Clear paint mode when leaving step 6 (solve & export)
        this.workflow.onStepChange = (step) => {
            if (step !== 6) {
                this.viewer.setPaintMode(null);
                document.getElementById('paintConstraint').classList.remove('active-tool');
                document.getElementById('paintForce').classList.remove('active-tool');
            }
        };
        
        // Setup event listeners
        this.setupEventListeners();
        await this._initGPUControls();

        // Initialize solution type UI to match default config
        this._applySolutionType(this.config.solutionType || 'topology');

        // Show benchmark button if history exists
        this.displayBenchmarkHistory();
        
        console.log('App initialized successfully');
    }

    _showMobileViewerToggle() {
        const el = document.getElementById('mobileToggleViewer');
        if (el) el.classList.add('visible');
    }

    setupEventListeners() {
        // Step 1: Get Started
        document.getElementById('startNewProject').addEventListener('click', () => {
            this.workflow.enableStep(2);
            this.workflow.goToStep(2);
        });
        document.getElementById('startImportProject').addEventListener('click', () => {
            const input = document.getElementById('importSetupFile');
            if (input) {
                input.value = '';
                input.click();
            }
        });
        document.getElementById('importSetupFile').addEventListener('change', async (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;
            await this.importSetup(file);
        });

        // Step 2: Import
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
        document.getElementById('useStressCubeTemplate').addEventListener('click', () => {
            this.loadTemplate('stress-cube');
        });
        document.getElementById('useGasketTemplate').addEventListener('click', () => {
            this.loadTemplate('gasket');
        });

        // Mesh method confirmation button
        document.getElementById('confirmMeshMethod').addEventListener('click', () => {
            this._confirmMeshMethod();
        });

        // Dimension mode radio buttons (2D/3D)
        document.querySelectorAll('input[name="dimensionMode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const is2D = e.target.value === '2d';
                const projSelector = document.getElementById('projectionSideSelector');
                if (projSelector) {
                    projSelector.classList.toggle('hidden', !is2D);
                }
            });
        });

        // Projection side buttons
        document.querySelectorAll('.projection-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.projection-btn').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
            });
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
            this._showMobileViewerToggle();
            
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
                
                // Check if it's a template with scaling info (must check before originalVertices
                // since templates now also have originalVertices for the reference mesh)
                if (this.currentModel.templateScale) {
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
                // Check if it's an STL model with original vertices
                else if (this.currentModel.originalVertices) {
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
                    this._showMobileViewerToggle();
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
        
        // Step 4: Material selection
        document.getElementById('materialSelect').addEventListener('change', (e) => {
            const material = e.target.value;
            this.config.material = material;
            if (MATERIAL_PRESETS[material]) {
                const preset = MATERIAL_PRESETS[material];
                this.config.youngsModulus = preset.youngsModulus;
                this.config.poissonsRatio = preset.poissonsRatio;
                this.config.yieldStrength = preset.yieldStrength;
                document.getElementById('youngsModulus').value = preset.youngsModulus;
                document.getElementById('poissonsRatio').value = preset.poissonsRatio;
                document.getElementById('yieldStrength').value = preset.yieldStrength;
                // Auto-select matching nonlinear material model
                if (preset.nonlinearType) {
                    this.config.nonlinearMaterialModel = preset.nonlinearType;
                }
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

        document.getElementById('yieldStrength').addEventListener('input', (e) => {
            this.config.yieldStrength = parseFloat(e.target.value) || 0;
            document.getElementById('materialSelect').value = 'custom';
            this.config.material = 'custom';
            this.viewer.setYieldStrength(this.config.yieldStrength);
        });

        // Step 6: Forces and constraints
        document.getElementById('volumeFraction').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.config.volumeFraction = value;
            document.getElementById('volumeFractionValue').textContent = `${Math.round(value * 100)}%`;
        });
        
        document.getElementById('forceDirection').addEventListener('change', (e) => {
            this.config.forceDirection = e.target.value;
            this.viewer.forceDirection = e.target.value;
            // Update the force vector inputs to match preset direction
            if (e.target.value !== 'custom') {
                const v = FORCE_DIRECTION_VECTORS[e.target.value] || [0, -1, 0];
                document.getElementById('forceVectorX').value = v[0];
                document.getElementById('forceVectorY').value = v[1];
                document.getElementById('forceVectorZ').value = v[2];
                this.config.forceVector = null;
                this.viewer.forceVector = null;
            } else {
                this._updateForceVector();
            }
            this.viewer.draw();
        });

        const updateForceVectorFromInputs = () => {
            this._updateForceVector();
            this.config.forceDirection = 'custom';
            document.getElementById('forceDirection').value = 'custom';
            this.viewer.forceDirection = 'custom';
            this.viewer.draw();
        };
        document.getElementById('forceVectorX').addEventListener('input', updateForceVectorFromInputs);
        document.getElementById('forceVectorY').addEventListener('input', updateForceVectorFromInputs);
        document.getElementById('forceVectorZ').addEventListener('input', updateForceVectorFromInputs);
        
        // Force type
        document.getElementById('forceType').addEventListener('change', (e) => {
            this.config.forceType = e.target.value;
            this.viewer.forceType = e.target.value;
            const label = document.getElementById('forceMagnitudeLabel');
            if (label) {
                label.textContent = e.target.value === 'pressure'
                    ? 'Pressure (N/mm² = MPa)'
                    : 'Force Magnitude (N)';
            }
        });

        document.getElementById('forceMagnitude').addEventListener('input', (e) => {
            this.config.forceMagnitude = parseFloat(e.target.value);
            this.viewer.forceMagnitude = parseFloat(e.target.value);
        });

        document.getElementById('constraintDOFs').addEventListener('change', (e) => {
            this.config.constraintDOFs = e.target.value;
            this._updateDOFPreview(e.target.value);
        });
        
        document.getElementById('constraintPosition').addEventListener('change', (e) => {
            this.config.constraintPosition = e.target.value;
        });

        // Template defaults checkbox
        const defaultsCheckbox = document.getElementById('useTemplateDefaults');
        if (defaultsCheckbox) {
            defaultsCheckbox.addEventListener('change', (e) => {
                this._applyTemplateDefaultsToggle(e.target.checked);
            });
        }

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

        // Angle-based selection
        document.getElementById('useAngleSelection').addEventListener('change', (e) => {
            this.viewer.useAngleSelection = e.target.checked;
            document.getElementById('angleToleranceControls').style.display = e.target.checked ? '' : 'none';
        });

        const updateAngleTolerance = () => {
            const val = parseInt(document.getElementById('angleTolerance').value);
            document.getElementById('angleToleranceInput').value = val;
            this.viewer.angleTolerance = val;
            this.viewer.updateAngleSelection();
        };
        document.getElementById('angleTolerance').addEventListener('input', () => {
            updateAngleTolerance();
        });
        document.getElementById('angleToleranceInput').addEventListener('input', (e) => {
            document.getElementById('angleTolerance').value = e.target.value;
            updateAngleTolerance();
        });

        document.getElementById('brushSize').addEventListener('input', (e) => {
            const size = parseInt(e.target.value);
            this.viewer.brushSize = size;
            document.getElementById('brushSizeValue').textContent = size;
        });

        // Selection Groups
        document.getElementById('addGroup').addEventListener('click', () => {
            const group = this.viewer.addSelectionGroup('force');
            this.viewer.setPaintMode('force');
            document.getElementById('paintForce').classList.add('active-tool');
            document.getElementById('paintConstraint').classList.remove('active-tool');
            this._renderGroupsList();
        });

        // Shape Selection
        document.getElementById('addShapeCube').addEventListener('click', () => {
            this.viewer.addSelectionShape('cube');
            this._renderShapeList();
        });
        document.getElementById('addShapeCylinder').addEventListener('click', () => {
            this.viewer.addSelectionShape('cylinder');
            this._renderShapeList();
        });
        document.getElementById('addShapeBall').addEventListener('click', () => {
            this.viewer.addSelectionShape('sphere');
            this._renderShapeList();
        });
        document.getElementById('applyAllShapesToGroup').addEventListener('click', () => {
            this.viewer.applyShapeSelectionToGroup();
            this._renderGroupsList();
        });
        
        // Step 3: Solution type card selection
        document.querySelectorAll('.solution-type-card').forEach(card => {
            card.addEventListener('click', () => {
                document.querySelectorAll('.solution-type-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this._applySolutionType(card.dataset.solution);
            });
        });

        // Step 3: Optimization algorithm selector (shown only for topology)
        document.getElementById('solverSelect').addEventListener('change', (e) => {
            const solverValue = e.target.value;
            if (solverValue === 'petsc-bddc') {
                this.config.solver = 'petsc';
                this.config.petscPC = 'bddc';
            } else if (solverValue === 'petsc-mg') {
                this.config.solver = 'petsc';
                this.config.petscPC = 'mg';
            } else if (solverValue === 'petsc-jacobi') {
                this.config.solver = 'petsc';
                this.config.petscPC = 'jacobi';
            } else {
                this.config.solver = solverValue;
                if (solverValue !== 'petsc') this.config.petscPC = 'bddc';
            }
            const geneticPanel = document.getElementById('geneticPanel');
            if (geneticPanel) {
                geneticPanel.style.display = solverValue === 'genetic' ? '' : 'none';
            }
        });

        document.getElementById('maxIterations').addEventListener('input', (e) => {
            this.config.maxIterations = parseInt(e.target.value);
        });
        
        document.getElementById('penaltyFactor').addEventListener('input', (e) => {
            this.config.penaltyFactor = parseFloat(e.target.value);
        });
        
        document.getElementById('filterRadius').addEventListener('input', (e) => {
            this.config.filterRadius = parseFloat(e.target.value);
        });

        const feaSolverBackendSelect = document.getElementById('feaSolverBackend');
        if (feaSolverBackendSelect) {
            feaSolverBackendSelect.addEventListener('change', (e) => {
                this.config.feaSolverBackend = e.target.value;
                // Legacy compat: set useGPU based on backend selection
                this.config.useGPU = (e.target.value === 'webgpu');
            });
        }

        const volumetricOutputModeSelect = document.getElementById('volumetricOutputMode');
        if (volumetricOutputModeSelect) {
            volumetricOutputModeSelect.value = this.config.volumetricOutputMode;
            volumetricOutputModeSelect.addEventListener('change', (e) => {
                this.config.volumetricOutputMode = e.target.value;
                this.optimizer.updateConfig({ volumetricOutputMode: this.config.volumetricOutputMode });
                this._updateVolumetricControlState();
            });
        }

        const extractVolumetricStressButton = document.getElementById('extractVolumetricStress');
        if (extractVolumetricStressButton) {
            extractVolumetricStressButton.addEventListener('click', () => {
                this.requestVolumetricSnapshot();
            });
        }
        this._updateVolumetricControlState();
        
        document.getElementById('minCrossSection').addEventListener('input', (e) => {
            this.config.minCrossSection = parseFloat(e.target.value);
        });

        // Step 5: Manufacturing method card selection
        document.querySelectorAll('.manufacturing-card').forEach(card => {
            card.addEventListener('click', () => {
                document.querySelectorAll('.manufacturing-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this._applyManufacturingPreset(card.dataset.method);
            });
        });

        document.getElementById('constrainToSolid').addEventListener('change', (e) => {
            this.config.constrainToSolid = e.target.checked;
        });

        document.getElementById('preventVoids').addEventListener('change', (e) => {
            this.config.preventVoids = e.target.checked;
        });

        // Manufacturing constraint controls
        document.getElementById('manufacturingConstraint').addEventListener('change', (e) => {
            this.config.manufacturingConstraint = e.target.checked;
            document.getElementById('manufacturingControls').style.display = e.target.checked ? '' : 'none';
        });

        const angleSlider = document.getElementById('manufacturingAngle');
        const angleInput = document.getElementById('manufacturingAngleInput');
        angleSlider.addEventListener('input', (e) => {
            this.config.manufacturingAngle = parseFloat(e.target.value);
            angleInput.value = e.target.value;
        });
        angleInput.addEventListener('input', (e) => {
            this.config.manufacturingAngle = parseFloat(e.target.value);
            angleSlider.value = e.target.value;
        });

        document.getElementById('manufacturingMinRadius').addEventListener('input', (e) => {
            this.config.manufacturingMinRadius = parseFloat(e.target.value) || 0;
        });
        document.getElementById('manufacturingMaxDepth').addEventListener('input', (e) => {
            this.config.manufacturingMaxDepth = parseFloat(e.target.value) || 0;
        });

        // Genetic algorithm parameter controls
        document.getElementById('populationSize').addEventListener('input', (e) => {
            this.config.populationSize = parseInt(e.target.value);
        });
        document.getElementById('eliteCount').addEventListener('input', (e) => {
            this.config.eliteCount = parseInt(e.target.value);
        });
        document.getElementById('mutationRate').addEventListener('input', (e) => {
            this.config.mutationRate = parseFloat(e.target.value);
        });
        document.getElementById('crossoverRate').addEventListener('input', (e) => {
            this.config.crossoverRate = parseFloat(e.target.value);
        });
        document.getElementById('tournamentSize').addEventListener('input', (e) => {
            this.config.tournamentSize = parseInt(e.target.value);
        });
        document.getElementById('volumePenalty').addEventListener('input', (e) => {
            this.config.volumePenalty = parseFloat(e.target.value);
        });

        // Collapsible sub-panel toggle
        document.querySelectorAll('.sub-panel-header[data-toggle]').forEach((header) => {
            header.addEventListener('click', () => {
                const panel = header.closest('.sub-panel');
                if (panel) panel.classList.toggle('collapsed');
            });
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

        // Advanced parameters
        document.getElementById('penalStart').addEventListener('input', (e) => {
            this.config.penalStart = parseFloat(e.target.value);
        });

        document.getElementById('continuationIters').addEventListener('input', (e) => {
            this.config.continuationIters = parseInt(e.target.value);
        });

        document.getElementById('useProjection').addEventListener('change', (e) => {
            this.config.useProjection = e.target.checked;
            const projControls = document.getElementById('projectionControls');
            const projControls2 = document.getElementById('projectionControls2');
            if (projControls) projControls.style.display = e.target.checked ? '' : 'none';
            if (projControls2) projControls2.style.display = e.target.checked ? '' : 'none';
        });

        document.getElementById('betaMax').addEventListener('input', (e) => {
            this.config.betaMax = parseInt(e.target.value);
        });

        document.getElementById('betaInterval').addEventListener('input', (e) => {
            this.config.betaInterval = parseInt(e.target.value);
        });

        const nonlinearLoadStepsEl = document.getElementById('nonlinearLoadSteps');
        if (nonlinearLoadStepsEl) {
            nonlinearLoadStepsEl.addEventListener('input', (e) => {
                this.config.nonlinearLoadSteps = parseInt(e.target.value) || 10;
            });
        }
        const nonlinearMaxNewtonIterEl = document.getElementById('nonlinearMaxNewtonIter');
        if (nonlinearMaxNewtonIterEl) {
            nonlinearMaxNewtonIterEl.addEventListener('input', (e) => {
                this.config.nonlinearMaxNewtonIter = parseInt(e.target.value) || 20;
            });
        }
        const nonlinearToleranceEl = document.getElementById('nonlinearTolerance');
        if (nonlinearToleranceEl) {
            nonlinearToleranceEl.addEventListener('input', (e) => {
                this.config.nonlinearTolerance = parseFloat(e.target.value) || 1e-6;
            });
        }

        // Linear FEA sub-type selector
        const feaSubTypeEl = document.getElementById('feaSubType');
        if (feaSubTypeEl) {
            feaSubTypeEl.addEventListener('change', (e) => {
                this.config.feaSubType = e.target.value;
                this._updateFeaSubTypeUI(e.target.value);
            });
        }
        // Linear FEA buckling parameters
        const feaBucklingModesEl = document.getElementById('feaBucklingModes');
        if (feaBucklingModesEl) {
            feaBucklingModesEl.addEventListener('input', (e) => {
                this.config.feaBucklingModes = parseInt(e.target.value) || 3;
            });
        }
        const feaBucklingImperfectionEl = document.getElementById('feaBucklingImperfection');
        if (feaBucklingImperfectionEl) {
            feaBucklingImperfectionEl.addEventListener('input', (e) => {
                this.config.feaBucklingImperfection = parseFloat(e.target.value) || 0.01;
            });
        }

        // Nonlinear analysis sub-type and material model
        const analysisTypeEl = document.getElementById('nonlinearAnalysisType');
        if (analysisTypeEl) {
            analysisTypeEl.addEventListener('change', (e) => {
                this.config.nonlinearAnalysisType = e.target.value;
                this._updateNonlinearSubTypeUI(e.target.value);
            });
        }
        // Fracture parameters
        const fractureMethodEl = document.getElementById('fractureMethod');
        if (fractureMethodEl) {
            fractureMethodEl.addEventListener('change', (e) => {
                this.config.fractureMethod = e.target.value;
            });
        }
        const fractureToughnessEl = document.getElementById('fractureToughness');
        if (fractureToughnessEl) {
            fractureToughnessEl.addEventListener('input', (e) => {
                this.config.fractureToughness = parseFloat(e.target.value) || 2700;
            });
        }
        const fractureLengthScaleEl = document.getElementById('fractureLengthScale');
        if (fractureLengthScaleEl) {
            fractureLengthScaleEl.addEventListener('input', (e) => {
                this.config.fractureLengthScale = parseFloat(e.target.value) || 0.01;
            });
        }
        // Buckling parameters
        const bucklingModesEl = document.getElementById('bucklingModes');
        if (bucklingModesEl) {
            bucklingModesEl.addEventListener('input', (e) => {
                this.config.bucklingModes = parseInt(e.target.value) || 3;
            });
        }
        const bucklingImperfectionEl = document.getElementById('bucklingImperfection');
        if (bucklingImperfectionEl) {
            bucklingImperfectionEl.addEventListener('input', (e) => {
                this.config.bucklingImperfection = parseFloat(e.target.value) || 0.01;
            });
        }
        // Plasticity parameters
        const hardeningModulusEl = document.getElementById('hardeningModulus');
        if (hardeningModulusEl) {
            hardeningModulusEl.addEventListener('input', (e) => {
                this.config.hardeningModulus = parseFloat(e.target.value) || 1000;
            });
        }

        // Result display mode selector
        const resultColorModeEl = document.getElementById('resultColorMode');
        if (resultColorModeEl) {
            resultColorModeEl.addEventListener('change', (e) => {
                const mode = e.target.value;
                this.config.resultColorMode = mode;
                this.viewer.setColorMode(mode);
                this._updateStressBarForMode(mode);
            });
        }

        // Result overlay toggles
        const resultShowDeformedEl = document.getElementById('resultShowDeformed');
        if (resultShowDeformedEl) {
            resultShowDeformedEl.addEventListener('change', (e) => {
                this.viewer.showDeformed = e.target.checked;
                this.viewer._needsRebuild = true;
                this.viewer.draw();
            });
        }
        const resultIsoSurfaceEl = document.getElementById('resultIsoSurface');
        if (resultIsoSurfaceEl) {
            resultIsoSurfaceEl.addEventListener('input', (e) => {
                const val = parseInt(e.target.value);
                const label = document.getElementById('resultIsoSurfaceValue');
                if (label) label.textContent = val + '%';
                this.viewer.isoSurfaceLevel = val / 100;
                this.viewer._needsRebuild = true;
                this.viewer.draw();
            });
        }
        const resultClipPlaneEl = document.getElementById('resultClipPlane');
        if (resultClipPlaneEl) {
            resultClipPlaneEl.addEventListener('change', (e) => {
                this.viewer.resultClipPlane = e.target.checked;
                this.viewer._needsRebuild = true;
                this.viewer.draw();
            });
        }
        
        document.getElementById('runOptimization').addEventListener('click', () => {
            this.runOptimization();
        });
        
        document.getElementById('cancelOptimization').addEventListener('click', () => {
            this.cancelOptimization();
        });

        document.getElementById('pauseOptimization').addEventListener('click', () => {
            this.togglePauseOptimization();
        });
        
        // Step 6: Export (now part of solve & export)
        document.getElementById('downloadSTL').addEventListener('click', () => {
            this.exportSTL();
        });
        
        document.getElementById('downloadJSON').addEventListener('click', () => {
            this.exportJSON();
        });

        document.getElementById('resetSolution').addEventListener('click', () => {
            this.resetSolution();
        });

        document.getElementById('exportSetup').addEventListener('click', () => {
            this.exportSetup();
        });

        document.getElementById('importSetup').addEventListener('click', () => {
            const input = document.getElementById('importSetupFile');
            if (input) {
                input.value = '';
                input.click();
            }
        });
        
        document.getElementById('resetApp').addEventListener('click', () => {
            this.reset();
        });

        // Benchmark history modal
        document.getElementById('showBenchmarkHistory').addEventListener('click', () => {
            this.displayBenchmarkHistory();
            document.getElementById('benchmarkModal').classList.remove('hidden');
        });
        document.getElementById('closeBenchmarkModal').addEventListener('click', () => {
            document.getElementById('benchmarkModal').classList.add('hidden');
        });
        document.getElementById('benchmarkModal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                document.getElementById('benchmarkModal').classList.add('hidden');
            }
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

        document.getElementById('toggleSection').addEventListener('click', () => {
            this.viewer.toggleSection();
            const btn = document.getElementById('toggleSection');
            btn.classList.toggle('active-tool', this.viewer.sectionEnabled);
            this._updateSectionSliders();
        });
        
        document.getElementById('toggleReference').addEventListener('click', () => {
            this.viewer.toggleReference();
            const btn = document.getElementById('toggleReference');
            btn.classList.toggle('active-tool', !this.viewer.showReference);
        });

        document.getElementById('resetCamera').addEventListener('click', () => {
            this.viewer.resetCamera();
            document.getElementById('toggleSection').classList.remove('active-tool');
            this._updateSectionSliders();
        });

        // Section plane sliders
        const sectionAzimuth = document.getElementById('sectionAzimuth');
        const sectionElevation = document.getElementById('sectionElevation');
        const sectionOffsetSlider = document.getElementById('sectionOffsetSlider');
        const sectionAzimuthValue = document.getElementById('sectionAzimuthValue');
        const sectionElevationValue = document.getElementById('sectionElevationValue');
        const sectionOffsetValue = document.getElementById('sectionOffsetValue');

        const applySectionSliders = () => {
            const az = parseFloat(sectionAzimuth.value) * Math.PI / 180;
            const el = parseFloat(sectionElevation.value) * Math.PI / 180;
            // Spherical to Cartesian (Y-up): azimuth rotates in XZ plane, elevation tilts toward Y
            const cosEl = Math.cos(el);
            this.viewer.sectionNormal = [cosEl * Math.sin(az), Math.sin(el), cosEl * Math.cos(az)];
            if (this.viewer.model) {
                const maxDim = Math.max(this.viewer.model.nx, this.viewer.model.ny, this.viewer.model.nz);
                this.viewer.sectionOffset = parseFloat(sectionOffsetSlider.value) / 100 * maxDim;
            }
            sectionAzimuthValue.textContent = `${sectionAzimuth.value}°`;
            sectionElevationValue.textContent = `${sectionElevation.value}°`;
            sectionOffsetValue.textContent = `${sectionOffsetSlider.value}%`;
            this.viewer.draw();
        };

        sectionAzimuth.addEventListener('input', applySectionSliders);
        sectionElevation.addEventListener('input', applySectionSliders);
        sectionOffsetSlider.addEventListener('input', applySectionSliders);


        // Strain range is now controlled by dragging handles on the stress scale bar
        // in the viewer. Register callback to stay in sync.
        this.viewer.onStrainRangeChange = (min, max) => {
            // Optional: external sync point
        };

        // Density threshold slider
        const densityThresholdSlider = document.getElementById('densityThresholdSlider');
        const densityThresholdValue = document.getElementById('densityThresholdValue');
        if (densityThresholdSlider) {
            densityThresholdSlider.addEventListener('input', () => {
                const val = parseFloat(densityThresholdSlider.value);
                this.viewer.setDensityThreshold(val);
                const effective = this.viewer.densityThreshold;
                densityThresholdSlider.value = effective.toFixed(2);
                densityThresholdValue.textContent = effective.toFixed(2);
            });
        }

        // Displacement toggle and scale slider
        const dispToggle = document.getElementById('toggleDisplacement');
        const dispScaleSlider = document.getElementById('displacementScaleSlider');
        const dispScaleValue = document.getElementById('displacementScaleValue');
        if (dispToggle) {
            dispToggle.addEventListener('change', () => {
                this.viewer.showDisplacement = dispToggle.checked;
                this.viewer._needsRebuild = true;
                this.viewer.draw();
            });
        }
        if (dispScaleSlider) {
            dispScaleSlider.addEventListener('input', () => {
                const val = parseInt(dispScaleSlider.value, 10);
                this.viewer.setDisplacementScale(val);
                if (dispScaleValue) dispScaleValue.textContent = `${val}×`;
            });
        }

        // Load step slider for nonlinear FEA time stepping
        const loadStepSlider = document.getElementById('loadStepSlider');
        if (loadStepSlider) {
            loadStepSlider.addEventListener('input', () => {
                this._applyLoadStep(parseInt(loadStepSlider.value, 10));
            });
        }
    }

    _updateForceVector() {
        const fx = parseFloat(document.getElementById('forceVectorX').value) || 0;
        const fy = parseFloat(document.getElementById('forceVectorY').value) || 0;
        const fz = parseFloat(document.getElementById('forceVectorZ').value) || 0;
        this.config.forceVector = [fx, fy, fz];
        this.viewer.forceVector = [fx, fy, fz];
    }

    _updateSectionSliders() {
        const container = document.getElementById('sectionSliderContainer');
        if (this.viewer.sectionEnabled) {
            container.classList.remove('hidden');
            // Convert current normal to azimuth/elevation
            const [nx, ny, nz] = this.viewer.sectionNormal;
            const elevation = Math.asin(Math.max(-1, Math.min(1, ny)));
            const azimuth = Math.atan2(nx, nz);
            const azDeg = ((azimuth * 180 / Math.PI) + 360) % 360;
            const elDeg = elevation * 180 / Math.PI;

            document.getElementById('sectionAzimuth').value = Math.round(azDeg);
            document.getElementById('sectionElevation').value = Math.round(elDeg);
            document.getElementById('sectionAzimuthValue').textContent = `${Math.round(azDeg)}°`;
            document.getElementById('sectionElevationValue').textContent = `${Math.round(elDeg)}°`;

            if (this.viewer.model) {
                const maxDim = Math.max(this.viewer.model.nx, this.viewer.model.ny, this.viewer.model.nz);
                const pct = maxDim > 0 ? (this.viewer.sectionOffset / maxDim) * 100 : 50;
                document.getElementById('sectionOffsetSlider').value = pct;
                document.getElementById('sectionOffsetValue').textContent = `${Math.round(pct)}%`;
            }
        } else {
            container.classList.add('hidden');
        }
    }

    _updateDOFPreview(dofValue) {
        const previewEl = document.getElementById('dofPreviewText');
        if (!previewEl) return;
        previewEl.textContent = DOF_DESCRIPTIONS[dofValue] || DOF_DESCRIPTIONS['all'];
    }

    /**
     * Toggle between template default and custom constraints/forces.
     * @param {boolean} useDefaults - Whether to use template defaults
     */
    _applyTemplateDefaultsToggle(useDefaults) {
        const forceSelect = document.getElementById('forceDirection');
        const constraintSelect = document.getElementById('constraintPosition');

        if (useDefaults && this.currentModel) {
            const model = this.currentModel;
            // Apply template defaults
            if (model.forceDirection) {
                this.config.forceDirection = model.forceDirection;
                this.viewer.forceDirection = model.forceDirection;
                forceSelect.value = model.forceDirection;
                const v = FORCE_DIRECTION_VECTORS[model.forceDirection] || [0, -1, 0];
                document.getElementById('forceVectorX').value = v[0];
                document.getElementById('forceVectorY').value = v[1];
                document.getElementById('forceVectorZ').value = v[2];
                this.config.forceVector = null;
                this.viewer.forceVector = null;
            }
            if (model.constraintPositions) {
                this.config.constraintPosition = model.constraintPositions;
                const optionExists = Array.from(constraintSelect.options).some(o => o.value === model.constraintPositions);
                if (optionExists) constraintSelect.value = model.constraintPositions;
            }

            // Disable manual controls when using defaults
            forceSelect.disabled = true;
            constraintSelect.disabled = true;
            document.getElementById('forceVectorX').disabled = true;
            document.getElementById('forceVectorY').disabled = true;
            document.getElementById('forceVectorZ').disabled = true;
        } else {
            // Re-enable manual controls
            forceSelect.disabled = false;
            constraintSelect.disabled = false;
            document.getElementById('forceVectorX').disabled = false;
            document.getElementById('forceVectorY').disabled = false;
            document.getElementById('forceVectorZ').disabled = false;
        }
        if (this.viewer) this.viewer.draw();
    }

    _updateVolumetricControlState() {
        const extractButton = document.getElementById('extractVolumetricStress');
        if (!extractButton) return;
        extractButton.disabled = this.config.volumetricOutputMode === 'every-iteration';
    }

    _applyVolumetricSnapshot(volumetricData, source) {
        if (!volumetricData) return;
        this.lastVolumetricData = volumetricData;
        if (source === 'pause') this.pausedVolumetricData = volumetricData;
        if (source === 'complete') this.finalVolumetricData = volumetricData;
        this.viewer.setVolumetricStressData(volumetricData);
    }

    /**
     * Apply a specific nonlinear load step snapshot to the viewer.
     * Updates displacement and stress data to reflect the state at the given step.
     */
    _applyLoadStep(stepIndex) {
        if (!this._nonlinearSnapshots || stepIndex < 1) return;
        const snapshots = this._nonlinearSnapshots;
        const idx = Math.min(stepIndex, snapshots.length) - 1;
        const snap = snapshots[idx];
        if (!snap) return;

        const totalSteps = snapshots.length;
        const loadStepValue = document.getElementById('loadStepValue');
        const loadStepFraction = document.getElementById('loadStepFraction');
        const loadStepStatus = document.getElementById('loadStepStatus');
        if (loadStepValue) loadStepValue.textContent = `${snap.step}/${totalSteps}`;
        if (loadStepFraction) loadStepFraction.textContent = `${(snap.loadFraction * 100).toFixed(1)}%`;

        // Show convergence/failure status
        if (loadStepStatus) {
            if (!snap.converged) {
                loadStepStatus.textContent = '⚠ Not converged — possible failure';
                loadStepStatus.style.color = '#dc2626';
            } else {
                loadStepStatus.textContent = '✓ Converged';
                loadStepStatus.style.color = '#10b981';
            }
        }

        // Update volumetric stress for this step
        if (snap.vonMisesStress && this._nonlinearNx) {
            const volData = {
                nx: this._nonlinearNx,
                ny: this._nonlinearNy,
                nz: this._nonlinearNz,
                iteration: snap.step,
                maxStress: 0,
                stress: snap.vonMisesStress
            };
            // Compute max stress for this step
            for (let i = 0; i < snap.vonMisesStress.length; i++) {
                if (snap.vonMisesStress[i] > volData.maxStress) {
                    volData.maxStress = snap.vonMisesStress[i];
                }
            }
            this.viewer.setVolumetricStressData(volData);
        }

        // Update triaxiality scalar field for this step
        if (snap.triaxiality) {
            let minTri = Infinity, maxTri = -Infinity;
            for (let i = 0; i < snap.triaxiality.length; i++) {
                if (snap.triaxiality[i] < minTri) minTri = snap.triaxiality[i];
                if (snap.triaxiality[i] > maxTri) maxTri = snap.triaxiality[i];
            }
            this.viewer.setScalarField('triaxiality', snap.triaxiality, minTri, maxTri);
        }

        // Update damage / phase-field for this step
        if (snap.damageField) {
            this.viewer.setDamageField(snap.damageField);
            this.viewer.setScalarField('damage', snap.damageField, 0, 1);
        }

        // Update eroded elements for this step
        if (snap.erodedElements) {
            this.viewer.setErodedElements(new Set(snap.erodedElements));
        } else {
            this.viewer.setErodedElements(null);
        }

        // Update displacement data for this step
        if (snap.displacement && this._nonlinearNx) {
            const U = snap.displacement;
            let maxU = 0;
            for (let i = 0; i < U.length; i++) {
                const v = Math.abs(U[i]);
                if (v > maxU) maxU = v;
            }
            const normalizedU = new Float32Array(U.length);
            if (maxU > 0) {
                for (let i = 0; i < U.length; i++) normalizedU[i] = U[i] / maxU;
            }
            this.viewer.setDisplacementData({
                U: normalizedU,
                nx: this._nonlinearNx,
                ny: this._nonlinearNy,
                nz: this._nonlinearNz
            });
            this.viewer._needsRebuild = true;
            this.viewer.draw();
        }
    }

    requestVolumetricSnapshot() {
        this.optimizer.requestVolumetricSnapshot();
    }

    _serializeModel(model) {
        if (!model) return null;
        return {
            ...model,
            elements: model.elements ? Array.from(model.elements) : null,
            originalVertices: model.originalVertices ? model.originalVertices : null
        };
    }

    _deserializeModel(serializedModel) {
        if (!serializedModel) return null;
        const model = { ...serializedModel };
        if (Array.isArray(serializedModel.elements)) {
            model.elements = new Float32Array(serializedModel.elements);
        }
        return model;
    }

    _applyConfigToUI() {
        const cfg = this.config;

        const solverSelect = document.getElementById('solverSelect');
        if (solverSelect) {
            let solverValue = cfg.solver || 'auto';
            if (solverValue === 'petsc') {
                if (cfg.petscPC === 'mg') solverValue = 'petsc-mg';
                else if (cfg.petscPC === 'jacobi') solverValue = 'petsc-jacobi';
                else solverValue = 'petsc-bddc';
            }
            solverSelect.value = solverValue;
            const geneticPanel = document.getElementById('geneticPanel');
            if (geneticPanel) {
                geneticPanel.style.display = solverValue === 'genetic' ? '' : 'none';
            }
        }

        const setValue = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined && value !== null) el.value = value;
        };
        const setChecked = (id, value) => {
            const el = document.getElementById(id);
            if (el && value !== undefined) el.checked = !!value;
        };

        setValue('materialSelect', cfg.material || 'custom');
        setValue('youngsModulus', cfg.youngsModulus);
        setValue('poissonsRatio', cfg.poissonsRatio);
        setValue('yieldStrength', cfg.yieldStrength);
        setValue('volumeFraction', cfg.volumeFraction);
        setValue('forceDirection', cfg.forceDirection || 'down');
        setValue('forceType', cfg.forceType || 'total');
        setValue('forceMagnitude', cfg.forceMagnitude);
        setValue('constraintPosition', cfg.constraintPosition || 'left');
        setValue('constraintDOFs', cfg.constraintDOFs || 'all');
        setValue('maxIterations', cfg.maxIterations);
        setValue('penaltyFactor', cfg.penaltyFactor);
        setValue('filterRadius', cfg.filterRadius);
        setValue('minCrossSection', cfg.minCrossSection);
        setValue('manufacturingAngle', cfg.manufacturingAngle);
        setValue('manufacturingAngleInput', cfg.manufacturingAngle);
        setValue('manufacturingMinRadius', cfg.manufacturingMinRadius);
        setValue('manufacturingMaxDepth', cfg.manufacturingMaxDepth);
        setValue('populationSize', cfg.populationSize);
        setValue('eliteCount', cfg.eliteCount);
        setValue('mutationRate', cfg.mutationRate);
        setValue('crossoverRate', cfg.crossoverRate);
        setValue('tournamentSize', cfg.tournamentSize);
        setValue('volumePenalty', cfg.volumePenalty);
        setValue('penalStart', cfg.penalStart);
        setValue('continuationIters', cfg.continuationIters);
        setValue('betaMax', cfg.betaMax);
        setValue('betaInterval', cfg.betaInterval);
        setValue('minGranuleSize', cfg.minGranuleSize);
        setValue('maxGranuleSize', cfg.maxGranuleSize);
        setValue('amrInterval', cfg.amrInterval);
        setValue('volumetricOutputMode', cfg.volumetricOutputMode || 'on-stop');

        setValue('feaSolverBackend', cfg.feaSolverBackend || 'auto');
        setChecked('constrainToSolid', cfg.constrainToSolid);
        setChecked('preventVoids', cfg.preventVoids);
        setChecked('manufacturingConstraint', cfg.manufacturingConstraint);
        setChecked('useProjection', cfg.useProjection !== false);
        setChecked('useAMR', cfg.useAMR !== false);

        const volumeFractionValue = document.getElementById('volumeFractionValue');
        if (volumeFractionValue) volumeFractionValue.textContent = `${Math.round((cfg.volumeFraction || 0) * 100)}%`;

        const manufacturingControls = document.getElementById('manufacturingControls');
        if (manufacturingControls) manufacturingControls.style.display = cfg.manufacturingConstraint ? '' : 'none';
        const projectionControls = document.getElementById('projectionControls');
        if (projectionControls) projectionControls.style.display = cfg.useProjection !== false ? '' : 'none';
        const projectionControls2 = document.getElementById('projectionControls2');
        if (projectionControls2) projectionControls2.style.display = cfg.useProjection !== false ? '' : 'none';
        const amrControls = document.getElementById('amrControls');
        if (amrControls) amrControls.style.display = cfg.useAMR !== false ? '' : 'none';
        const amrControls2 = document.getElementById('amrControls2');
        if (amrControls2) amrControls2.style.display = cfg.useAMR !== false ? '' : 'none';
        const amrControls3 = document.getElementById('amrControls3');
        if (amrControls3) amrControls3.style.display = cfg.useAMR !== false ? '' : 'none';

        const forceLabel = document.getElementById('forceMagnitudeLabel');
        if (forceLabel) {
            forceLabel.textContent = (cfg.forceType === 'pressure') ? 'Pressure (N/mm² = MPa)' : 'Force Magnitude (N)';
        }

        if (Array.isArray(cfg.forceVector) && cfg.forceVector.length >= 3) {
            setValue('forceVectorX', cfg.forceVector[0]);
            setValue('forceVectorY', cfg.forceVector[1]);
            setValue('forceVectorZ', cfg.forceVector[2]);
        }

        this._updateDOFPreview(cfg.constraintDOFs || 'all');
        this._updateVolumetricControlState();

        this.viewer.forceDirection = cfg.forceDirection || 'down';
        this.viewer.forceType = cfg.forceType || 'total';
        this.viewer.forceMagnitude = cfg.forceMagnitude || 1000;
        this.viewer.forceVector = Array.isArray(cfg.forceVector) ? cfg.forceVector : null;
        this.viewer.draw();

        // Restore solution type cards
        const solutionType = cfg.solutionType || 'topology';
        document.querySelectorAll('.solution-type-card').forEach(c => {
            c.classList.toggle('selected', c.dataset.solution === solutionType);
        });
        this._applySolutionType(solutionType);
        // Restore linear FEA sub-type
        setValue('feaSubType', cfg.feaSubType || 'static');
        setValue('feaBucklingModes', cfg.feaBucklingModes || 3);
        setValue('feaBucklingImperfection', cfg.feaBucklingImperfection || 0.01);
        this._updateFeaSubTypeUI(cfg.feaSubType || 'static');
        // Restore nonlinear FEA params
        setValue('nonlinearLoadSteps', cfg.nonlinearLoadSteps || 10);
        setValue('nonlinearMaxNewtonIter', cfg.nonlinearMaxNewtonIter || 20);
        setValue('nonlinearTolerance', cfg.nonlinearTolerance || 1e-6);
        setValue('nonlinearAnalysisType', cfg.nonlinearAnalysisType || 'large-deformation');
        setValue('fractureMethod', cfg.fractureMethod || 'phase-field');
        setValue('fractureToughness', cfg.fractureToughness || 2700);
        setValue('fractureLengthScale', cfg.fractureLengthScale || 0.01);
        setValue('bucklingModes', cfg.bucklingModes || 3);
        setValue('bucklingImperfection', cfg.bucklingImperfection || 0.01);
        setValue('hardeningModulus', cfg.hardeningModulus || 1000);
        this._updateNonlinearSubTypeUI(cfg.nonlinearAnalysisType || 'large-deformation');
        // Restore viewer result color mode
        setValue('resultColorMode', cfg.resultColorMode || 'stress');
        if (this.viewer) {
            this.viewer.setColorMode(cfg.resultColorMode || 'stress');
            this._updateStressBarForMode(cfg.resultColorMode || 'stress');
        }
    }

    _applySelectionState(selectionState) {
        const state = selectionState || {};
        const groups = Array.isArray(state.selectionGroups) ? state.selectionGroups : [];

        this.viewer.selectionGroups = groups.map((g) => ({
            ...g,
            faces: new Set(Array.isArray(g.faces) ? g.faces : [])
        }));
        this.viewer._nextGroupId = this.viewer.selectionGroups.reduce((maxId, g) => Math.max(maxId, g.id || 0), 0) + 1;
        this.viewer.activeGroupId = state.activeGroupId || (this.viewer.selectionGroups[0]?.id ?? null);

        if (this.viewer.selectionGroups.length > 0) {
            this.viewer._syncGroupsToFaceSets();
        } else {
            this.viewer.paintedConstraintFaces = new Set(state.paintedConstraints || []);
            this.viewer.paintedForceFaces = new Set(state.paintedForces || []);
            this.viewer.paintedKeepFaces = new Set(state.paintedKeep || []);
        }

        if (state.useAngleSelection !== undefined) {
            this.viewer.useAngleSelection = !!state.useAngleSelection;
            const useAngleSelection = document.getElementById('useAngleSelection');
            if (useAngleSelection) useAngleSelection.checked = this.viewer.useAngleSelection;
            const angleToleranceControls = document.getElementById('angleToleranceControls');
            if (angleToleranceControls) {
                angleToleranceControls.style.display = this.viewer.useAngleSelection ? '' : 'none';
            }
        }
        if (state.angleTolerance !== undefined) {
            this.viewer.angleTolerance = state.angleTolerance;
            const angleTolerance = document.getElementById('angleTolerance');
            const angleToleranceInput = document.getElementById('angleToleranceInput');
            if (angleTolerance) angleTolerance.value = state.angleTolerance;
            if (angleToleranceInput) angleToleranceInput.value = state.angleTolerance;
        }
        if (state.brushSize !== undefined) {
            this.viewer.brushSize = state.brushSize;
            const brushSize = document.getElementById('brushSize');
            const brushSizeValue = document.getElementById('brushSizeValue');
            if (brushSize) brushSize.value = state.brushSize;
            if (brushSizeValue) brushSizeValue.textContent = state.brushSize;
        }
        if (Array.isArray(state.selectionShapes)) {
            this.viewer.selectionShapes = state.selectionShapes.map(s => ({
                id: s.id,
                shapeType: s.shapeType || 'cube',
                position: Array.isArray(s.position) ? [...s.position] : [0, 0, 0],
                size: Array.isArray(s.size) ? [...s.size] : [1, 1, 1],
                rotation: Array.isArray(s.rotation) ? [...s.rotation] : [0, 0, 0]
            }));
            this.viewer._nextShapeId = this.viewer.selectionShapes.reduce((m, s) => Math.max(m, s.id || 0), 0) + 1;
            this.viewer._computeShapeHighlights();
        }
        this._renderGroupsList();
        this._renderShapeList();
    }

    async importSetup(file) {
        try {
            const text = await file.text();
            const setup = JSON.parse(text);

            if (!setup || !setup.model || !setup.config) {
                throw new Error('Invalid setup file: missing model/config');
            }

            const importedModel = this._deserializeModel(setup.model);
            if (!importedModel || !importedModel.nx || !importedModel.ny || !importedModel.nz) {
                throw new Error('Invalid setup file: malformed model dimensions');
            }

            this.currentModel = importedModel;
            this.optimizedModel = null;
            this.pausedVolumetricData = null;
            this.finalVolumetricData = null;
            this.lastVolumetricData = null;

            this.config = {
                ...this.config,
                ...setup.config
            };

            this.viewer.setModel(importedModel);
            this._showMobileViewerToggle();
            if (setup.referenceModel && Array.isArray(setup.referenceModel.vertices)) {
                this.viewer.setReferenceModel(setup.referenceModel.vertices, setup.referenceModel.bounds || null);
            }

            this._applyConfigToUI();
            this._applySelectionState(setup.selectionState || {});
            this.viewer.setVolumetricStressData(null);

            const info = document.getElementById('modelInfo');
            info.classList.remove('hidden');
            info.innerHTML = `
                <strong>Setup imported:</strong> ${file.name}<br>
                <strong>Grid:</strong> ${importedModel.nx} × ${importedModel.ny} × ${importedModel.nz}<br>
                <strong>Elements:</strong> ${importedModel.nx * importedModel.ny * importedModel.nz}<br>
                <em>Ready to re-run with restored parameters, forces and constraints.</em>
            `;

            document.getElementById('progressContainer').classList.add('hidden');
            document.getElementById('exportSection').classList.add('hidden');
            document.getElementById('optimizationResults').innerHTML = '<em>Imported setup. Ready to solve.</em>';
            this.workflow.enableStep(2);
            this.workflow.enableStep(3);
            this.workflow.enableStep(4);
            this.workflow.enableStep(5);
            this.workflow.enableStep(6);
            this.workflow.goToStep(6);
        } catch (error) {
            console.error('Failed to import setup:', error);
            alert(`Failed to import setup: ${error.message}`);
        }
    }

    exportSetup() {
        if (!this.currentModel) {
            alert('Please import or create a model first');
            return;
        }

        const payload = {
            schema: 'topology-setup-v1',
            timestamp: new Date().toISOString(),
            config: { ...this.config },
            model: this._serializeModel(this.currentModel),
            referenceModel: {
                vertices: this.viewer.referenceVertices || null,
                bounds: this.viewer._referenceBounds || null
            },
            selectionState: {
                paintedConstraints: Array.from(this.viewer.paintedConstraintFaces || []),
                paintedForces: Array.from(this.viewer.paintedForceFaces || []),
                paintedKeep: Array.from(this.viewer.paintedKeepFaces || []),
                selectionGroups: (this.viewer.selectionGroups || []).map((g) => ({
                    ...g,
                    faces: Array.from(g.faces || [])
                })),
                activeGroupId: this.viewer.activeGroupId,
                useAngleSelection: this.viewer.useAngleSelection,
                angleTolerance: this.viewer.angleTolerance,
                brushSize: this.viewer.brushSize,
                selectionShapes: (this.viewer.selectionShapes || []).map(s => ({ ...s }))
            }
        };

        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const ts = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const filename = `setup-${ts.getFullYear()}${pad(ts.getMonth() + 1)}${pad(ts.getDate())}-${pad(ts.getHours())}${pad(ts.getMinutes())}${pad(ts.getSeconds())}.json`;
        this.exporter.downloadBlob(blob, filename);
    }

    _applySolutionType(solutionType) {
        this.config.solutionType = solutionType;

        const topoAlgorithmGroup = document.getElementById('topoAlgorithmGroup');
        const feaParamsGroup = document.getElementById('feaParamsGroup');
        const nonlinearParamsGroup = document.getElementById('nonlinearParamsGroup');
        const geneticPanel = document.getElementById('geneticPanel');
        const solverParamsPanel = document.getElementById('solverParamsPanel');
        const runButton = document.getElementById('runOptimization');
        const manufacturingStep = document.querySelector('[data-step="5"]');

        // Show/hide algorithm selector
        if (topoAlgorithmGroup) {
            topoAlgorithmGroup.style.display = solutionType === 'topology' ? '' : 'none';
        }
        // Show/hide linear FEA params
        if (feaParamsGroup) {
            feaParamsGroup.style.display = solutionType === 'fea' ? '' : 'none';
        }
        // Show/hide nonlinear params
        if (nonlinearParamsGroup) {
            nonlinearParamsGroup.style.display = solutionType === 'nonlinear' ? '' : 'none';
        }
        // Hide genetic panel when not topology+genetic
        if (geneticPanel) {
            const isGenetic = solutionType === 'topology' &&
                document.getElementById('solverSelect')?.value === 'genetic';
            geneticPanel.style.display = isGenetic ? '' : 'none';
        }
        // Solver params (volume fraction etc.) only relevant for topology optimization
        if (solverParamsPanel) {
            solverParamsPanel.style.display = solutionType === 'topology' ? '' : 'none';
        }
        // Manufacturing step only relevant for topology optimization
        if (manufacturingStep) {
            manufacturingStep.dataset.skip = solutionType === 'topology' ? '' : 'true';
        }

        // Update run button label
        const labels = {
            topology: 'Run Optimization',
            fea: 'Run Linear FEA',
            nonlinear: 'Run Nonlinear FEA',
            fatigue: 'Run Fatigue Analysis'
        };
        if (runButton) runButton.textContent = labels[solutionType] || 'Run';

        // Map solution type to internal solver key
        if (solutionType === 'topology') {
            // Defer to solverSelect value
            const sel = document.getElementById('solverSelect');
            if (sel) sel.dispatchEvent(new Event('change'));
        } else if (solutionType === 'fea') {
            this.config.solver = 'fea';
        } else if (solutionType === 'nonlinear') {
            this.config.solver = 'nonlinear';
        } else if (solutionType === 'fatigue') {
            this.config.solver = 'fatigue';
        }
    }

    /**
     * Show/hide nonlinear analysis sub-type parameter groups based on selected analysis type.
     * @param {string} analysisType - 'large-deformation'|'buckling'|'plastic-deformation'|'fracture'|'shear'
     */
    _updateNonlinearSubTypeUI(analysisType) {
        const fractureParams = document.getElementById('fractureParamsGroup');
        const bucklingParams = document.getElementById('bucklingParamsGroup');
        const plasticParams = document.getElementById('plasticParamsGroup');

        if (fractureParams) fractureParams.style.display = analysisType === 'fracture' ? '' : 'none';
        if (bucklingParams) bucklingParams.style.display = analysisType === 'buckling' ? '' : 'none';
        if (plasticParams) plasticParams.style.display = analysisType === 'plastic-deformation' ? '' : 'none';

        // Auto-select appropriate material model for analysis type
        if (analysisType === 'plastic-deformation') {
            this.config.nonlinearMaterialModel = 'j2-plasticity';
        } else if (analysisType === 'fracture') {
            this.config.nonlinearMaterialModel = 'linear-elastic';
        } else if (analysisType === 'buckling' || analysisType === 'shear') {
            this.config.nonlinearMaterialModel = 'linear-elastic';
        } else {
            this.config.nonlinearMaterialModel = 'neo-hookean';
        }
    }

    /**
     * Show/hide linear FEA sub-type parameter groups based on selected sub-type.
     * @param {string} subType - 'static'|'buckling'|'shear'
     */
    _updateFeaSubTypeUI(subType) {
        const staticParams = document.getElementById('feaStaticParamsGroup');
        const bucklingParams = document.getElementById('feaBucklingParamsGroup');
        const shearParams = document.getElementById('feaShearParamsGroup');

        if (staticParams) staticParams.style.display = subType === 'static' ? '' : 'none';
        if (bucklingParams) bucklingParams.style.display = subType === 'buckling' ? '' : 'none';
        if (shearParams) shearParams.style.display = subType === 'shear' ? '' : 'none';
    }

    /**
     * Update the stress bar label and unit based on the selected result color mode.
     * @param {string} mode - Color mode identifier
     */
    _updateStressBarForMode(mode) {
        const labels = {
            stress: ['Stress (N/mm² = MPa)', 'MPa'],
            strain: ['Strain Energy Density', ''],
            displacement: ['Displacement Magnitude', 'mm'],
            density: ['Material Density', ''],
            damage: ['Damage (0=intact, 1=failed)', ''],
            triaxiality: ['Stress Triaxiality', ''],
            plasticStrain: ['Equivalent Plastic Strain', '']
        };
        const [label, unit] = labels[mode] || labels.stress;
        this.viewer.setStressBarLabel(label, unit);
    }

    _applyManufacturingPreset(method) {
        const constrainToSolid = document.getElementById('constrainToSolid');
        const preventVoids = document.getElementById('preventVoids');
        const manufacturingConstraint = document.getElementById('manufacturingConstraint');
        const manufacturingAngle = document.getElementById('manufacturingAngle');
        const manufacturingAngleInput = document.getElementById('manufacturingAngleInput');
        const manufacturingControls = document.getElementById('manufacturingControls');

        switch (method) {
            case '3d-print':
                this.config.manufacturingConstraint = true;
                this.config.manufacturingAngle = 45;
                this.config.preventVoids = true;
                this.config.constrainToSolid = false;
                constrainToSolid.checked = false;
                preventVoids.checked = true;
                manufacturingConstraint.checked = true;
                manufacturingAngle.value = 45;
                manufacturingAngleInput.value = 45;
                manufacturingControls.style.display = '';
                break;
            case 'cnc':
                this.config.manufacturingConstraint = true;
                this.config.manufacturingAngle = 90;
                this.config.preventVoids = false;
                this.config.constrainToSolid = true;
                constrainToSolid.checked = true;
                preventVoids.checked = false;
                manufacturingConstraint.checked = true;
                manufacturingAngle.value = 90;
                manufacturingAngleInput.value = 90;
                manufacturingControls.style.display = '';
                break;
            case 'injection':
                this.config.manufacturingConstraint = true;
                this.config.manufacturingAngle = 88;
                this.config.preventVoids = true;
                this.config.constrainToSolid = false;
                constrainToSolid.checked = false;
                preventVoids.checked = true;
                manufacturingConstraint.checked = true;
                manufacturingAngle.value = 88;
                manufacturingAngleInput.value = 88;
                manufacturingControls.style.display = '';
                break;
            case 'none':
            default:
                this.config.manufacturingConstraint = false;
                this.config.preventVoids = false;
                this.config.constrainToSolid = false;
                constrainToSolid.checked = false;
                preventVoids.checked = false;
                manufacturingConstraint.checked = false;
                manufacturingControls.style.display = 'none';
                break;
        }
    }

    resetSolution() {
        this.optimizer.cancel();
        this._optimizationPaused = false;
        this.optimizedModel = null;
        this.pausedVolumetricData = null;
        this.finalVolumetricData = null;
        this.lastVolumetricData = null;

        this.viewer.meshData = null;
        this.viewer.densities = null;
        this.viewer.amrCells = null;
        this.viewer.maxStress = 0;
        this.viewer.setVolumetricStressData(null);
        this.viewer._needsRebuild = true;
        this.viewer.draw();

        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const complianceText = document.getElementById('complianceText');
        const runButton = document.getElementById('runOptimization');
        const cancelButton = document.getElementById('cancelOptimization');
        const pauseButton = document.getElementById('pauseOptimization');

        progressContainer.classList.add('hidden');
        progressFill.style.width = '0%';
        progressText.textContent = 'Iteration 0 / 100';
        complianceText.textContent = '';
        runButton.classList.remove('hidden');
        cancelButton.classList.add('hidden');
        if (pauseButton) {
            pauseButton.classList.add('hidden');
            pauseButton.textContent = 'Pause';
        }

        document.getElementById('optimizationResults').innerHTML = '<em>Solution reset. Configuration, forces, constraints and model are preserved.</em>';
        document.getElementById('exportSection').classList.add('hidden');

        // Hide result display controls
        const resultDisplayReset = document.getElementById('resultDisplayContainer');
        if (resultDisplayReset) resultDisplayReset.classList.add('hidden');
        // Hide displacement and load step containers
        const dispReset = document.getElementById('displacementContainer');
        if (dispReset) dispReset.classList.add('hidden');
        const loadStepReset = document.getElementById('loadStepContainer');
        if (loadStepReset) loadStepReset.classList.add('hidden');
        // Hide density threshold container
        const densityReset = document.getElementById('densityThresholdContainer');
        if (densityReset) densityReset.classList.add('hidden');

        // Stay in step 6 (solve & export) for re-running
        this.workflow.goToStep(6);
    }

    async _initGPUControls() {
        const feaBackendSelect = document.getElementById('feaSolverBackend');
        const feaStatus = document.getElementById('feaSolverStatus');
        if (!feaBackendSelect || !feaStatus) return;

        // Probe available backends
        const backends = ['js']; // JS always available
        let gpuAvailable = false;

        // Check WASM (always available if the file exists — worker will probe at runtime)
        backends.unshift('wasm');

        // Check WebGPU
        if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    gpuAvailable = true;
                    backends.unshift('webgpu');
                }
            } catch (_err) { /* WebGPU not usable */ }
        }

        // Disable unavailable options
        for (const opt of feaBackendSelect.options) {
            if (opt.value === 'auto') continue;
            opt.disabled = !backends.includes(opt.value);
            if (opt.disabled) opt.textContent += ' (unavailable)';
        }

        // Build status text
        const parts = [];
        if (gpuAvailable) parts.push('WebGPU');
        parts.push('WASM', 'JS');
        feaStatus.textContent = `Available: ${parts.join(', ')}. Auto will pick: ${parts[0]}.`;

        // Set initial config
        if (this.config.feaSolverBackend === 'auto') {
            // Resolve auto for useGPU legacy compat
            this.config.useGPU = gpuAvailable;
        }
    }

    _renderGroupsList() {
        const container = document.getElementById('selectionGroupsList');
        if (!container) return;
        container.innerHTML = '';

        for (const group of this.viewer.selectionGroups) {
            const item = document.createElement('div');
            item.className = 'group-item' + (group.id === this.viewer.activeGroupId ? ' active-group' : '');

            const groupColors = { constraint: '#10b981', force: '#f97316', keep: '#3b82f6' };
            const colorDot = document.createElement('span');
            colorDot.className = 'group-color';
            colorDot.style.background = groupColors[group.type] || '#f97316';

            const typeSelect = document.createElement('select');
            typeSelect.className = 'group-type-select';
            typeSelect.style.cssText = 'font-size: 0.7rem; padding: 0.1rem 0.2rem; border-radius: 4px; border: 1px solid var(--border-color); background: var(--bg-secondary); cursor: pointer;';
            for (const opt of ['force', 'constraint', 'keep']) {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt.charAt(0).toUpperCase() + opt.slice(1);
                if (opt === group.type) option.selected = true;
                typeSelect.appendChild(option);
            }
            typeSelect.addEventListener('click', (e) => e.stopPropagation());
            typeSelect.addEventListener('change', (e) => {
                e.stopPropagation();
                const newType = e.target.value;
                group.type = newType;
                group.name = `${newType} ${group.id}`;
                if (newType === 'force') {
                    group.params = { direction: 'down', vector: null, magnitude: 1000, forceType: 'total', dofs: 'all' };
                } else {
                    group.params = { dofs: 'all' };
                }
                this.viewer._syncGroupsToFaceSets();
                this.viewer._needsRebuild = true;
                this.viewer.draw();
                if (group.id === this.viewer.activeGroupId) {
                    this.viewer.setPaintMode(newType);
                    this._updatePaintToolButtons(newType);
                }
                this._renderGroupsList();
            });

            const countSpan = document.createElement('span');
            countSpan.className = 'group-count';
            countSpan.textContent = `${group.faces.size} faces`;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'group-remove';
            removeBtn.textContent = '×';
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.viewer.removeSelectionGroup(group.id);
                this._renderGroupsList();
            });

            item.addEventListener('click', () => {
                this.viewer.setActiveGroup(group.id);
                this.viewer.setPaintMode(group.type);
                this._updatePaintToolButtons(group.type);
                this._renderGroupsList();
            });

            item.appendChild(colorDot);
            item.appendChild(typeSelect);
            item.appendChild(countSpan);
            item.appendChild(removeBtn);
            container.appendChild(item);
        }
    }

    _updatePaintToolButtons(type) {
        const paintForce = document.getElementById('paintForce');
        const paintConstraint = document.getElementById('paintConstraint');
        paintForce.classList.remove('active-tool');
        paintConstraint.classList.remove('active-tool');
        if (type === 'force') {
            paintForce.classList.add('active-tool');
        } else if (type === 'constraint') {
            paintConstraint.classList.add('active-tool');
        }
    }

    _renderShapeList() {
        const container = document.getElementById('shapeSelectionList');
        if (!container) return;
        container.innerHTML = '';

        const applyAllBtn = document.getElementById('applyAllShapesToGroup');
        if (applyAllBtn) {
            applyAllBtn.style.display = this.viewer.selectionShapes.length > 0 ? '' : 'none';
        }

        const shapeIcons = { cube: '⬛', sphere: '⚽', cylinder: '🔵' };

        for (const shape of this.viewer.selectionShapes) {
            const item = document.createElement('div');
            item.className = 'shape-item';

            // Header row: icon + type label + apply btn + delete btn
            const header = document.createElement('div');
            header.style.cssText = 'display:flex; align-items:center; gap:0.35rem; margin-bottom:0.4rem;';

            const icon = document.createElement('span');
            icon.textContent = shapeIcons[shape.shapeType] || '◆';
            icon.style.fontSize = '0.8rem';

            const label = document.createElement('span');
            label.style.cssText = 'font-size:0.75rem; font-weight:600; flex:1; text-transform:capitalize;';
            label.textContent = shape.shapeType + ' #' + shape.id;

            const applyBtn = document.createElement('button');
            applyBtn.className = 'btn-secondary';
            applyBtn.style.cssText = 'font-size:0.65rem; padding:0.1rem 0.4rem;';
            applyBtn.textContent = 'Apply';
            applyBtn.title = 'Apply this shape\'s face selection to the active group';
            applyBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.viewer.applyShapeSelectionToGroup(undefined, shape.id);
                this._renderGroupsList();
            });

            const removeBtn = document.createElement('button');
            removeBtn.className = 'group-remove';
            removeBtn.textContent = '×';
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.viewer.removeSelectionShape(shape.id);
                this._renderShapeList();
            });

            header.appendChild(icon);
            header.appendChild(label);
            header.appendChild(applyBtn);
            header.appendChild(removeBtn);
            item.appendChild(header);

            // Controls grid: position, size, rotation
            const makeRow = (labelText, values, onChange) => {
                const row = document.createElement('div');
                row.style.cssText = 'display:grid; grid-template-columns:4rem 1fr 1fr 1fr; gap:0.2rem; align-items:center; margin-bottom:0.2rem;';
                const lbl = document.createElement('label');
                lbl.style.cssText = 'font-size:0.65rem; color:var(--text-secondary);';
                lbl.textContent = labelText;
                row.appendChild(lbl);
                ['X', 'Y', 'Z'].forEach((axis, i) => {
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.step = '0.5';
                    input.style.cssText = 'font-size:0.65rem; padding:0.1rem 0.2rem; border:1px solid var(--border-color); border-radius:3px; background:var(--bg-secondary); width:100%; min-width:0;';
                    input.value = Math.round(values[i] * 100) / 100;
                    input.title = axis;
                    input.addEventListener('input', (e) => {
                        const v = parseFloat(e.target.value);
                        if (!isNaN(v)) onChange(i, v);
                    });
                    row.appendChild(input);
                });
                return row;
            };

            item.appendChild(makeRow('Position', shape.position, (i, v) => {
                shape.position[i] = v;
                this.viewer.updateSelectionShape(shape.id, { position: [...shape.position] });
            }));

            item.appendChild(makeRow('Size', shape.size, (i, v) => {
                if (v >= 0.01) {
                    shape.size[i] = v;
                    this.viewer.updateSelectionShape(shape.id, { size: [...shape.size] });
                }
            }));

            item.appendChild(makeRow('Rotation°', shape.rotation, (i, v) => {
                shape.rotation[i] = v;
                this.viewer.updateSelectionShape(shape.id, { rotation: [...shape.rotation] });
            }));

            container.appendChild(item);
        }
    }

    async handleFileImport(file) {
        try {
            console.log('Importing file:', file.name);
            // Parse file (STL, STEP, DXF, or SVG), then voxelize with mm-based voxel size
            const model = await this.importer.importFile(file, null);

            // Hide template defaults checkbox for imported files
            const defaultsGroup = document.getElementById('templateDefaultsGroup');
            if (defaultsGroup) defaultsGroup.classList.add('hidden');
            const defaultsCheckbox = document.getElementById('useTemplateDefaults');
            if (defaultsCheckbox) {
                defaultsCheckbox.checked = false;
                this._applyTemplateDefaultsToggle(false);
            }

            // Check if this is a 2D file (DXF/SVG) - skip mesh method selection
            const is2D = /\.(dxf|svg)$/i.test(file.name);

            // Store parsed data for re-meshing after mesh method selection
            this._pendingImport = { file, model };

            // Show transform controls for imported models (3D only)
            if (is2D) {
                document.getElementById('transformControls').classList.add('hidden');
            } else {
                document.getElementById('transformControls').classList.remove('hidden');
            }

            // Show mesh method selector (3D files only)
            const selector = document.getElementById('meshMethodSelector');
            const dimSelector = document.getElementById('dimensionModeSelector');
            if (is2D) {
                selector.classList.add('hidden');
                if (dimSelector) dimSelector.classList.add('hidden');
            } else {
                selector.classList.remove('hidden');
                if (dimSelector) {
                    dimSelector.classList.remove('hidden');
                    // Reset to 3D mode
                    const radio3D = document.querySelector('input[name="dimensionMode"][value="3d"]');
                    if (radio3D) radio3D.checked = true;
                    const projSelector = document.getElementById('projectionSideSelector');
                    if (projSelector) projSelector.classList.add('hidden');
                }
            }

            // Enable/disable blended curvature option based on file type
            const isSTEP = /\.(stp|step)$/i.test(file.name);
            const blendedOption = document.getElementById('blendedCurvatureOption');
            if (isSTEP) {
                blendedOption.classList.remove('disabled');
                blendedOption.querySelector('input').disabled = false;
            } else {
                blendedOption.classList.add('disabled');
                blendedOption.querySelector('input').disabled = true;
                // Force box mesh for non-STEP files
                document.querySelector('input[name="meshMethod"][value="box"]').checked = true;
            }

            // For 2D files, the model is already voxelized - use directly
            let displayModel;
            if (is2D) {
                displayModel = model;
            } else {
                // Display preliminary model info for 3D files
                const voxelSizeMM = this.config.voxelSizeMM;
                displayModel = this.importer.voxelizeVertices(model.originalVertices, null, voxelSizeMM);
                displayModel.originalVertices = model.originalVertices;
            }

            const info = document.getElementById('modelInfo');
            info.classList.remove('hidden');
            const physX = displayModel.bounds ? (displayModel.bounds.maxX - displayModel.bounds.minX).toFixed(1) : '?';
            const physY = displayModel.bounds ? (displayModel.bounds.maxY - displayModel.bounds.minY).toFixed(1) : '?';
            const physZ = displayModel.bounds ? (displayModel.bounds.maxZ - displayModel.bounds.minZ).toFixed(1) : '?';
            const voxelSizeStr = displayModel.voxelSize ? displayModel.voxelSize.toFixed(2) : '?';
            let infoHTML = `
                <strong>Model loaded:</strong> ${file.name}<br>
                <strong>Size:</strong> ${physX} × ${physY} × ${physZ} mm<br>
                <strong>Voxel size:</strong> ${voxelSizeStr} mm<br>
                <strong>Elements:</strong> ${displayModel.nx * displayModel.ny * displayModel.nz}<br>
                <strong>Grid:</strong> ${displayModel.nx} × ${displayModel.ny} × ${displayModel.nz}
            `;
            if (!is2D) {
                infoHTML += `<br><em>Select meshing method above and click Continue.</em>`;
            } else {
                infoHTML += `<br><strong>Mode:</strong> 2D profile`;
            }
            info.innerHTML = infoHTML;

            // Preview the model
            this.currentModel = displayModel;
            this.viewer.setModel(displayModel);

            // Show the mobile viewer toggle button now that a model is loaded
            this._showMobileViewerToggle();

            // For 2D files, skip mesh method selection and go directly to preview
            if (is2D) {
                this._pendingImport = null;
                this.workflow.enableStep(3);
                this.workflow.goToStep(3);
            }

        } catch (error) {
            console.error('Import error:', error);
            alert('Failed to import file: ' + error.message);
        }
    }

    /**
     * Finalize import after user selects a mesh method and clicks Continue.
     */
    _confirmMeshMethod() {
        if (!this._pendingImport) return;

        const { file, model } = this._pendingImport;
        const meshMethod = document.querySelector('input[name="meshMethod"]:checked').value;
        const voxelSizeMM = this.config.voxelSizeMM;

        // Check dimension mode
        const dimModeRadio = document.querySelector('input[name="dimensionMode"]:checked');
        const dimMode = dimModeRadio ? dimModeRadio.value : '3d';
        const projSideBtn = document.querySelector('.projection-btn.selected');
        const projSide = projSideBtn ? projSideBtn.dataset.side : 'top';

        let finalModel;
        if (meshMethod === 'blended-curvature') {
            finalModel = this.importer.blendedCurvatureMesh(model.originalVertices, null, voxelSizeMM);
        } else {
            finalModel = this.importer.voxelizeVertices(model.originalVertices, null, voxelSizeMM);
        }
        finalModel.originalVertices = model.originalVertices;
        if (model.sourceFormat) finalModel.sourceFormat = model.sourceFormat;
        if (model.protocol) finalModel.protocol = model.protocol;

        // If 2D mode selected on a 3D model, project to single depth
        if (dimMode === '2d') {
            finalModel = this._projectTo2D(finalModel, projSide);
        }

        this.currentModel = finalModel;

        // Update info display
        const info = document.getElementById('modelInfo');
        info.classList.remove('hidden');
        const physX = finalModel.bounds ? (finalModel.bounds.maxX - finalModel.bounds.minX).toFixed(1) : '?';
        const physY = finalModel.bounds ? (finalModel.bounds.maxY - finalModel.bounds.minY).toFixed(1) : '?';
        const physZ = finalModel.bounds ? (finalModel.bounds.maxZ - finalModel.bounds.minZ).toFixed(1) : '?';
        const voxelSizeStr = finalModel.voxelSize ? finalModel.voxelSize.toFixed(2) : '?';
        const meshLabel = meshMethod === 'blended-curvature' ? 'Blended Curvature' : 'Box (Voxelized)';
        let modeLabel = dimMode === '2d' ? `2D (${projSide})` : '3D';
        info.innerHTML = `
            <strong>Model loaded:</strong> ${file.name}<br>
            <strong>Mesh type:</strong> ${meshLabel}<br>
            <strong>Mode:</strong> ${modeLabel}<br>
            <strong>Size:</strong> ${physX} × ${physY} × ${physZ} mm<br>
            <strong>Voxel size:</strong> ${voxelSizeStr} mm<br>
            <strong>Elements:</strong> ${finalModel.nx * finalModel.ny * finalModel.nz}<br>
            <strong>Grid:</strong> ${finalModel.nx} × ${finalModel.ny} × ${finalModel.nz}
        `;

        // Visualize
        this.viewer.setModel(finalModel);
        this._showMobileViewerToggle();

        // Hide mesh method selector and dimension mode selector
        document.getElementById('meshMethodSelector').classList.add('hidden');
        const dimSelector = document.getElementById('dimensionModeSelector');
        if (dimSelector) dimSelector.classList.add('hidden');

        // Clean up pending import
        this._pendingImport = null;

        // Enable and navigate to step 3 (preview)
        this.workflow.enableStep(3);
        this.workflow.goToStep(3);
    }

    /**
     * Project a 3D model to 2D (depth=1) by collapsing one axis based on the projection side.
     * The projection flattens the model along the selected viewing direction.
     */
    _projectTo2D(model, side) {
        const { nx, ny, nz, elements } = model;

        let outNx, outNy, outNz;
        let projectedElements;

        // Determine which axis to collapse based on the projection side
        switch (side) {
            case 'front':
            case 'back': {
                // XY plane: collapse Z axis → outNz=1
                outNx = nx; outNy = ny; outNz = 1;
                projectedElements = new Float32Array(outNx * outNy);
                for (let iy = 0; iy < ny; iy++) {
                    for (let ix = 0; ix < nx; ix++) {
                        let maxVal = 0;
                        for (let iz = 0; iz < nz; iz++) {
                            const idx = iz * ny * nx + iy * nx + ix;
                            if (elements[idx] > maxVal) maxVal = elements[idx];
                        }
                        projectedElements[iy * outNx + ix] = maxVal;
                    }
                }
                break;
            }
            case 'top':
            case 'bottom': {
                // XZ plane: collapse Y axis → outNy=1
                outNx = nx; outNy = 1; outNz = nz;
                projectedElements = new Float32Array(outNx * outNz);
                for (let iz = 0; iz < nz; iz++) {
                    for (let ix = 0; ix < nx; ix++) {
                        let maxVal = 0;
                        for (let iy = 0; iy < ny; iy++) {
                            const idx = iz * ny * nx + iy * nx + ix;
                            if (elements[idx] > maxVal) maxVal = elements[idx];
                        }
                        projectedElements[iz * outNx + ix] = maxVal;
                    }
                }
                break;
            }
            case 'right':
            case 'left': {
                // YZ plane: collapse X axis → outNx=1
                outNx = 1; outNy = ny; outNz = nz;
                projectedElements = new Float32Array(outNy * outNz);
                for (let iz = 0; iz < nz; iz++) {
                    for (let iy = 0; iy < ny; iy++) {
                        let maxVal = 0;
                        for (let ix = 0; ix < nx; ix++) {
                            const idx = iz * ny * nx + iy * nx + ix;
                            if (elements[idx] > maxVal) maxVal = elements[idx];
                        }
                        projectedElements[iz * outNy + iy] = maxVal;
                    }
                }
                break;
            }
            default: {
                // Default: collapse Z
                outNx = nx; outNy = ny; outNz = 1;
                projectedElements = new Float32Array(outNx * outNy);
                for (let iy = 0; iy < ny; iy++) {
                    for (let ix = 0; ix < nx; ix++) {
                        let maxVal = 0;
                        for (let iz = 0; iz < nz; iz++) {
                            const idx = iz * ny * nx + iy * nx + ix;
                            if (elements[idx] > maxVal) maxVal = elements[idx];
                        }
                        projectedElements[iy * outNx + ix] = maxVal;
                    }
                }
                break;
            }
        }

        return {
            ...model,
            nx: outNx,
            ny: outNy,
            nz: outNz,
            elements: projectedElements
        };
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
        
        // Apply template default boundary conditions
        if (model.forceDirection) {
            this.config.forceDirection = model.forceDirection;
            this.viewer.forceDirection = model.forceDirection;
            document.getElementById('forceDirection').value = model.forceDirection;
            const v = FORCE_DIRECTION_VECTORS[model.forceDirection] || [0, -1, 0];
            document.getElementById('forceVectorX').value = v[0];
            document.getElementById('forceVectorY').value = v[1];
            document.getElementById('forceVectorZ').value = v[2];
            this.config.forceVector = null;
            this.viewer.forceVector = null;
        }
        if (model.constraintPositions) {
            this.config.constraintPosition = model.constraintPositions;
            // Update dropdown if the value exists in the options
            const constraintSelect = document.getElementById('constraintPosition');
            const optionExists = Array.from(constraintSelect.options).some(o => o.value === model.constraintPositions);
            if (optionExists) {
                constraintSelect.value = model.constraintPositions;
            }
        }

        // Apply painted force/constraint faces from templates
        if (model.paintedForces && model.paintedForces.length > 0) {
            this.viewer.paintedForceFaces = new Set(model.paintedForces);
        }
        if (model.paintedConstraints && model.paintedConstraints.length > 0) {
            this.viewer.paintedConstraintFaces = new Set(model.paintedConstraints);
        }

        // Pre-select recommended simulation parameters (user can still change them)
        if (model.recommendedParams) {
            const params = model.recommendedParams;
            if (params.solutionType) {
                this._applySolutionType(params.solutionType);
                document.querySelectorAll('.solution-type-card').forEach(c => {
                    c.classList.toggle('selected', c.dataset.solution === params.solutionType);
                });
            }
            if (params.nonlinearAnalysisType) {
                this.config.nonlinearAnalysisType = params.nonlinearAnalysisType;
                const el = document.getElementById('nonlinearAnalysisType');
                if (el) el.value = params.nonlinearAnalysisType;
                this._updateNonlinearSubTypeUI(params.nonlinearAnalysisType);
            }
            if (params.nonlinearLoadSteps) {
                this.config.nonlinearLoadSteps = params.nonlinearLoadSteps;
                const el = document.getElementById('nonlinearLoadSteps');
                if (el) el.value = params.nonlinearLoadSteps;
            }
            if (params.nonlinearMaxNewtonIter) {
                this.config.nonlinearMaxNewtonIter = params.nonlinearMaxNewtonIter;
                const el = document.getElementById('nonlinearMaxNewtonIter');
                if (el) el.value = params.nonlinearMaxNewtonIter;
            }
            if (params.nonlinearTolerance) {
                this.config.nonlinearTolerance = params.nonlinearTolerance;
                const el = document.getElementById('nonlinearTolerance');
                if (el) el.value = params.nonlinearTolerance;
            }
            if (params.forceMagnitude) {
                this.config.forceMagnitude = params.forceMagnitude;
                const el = document.getElementById('forceMagnitude');
                if (el) el.value = params.forceMagnitude;
            }
            if (params.material && MATERIAL_PRESETS[params.material]) {
                const preset = MATERIAL_PRESETS[params.material];
                this.config.material = params.material;
                this.config.youngsModulus = preset.youngsModulus;
                this.config.poissonsRatio = preset.poissonsRatio;
                this.config.yieldStrength = preset.yieldStrength;
                if (preset.nonlinearType) {
                    this.config.nonlinearMaterialModel = preset.nonlinearType;
                }
                const matEl = document.getElementById('materialSelect');
                if (matEl) matEl.value = params.material;
                const eEl = document.getElementById('youngsModulus');
                if (eEl) eEl.value = preset.youngsModulus;
                const nuEl = document.getElementById('poissonsRatio');
                if (nuEl) nuEl.value = preset.poissonsRatio;
                const yEl = document.getElementById('yieldStrength');
                if (yEl) yEl.value = preset.yieldStrength;
            }
        }

        // Enable template defaults checkbox and check it
        const defaultsCheckbox = document.getElementById('useTemplateDefaults');
        if (defaultsCheckbox) {
            defaultsCheckbox.checked = true;
            const defaultsGroup = document.getElementById('templateDefaultsGroup');
            if (defaultsGroup) defaultsGroup.classList.remove('hidden');
            this._applyTemplateDefaultsToggle(true);
        }
        
        // Display model info including default constraints/forces
        const info = document.getElementById('modelInfo');
        info.classList.remove('hidden');
        let infoHTML = `
            <strong>Template loaded:</strong> ${type}<br>
            <strong>Elements:</strong> ${model.nx * model.ny * model.nz}<br>
            <strong>Dimensions:</strong> ${model.nx} x ${model.ny} x ${model.nz}
        `;
        if (model.forcePosition || model.constraintPositions) {
            infoHTML += `<br><strong>Defaults:</strong>`;
            if (model.forcePosition) infoHTML += ` Force: ${model.forcePosition} (${model.forceDirection || 'down'})`;
            if (model.constraintPositions) infoHTML += ` | Constraint: ${model.constraintPositions}`;
        }
        if (model.recommendedParams && model.recommendedParams.solutionType) {
            infoHTML += `<br><strong>Recommended:</strong> ${model.recommendedParams.solutionType} analysis`;
        }
        info.innerHTML = infoHTML;
        
        // Visualize and reset camera for fresh template view
        this.viewer.setModel(model);
        this._showMobileViewerToggle();
        this.viewer.resetCamera();
        document.getElementById('toggleSection').classList.remove('active-tool');
        this._updateSectionSliders();
        
        // Enable and navigate to step 3 (preview)
        this.workflow.enableStep(3);
        this.workflow.goToStep(3);
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
        this.viewer.setVolumetricStressData(null);
        this.viewer.setDisplacementData(null);
        this.viewer.setStressBarLabel('Stress (N/mm² = MPa)', 'MPa');
        const dispContainerReset = document.getElementById('displacementContainer');
        if (dispContainerReset) dispContainerReset.classList.add('hidden');
        const loadStepContainerReset = document.getElementById('loadStepContainer');
        if (loadStepContainerReset) loadStepContainerReset.classList.add('hidden');
        const resultDisplayReset = document.getElementById('resultDisplayContainer');
        if (resultDisplayReset) resultDisplayReset.classList.add('hidden');
        const densityContainerReset = document.getElementById('densityThresholdContainer');
        if (densityContainerReset) densityContainerReset.classList.add('hidden');
        this._nonlinearSnapshots = null;
        this.viewer.draw();
        
        // Show progress
        const progressContainer = document.getElementById('progressContainer');
        progressContainer.classList.remove('hidden');
        
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const complianceText = document.getElementById('complianceText');
        
        // Toggle buttons: hide Run, show Cancel + Pause
        const runButton = document.getElementById('runOptimization');
        const cancelButton = document.getElementById('cancelOptimization');
        const pauseButton = document.getElementById('pauseOptimization');
        runButton.classList.add('hidden');
        cancelButton.classList.remove('hidden');
        if (pauseButton) { pauseButton.classList.remove('hidden'); pauseButton.textContent = 'Pause'; }
        this._optimizationPaused = false;
        this.pausedVolumetricData = null;
        this.finalVolumetricData = null;
        this.lastVolumetricData = null;
        
        try {
            // Include painted constraint/force/keep data and selection groups in config
            const optimConfig = {
                ...this.config,
                paintedConstraints: Array.from(this.viewer.paintedConstraintFaces),
                paintedForces: Array.from(this.viewer.paintedForceFaces),
                paintedKeep: Array.from(this.viewer.paintedKeepFaces),
                selectionGroups: this.viewer.selectionGroups.map(g => ({
                    ...g,
                    faces: Array.from(g.faces)
                }))
            };

            // Run optimization in worker
            const result = await this.optimizer.optimize(
                this.currentModel,
                optimConfig,
                (iteration, compliance, meshData, timing, maxStress, volumetricData) => {
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
                        this.viewer.updateMesh(meshData, maxStress);
                    }
                    if (volumetricData) {
                        this._applyVolumetricSnapshot(volumetricData, 'iteration');
                    }
                }
            );
            
            this.optimizedModel = result;
            this.finalVolumetricData = result.volumetricData || null;
            if (!this.finalVolumetricData && result.elementStress && result.nx && result.ny && result.nz) {
                this.finalVolumetricData = {
                    nx: result.nx,
                    ny: result.ny,
                    nz: result.nz,
                    iteration: result.iterations || 0,
                    maxStress: result.maxStress || 0,
                    stress: result.elementStress
                };
            }
            this._applyVolumetricSnapshot(this.finalVolumetricData, 'complete');

            // Apply fatigue mode label to stress bar if applicable
            if (result.fatigueMode) {
                this.viewer.setStressBarLabel('Fatigue Risk (0=safe, 1=high)', '');
            } else {
                this.viewer.setStressBarLabel('Stress (N/mm² = MPa)', 'MPa');
            }

            // Set yield strength for elastic/plastic deformation visualization
            this.viewer.setYieldStrength(this.config.yieldStrength || 0);

            // Store displacement data from FEA/fatigue solve and show controls
            if (result.displacementU && result.nx && result.ny && result.nz) {
                const U = result.displacementU;
                // Normalize U so that max|U| = 1 (display as deformation mode shape).
                // The FEA solver uses mixed units (GPa stiffness × voxel-space geometry × N force),
                // so the raw displacement magnitudes are not physically meaningful for visualization.
                // Normalizing produces a unit mode shape; the scale slider then controls how many
                // voxels the maximum nodal displacement should appear as on screen.
                let maxU = 0;
                for (let i = 0; i < U.length; i++) {
                    const v = Math.abs(U[i]);
                    if (v > maxU) maxU = v;
                }
                const normalizedU = new Float32Array(U.length);
                if (maxU > 0) {
                    for (let i = 0; i < U.length; i++) normalizedU[i] = U[i] / maxU;
                }

                // Default scale: 10% of the longest model dimension.
                // This makes the peak displacement clearly visible without distorting the shape.
                const DISP_SCALE_RATIO = 0.1; // fraction of longest model dimension for default scale
                const maxDim = Math.max(result.nx, result.ny, result.nz);
                const defaultScale = Math.max(1, Math.round(DISP_SCALE_RATIO * maxDim));

                this.viewer.setDisplacementData({
                    U: normalizedU,
                    nx: result.nx,
                    ny: result.ny,
                    nz: result.nz
                });
                this.viewer.setDisplacementScale(defaultScale);

                const dispContainer = document.getElementById('displacementContainer');
                if (dispContainer) dispContainer.classList.remove('hidden');
                // Sync UI controls with computed defaults
                const dispToggle = document.getElementById('toggleDisplacement');
                if (dispToggle) dispToggle.checked = this.viewer.showDisplacement;
                const dispScaleSlider = document.getElementById('displacementScaleSlider');
                const dispScaleValue = document.getElementById('displacementScaleValue');
                if (dispScaleSlider) dispScaleSlider.value = defaultScale;
                if (dispScaleValue) dispScaleValue.textContent = `${defaultScale}×`;
            } else {
                this.viewer.setDisplacementData(null);
                const dispContainer = document.getElementById('displacementContainer');
                if (dispContainer) dispContainer.classList.add('hidden');
            }

            // Show load step slider for nonlinear FEA results with time stepping
            const loadStepContainer = document.getElementById('loadStepContainer');
            if (result.nonlinearMode && result.stepSnapshots && result.stepSnapshots.length > 0) {
                this._nonlinearSnapshots = result.stepSnapshots;
                this._nonlinearNx = result.nx;
                this._nonlinearNy = result.ny;
                this._nonlinearNz = result.nz;

                const totalSteps = result.stepSnapshots.length;
                const loadStepSlider = document.getElementById('loadStepSlider');
                if (loadStepSlider) {
                    loadStepSlider.max = totalSteps;
                    loadStepSlider.value = totalSteps;
                }
                if (loadStepContainer) loadStepContainer.classList.remove('hidden');

                // Show final step status
                this._applyLoadStep(totalSteps);
            } else {
                this._nonlinearSnapshots = null;
                if (loadStepContainer) loadStepContainer.classList.add('hidden');
            }

            // Set triaxiality scalar field for nonlinear results
            if (result.nonlinearMode && result.triaxiality) {
                let minTri = Infinity, maxTri = -Infinity;
                for (let i = 0; i < result.triaxiality.length; i++) {
                    if (result.triaxiality[i] < minTri) minTri = result.triaxiality[i];
                    if (result.triaxiality[i] > maxTri) maxTri = result.triaxiality[i];
                }
                this.viewer.setScalarField('triaxiality', result.triaxiality, minTri, maxTri);
            }

            // Set damage / phase-field data for nonlinear fracture results
            if (result.nonlinearMode && result.damageField) {
                this.viewer.setDamageField(result.damageField);
                this.viewer.setScalarField('damage', result.damageField, 0, 1);
            }

            // Set eroded elements for crack visualization
            if (result.nonlinearMode && result.erodedElements) {
                this.viewer.setErodedElements(new Set(result.erodedElements));
            }
            
            // Update viewer with final mesh
            if (result.meshData) {
                this.viewer.updateMesh(result.meshData, result.maxStress);
            }
            
            // Show results
            let resultsTitle;
            if (result.feaOnly) {
                if (result.fatigueMode) resultsTitle = 'Fatigue Analysis Complete!';
                else if (result.nonlinearMode) {
                    if (!result.converged) {
                        resultsTitle = `⚠ Nonlinear FEA — Catastrophic Failure Detected`;
                    } else {
                        resultsTitle = 'Nonlinear FEA Complete!';
                    }
                }
                else resultsTitle = 'FEA Analysis Complete!';
            } else {
                resultsTitle = 'Optimization Complete!';
            }
            let resultsHTML = `
                <strong>${resultsTitle}</strong><br>
                Final Compliance: ${result.finalCompliance.toFixed(2)}<br>
                Iterations: ${result.iterations}<br>
                Volume Fraction: ${(result.volumeFraction * 100).toFixed(1)}%
            `;

            // Add nonlinear failure details
            if (result.nonlinearMode && !result.converged) {
                const failStep = result.failedAtStep || '?';
                const totalSteps = this.config.nonlinearLoadSteps || 10;
                const failLoad = result.failedAtStep > 0 ? ((result.failedAtStep / totalSteps) * 100).toFixed(0) : '?';
                resultsHTML += `<br><br><span style="color: #dc2626;"><strong>⚠ Failure at load step ${failStep}/${totalSteps} (${failLoad}% load)</strong></span>`;
                resultsHTML += `<br><small>The solver did not converge, indicating structural failure or crack propagation. Use the Load Step slider to inspect the progression.</small>`;
            }

            // Count elements exceeding yield (cracks indicator)
            if (result.nonlinearMode && result.elementStress && this.config.yieldStrength > 0) {
                const yieldStress = this.config.yieldStrength;
                let yieldedCount = 0;
                for (let i = 0; i < result.elementStress.length; i++) {
                    if (result.elementStress[i] > yieldStress) yieldedCount++;
                }
                if (yieldedCount > 0) {
                    const pct = ((yieldedCount / result.elementStress.length) * 100).toFixed(1);
                    resultsHTML += `<br><span style="color: #f59e0b;"><strong>⚡ ${yieldedCount} elements (${pct}%) exceeded yield strength (${yieldStress} MPa)</strong></span>`;
                }
            }
            
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
            
            // Show export section within step 6
            document.getElementById('exportSection').classList.remove('hidden');

            // Show result display controls for toggling color mode
            const resultDisplayContainer = document.getElementById('resultDisplayContainer');
            if (resultDisplayContainer) resultDisplayContainer.classList.remove('hidden');

            // Show density threshold only for topology optimization results
            const densityContainer = document.getElementById('densityThresholdContainer');
            if (densityContainer) {
                if (!result.feaOnly) {
                    densityContainer.classList.remove('hidden');
                } else {
                    densityContainer.classList.add('hidden');
                }
            }
            
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
            // Reset pause state
            this._optimizationPaused = false;
            const pauseButton = document.getElementById('pauseOptimization');
            if (pauseButton) { pauseButton.classList.add('hidden'); pauseButton.textContent = 'Pause'; }
        }
    }

    cancelOptimization() {
        console.log('Cancelling optimization...');
        this._optimizationPaused = false;
        this.optimizer.cancel();
    }

    togglePauseOptimization() {
        const pauseButton = document.getElementById('pauseOptimization');
        if (this._optimizationPaused) {
            this._optimizationPaused = false;
            this.optimizer.resume();
            if (pauseButton) pauseButton.textContent = 'Pause';
            console.log('Optimization resumed');
        } else {
            this._optimizationPaused = true;
            this.optimizer.pause();
            if (pauseButton) pauseButton.textContent = 'Resume';
            console.log('Optimization paused');
        }
    }

    exportSTL() {
        if (!this.optimizedModel) {
            alert('Please run optimization first');
            return;
        }
        
        console.log('Exporting STL...');
        this.exporter.exportSTL(this.optimizedModel, 'optimized_model.stl', {
            threshold: this.viewer.densityThreshold
        });
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
        this._optimizationPaused = false;
        
        this.currentModel = null;
        this.optimizedModel = null;
        this.pausedVolumetricData = null;
        this.finalVolumetricData = null;
        this.lastVolumetricData = null;
        
        // Clear UI
        document.getElementById('modelInfo').classList.add('hidden');
        document.getElementById('progressContainer').classList.add('hidden');
        document.getElementById('fileInput').value = '';
        document.getElementById('runOptimization').classList.remove('hidden');
        document.getElementById('cancelOptimization').classList.add('hidden');
        const pauseBtn = document.getElementById('pauseOptimization');
        if (pauseBtn) { pauseBtn.classList.add('hidden'); pauseBtn.textContent = 'Pause'; }
        document.getElementById('transformControls').classList.add('hidden');
        document.getElementById('meshMethodSelector').classList.add('hidden');
        const dimSelector = document.getElementById('dimensionModeSelector');
        if (dimSelector) dimSelector.classList.add('hidden');
        document.getElementById('exportSection').classList.add('hidden');
        // Clear manufacturing card selection
        document.querySelectorAll('.manufacturing-card').forEach(c => c.classList.remove('selected'));
        this._pendingImport = null;
        
        // Reset transform inputs
        document.getElementById('modelScale').value = 1;
        document.getElementById('rotateX').value = 0;
        document.getElementById('rotateY').value = 0;
        document.getElementById('rotateZ').value = 0;
        
        // Reset viewer control button states
        document.getElementById('toggleViewMode').classList.remove('active-tool');
        document.getElementById('toggleMeshVisibility').classList.remove('active-tool');
        document.getElementById('toggleSection').classList.remove('active-tool');
        
        // Reset viewer
        this.viewer.clear();
        this._updateSectionSliders();
        
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
        const showBtn = document.getElementById('showBenchmarkHistory');
        const benchmarkResults = document.getElementById('benchmarkResults');
        
        if (this.benchmarkHistory.length === 0) {
            if (showBtn) showBtn.classList.add('hidden');
            return;
        }
        
        if (showBtn) showBtn.classList.remove('hidden');
        
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
