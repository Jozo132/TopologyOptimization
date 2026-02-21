// Advanced solver tests for material-models, nonlinear-solver, fracture-solver, dynamics-solver
// Run with: node tests/test-advanced-solvers.js

import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toURL = (p) => pathToFileURL(p).href;

const materialModels = await import(toURL(join(__dirname, '..', 'js', 'material-models.js')));
const nonlinearSolver = await import(toURL(join(__dirname, '..', 'js', 'nonlinear-solver.js')));
const fractureSolver = await import(toURL(join(__dirname, '..', 'js', 'fracture-solver.js')));
const dynamicsSolver = await import(toURL(join(__dirname, '..', 'js', 'dynamics-solver.js')));

let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        passed++;
        console.log(`  ✓ ${message}`);
    } else {
        failed++;
        console.error(`  ✗ ${message}`);
    }
}

const TOLERANCE = 1e-6;
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// ──────────────────────────────────────────────────
// Material Models Tests
// ──────────────────────────────────────────────────

// ──────────────────────────────────────────────────
// Test 1: LinearElastic creation
// ──────────────────────────────────────────────────
console.log('Test 1: LinearElastic creation');
{
    const mat = materialModels.createMaterial('linear-elastic', { E: 200e9, nu: 0.3 });
    assert(mat !== null && mat !== undefined, 'Should return a valid object');
    assert(typeof mat.computeStress === 'function', 'Should have computeStress method');
    assert(typeof mat.getConstitutiveMatrix === 'function', 'Should have getConstitutiveMatrix method');
}

// ──────────────────────────────────────────────────
// Test 2: LinearElastic constitutive matrix
// ──────────────────────────────────────────────────
console.log('Test 2: LinearElastic constitutive matrix');
{
    const mat = materialModels.createMaterial('linear-elastic', { E: 200e9, nu: 0.3 });
    const C = mat.getConstitutiveMatrix();
    assert(C.length === 36, `C matrix should have 36 elements, got ${C.length}`);
    assert(C[0] > 0, `C[0] should be positive, got ${C[0]}`);
    // Check symmetry: C[i*6+j] === C[j*6+i]
    let symmetric = true;
    for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
            if (Math.abs(C[i * 6 + j] - C[j * 6 + i]) > TOLERANCE) symmetric = false;
        }
    }
    assert(symmetric, 'C matrix should be symmetric');
}

// ──────────────────────────────────────────────────
// Test 3: LinearElastic stress computation with identity F
// ──────────────────────────────────────────────────
console.log('Test 3: LinearElastic stress computation - identity F');
{
    const mat = materialModels.createMaterial('linear-elastic', { E: 200e9, nu: 0.3 });
    const result = mat.computeStress([...I3], null);
    const allZero = result.stress.every(s => Math.abs(s) < TOLERANCE);
    assert(allZero, `Identity F should produce zero stress, got [${result.stress.map(s => s.toExponential(2))}]`);
}

// ──────────────────────────────────────────────────
// Test 4: NeoHookean creation
// ──────────────────────────────────────────────────
console.log('Test 4: NeoHookean creation');
{
    const mat = materialModels.createMaterial('neo-hookean', { E: 1e6, nu: 0.45 });
    assert(mat !== null && mat !== undefined, 'Should return a valid object');
    assert(typeof mat.computeStress === 'function', 'Should have computeStress method');
}

// ──────────────────────────────────────────────────
// Test 5: NeoHookean zero deformation stress
// ──────────────────────────────────────────────────
console.log('Test 5: NeoHookean zero deformation - identity F');
{
    const mat = materialModels.createMaterial('neo-hookean', { E: 1e6, nu: 0.45 });
    const result = mat.computeStress([...I3], null);
    const allZero = result.stress.every(s => Math.abs(s) < TOLERANCE);
    assert(allZero, `Identity F should produce zero Cauchy stress, got [${result.stress.map(s => s.toExponential(2))}]`);
}

// ──────────────────────────────────────────────────
// Test 6: J2Plasticity creation
// ──────────────────────────────────────────────────
console.log('Test 6: J2Plasticity creation');
{
    const mat = materialModels.createMaterial('j2-plasticity', { E: 200e9, nu: 0.3, sigY: 250e6 });
    assert(mat !== null && mat !== undefined, 'Should return a valid object');
    assert(typeof mat.computeStress === 'function', 'Should have computeStress method');
}

// ──────────────────────────────────────────────────
// Test 7: J2Plasticity elastic response
// ──────────────────────────────────────────────────
console.log('Test 7: J2Plasticity elastic response - small strain');
{
    const mat = materialModels.createMaterial('j2-plasticity', { E: 200e9, nu: 0.3, sigY: 250e6 });
    // Small uniaxial strain: F11 = 1 + small_eps
    const smallEps = 1e-6;
    const F = [1 + smallEps, 0, 0, 0, 1, 0, 0, 0, 1];
    const state = new materialModels.MaterialState();
    const result = mat.computeStress(F, state);
    const vm = nonlinearSolver.vonMises(result.stress);
    assert(vm < 250e6, `Von Mises stress (${vm.toExponential(2)}) should be below yield (250e6)`);
}

// ──────────────────────────────────────────────────
// Test 8: DruckerPrager creation
// ──────────────────────────────────────────────────
console.log('Test 8: DruckerPrager creation');
{
    const mat = materialModels.createMaterial('drucker-prager', { E: 30000, nu: 0.2, c: 100, phi: Math.PI / 6 });
    assert(mat !== null && mat !== undefined, 'Should create with cohesion and friction angle');
    assert(typeof mat.computeStress === 'function', 'Should have computeStress method');
}

// ──────────────────────────────────────────────────
// Test 9: MooneyRivlin creation
// ──────────────────────────────────────────────────
console.log('Test 9: MooneyRivlin creation');
{
    const mat = materialModels.createMaterial('mooney-rivlin', { C10: 0.5e6, C01: 0.1e6 });
    assert(mat !== null && mat !== undefined, 'Should return a valid object');
    assert(typeof mat.computeStress === 'function', 'Should have computeStress method');
}

// ──────────────────────────────────────────────────
// Test 10: Ogden creation
// ──────────────────────────────────────────────────
console.log('Test 10: Ogden creation');
{
    const mat = materialModels.createMaterial('ogden', { mu: 1e6, alpha: 2 });
    assert(mat !== null && mat !== undefined, 'Should return a valid object');
    assert(typeof mat.computeStress === 'function', 'Should have computeStress method');
}

// ──────────────────────────────────────────────────
// Test 11: MaterialState
// ──────────────────────────────────────────────────
console.log('Test 11: MaterialState default values');
{
    const st = new materialModels.MaterialState();
    assert(st.epsPl === 0, 'Plastic strain should be zero');
    assert(st.damage === 0, 'Damage should be zero');
}

// ──────────────────────────────────────────────────
// Test 12: MaterialRegistry
// ──────────────────────────────────────────────────
console.log('Test 12: MaterialRegistry contains all types');
{
    const reg = materialModels.MaterialRegistry;
    const expectedTypes = ['linear-elastic', 'neo-hookean', 'j2-plasticity', 'drucker-prager', 'mooney-rivlin', 'ogden'];
    const allPresent = expectedTypes.every(t => t in reg);
    assert(allPresent, `Registry should contain all 6 types, has: [${Object.keys(reg)}]`);
    assert(Object.keys(reg).length === 6, `Registry should have exactly 6 entries, got ${Object.keys(reg).length}`);
}

// ──────────────────────────────────────────────────
// Test 13: createMaterial invalid type
// ──────────────────────────────────────────────────
console.log('Test 13: createMaterial invalid type throws');
{
    let threw = false;
    try {
        materialModels.createMaterial('unknown-material', {});
    } catch (e) {
        threw = true;
    }
    assert(threw, 'Should throw for unknown material type');
}

// ──────────────────────────────────────────────────
// Nonlinear Solver Tests
// ──────────────────────────────────────────────────

// ──────────────────────────────────────────────────
// Test 14: NonlinearSolver instantiation
// ──────────────────────────────────────────────────
console.log('Test 14: NonlinearSolver instantiation');
{
    const solver = new nonlinearSolver.NonlinearSolver({});
    assert(solver !== null && solver !== undefined, 'Should create a valid object');
    assert(solver.maxNewtonIter === 20, `Default maxNewtonIter should be 20, got ${solver.maxNewtonIter}`);
}

// ──────────────────────────────────────────────────
// Test 15: Deformation gradient at zero displacement → identity
// ──────────────────────────────────────────────────
console.log('Test 15: Deformation gradient identity at zero displacement');
{
    // Unit cube element nodes (8-node hex)
    const nodeCoords = new Float64Array([
        0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
        0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1
    ]);
    const u_elem = new Float64Array(24); // zero displacement
    const result = nonlinearSolver.computeDeformationGradient(nodeCoords, u_elem, 0, 0, 0);
    let isIdentity = true;
    for (let i = 0; i < 9; i++) {
        if (Math.abs(result.F[i] - I3[i]) > TOLERANCE) isIdentity = false;
    }
    assert(isIdentity, `F should be identity at zero displacement, got [${result.F.map(v => v.toFixed(4))}]`);
}

// ──────────────────────────────────────────────────
// Test 16: Green-Lagrange strain at identity
// ──────────────────────────────────────────────────
console.log('Test 16: Green-Lagrange strain at identity F');
{
    const E = nonlinearSolver.greenLagrangeStrain([...I3]);
    const allZero = E.every(e => Math.abs(e) < TOLERANCE);
    assert(allZero, `E should be zero for F=I, got [${E.map(v => v.toExponential(2))}]`);
}

// ──────────────────────────────────────────────────
// Test 17: Von Mises stress computation
// ──────────────────────────────────────────────────
console.log('Test 17: Von Mises stress - uniaxial case');
{
    // Uniaxial stress: σ = [100, 0, 0, 0, 0, 0]
    const sigma = [100, 0, 0, 0, 0, 0];
    const vm = nonlinearSolver.vonMises(sigma);
    assert(Math.abs(vm - 100) < TOLERANCE, `Von Mises of uniaxial [100,0,0,0,0,0] should be 100, got ${vm}`);
}

// ──────────────────────────────────────────────────
// Test 18: Principal stresses
// ──────────────────────────────────────────────────
console.log('Test 18: Principal stresses for diagonal stress');
{
    // Diagonal stress state: σ = [300, 200, 100, 0, 0, 0]
    const sigma = [300, 200, 100, 0, 0, 0];
    const p = nonlinearSolver.principalStresses(sigma);
    assert(Math.abs(p[0] - 300) < TOLERANCE, `σ1 should be 300, got ${p[0]}`);
    assert(Math.abs(p[1] - 200) < TOLERANCE, `σ2 should be 200, got ${p[1]}`);
    assert(Math.abs(p[2] - 100) < TOLERANCE, `σ3 should be 100, got ${p[2]}`);
}

// ──────────────────────────────────────────────────
// Test 19: Stress triaxiality
// ──────────────────────────────────────────────────
console.log('Test 19: Stress triaxiality for uniaxial tension');
{
    // Uniaxial tension: σ = [σ, 0, 0, 0, 0, 0] → η = (σ/3)/σ = 1/3
    const sigma = [100, 0, 0, 0, 0, 0];
    const eta = nonlinearSolver.stressTriaxiality(sigma);
    assert(Math.abs(eta - 1.0 / 3.0) < TOLERANCE, `Triaxiality should be 1/3, got ${eta}`);
}

// ──────────────────────────────────────────────────
// Fracture Solver Tests
// ──────────────────────────────────────────────────

// ──────────────────────────────────────────────────
// Test 20: PhaseFieldFracture creation
// ──────────────────────────────────────────────────
console.log('Test 20: PhaseFieldFracture creation');
{
    const pf = new fractureSolver.PhaseFieldFracture({ Gc: 2700, lengthScale: 0.01 });
    assert(pf !== null && pf !== undefined, 'Should create a valid object');
    assert(pf.Gc === 2700, `Gc should be 2700, got ${pf.Gc}`);
    assert(pf.lengthScale === 0.01, `lengthScale should be 0.01, got ${pf.lengthScale}`);
}

// ──────────────────────────────────────────────────
// Test 21: PhaseFieldFracture field initialization
// ──────────────────────────────────────────────────
console.log('Test 21: PhaseFieldFracture field initialization');
{
    const pf = new fractureSolver.PhaseFieldFracture({ Gc: 2700, lengthScale: 0.01 });
    const field = pf.initializeField(100);
    assert(field.d.length === 100, `Damage field should have 100 elements, got ${field.d.length}`);
    const allZero = Array.from(field.d).every(v => v === 0);
    assert(allZero, 'Initial damage should be all zeros');
}

// ──────────────────────────────────────────────────
// Test 22: CohesiveZoneModel creation
// ──────────────────────────────────────────────────
console.log('Test 22: CohesiveZoneModel creation');
{
    const czm = new fractureSolver.CohesiveZoneModel({ sigmaMax: 500, deltaC: 0.005, GIc: 1.25 });
    assert(czm !== null && czm !== undefined, 'Should create a valid object');
    assert(czm.sigmaMax === 500, `sigmaMax should be 500, got ${czm.sigmaMax}`);
    assert(czm.deltaC === 0.005, `deltaC should be 0.005, got ${czm.deltaC}`);
}

// ──────────────────────────────────────────────────
// Test 23: CohesiveZoneModel traction at zero opening
// ──────────────────────────────────────────────────
console.log('Test 23: CohesiveZoneModel traction at zero opening');
{
    const czm = new fractureSolver.CohesiveZoneModel({ sigmaMax: 500, deltaC: 0.005, GIc: 1.25 });
    const result = czm.computeTraction(0, 0, null);
    assert(Math.abs(result.traction_n) < TOLERANCE, `Normal traction should be zero, got ${result.traction_n}`);
    assert(Math.abs(result.traction_t) < TOLERANCE, `Tangential traction should be zero, got ${result.traction_t}`);
}

// ──────────────────────────────────────────────────
// Test 24: ElementErosion
// ──────────────────────────────────────────────────
console.log('Test 24: ElementErosion identifies elements above threshold');
{
    const erosion = new fractureSolver.ElementErosion({ threshold: 0.95 });
    const damage = new Float64Array([0.1, 0.5, 0.96, 0.3, 0.99]);
    const result = erosion.checkAndErode(damage);
    assert(result.newlyEroded.includes(2), 'Element 2 (damage=0.96) should be eroded');
    assert(result.newlyEroded.includes(4), 'Element 4 (damage=0.99) should be eroded');
    assert(!result.newlyEroded.includes(0), 'Element 0 (damage=0.1) should not be eroded');
}

// ──────────────────────────────────────────────────
// Test 25: JohnsonCookDamage failure strain
// ──────────────────────────────────────────────────
console.log('Test 25: JohnsonCookDamage failure strain > 0');
{
    const jc = new fractureSolver.JohnsonCookDamage({ D1: 0.05, D2: 3.44, D3: -2.12, D4: 0.002, D5: 0.61 });
    const epsF = jc.computeFailureStrain(1.0 / 3.0);
    assert(epsF > 0, `Failure strain should be > 0, got ${epsF}`);
}

// ──────────────────────────────────────────────────
// Test 26: GursonModel yield with f=0 reduces to von Mises
// ──────────────────────────────────────────────────
console.log('Test 26: GursonModel yield function with f=0');
{
    const gurson = new fractureSolver.GursonModel({ q1: 1.5, q2: 1.0, sigmaY: 250 });
    // With f=0: Φ = (σ_eq/σ_y)² + 0 - 1 - 0 = (σ_eq/σ_y)² - 1
    // At σ_eq = σ_y, Φ should be ~0 (cosh(0)=1 but 2*q1*0*cosh=0)
    const phi = gurson.yieldFunction(250, 0, 250, 0);
    assert(Math.abs(phi) < TOLERANCE, `Yield function at f=0, σ_eq=σ_y should be ~0, got ${phi}`);
}

// ──────────────────────────────────────────────────
// Test 27: LemaitreDamage zero plastic strain rate
// ──────────────────────────────────────────────────
console.log('Test 27: LemaitreDamage zero plastic strain rate → zero damage rate');
{
    const lemaitre = new fractureSolver.LemaitreDamage({ E: 210000, nu: 0.3, S: 2.0, s: 1.0 });
    const sigma = [100, 0, 0, 0, 0, 0];
    const dDot = lemaitre.computeDamageRate(sigma, 0.0, 0.1, 0.0);
    assert(Math.abs(dDot) < TOLERANCE, `Damage rate should be zero with zero plastic strain rate, got ${dDot}`);
}

// ──────────────────────────────────────────────────
// Test 28: FailureCriteria max principal stress
// ──────────────────────────────────────────────────
console.log('Test 28: FailureCriteria max principal stress');
{
    // Use equally-spaced eigenvalues [500, 300, 100] so q=0 in the cubic solver
    const sigma = [500, 300, 100, 0, 0, 0];
    const result = fractureSolver.FailureCriteria.maxPrincipalStress(sigma, 400);
    assert(result.failed === true, 'Should detect failure when σ1 > threshold');
    assert(Math.abs(result.sigma1 - 500) < TOLERANCE, `σ1 should be 500, got ${result.sigma1}`);
}

// ──────────────────────────────────────────────────
// Test 29: FailureCriteria Mohr-Coulomb
// ──────────────────────────────────────────────────
console.log('Test 29: FailureCriteria Mohr-Coulomb');
{
    const sigma = [300, 0, -100, 0, 0, 0];
    const cohesion = 50;
    const phi = Math.PI / 6; // 30 degrees
    const result = fractureSolver.FailureCriteria.mohrCoulomb(sigma, cohesion, phi);
    assert(typeof result.failed === 'boolean', 'Should return a boolean failed property');
    assert(typeof result.tau === 'number', 'Should return tau');
    assert(typeof result.strength === 'number', 'Should return strength');
}

// ──────────────────────────────────────────────────
// Dynamics Solver Tests
// ──────────────────────────────────────────────────

// ──────────────────────────────────────────────────
// Test 30: ExplicitDynamics instantiation
// ──────────────────────────────────────────────────
console.log('Test 30: ExplicitDynamics instantiation');
{
    const solver = new dynamicsSolver.ExplicitDynamics({ density: 7800 });
    assert(solver !== null && solver !== undefined, 'Should create a valid object');
    assert(solver.density === 7800, `Density should be 7800, got ${solver.density}`);
}

// ──────────────────────────────────────────────────
// Test 31: ExplicitDynamics critical time step
// ──────────────────────────────────────────────────
console.log('Test 31: ExplicitDynamics critical time step');
{
    const solver = new dynamicsSolver.ExplicitDynamics({ density: 7800, safetyFactor: 0.8 });
    // Create a simple single-element mesh (unit cube)
    const mesh = {
        nodeCount: 8,
        elemCount: 1,
        getElementNodes: () => [0, 1, 2, 3, 4, 5, 6, 7],
        getNodeCoords: (n) => {
            const coords = [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ];
            return coords[n];
        }
    };
    const material = { E: 200e9, nu: 0.3 };
    const dt = solver.criticalTimeStep(mesh, material, 7800);
    assert(dt > 0, `Critical time step should be positive, got ${dt}`);
    assert(isFinite(dt), `Critical time step should be finite, got ${dt}`);
}

// ──────────────────────────────────────────────────
// Test 32: ImplicitQuasiStatic instantiation
// ──────────────────────────────────────────────────
console.log('Test 32: ImplicitQuasiStatic instantiation');
{
    const solver = new dynamicsSolver.ImplicitQuasiStatic({ loadSteps: 10 });
    assert(solver !== null && solver !== undefined, 'Should create a valid object');
    assert(solver.loadSteps === 10, `loadSteps should be 10, got ${solver.loadSteps}`);
}

// ──────────────────────────────────────────────────
// Test 33: NewmarkDynamics instantiation
// ──────────────────────────────────────────────────
console.log('Test 33: NewmarkDynamics instantiation');
{
    const solver = new dynamicsSolver.NewmarkDynamics({ beta: 0.25, gamma: 0.5 });
    assert(solver !== null && solver !== undefined, 'Should create a valid object');
    assert(solver.beta === 0.25, `beta should be 0.25, got ${solver.beta}`);
    assert(solver.gamma === 0.5, `gamma should be 0.5, got ${solver.gamma}`);
}

// ──────────────────────────────────────────────────
// Test 34: Lumped mass computation
// ──────────────────────────────────────────────────
console.log('Test 34: Lumped mass computation for unit cube');
{
    const solver = new dynamicsSolver.ExplicitDynamics({ density: 1.0 });
    // Unit cube element
    const nodeCoords = new Float64Array([
        0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
        0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1
    ]);
    const elemMass = solver.computeElementMass(nodeCoords, 1.0);
    assert(elemMass.length === 8, `Should have 8 nodal masses, got ${elemMass.length}`);
    const allPositive = Array.from(elemMass).every(m => m > 0);
    assert(allPositive, 'All nodal masses should be positive');
    // Total mass should equal volume * density = 1.0 * 1.0 = 1.0
    const totalMass = Array.from(elemMass).reduce((a, b) => a + b, 0);
    assert(Math.abs(totalMass - 1.0) < TOLERANCE, `Total mass should be 1.0, got ${totalMass}`);
}

// ──────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
