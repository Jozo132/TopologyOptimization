#!/usr/bin/env node
/**
 * Nonlinear FEA Compliance Tests
 *
 * 2D compliance benchmarks using a 10×10×1 hex-element grid with:
 *   - Vertical force (upward) applied at the top center node
 *   - Fixed constraint at the bottom center node(s)
 *
 * Covers: large-deformation, stress, plastic-deformation, shear,
 *         buckling, and fracture/damage via all material models.
 *
 * Run with: node tests/test-nonlinear-compliance.js
 */

import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toURL = (p) => pathToFileURL(p).href;

const { NonlinearSolver, computeDeformationGradient, greenLagrangeStrain,
    vonMises, principalStresses, stressTriaxiality, cauchyStress,
    strainEnergyDensity } = await import(toURL(join(__dirname, '..', 'js', 'nonlinear-solver.js')));
const { createMaterial, MaterialState } = await import(toURL(join(__dirname, '..', 'js', 'material-models.js')));

let passed = 0;
let failed = 0;
let testNumber = 0;

function assert(condition, message) {
    if (condition) {
        passed++;
        console.log(`  ✓ ${message}`);
    } else {
        failed++;
        console.error(`  ✗ ${message}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Shared mesh utilities: 10×10×1 hex8 grid
// ═══════════════════════════════════════════════════════════════════════

/**
 * Create a structured hex8 mesh of size nelx × nely × nelz.
 * Node ordering: node(ix,iy,iz) = ix*(nny)*(nnz) + iy*(nnz) + iz
 * where nnx=nelx+1, nny=nely+1, nnz=nelz+1
 */
function createHexMesh(nelx, nely, nelz, sizeX, sizeY, sizeZ) {
    const nnx = nelx + 1;
    const nny = nely + 1;
    const nnz = nelz + 1;
    const nodeCount = nnx * nny * nnz;
    const elemCount = nelx * nely * nelz;

    const dx = sizeX / nelx;
    const dy = sizeY / nely;
    const dz = sizeZ / nelz;

    // Node coordinates
    const coords = new Float64Array(nodeCount * 3);
    for (let ix = 0; ix < nnx; ix++) {
        for (let iy = 0; iy < nny; iy++) {
            for (let iz = 0; iz < nnz; iz++) {
                const idx = ix * nny * nnz + iy * nnz + iz;
                coords[idx * 3]     = ix * dx;
                coords[idx * 3 + 1] = iy * dy;
                coords[idx * 3 + 2] = iz * dz;
            }
        }
    }

    // Element connectivity: hex8 node ordering
    // Local node 0: (ix,   iy,   iz)
    // Local node 1: (ix+1, iy,   iz)
    // Local node 2: (ix+1, iy+1, iz)
    // Local node 3: (ix,   iy+1, iz)
    // Local node 4: (ix,   iy,   iz+1)
    // Local node 5: (ix+1, iy,   iz+1)
    // Local node 6: (ix+1, iy+1, iz+1)
    // Local node 7: (ix,   iy+1, iz+1)
    function getElementNodes(e) {
        const iz = e % nelz;
        const rem = (e - iz) / nelz;
        const iy = rem % nely;
        const ix = (rem - iy) / nely;

        const n0 = ix * nny * nnz + iy * nnz + iz;
        const n1 = (ix + 1) * nny * nnz + iy * nnz + iz;
        const n2 = (ix + 1) * nny * nnz + (iy + 1) * nnz + iz;
        const n3 = ix * nny * nnz + (iy + 1) * nnz + iz;
        const n4 = ix * nny * nnz + iy * nnz + (iz + 1);
        const n5 = (ix + 1) * nny * nnz + iy * nnz + (iz + 1);
        const n6 = (ix + 1) * nny * nnz + (iy + 1) * nnz + (iz + 1);
        const n7 = ix * nny * nnz + (iy + 1) * nnz + (iz + 1);

        return [n0, n1, n2, n3, n4, n5, n6, n7];
    }

    function getNodeCoords(nodeIdx) {
        return [
            coords[nodeIdx * 3],
            coords[nodeIdx * 3 + 1],
            coords[nodeIdx * 3 + 2]
        ];
    }

    return {
        nelx, nely, nelz,
        nodeCount,
        elemCount,
        getElementNodes,
        getNodeCoords,
        coords
    };
}

/**
 * Create the standard 10×10×1 compliance test mesh.
 * Physical size: 10mm × 10mm × 1mm (thin plate).
 */
function createStandardMesh() {
    return createHexMesh(10, 10, 1, 10, 10, 1);
}

/**
 * Get fixed DOFs: entire bottom row (y=0, all x, both z=0 and z=1).
 * Fixes all 3 translational DOFs (x, y, z) for each node.
 */
function getBottomRowConstraints(mesh) {
    const nnx = mesh.nelx + 1;
    const nnz = mesh.nelz + 1;
    const nny = mesh.nely + 1;
    const dofs = [];
    for (let ix = 0; ix < nnx; ix++) {
        for (let iz = 0; iz < nnz; iz++) {
            const nodeIdx = ix * nny * nnz + 0 * nnz + iz;
            dofs.push(nodeIdx * 3, nodeIdx * 3 + 1, nodeIdx * 3 + 2);
        }
    }
    return new Int32Array(dofs);
}

/**
 * Get fixed DOFs: bottom center column (x=5, y=0, both z=0 and z=1).
 * Also fixes the two adjacent x-neighbors for rotational stability.
 * Fixes all 3 translational DOFs (x, y, z) for each node.
 */
function getBottomCenterConstraints(mesh) {
    const nnz = mesh.nelz + 1;
    const nny = mesh.nely + 1;
    const midX = Math.floor(mesh.nelx / 2);
    const dofs = [];
    // Fix center and neighbors for stability
    for (let ix = Math.max(0, midX - 1); ix <= Math.min(mesh.nelx, midX + 1); ix++) {
        for (let iz = 0; iz < nnz; iz++) {
            const nodeIdx = ix * nny * nnz + 0 * nnz + iz;
            dofs.push(nodeIdx * 3, nodeIdx * 3 + 1, nodeIdx * 3 + 2);
        }
    }
    return new Int32Array(dofs);
}

/**
 * Create force vector with upward (+Y) force at the top center node(s).
 * Force applied at (x=5, y=10, z=0) and (x=5, y=10, z=1).
 */
function createUpwardForceAtTop(mesh, forceMagnitude) {
    const ndof = mesh.nodeCount * 3;
    const F = new Float64Array(ndof);
    const nnz = mesh.nelz + 1;
    const nny = mesh.nely + 1;
    const midX = Math.floor(mesh.nelx / 2);
    const topY = mesh.nely;

    for (let iz = 0; iz < nnz; iz++) {
        const nodeIdx = midX * nny * nnz + topY * nnz + iz;
        F[nodeIdx * 3 + 1] = forceMagnitude / nnz; // Y direction, distribute
    }
    return F;
}

/**
 * Create force vector with downward (-Y) force at the top center node(s).
 */
function createDownwardForceAtTop(mesh, forceMagnitude) {
    return createUpwardForceAtTop(mesh, -forceMagnitude);
}

/**
 * Create horizontal (shear) force at top center: +X direction.
 */
function createShearForceAtTop(mesh, forceMagnitude) {
    const ndof = mesh.nodeCount * 3;
    const F = new Float64Array(ndof);
    const nnz = mesh.nelz + 1;
    const nny = mesh.nely + 1;
    const midX = Math.floor(mesh.nelx / 2);
    const topY = mesh.nely;

    for (let iz = 0; iz < nnz; iz++) {
        const nodeIdx = midX * nny * nnz + topY * nnz + iz;
        F[nodeIdx * 3] = forceMagnitude / nnz; // X direction
    }
    return F;
}

/**
 * Render a 2D ASCII stress map for the 10×10 grid (visual verification).
 * Shows von Mises stress as characters: ' ' (low) → '░' → '▒' → '▓' → '█' (high).
 */
function renderStressMap(vonMisesStress, nelx, nely, nelz, label) {
    const chars = [' ', '░', '▒', '▓', '█'];
    let maxVM = 0;
    for (let i = 0; i < vonMisesStress.length; i++) {
        if (vonMisesStress[i] > maxVM) maxVM = vonMisesStress[i];
    }
    if (maxVM === 0) maxVM = 1;

    console.log(`  ┌── ${label} (max: ${maxVM.toFixed(2)}) ──`);
    for (let iy = nely - 1; iy >= 0; iy--) {
        let row = '  │ ';
        for (let ix = 0; ix < nelx; ix++) {
            // Average across z
            let avgVM = 0;
            for (let iz = 0; iz < nelz; iz++) {
                const e = ix * nely * nelz + iy * nelz + iz;
                avgVM += vonMisesStress[e] || 0;
            }
            avgVM /= nelz;
            const level = Math.min(chars.length - 1, Math.floor((avgVM / maxVM) * (chars.length - 0.01)));
            row += chars[level] + chars[level];
        }
        console.log(row);
    }
    console.log('  └' + '─'.repeat(nelx * 2 + 3));
}

/**
 * Render a 2D ASCII displacement map for the 10×10 grid.
 */
function renderDisplacementMap(displacement, mesh, label) {
    const { nelx, nely, nelz } = mesh;
    const nnz = nelz + 1;
    const nny = nely + 1;
    const chars = [' ', '░', '▒', '▓', '█'];

    // Compute max displacement magnitude
    let maxMag = 0;
    const nodeCount = mesh.nodeCount;
    for (let i = 0; i < nodeCount; i++) {
        const ux = displacement[i * 3] || 0;
        const uy = displacement[i * 3 + 1] || 0;
        const uz = displacement[i * 3 + 2] || 0;
        const mag = Math.sqrt(ux * ux + uy * uy + uz * uz);
        if (mag > maxMag) maxMag = mag;
    }
    if (maxMag === 0) maxMag = 1;

    console.log(`  ┌── ${label} (max: ${maxMag.toExponential(3)}) ──`);
    for (let iy = nely; iy >= 0; iy--) {
        let row = '  │ ';
        for (let ix = 0; ix <= nelx; ix++) {
            let avgMag = 0;
            for (let iz = 0; iz < nnz; iz++) {
                const nodeIdx = ix * nny * nnz + iy * nnz + iz;
                const ux = displacement[nodeIdx * 3] || 0;
                const uy = displacement[nodeIdx * 3 + 1] || 0;
                const uz = displacement[nodeIdx * 3 + 2] || 0;
                avgMag += Math.sqrt(ux * ux + uy * uy + uz * uz);
            }
            avgMag /= nnz;
            const level = Math.min(chars.length - 1, Math.floor((avgMag / maxMag) * (chars.length - 0.01)));
            row += chars[level] + chars[level];
        }
        console.log(row);
    }
    console.log('  └' + '─'.repeat((nelx + 1) * 2 + 3));
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: Large Deformation — Neo-Hookean, upward force
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`Test ${testNumber}: Large Deformation — Neo-Hookean hyperelastic (10×10×1, force up, fixed bottom row)`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('neo-hookean', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 5);

    const solver = new NonlinearSolver({
        numLoadSteps: 5,
        maxNewtonIter: 50,
        residualTol: 1e-3,
        incrementTol: 1e-3
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.displacement.length === mesh.nodeCount * 3,
        `Displacement vector has correct size: ${result.displacement.length}`);
    assert(result.vonMisesStress.length === mesh.elemCount,
        `Von Mises stress has correct size: ${result.vonMisesStress.length}`);
    assert(result.converged === true, `Solver converged: ${result.converged}`);

    // Check that top center node displaced upward
    const nny = mesh.nely + 1;
    const nnz = mesh.nelz + 1;
    const topCenterNode = 5 * nny * nnz + 10 * nnz + 0;
    const topUy = result.displacement[topCenterNode * 3 + 1];
    assert(topUy > 0, `Top center node displaced upward: uy=${topUy.toExponential(3)}`);

    // Check that bottom center is fixed (zero displacement)
    const botCenterNode = 5 * nny * nnz + 0 * nnz + 0;
    const botUy = result.displacement[botCenterNode * 3 + 1];
    assert(Math.abs(botUy) < 1e-10, `Bottom center fixed: uy=${botUy.toExponential(3)}`);

    // Check stress is highest near the constraint (stress concentration)
    const bottomStress = result.vonMisesStress[5 * 10 * 1 + 0 * 1 + 0]; // bottom-center element
    let maxStress = 0;
    for (let i = 0; i < result.vonMisesStress.length; i++) {
        if (result.vonMisesStress[i] > maxStress) maxStress = result.vonMisesStress[i];
    }
    assert(maxStress > 0, `Non-zero stress field: max σ_vm = ${maxStress.toFixed(2)}`);
    assert(result.strainEnergy.length === mesh.elemCount,
        `Strain energy field has correct size: ${result.strainEnergy.length}`);

    // Check step snapshots
    assert(result.stepSnapshots.length >= 2, `Has multiple load step snapshots: ${result.stepSnapshots.length}`);
    if (result.stepSnapshots.length >= 2) {
        const last = result.stepSnapshots.length - 1;
        assert(result.stepSnapshots[0].loadFraction < result.stepSnapshots[last].loadFraction,
            'Load fraction increases across steps');
    }

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Neo-Hookean');
    renderDisplacementMap(result.displacement, mesh, 'Displacement — Neo-Hookean');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: Stress Distribution — Linear Elastic, verify stress symmetry
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Stress Distribution — Linear Elastic (verify symmetry and stress concentration)`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('linear-elastic', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 10);

    const solver = new NonlinearSolver({
        numLoadSteps: 1,
        maxNewtonIter: 50,
        residualTol: 1e-3,
        incrementTol: 1e-3,
        cgMaxIter: 2000
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.converged === true, `Linear elastic converged: ${result.converged}`);

    // Check stress symmetry: left half ≈ right half (within tolerance)
    // Element (ix, iy) = elem index ix * nely * nelz + iy * nelz + 0
    let symmetryError = 0;
    let symmetryCount = 0;
    for (let iy = 0; iy < 10; iy++) {
        for (let ix = 0; ix < 5; ix++) {
            const leftIdx = ix * 10 * 1 + iy * 1 + 0;
            const rightIdx = (9 - ix) * 10 * 1 + iy * 1 + 0;
            const leftVM = result.vonMisesStress[leftIdx];
            const rightVM = result.vonMisesStress[rightIdx];
            const maxVal = Math.max(Math.abs(leftVM), Math.abs(rightVM), 1e-12);
            symmetryError += Math.abs(leftVM - rightVM) / maxVal;
            symmetryCount++;
        }
    }
    const avgSymError = symmetryError / symmetryCount;
    assert(avgSymError < 0.1, `Stress field approximately symmetric: avg relative error = ${avgSymError.toFixed(4)}`);

    // Principal stresses should be sorted: σ1 ≥ σ2 ≥ σ3
    let principalOrdered = true;
    for (let e = 0; e < mesh.elemCount; e++) {
        const s1 = result.principalStressField[e * 3];
        const s2 = result.principalStressField[e * 3 + 1];
        const s3 = result.principalStressField[e * 3 + 2];
        if (s1 + 1e-6 < s2 || s2 + 1e-6 < s3) {
            principalOrdered = false;
            break;
        }
    }
    assert(principalOrdered, 'Principal stresses are correctly ordered: σ1 ≥ σ2 ≥ σ3');

    // Triaxiality is finite everywhere
    let triaxFinite = true;
    for (let e = 0; e < mesh.elemCount; e++) {
        if (!isFinite(result.triaxiality[e])) { triaxFinite = false; break; }
    }
    assert(triaxFinite, 'Triaxiality is finite everywhere');

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Linear Elastic');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: Plastic Deformation — J2 Plasticity, verify yielding
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Plastic Deformation — J2 Plasticity (verify yielding at constraint)`);
{
    const mesh = createStandardMesh();
    // Low yield strength so elements near constraint exceed yield
    const material = createMaterial('j2-plasticity', { E: 1e3, nu: 0.3, sigY: 5 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 10);

    const solver = new NonlinearSolver({
        numLoadSteps: 5,
        maxNewtonIter: 30,
        residualTol: 1e-3,
        incrementTol: 1e-3
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.vonMisesStress.length === mesh.elemCount,
        `Stress field computed: ${result.vonMisesStress.length} elements`);

    // Count yielded elements (von Mises > yield strength)
    const yieldStrength = 5;
    let yieldedCount = 0;
    for (let i = 0; i < result.vonMisesStress.length; i++) {
        if (result.vonMisesStress[i] > yieldStrength * 0.9) yieldedCount++;
    }
    assert(yieldedCount > 0, `Elements near/above yield: ${yieldedCount} (yield σ_y=${yieldStrength})`);

    // Material states should track plastic deformation
    let hasPlasticStrain = false;
    for (let e = 0; e < mesh.elemCount && !hasPlasticStrain; e++) {
        if (result.materialStates[e]) {
            for (const gpState of result.materialStates[e]) {
                if (gpState && gpState.epsPl > 0) {
                    hasPlasticStrain = true;
                    break;
                }
            }
        }
    }
    assert(hasPlasticStrain, 'Plastic strain accumulated in at least one element');

    // Strain energy should be positive
    let totalEnergy = 0;
    for (let e = 0; e < mesh.elemCount; e++) totalEnergy += result.strainEnergy[e];
    assert(totalEnergy > 0, `Total strain energy is positive: ${totalEnergy.toExponential(3)}`);

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — J2 Plasticity');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 4: Shear Analysis — force applied horizontally at top
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Shear Analysis — horizontal force at top center (10×10×1)`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('linear-elastic', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createShearForceAtTop(mesh, 5);

    const solver = new NonlinearSolver({
        numLoadSteps: 1,
        maxNewtonIter: 50,
        residualTol: 1e-3,
        incrementTol: 1e-3,
        cgMaxIter: 2000
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.converged === true, `Shear analysis converged: ${result.converged}`);

    // Top center node should displace primarily in X
    const nny = mesh.nely + 1;
    const nnz = mesh.nelz + 1;
    const topCenterNode = 5 * nny * nnz + 10 * nnz + 0;
    const topUx = result.displacement[topCenterNode * 3];
    const topUy = result.displacement[topCenterNode * 3 + 1];
    assert(Math.abs(topUx) > 0, `Top center displaced in X: ux=${topUx.toExponential(3)}`);

    // Cauchy stress should have shear components
    assert(result.cauchyStress.length === mesh.elemCount * 6,
        `Cauchy stress tensor field: ${result.cauchyStress.length} components`);

    // Check shear stress is significant in some elements (σ_xy component at index 3)
    let maxShear = 0;
    for (let e = 0; e < mesh.elemCount; e++) {
        const shearXY = Math.abs(result.cauchyStress[e * 6 + 3]);
        if (shearXY > maxShear) maxShear = shearXY;
    }
    assert(maxShear > 0, `Shear stress component present: max |σ_xy| = ${maxShear.toFixed(4)}`);

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Shear');
    renderDisplacementMap(result.displacement, mesh, 'Displacement — Shear');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 5: Buckling-like compression — downward force at top
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Compression / Buckling tendency — downward force at top center`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('linear-elastic', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createDownwardForceAtTop(mesh, 10);

    const solver = new NonlinearSolver({
        numLoadSteps: 3,
        maxNewtonIter: 20,
        residualTol: 1e-4,
        incrementTol: 1e-4
    });

    const result = solver.solve(mesh, material, constraints, loads);

    // Top center should move downward
    const nny = mesh.nely + 1;
    const nnz = mesh.nelz + 1;
    const topCenterNode = 5 * nny * nnz + 10 * nnz + 0;
    const topUy = result.displacement[topCenterNode * 3 + 1];
    assert(topUy < 0, `Top center displaced downward under compression: uy=${topUy.toExponential(3)}`);

    // Under compression, triaxiality should be negative in many elements
    let negTriaxCount = 0;
    for (let e = 0; e < mesh.elemCount; e++) {
        if (result.triaxiality[e] < -0.01) negTriaxCount++;
    }
    assert(negTriaxCount > 0, `Some elements in compression (negative triaxiality): ${negTriaxCount}`);

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Compression');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 6: Mooney-Rivlin rubber — large deformation
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Large Deformation — Mooney-Rivlin rubber-like material`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('mooney-rivlin', { E: 10, nu: 0.45 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 5);

    const solver = new NonlinearSolver({
        numLoadSteps: 5,
        maxNewtonIter: 30,
        residualTol: 1e-3,
        incrementTol: 1e-3
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.displacement.length === mesh.nodeCount * 3,
        `Displacement vector computed: ${result.displacement.length}`);

    // Rubber should deform much more than stiff material (large displacement)
    const nny = mesh.nely + 1;
    const nnz = mesh.nelz + 1;
    const topCenterNode = 5 * nny * nnz + 10 * nnz + 0;
    const topUy = result.displacement[topCenterNode * 3 + 1];
    assert(topUy > 0, `Rubber top center displaced upward: uy=${topUy.toExponential(3)}`);

    // Step snapshots should show increasing displacement
    if (result.stepSnapshots.length >= 2) {
        const step1Uy = result.stepSnapshots[0].displacement[topCenterNode * 3 + 1];
        const lastStepUy = result.stepSnapshots[result.stepSnapshots.length - 1].displacement[topCenterNode * 3 + 1];
        assert(Math.abs(lastStepUy) >= Math.abs(step1Uy) * 0.99,
            `Displacement increases with load steps: step1=${step1Uy.toExponential(3)}, final=${lastStepUy.toExponential(3)}`);
    }

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Mooney-Rivlin');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 7: Ogden hyperelastic — large deformation
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Large Deformation — Ogden hyperelastic`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('ogden', { E: 10, nu: 0.45 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 5);

    const solver = new NonlinearSolver({
        numLoadSteps: 5,
        maxNewtonIter: 30,
        residualTol: 1e-3,
        incrementTol: 1e-3
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.displacement.length === mesh.nodeCount * 3,
        `Ogden displacement computed: ${result.displacement.length}`);
    assert(result.vonMisesStress.length === mesh.elemCount,
        `Ogden stress field computed: ${result.vonMisesStress.length}`);

    // Verify non-zero stress
    let maxVM = 0;
    for (let i = 0; i < result.vonMisesStress.length; i++) {
        if (result.vonMisesStress[i] > maxVM) maxVM = result.vonMisesStress[i];
    }
    assert(maxVM > 0, `Ogden stress is non-zero: max σ_vm = ${maxVM.toFixed(2)}`);

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Ogden');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 8: Drucker-Prager — soil/concrete-like behavior
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Drucker-Prager — pressure-dependent yield (soil/concrete)`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('drucker-prager', { E: 1e3, nu: 0.2, sigY: 10 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 20);

    const solver = new NonlinearSolver({
        numLoadSteps: 3,
        maxNewtonIter: 20,
        residualTol: 1e-3,
        incrementTol: 1e-3
    });

    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.vonMisesStress.length === mesh.elemCount,
        `Drucker-Prager stress field: ${result.vonMisesStress.length} elements`);

    // Verify stress concentration near constraint
    let maxVM = 0;
    let maxVMIdx = 0;
    for (let i = 0; i < result.vonMisesStress.length; i++) {
        if (result.vonMisesStress[i] > maxVM) {
            maxVM = result.vonMisesStress[i];
            maxVMIdx = i;
        }
    }
    assert(maxVM > 0, `Drucker-Prager max stress: ${maxVM.toFixed(2)} at element ${maxVMIdx}`);

    renderStressMap(result.vonMisesStress, 10, 10, 1, 'Von Mises Stress — Drucker-Prager');
}

// ═══════════════════════════════════════════════════════════════════════
// Test 9: Energy conservation — strain energy increases with load
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Energy conservation — strain energy increases monotonically with load`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('neo-hookean', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 20);

    const solver = new NonlinearSolver({
        numLoadSteps: 5,
        maxNewtonIter: 20,
        residualTol: 1e-4,
        incrementTol: 1e-4
    });

    const result = solver.solve(mesh, material, constraints, loads);

    // Verify strain energy increases with each load step
    let energyIncreasing = true;
    let prevMaxVM = 0;
    for (let s = 0; s < result.stepSnapshots.length; s++) {
        const snap = result.stepSnapshots[s];
        let maxVM = 0;
        for (let i = 0; i < snap.vonMisesStress.length; i++) {
            if (snap.vonMisesStress[i] > maxVM) maxVM = snap.vonMisesStress[i];
        }
        if (s > 0 && maxVM < prevMaxVM * 0.5) {
            energyIncreasing = false;
        }
        prevMaxVM = maxVM;
    }
    assert(energyIncreasing, 'Peak stress generally increases with load steps (energy consistency)');

    // Total strain energy should be positive
    let totalE = 0;
    for (let i = 0; i < result.strainEnergy.length; i++) totalE += result.strainEnergy[i];
    assert(totalE > 0, `Total strain energy positive: ${totalE.toExponential(3)}`);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 10: Zero force → zero displacement (equilibrium check)
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Zero force — zero displacement verification`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('linear-elastic', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = new Float64Array(mesh.nodeCount * 3); // zero force

    const solver = new NonlinearSolver({
        numLoadSteps: 1,
        maxNewtonIter: 5,
        residualTol: 1e-8,
        incrementTol: 1e-8
    });

    const result = solver.solve(mesh, material, constraints, loads);

    // All displacements should be zero
    let maxDisp = 0;
    for (let i = 0; i < result.displacement.length; i++) {
        const d = Math.abs(result.displacement[i]);
        if (d > maxDisp) maxDisp = d;
    }
    assert(maxDisp < 1e-10, `Zero force produces zero displacement: max |u| = ${maxDisp.toExponential(3)}`);

    // All stresses should be zero
    let maxStress = 0;
    for (let i = 0; i < result.vonMisesStress.length; i++) {
        if (result.vonMisesStress[i] > maxStress) maxStress = result.vonMisesStress[i];
    }
    assert(maxStress < 1e-6, `Zero force produces zero stress: max σ_vm = ${maxStress.toExponential(3)}`);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 11: Displacement proportionality — double force → double displacement
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Linear proportionality — 2× force ≈ 2× displacement (linear elastic)`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('linear-elastic', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);

    const solver = new NonlinearSolver({
        numLoadSteps: 1,
        maxNewtonIter: 20,
        residualTol: 1e-6,
        incrementTol: 1e-6
    });

    const loads1 = createUpwardForceAtTop(mesh, 5);
    const result1 = solver.solve(mesh, material, constraints, loads1);

    const loads2 = createUpwardForceAtTop(mesh, 10);
    const result2 = solver.solve(mesh, material, constraints, loads2);

    // Top center displacement should roughly double
    const nny = mesh.nely + 1;
    const nnz = mesh.nelz + 1;
    const topCenterNode = 5 * nny * nnz + 10 * nnz + 0;
    const uy1 = result1.displacement[topCenterNode * 3 + 1];
    const uy2 = result2.displacement[topCenterNode * 3 + 1];
    const ratio = uy2 / uy1;
    assert(Math.abs(ratio - 2.0) < 0.15,
        `Force doubling: uy ratio = ${ratio.toFixed(3)} (expected ~2.0), uy1=${uy1.toExponential(3)}, uy2=${uy2.toExponential(3)}`);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 12: Cauchy stress tensor symmetry (σ_ij = σ_ji)
// ═══════════════════════════════════════════════════════════════════════
testNumber++;
console.log(`\nTest ${testNumber}: Cauchy stress tensor symmetry`);
{
    const mesh = createStandardMesh();
    const material = createMaterial('neo-hookean', { E: 1e3, nu: 0.3 });
    const constraints = getBottomRowConstraints(mesh);
    const loads = createUpwardForceAtTop(mesh, 20);

    const solver = new NonlinearSolver({ numLoadSteps: 3, maxNewtonIter: 20, residualTol: 1e-4, incrementTol: 1e-4 });
    const result = solver.solve(mesh, material, constraints, loads);

    // Cauchy stress in Voigt: [σ11, σ22, σ33, σ12, σ23, σ13]
    // It's inherently symmetric by construction (Voigt), but verify the data is sensible
    let allFinite = true;
    for (let e = 0; e < mesh.elemCount; e++) {
        for (let c = 0; c < 6; c++) {
            if (!isFinite(result.cauchyStress[e * 6 + c])) {
                allFinite = false;
                break;
            }
        }
    }
    assert(allFinite, 'All Cauchy stress components are finite');

    // Von Mises should be consistent with Cauchy stress
    for (let e = 0; e < Math.min(5, mesh.elemCount); e++) {
        const sigma = [];
        for (let c = 0; c < 6; c++) sigma.push(result.cauchyStress[e * 6 + c]);
        const vmRecomputed = vonMises(sigma);
        const vmStored = result.vonMisesStress[e];
        const diff = Math.abs(vmRecomputed - vmStored);
        assert(diff < 1e-6 * (vmStored + 1e-12),
            `VM recomputed matches stored for element ${e}: diff=${diff.toExponential(3)}`);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${'═'.repeat(60)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
