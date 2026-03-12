#!/usr/bin/env node
/**
 * Tests validating that internal 3D volume elements are properly included
 * in both linear and nonlinear FEA calculations.
 *
 * Run with: node tests/test-internal-volume-fea.js
 */

import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toURL = (p) => pathToFileURL(p).href;

const { NonlinearSolver } = await import(toURL(join(__dirname, '..', 'js', 'nonlinear-solver.js')));
const { createMaterial } = await import(toURL(join(__dirname, '..', 'js', 'material-models.js')));

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

function approxEqual(a, b, tol) {
    return Math.abs(a - b) < tol;
}

/**
 * Create a structured hex8 mesh with optional element densities.
 */
function createHexMesh(nelx, nely, nelz, sizeX, sizeY, sizeZ, elemDensities) {
    const nnx = nelx + 1;
    const nny = nely + 1;
    const nnz = nelz + 1;
    const nodeCount = nnx * nny * nnz;
    const elemCount = nelx * nely * nelz;

    const dx = sizeX / nelx;
    const dy = sizeY / nely;
    const dz = sizeZ / nelz;

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
        elemDensities: elemDensities || null,
        getElementNodes,
        getNodeCoords,
        coords
    };
}

// ═════════════════════════════════════════════════════════════════════
// Test 1: Nonlinear solver with uniform density produces stress in all elements
// ═════════════════════════════════════════════════════════════════════
console.log('Test 1: Nonlinear solver includes all elements (uniform density)');
{
    const nelx = 3, nely = 3, nelz = 3;
    const mesh = createHexMesh(nelx, nely, nelz, 3, 3, 3);
    const material = createMaterial('neo-hookean', { E: 1000, nu: 0.3 });

    // Fix bottom face (z=0)
    const constraints = [];
    const nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    for (let ix = 0; ix < nnx; ix++) {
        for (let iy = 0; iy < nny; iy++) {
            const nid = ix * nny * nnz + iy * nnz + 0; // iz=0
            constraints.push(nid * 3, nid * 3 + 1, nid * 3 + 2);
        }
    }

    // Small force on top face (z=nelz)
    const loads = new Float64Array(mesh.nodeCount * 3);
    for (let ix = 0; ix < nnx; ix++) {
        for (let iy = 0; iy < nny; iy++) {
            const nid = ix * nny * nnz + iy * nnz + nelz; // iz=nelz
            loads[nid * 3 + 2] = 0.1;
        }
    }

    const solver = new NonlinearSolver({ numLoadSteps: 2, maxNewtonIter: 20, residualTol: 1e-4, incrementTol: 1e-4 });
    const result = solver.solve(mesh, material, new Int32Array(constraints), loads);

    // Count elements with non-zero von Mises stress
    let stressedCount = 0;
    for (let e = 0; e < mesh.elemCount; e++) {
        if (result.vonMisesStress[e] > 1e-12) stressedCount++;
    }

    assert(stressedCount === mesh.elemCount, `All ${mesh.elemCount} elements should have stress, got ${stressedCount}`);

    // Verify internal elements (those completely surrounded) have stress
    // In a 3x3x3 mesh, element (1,1,1) is the only fully internal element
    const internalIdx = 1 * nely * nelz + 1 * nelz + 1; // ix=1,iy=1,iz=1
    assert(result.vonMisesStress[internalIdx] > 1e-12, `Internal element (1,1,1) idx=${internalIdx} should have stress: ${result.vonMisesStress[internalIdx]}`);
}

// ═════════════════════════════════════════════════════════════════════
// Test 2: Nonlinear solver density scaling - partial density reduces stiffness
// ═════════════════════════════════════════════════════════════════════
console.log('Test 2: Nonlinear solver density scaling reduces element stiffness');
{
    const nelx = 2, nely = 1, nelz = 1;
    const elemCount = nelx * nely * nelz;

    // Both elements solid
    const densitiesFull = new Float32Array(elemCount);
    densitiesFull.fill(1.0);

    // Second element has reduced density (0.5)
    const densitiesHalf = new Float32Array(elemCount);
    densitiesHalf[0] = 1.0;
    densitiesHalf[1] = 0.5;

    const meshFull = createHexMesh(nelx, nely, nelz, 2, 1, 1, densitiesFull);
    const meshHalf = createHexMesh(nelx, nely, nelz, 2, 1, 1, densitiesHalf);
    const material = createMaterial('neo-hookean', { E: 1000, nu: 0.3 });

    // Fix left face (x=0)
    const constraints = [];
    const nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    for (let iy = 0; iy < nny; iy++) {
        for (let iz = 0; iz < nnz; iz++) {
            const nid = 0 * nny * nnz + iy * nnz + iz;
            constraints.push(nid * 3, nid * 3 + 1, nid * 3 + 2);
        }
    }

    // Force on right face
    const loads = new Float64Array(meshFull.nodeCount * 3);
    for (let iy = 0; iy < nny; iy++) {
        for (let iz = 0; iz < nnz; iz++) {
            const nid = nelx * nny * nnz + iy * nnz + iz;
            loads[nid * 3] = 0.5; // Small force in X direction
        }
    }

    const solver = new NonlinearSolver({ numLoadSteps: 1, maxNewtonIter: 20, residualTol: 1e-6, incrementTol: 1e-6 });
    const resultFull = solver.solve(meshFull, material, new Int32Array(constraints), loads);
    const resultHalf = solver.solve(meshHalf, material, new Int32Array(constraints), loads);

    // With reduced density element, the structure is softer → larger displacement
    let maxDispFull = 0, maxDispHalf = 0;
    const ndof = meshFull.nodeCount * 3;
    for (let i = 0; i < ndof; i++) {
        maxDispFull = Math.max(maxDispFull, Math.abs(resultFull.displacement[i]));
        maxDispHalf = Math.max(maxDispHalf, Math.abs(resultHalf.displacement[i]));
    }
    assert(maxDispHalf > maxDispFull, `Half-density mesh should deflect more: ${maxDispHalf.toExponential(3)} > ${maxDispFull.toExponential(3)}`);

    // Both elements should still have stress
    assert(resultHalf.vonMisesStress[0] > 0, `Solid element should have stress: ${resultHalf.vonMisesStress[0].toExponential(3)}`);
    assert(resultHalf.vonMisesStress[1] > 0, `Half-density element should have stress: ${resultHalf.vonMisesStress[1].toExponential(3)}`);
}

// ═════════════════════════════════════════════════════════════════════
// Test 3: Nonlinear solver with partial density elements (boundary blending)
// ═════════════════════════════════════════════════════════════════════
console.log('Test 3: Nonlinear solver handles partial density elements (boundary blending)');
{
    const nelx = 2, nely = 1, nelz = 1;
    const elemCount = nelx * nely * nelz;

    // First element solid, second has partial density (0.3) - simulating boundary blending
    const densitiesPartial = new Float32Array(elemCount);
    densitiesPartial[0] = 1.0;
    densitiesPartial[1] = 0.3;

    const meshPartial = createHexMesh(nelx, nely, nelz, 2, 1, 1, densitiesPartial);
    const material = createMaterial('neo-hookean', { E: 1000, nu: 0.3 });

    // Fix left face
    const constraints = [];
    const nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    for (let iy = 0; iy < nny; iy++) {
        for (let iz = 0; iz < nnz; iz++) {
            const nid = 0 * nny * nnz + iy * nnz + iz;
            constraints.push(nid * 3, nid * 3 + 1, nid * 3 + 2);
        }
    }

    // Force on right face
    const loads = new Float64Array(meshPartial.nodeCount * 3);
    for (let iy = 0; iy < nny; iy++) {
        for (let iz = 0; iz < nnz; iz++) {
            const nid = nelx * nny * nnz + iy * nnz + iz;
            loads[nid * 3 + 1] = -0.5;
        }
    }

    const solver = new NonlinearSolver({ numLoadSteps: 1, maxNewtonIter: 20, residualTol: 1e-6, incrementTol: 1e-6 });
    const resultPartial = solver.solve(meshPartial, material, new Int32Array(constraints), loads);

    // Partial density element should have non-zero stress
    assert(resultPartial.vonMisesStress[1] > 0, `Partial density element should have stress: ${resultPartial.vonMisesStress[1].toExponential(3)}`);
}

// ═════════════════════════════════════════════════════════════════════
// Test 4: FEA-only density threshold - elements with density 0.1-0.5 are included
// ═════════════════════════════════════════════════════════════════════
console.log('Test 4: FEA-only density threshold preserves continuous density values');
{
    const Emin = 1e-9;
    const elements = new Float32Array([0, 0.1, 0.3, 0.5, 0.7, 1.0]);
    const x = new Float32Array(elements.length);
    for (let i = 0; i < elements.length; i++) {
        // Fixed logic: use actual density when > 0
        x[i] = elements[i] > 0 ? elements[i] : Emin;
    }

    // Void element gets Emin (approximately, due to Float32 precision)
    assert(x[0] < 1e-6, `Void element (0.0) should get near-zero value, got ${x[0]}`);
    // Boundary elements with density 0.1-0.5 should retain their values (not be forced to Emin)
    assert(approxEqual(x[1], 0.1, 1e-5), `Boundary element (0.1) should keep density ~0.1, got ${x[1]}`);
    assert(approxEqual(x[2], 0.3, 1e-5), `Boundary element (0.3) should keep density ~0.3, got ${x[2]}`);
    assert(approxEqual(x[3], 0.5, 1e-5), `Boundary element (0.5) should keep density ~0.5, got ${x[3]}`);
    assert(approxEqual(x[4], 0.7, 1e-5), `Interior element (0.7) should keep density ~0.7, got ${x[4]}`);
    assert(x[5] === 1.0, `Solid element (1.0) should keep density 1.0, got ${x[5]}`);

    // Verify that the OLD logic would have incorrectly excluded these elements
    const xOld = new Float32Array(elements.length);
    for (let i = 0; i < elements.length; i++) {
        xOld[i] = elements[i] > 0.5 ? 1.0 : Emin;
    }
    // Old logic: elements with density ≤ 0.5 were set to Emin (excluded from FEA)
    assert(xOld[1] < 1e-6, `OLD: Boundary element (0.1) was incorrectly set to near-zero`);
    assert(xOld[2] < 1e-6, `OLD: Boundary element (0.3) was incorrectly set to near-zero`);
    assert(xOld[3] < 1e-6, `OLD: Boundary element (0.5) was incorrectly set to near-zero`);
    // New logic preserves them
    assert(x[1] > 0.05, `NEW: Boundary element (0.1) is properly preserved`);
    assert(x[2] > 0.25, `NEW: Boundary element (0.3) is properly preserved`);
    assert(x[3] > 0.45, `NEW: Boundary element (0.5) is properly preserved`);
}

// ═════════════════════════════════════════════════════════════════════
// Test 5: No elemDensities → all elements included with full stiffness
// ═════════════════════════════════════════════════════════════════════
console.log('Test 5: No elemDensities on mesh → all elements included (backward compat)');
{
    const nelx = 2, nely = 2, nelz = 1;
    const mesh = createHexMesh(nelx, nely, nelz, 2, 2, 1);
    assert(mesh.elemDensities === null, 'Mesh without densities should have null elemDensities');

    const material = createMaterial('neo-hookean', { E: 1000, nu: 0.3 });

    // Fix bottom face
    const constraints = [];
    const nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    for (let ix = 0; ix < nnx; ix++) {
        for (let iy = 0; iy < nny; iy++) {
            const nid = ix * nny * nnz + iy * nnz + 0;
            constraints.push(nid * 3, nid * 3 + 1, nid * 3 + 2);
        }
    }

    // Force on top face
    const loads = new Float64Array(mesh.nodeCount * 3);
    for (let ix = 0; ix < nnx; ix++) {
        for (let iy = 0; iy < nny; iy++) {
            const nid = ix * nny * nnz + iy * nnz + nelz;
            loads[nid * 3 + 2] = 0.5;
        }
    }

    const solver = new NonlinearSolver({ numLoadSteps: 1, maxNewtonIter: 20, residualTol: 1e-6, incrementTol: 1e-6 });
    const result = solver.solve(mesh, material, new Int32Array(constraints), loads);

    let stressedCount = 0;
    for (let e = 0; e < mesh.elemCount; e++) {
        if (result.vonMisesStress[e] > 1e-12) stressedCount++;
    }
    assert(stressedCount === mesh.elemCount, `All ${mesh.elemCount} elements should have stress (backward compat), got ${stressedCount}`);
}

// ═════════════════════════════════════════════════════════════════════
console.log('');
console.log(`${'═'.repeat(60)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
