// Voxelization tests for ModelImporter
// Run with: node tests/index.js

// Provide a minimal FileReader stub so ModelImporter can be instantiated in Node.js
globalThis.FileReader = class FileReader {};

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const { ModelImporter } = await import(join(__dirname, '..', 'js', 'importer.js'));

const importer = new ModelImporter();

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

// ──────────────────────────────────────────────────
// Test 1: Ray-triangle intersection (basic hit)
// ──────────────────────────────────────────────────
console.log('Test 1: Ray-triangle intersection - basic hit');
{
    // Triangle lying in z=5 plane, spanning x=[0,10], y=[0,10]
    const v0 = { x: 0, y: 0, z: 5 };
    const v1 = { x: 10, y: 0, z: 5 };
    const v2 = { x: 0, y: 10, z: 5 };

    const hit = importer._rayTriangleIntersectZ(2, 2, v0, v1, v2);
    assert(hit !== null, 'Ray should hit the triangle');
    assert(Math.abs(hit - 5) < 1e-6, `Intersection Z should be 5, got ${hit}`);
}

// ──────────────────────────────────────────────────
// Test 2: Ray-triangle intersection (miss)
// ──────────────────────────────────────────────────
console.log('Test 2: Ray-triangle intersection - miss');
{
    const v0 = { x: 0, y: 0, z: 5 };
    const v1 = { x: 10, y: 0, z: 5 };
    const v2 = { x: 0, y: 10, z: 5 };

    // Ray at (20, 20) is well outside the triangle
    const hit = importer._rayTriangleIntersectZ(20, 20, v0, v1, v2);
    assert(hit === null, 'Ray should miss the triangle');
}

// ──────────────────────────────────────────────────
// Test 3: Voxelize an axis-aligned cube (box mesh)
//   A cube from (0,0,0) to (10,10,10) made of 12 triangles (2 per face)
// ──────────────────────────────────────────────────
console.log('Test 3: Voxelize a cube mesh');
{
    // Build 12 triangles for a cube [0,10]^3
    function quad(a, b, c, d) {
        return [a, b, c, a, c, d]; // Two triangles
    }
    const cubeVertices = [
        // -Z face (z=0)
        ...quad({ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, { x: 10, y: 10, z: 0 }, { x: 0, y: 10, z: 0 }),
        // +Z face (z=10)
        ...quad({ x: 0, y: 0, z: 10 }, { x: 0, y: 10, z: 10 }, { x: 10, y: 10, z: 10 }, { x: 10, y: 0, z: 10 }),
        // -X face (x=0)
        ...quad({ x: 0, y: 0, z: 0 }, { x: 0, y: 10, z: 0 }, { x: 0, y: 10, z: 10 }, { x: 0, y: 0, z: 10 }),
        // +X face (x=10)
        ...quad({ x: 10, y: 0, z: 0 }, { x: 10, y: 0, z: 10 }, { x: 10, y: 10, z: 10 }, { x: 10, y: 10, z: 0 }),
        // -Y face (y=0)
        ...quad({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 10 }, { x: 10, y: 0, z: 10 }, { x: 10, y: 0, z: 0 }),
        // +Y face (y=10)
        ...quad({ x: 0, y: 10, z: 0 }, { x: 10, y: 10, z: 0 }, { x: 10, y: 10, z: 10 }, { x: 0, y: 10, z: 10 }),
    ];

    const result = importer.voxelizeVertices(cubeVertices, 10);
    const { nx, ny, nz, elements } = result;

    assert(nx === 10, `Expected nx=10, got ${nx}`);
    assert(ny === 10, `Expected ny=10, got ${ny}`);
    assert(nz === 10, `Expected nz=10, got ${nz}`);

    // For a full cube, all interior voxels should be solid
    let solidCount = 0;
    for (let i = 0; i < elements.length; i++) {
        if (elements[i] > 0.5) solidCount++;
    }
    const totalVoxels = nx * ny * nz;
    assert(solidCount === totalVoxels, `All ${totalVoxels} voxels should be solid for a cube, got ${solidCount}`);
}

// ──────────────────────────────────────────────────
// Test 4: Voxelize a sphere-like mesh (not a full cube)
//   Use a tetrahedron to verify only interior voxels are filled
// ──────────────────────────────────────────────────
console.log('Test 4: Voxelize a tetrahedron (should NOT fill entire bounding box)');
{
    // Regular tetrahedron with vertices:
    // A = (0, 0, 0), B = (10, 0, 0), C = (5, 10, 0), D = (5, 5, 10)
    const A = { x: 0, y: 0, z: 0 };
    const B = { x: 10, y: 0, z: 0 };
    const C = { x: 5, y: 10, z: 0 };
    const D = { x: 5, y: 5, z: 10 };

    // 4 triangular faces of the tetrahedron
    const tetVertices = [
        // Face 1: A, B, C (bottom)
        A, B, C,
        // Face 2: A, B, D
        A, B, D,
        // Face 3: A, C, D
        A, C, D,
        // Face 4: B, C, D
        B, C, D,
    ];

    const result = importer.voxelizeVertices(tetVertices, 10);
    const { nx, ny, nz, elements } = result;

    let solidCount = 0;
    for (let i = 0; i < elements.length; i++) {
        if (elements[i] > 0.5) solidCount++;
    }
    const totalVoxels = nx * ny * nz;

    // A tetrahedron occupies roughly 1/6 of its bounding box volume
    // so the solid voxels should be significantly less than the total
    assert(solidCount < totalVoxels, `Tetrahedron should NOT fill entire bounding box (solid: ${solidCount} / ${totalVoxels})`);
    assert(solidCount > 0, `Tetrahedron should have some solid voxels (got ${solidCount})`);

    const fillRatio = solidCount / totalVoxels;
    assert(fillRatio < 0.5, `Fill ratio should be < 0.5 for a tetrahedron (got ${fillRatio.toFixed(3)})`);

    console.log(`  (Fill ratio: ${fillRatio.toFixed(3)}, solid: ${solidCount}/${totalVoxels})`);
}

// ──────────────────────────────────────────────────
// Test 5: Fallback — empty vertex list fills all solid
// ──────────────────────────────────────────────────
console.log('Test 5: Fallback with empty vertices');
{
    // When no vertices are provided, we can't do ray-casting
    // but the function needs at least some vertices for bounds.
    // With 0 vertices, numTriangles=0 so fallback fills all.
    // We need at least some vertices for bounds, so test with 1 or 2 vertices
    // (less than 3, so numTriangles=0).
    const result = importer.voxelizeVertices([
        { x: 0, y: 0, z: 0 },
        { x: 10, y: 10, z: 10 },
    ], 5);

    const { elements } = result;
    let solidCount = 0;
    for (let i = 0; i < elements.length; i++) {
        if (elements[i] > 0.5) solidCount++;
    }
    assert(solidCount === elements.length, `Fallback should fill all voxels (${solidCount}/${elements.length})`);
}

// ──────────────────────────────────────────────────
// Test 6: Templates should still be all-solid (unchanged behavior)
// ──────────────────────────────────────────────────
console.log('Test 6: Templates remain all-solid');
{
    const beam = importer.createBeamTemplate(20);
    let solidCount = 0;
    for (let i = 0; i < beam.elements.length; i++) {
        if (beam.elements[i] > 0.5) solidCount++;
    }
    assert(solidCount === beam.elements.length, `Beam template should be all solid (${solidCount}/${beam.elements.length})`);

    const cube = importer.createCubeTemplate(20);
    solidCount = 0;
    for (let i = 0; i < cube.elements.length; i++) {
        if (cube.elements[i] > 0.5) solidCount++;
    }
    assert(solidCount === cube.elements.length, `Cube template should be all solid (${solidCount}/${cube.elements.length})`);
}

// ──────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
