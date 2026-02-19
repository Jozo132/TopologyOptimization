// Voxelization tests for ModelImporter
// Run with: node tests/index.js

// Provide a minimal FileReader stub so ModelImporter can be instantiated in Node.js
globalThis.FileReader = class FileReader {};

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const { ModelImporter } = await import(join(__dirname, '..', 'js', 'importer.js'));
const { TopologySolver } = await import(join(__dirname, '..', 'lib', 'topology-solver.js'));

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
    const beam = importer.createBeamTemplate(30);
    let solidCount = 0;
    for (let i = 0; i < beam.elements.length; i++) {
        if (beam.elements[i] > 0.5) solidCount++;
    }
    assert(solidCount === beam.elements.length, `Beam template should be all solid (${solidCount}/${beam.elements.length})`);

    const cube = importer.createCubeTemplate(50);
    solidCount = 0;
    for (let i = 0; i < cube.elements.length; i++) {
        if (cube.elements[i] > 0.5) solidCount++;
    }
    assert(solidCount === cube.elements.length, `Cube template should be all solid (${solidCount}/${cube.elements.length})`);
    assert(cube.nx === 50, `Cube template at resolution 50 should have nx=50, got ${cube.nx}`);
    assert(cube.ny === 50, `Cube template at resolution 50 should have ny=50, got ${cube.ny}`);
    assert(cube.nz === 50, `Cube template at resolution 50 should have nz=50, got ${cube.nz}`);
}

// ──────────────────────────────────────────────────
// Test 7: Voxelize with mm-based voxel size
// ──────────────────────────────────────────────────
console.log('Test 7: Voxelize with mm-based voxel size');
{
    // Cube from (0,0,0) to (10,10,10), voxel size = 2mm → should get 5x5x5
    function quad(a, b, c, d) {
        return [a, b, c, a, c, d];
    }
    const cubeVertices = [
        ...quad({ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, { x: 10, y: 10, z: 0 }, { x: 0, y: 10, z: 0 }),
        ...quad({ x: 0, y: 0, z: 10 }, { x: 0, y: 10, z: 10 }, { x: 10, y: 10, z: 10 }, { x: 10, y: 0, z: 10 }),
        ...quad({ x: 0, y: 0, z: 0 }, { x: 0, y: 10, z: 0 }, { x: 0, y: 10, z: 10 }, { x: 0, y: 0, z: 10 }),
        ...quad({ x: 10, y: 0, z: 0 }, { x: 10, y: 0, z: 10 }, { x: 10, y: 10, z: 10 }, { x: 10, y: 10, z: 0 }),
        ...quad({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 10 }, { x: 10, y: 0, z: 10 }, { x: 10, y: 0, z: 0 }),
        ...quad({ x: 0, y: 10, z: 0 }, { x: 10, y: 10, z: 0 }, { x: 10, y: 10, z: 10 }, { x: 0, y: 10, z: 10 }),
    ];

    const result = importer.voxelizeVertices(cubeVertices, null, 2);
    assert(result.nx === 5, `Expected nx=5 with 2mm voxels on 10mm cube, got ${result.nx}`);
    assert(result.ny === 5, `Expected ny=5 with 2mm voxels on 10mm cube, got ${result.ny}`);
    assert(result.nz === 5, `Expected nz=5 with 2mm voxels on 10mm cube, got ${result.nz}`);
    assert(Math.abs(result.voxelSize - 2) < 0.001, `Voxel size should be 2mm, got ${result.voxelSize}`);
}

// ──────────────────────────────────────────────────
// Test 8: Transform vertices - scale
// ──────────────────────────────────────────────────
console.log('Test 8: Transform vertices - scale');
{
    const vertices = [
        { x: 1, y: 2, z: 3 },
        { x: 4, y: 5, z: 6 }
    ];
    const scaled = importer.transformVertices(vertices, 2, 0, 0, 0);
    assert(Math.abs(scaled[0].x - 2) < 1e-6, `Scaled x should be 2, got ${scaled[0].x}`);
    assert(Math.abs(scaled[0].y - 4) < 1e-6, `Scaled y should be 4, got ${scaled[0].y}`);
    assert(Math.abs(scaled[0].z - 6) < 1e-6, `Scaled z should be 6, got ${scaled[0].z}`);
    assert(Math.abs(scaled[1].x - 8) < 1e-6, `Scaled x should be 8, got ${scaled[1].x}`);
}

// ──────────────────────────────────────────────────
// Test 9: Transform vertices - identity (no change)
// ──────────────────────────────────────────────────
console.log('Test 9: Transform vertices - identity');
{
    const vertices = [
        { x: 1, y: 2, z: 3 }
    ];
    const result = importer.transformVertices(vertices, 1, 0, 0, 0);
    assert(result === vertices, 'Identity transform should return same array reference');
}

// ──────────────────────────────────────────────────
// Test 10: Transform vertices - rotation Z 90°
// ──────────────────────────────────────────────────
console.log('Test 10: Transform vertices - rotation Z 90°');
{
    const vertices = [
        { x: 1, y: 0, z: 0 }
    ];
    const rotated = importer.transformVertices(vertices, 1, 0, 0, 90);
    assert(Math.abs(rotated[0].x - 0) < 1e-6, `Rotated x should be ~0, got ${rotated[0].x}`);
    assert(Math.abs(rotated[0].y - 1) < 1e-6, `Rotated y should be ~1, got ${rotated[0].y}`);
    assert(Math.abs(rotated[0].z - 0) < 1e-6, `Rotated z should be ~0, got ${rotated[0].z}`);
}

// ──────────────────────────────────────────────────
// Test 11: Voxelization performance — spatial index produces same result
//   Large cube with many triangles should still voxelize correctly
// ──────────────────────────────────────────────────
console.log('Test 11: Spatial-index voxelization consistency (large mesh)');
{
    function quad(a, b, c, d) {
        return [a, b, c, a, c, d];
    }
    // Build subdivided cube faces for more triangles
    const cubeVertices = [];
    const subdivisions = 5; // 5x5 quads per face = 300 triangles total
    const size = 20;
    const step = size / subdivisions;

    // Generate +Z and -Z faces with subdivision
    for (let i = 0; i < subdivisions; i++) {
        for (let j = 0; j < subdivisions; j++) {
            const x0 = i * step, x1 = (i + 1) * step;
            const y0 = j * step, y1 = (j + 1) * step;
            // -Z face
            cubeVertices.push(...quad(
                { x: x0, y: y0, z: 0 }, { x: x1, y: y0, z: 0 },
                { x: x1, y: y1, z: 0 }, { x: x0, y: y1, z: 0 }
            ));
            // +Z face
            cubeVertices.push(...quad(
                { x: x0, y: y0, z: size }, { x: x0, y: y1, z: size },
                { x: x1, y: y1, z: size }, { x: x1, y: y0, z: size }
            ));
        }
    }
    // +X and -X faces
    for (let i = 0; i < subdivisions; i++) {
        for (let j = 0; j < subdivisions; j++) {
            const y0 = i * step, y1 = (i + 1) * step;
            const z0 = j * step, z1 = (j + 1) * step;
            cubeVertices.push(...quad(
                { x: 0, y: y0, z: z0 }, { x: 0, y: y1, z: z0 },
                { x: 0, y: y1, z: z1 }, { x: 0, y: y0, z: z1 }
            ));
            cubeVertices.push(...quad(
                { x: size, y: y0, z: z0 }, { x: size, y: y0, z: z1 },
                { x: size, y: y1, z: z1 }, { x: size, y: y1, z: z0 }
            ));
        }
    }
    // +Y and -Y faces
    for (let i = 0; i < subdivisions; i++) {
        for (let j = 0; j < subdivisions; j++) {
            const x0 = i * step, x1 = (i + 1) * step;
            const z0 = j * step, z1 = (j + 1) * step;
            cubeVertices.push(...quad(
                { x: x0, y: 0, z: z0 }, { x: x0, y: 0, z: z1 },
                { x: x1, y: 0, z: z1 }, { x: x1, y: 0, z: z0 }
            ));
            cubeVertices.push(...quad(
                { x: x0, y: size, z: z0 }, { x: x1, y: size, z: z0 },
                { x: x1, y: size, z: z1 }, { x: x0, y: size, z: z1 }
            ));
        }
    }

    const numTris = cubeVertices.length / 3;
    assert(numTris === 300, `Should have 300 triangles from subdivided cube, got ${numTris}`);

    const result = importer.voxelizeVertices(cubeVertices, null, 2);
    const { nx, ny, nz, elements } = result;
    assert(nx === 10, `Expected nx=10, got ${nx}`);
    assert(ny === 10, `Expected ny=10, got ${ny}`);
    assert(nz === 10, `Expected nz=10, got ${nz}`);

    let solidCount = 0;
    for (let i = 0; i < elements.length; i++) {
        if (elements[i] > 0.5) solidCount++;
    }
    assert(solidCount === 1000, `All 1000 voxels should be solid for subdivided cube, got ${solidCount}`);
}

// ──────────────────────────────────────────────────
// Test 12: Template scaling responds correctly to voxel size
//   Changing voxel size should change template dimensions proportionally
// ──────────────────────────────────────────────────
console.log('Test 12: Template scaling responds to voxel size');
{
    // Cube template: 50×50×50mm base, with 1mm voxels → 50×50×50
    const cube1mm = importer.createCubeTemplate(50);
    assert(cube1mm.nx === 50, `Cube at 1mm voxels (res=50) should be 50, got ${cube1mm.nx}`);

    // Cube template: 50×50×50mm base, with 2mm voxels → resolution=25 → 25×25×25
    const cube2mm = importer.createCubeTemplate(25);
    assert(cube2mm.nx === 25, `Cube at 2mm voxels (res=25) should be 25, got ${cube2mm.nx}`);

    // Cube template: 50×50×50mm base, with 5mm voxels → resolution=10 → 10×10×10
    const cube5mm = importer.createCubeTemplate(10);
    assert(cube5mm.nx === 10, `Cube at 5mm voxels (res=10) should be 10, got ${cube5mm.nx}`);

    // Beam template: 30×10×10mm base, with 1mm voxels → resolution=30 → 30×10×10
    const beam1mm = importer.createBeamTemplate(30);
    assert(beam1mm.nx === 30, `Beam at 1mm voxels (res=30) should have nx=30, got ${beam1mm.nx}`);
    assert(beam1mm.ny === 10, `Beam at 1mm voxels (res=30) should have ny=10, got ${beam1mm.ny}`);
    assert(beam1mm.nz === 10, `Beam at 1mm voxels (res=30) should have nz=10, got ${beam1mm.nz}`);

    // Beam template: 30×10×10mm base, with 2mm voxels → resolution=15 → 15×5×5
    const beam2mm = importer.createBeamTemplate(15);
    assert(beam2mm.nx === 15, `Beam at 2mm voxels (res=15) should have nx=15, got ${beam2mm.nx}`);
    assert(beam2mm.ny === 5, `Beam at 2mm voxels (res=15) should have ny=5, got ${beam2mm.ny}`);

    // Bridge template: 40×15×8mm base, with 2mm voxels → resolution=20 → 20×8×4
    const bridge2mm = importer.createBridgeTemplate(20);
    assert(bridge2mm.nx === 20, `Bridge at 2mm voxels (res=20) should have nx=20, got ${bridge2mm.nx}`);
    assert(bridge2mm.ny === 8, `Bridge at 2mm voxels (res=20) should have ny=8, got ${bridge2mm.ny}`);
    assert(bridge2mm.nz === 4, `Bridge at 2mm voxels (res=20) should have nz=4, got ${bridge2mm.nz}`);
}

// ──────────────────────────────────────────────────
// Test 13: TopologySolver library – environment detection
// ──────────────────────────────────────────────────
console.log('Test 13: TopologySolver library – environment detection');
{
    const info = TopologySolver.debug.detectEnvironment();
    assert(info.env === 'node', `env should be 'node', got '${info.env}'`);
    assert(info.workerType === 'worker_threads', `workerType should be 'worker_threads', got '${info.workerType}'`);

    const url2d = TopologySolver.debug.workerUrl('2d');
    const url3d = TopologySolver.debug.workerUrl('3d');
    assert(url2d.endsWith('optimizer-worker.js'), `2D worker URL should end with optimizer-worker.js`);
    assert(url3d.endsWith('optimizer-worker-3d.js'), `3D worker URL should end with optimizer-worker-3d.js`);
}

// ──────────────────────────────────────────────────
// Test 14: TopologySolver library – 2D optimization via worker_threads
// ──────────────────────────────────────────────────
console.log('Test 14: TopologySolver library – 2D optimization via Node.js worker_threads');
{
    const solver = new TopologySolver();
    const nx = 8, ny = 4, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: '2d',
        volumeFraction: 0.3,
        maxIterations: 3,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        penalStart: 1.5,
        continuationIters: 3,
        useProjection: false,
    };

    let progressCalled = false;
    const result = await solver.optimize(model, config, (iter) => { progressCalled = true; });

    assert(progressCalled, 'Progress callback should be called during optimization');
    assert(result.iterations >= 1, `Should complete at least 1 iteration, got ${result.iterations}`);
    assert(typeof result.finalCompliance === 'number' && result.finalCompliance > 0, `finalCompliance should be a positive number`);
    assert(result.densities instanceof Float32Array, 'result.densities should be a Float32Array');
}

// ──────────────────────────────────────────────────
// Test 15: TopologySolver library – solver selection ('auto' uses 3D for cube)
// ──────────────────────────────────────────────────
console.log('Test 15: TopologySolver library – solver selection auto/2d/3d');
{
    // 'auto' with nz>1 should pick 3D worker
    const url3d = TopologySolver.debug.workerUrl('3d');
    const url2d = TopologySolver.debug.workerUrl('2d');
    assert(url3d.includes('optimizer-worker-3d'), 'workerUrl(3d) should reference 3D worker');
    assert(url2d.includes('optimizer-worker.js'), 'workerUrl(2d) should reference 2D worker');
    assert(!url2d.includes('3d'), 'workerUrl(2d) should NOT contain "3d"');

    // Verify solver property is removed from workerConfig by running a quick 2D opt
    const solver2 = new TopologySolver();
    const nx = 6, ny = 3, nz = 1;
    const model = { nx, ny, nz, elements: new Float32Array(nx * ny).fill(1) };
    const result = await solver2.optimize(model, {
        solver: '2d', volumeFraction: 0.5, maxIterations: 2, penaltyFactor: 3,
        filterRadius: 0.9, forceDirection: 'down', forceMagnitude: 100,
        constraintPosition: 'left', useAMR: false, youngsModulus: 2.3,
        poissonsRatio: 0.35, useProjection: false,
    });
    assert(typeof result.finalCompliance === 'number', 'Optimization with solver:2d should succeed');
}

// ──────────────────────────────────────────────────
// Test 16: TopologySolver library – cancel
// ──────────────────────────────────────────────────
console.log('Test 16: TopologySolver library – cancel');
{
    const solver = new TopologySolver();
    const nx = 10, ny = 5, nz = 1;
    const model = { nx, ny, nz, elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: '2d', volumeFraction: 0.3, maxIterations: 100,
        penaltyFactor: 3, filterRadius: 0.9, forceDirection: 'down', forceMagnitude: 100,
        constraintPosition: 'left', useAMR: false, youngsModulus: 2.3,
        poissonsRatio: 0.35, useProjection: false,
    };

    let cancelled = false;
    const optimPromise = solver.optimize(model, config, (iter) => {
        // Cancel after first iteration
        if (iter === 1) solver.cancel();
    });

    try {
        await optimPromise;
    } catch (err) {
        cancelled = err.message === 'Optimization cancelled';
    }
    assert(cancelled, 'Cancelling optimization should reject the promise with "Optimization cancelled"');
}

// ──────────────────────────────────────────────────
// Test 17: TopologySolver library – preventVoids parameter
// ──────────────────────────────────────────────────
console.log('Test 17: TopologySolver library – preventVoids parameter');
{
    const solver = new TopologySolver();
    const nx = 8, ny = 4, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: '2d',
        volumeFraction: 0.3,
        maxIterations: 3,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        penalStart: 1.5,
        continuationIters: 3,
        useProjection: false,
        preventVoids: true,
    };

    const result = await solver.optimize(model, config, () => {});

    assert(result.iterations >= 1, `preventVoids: should complete at least 1 iteration, got ${result.iterations}`);
    assert(typeof result.finalCompliance === 'number' && result.finalCompliance > 0, 'preventVoids: finalCompliance should be a positive number');
    assert(result.densities instanceof Float32Array, 'preventVoids: result.densities should be a Float32Array');

    // Verify no internal voids: flood-fill from boundary voids and check
    // that all low-density elements are reachable from the boundary
    const densities = result.densities;
    const threshold = 0.3;
    // 2D worker uses column-major: ey + ex * ny
    const visited = new Uint8Array(nx * ny);
    const queue = [];
    for (let ex = 0; ex < nx; ex++) {
        for (let ey = 0; ey < ny; ey++) {
            if (ex === 0 || ex === nx - 1 || ey === 0 || ey === ny - 1) {
                const idx = ey + ex * ny;
                if (densities[idx] < threshold) {
                    visited[idx] = 1;
                    queue.push(idx);
                }
            }
        }
    }
    while (queue.length > 0) {
        const idx = queue.pop();
        const ex = Math.floor(idx / ny);
        const ey = idx % ny;
        const neighbors = [
            ey > 0 ? (ey - 1) + ex * ny : -1,
            ey < ny - 1 ? (ey + 1) + ex * ny : -1,
            ex > 0 ? ey + (ex - 1) * ny : -1,
            ex < nx - 1 ? ey + (ex + 1) * ny : -1,
        ];
        for (const ni of neighbors) {
            if (ni >= 0 && !visited[ni] && densities[ni] < threshold) {
                visited[ni] = 1;
                queue.push(ni);
            }
        }
    }
    let internalVoids = 0;
    for (let i = 0; i < nx * ny; i++) {
        if (densities[i] < threshold && !visited[i]) internalVoids++;
    }
    assert(internalVoids === 0, `preventVoids: should have no internal voids, found ${internalVoids}`);
}

// ──────────────────────────────────────────────────
// Test 18: Manufacturing constraint (90°) – no enclosed cavities
//   At 90° every void element must be reachable from the top surface (ey=0)
// ──────────────────────────────────────────────────
console.log('Test 18: Manufacturing constraint (90°) – no enclosed cavities');
{
    const solver = new TopologySolver();
    const nx = 10, ny = 5, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: '2d',
        volumeFraction: 0.4,
        maxIterations: 5,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        useProjection: false,
        manufacturingConstraint: true,
        manufacturingAngle: 90,
    };

    const result = await solver.optimize(model, config, () => {});

    assert(result.iterations >= 1, `mfg90: should complete at least 1 iteration, got ${result.iterations}`);
    assert(result.densities instanceof Float32Array, 'mfg90: result.densities should be a Float32Array');

    // Verify: every void element should be reachable from the top row (ey=0)
    // using a top-to-bottom flood fill through void elements (strict vertical at 90°)
    // Returned densities use x-major indexing: idx = ex + ey * nx (nz=1)
    const densities = result.densities;
    const threshold = 0.3;
    const accessible = new Uint8Array(nx * ny);
    // Seed: void elements on top row (ey=0)
    for (let ex = 0; ex < nx; ex++) {
        const idx = ex; // ey=0, x-major
        if (densities[idx] < threshold) accessible[idx] = 1;
    }
    // Sweep top-to-bottom: at 90° only check directly above (dx=0)
    for (let ey = 1; ey < ny; ey++) {
        for (let ex = 0; ex < nx; ex++) {
            const idx = ex + ey * nx;
            if (densities[idx] >= threshold) continue;
            const aboveIdx = ex + (ey - 1) * nx;
            if (accessible[aboveIdx]) accessible[idx] = 1;
        }
    }
    let enclosedCavities = 0;
    for (let i = 0; i < nx * ny; i++) {
        if (densities[i] < threshold && !accessible[i]) enclosedCavities++;
    }
    assert(enclosedCavities === 0, `mfg90: should have no enclosed cavities, found ${enclosedCavities}`);
}

// ──────────────────────────────────────────────────
// Test 19: Manufacturing max depth constraint
//   No void element should exist at or beyond the max depth
// ──────────────────────────────────────────────────
console.log('Test 19: Manufacturing max depth constraint');
{
    const solver = new TopologySolver();
    const nx = 8, ny = 6, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const maxDepth = 3;
    const config = {
        solver: '2d',
        volumeFraction: 0.4,
        maxIterations: 5,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        useProjection: false,
        manufacturingConstraint: true,
        manufacturingAngle: 90,
        manufacturingMaxDepth: maxDepth,
    };

    const result = await solver.optimize(model, config, () => {});
    assert(result.iterations >= 1, `maxDepth: should complete at least 1 iteration`);

    const densities = result.densities;
    const threshold = 0.3;
    let deepVoids = 0;
    // Returned densities use x-major indexing: idx = ex + ey * nx (nz=1)
    for (let ex = 0; ex < nx; ex++) {
        for (let ey = maxDepth; ey < ny; ey++) {
            const idx = ex + ey * nx;
            if (densities[idx] < threshold) deepVoids++;
        }
    }
    assert(deepVoids === 0, `maxDepth: no voids should exist at depth >= ${maxDepth}, found ${deepVoids}`);
}

// ──────────────────────────────────────────────────
// Test 20: Manufacturing constraint with min radius completes successfully
// ──────────────────────────────────────────────────
console.log('Test 20: Manufacturing constraint with min radius');
{
    const solver = new TopologySolver();
    const nx = 8, ny = 4, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: '2d',
        volumeFraction: 0.4,
        maxIterations: 3,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        useProjection: false,
        manufacturingConstraint: true,
        manufacturingAngle: 90,
        manufacturingMinRadius: 1,
    };

    const result = await solver.optimize(model, config, () => {});
    assert(result.iterations >= 1, `minRadius: should complete at least 1 iteration, got ${result.iterations}`);
    assert(typeof result.finalCompliance === 'number' && result.finalCompliance > 0, 'minRadius: finalCompliance should be a positive number');
    assert(result.densities instanceof Float32Array, 'minRadius: result.densities should be a Float32Array');
}

// ──────────────────────────────────────────────────
// Test 21: paintedKeep faces are preserved during optimization
//   Voxels marked as "keep" should remain solid (density ≥ 0.99)
// ──────────────────────────────────────────────────
console.log('Test 21: paintedKeep faces preserve voxels during optimization');
{
    const solver = new TopologySolver();
    const nx = 8, ny = 4, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    // Mark some interior voxels as "keep" (these would normally be optimized away at low volfrac)
    const keepKeys = ['3,1,0,3', '4,1,0,3', '3,2,0,2', '4,2,0,2'];
    const config = {
        solver: '2d',
        volumeFraction: 0.1,
        maxIterations: 5,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        useProjection: false,
        paintedKeep: keepKeys,
    };

    const result = await solver.optimize(model, config, () => {});
    assert(result.iterations >= 1, `keepFaces: should complete at least 1 iteration, got ${result.iterations}`);
    assert(result.densities instanceof Float32Array, 'keepFaces: result.densities should be a Float32Array');

    // Verify that preserved voxels are at full density (result uses row-major 3D indexing: x + y * nx)
    let allPreserved = true;
    for (const key of keepKeys) {
        const parts = key.split(',');
        const vx = parseInt(parts[0], 10);
        const vy = parseInt(parts[1], 10);
        const idx = vx + vy * nx;
        if (result.densities[idx] < 0.99) {
            allPreserved = false;
            break;
        }
    }
    assert(allPreserved, 'keepFaces: all keep-marked voxels should remain at full density');
}

// ──────────────────────────────────────────────────
// Test 22: Manufacturing constraint respects preserved elements
//   Preserved voxels should not be zeroed by overhang constraint
// ──────────────────────────────────────────────────
console.log('Test 22: Manufacturing constraint respects preserved elements');
{
    const solver = new TopologySolver();
    const nx = 6, ny = 4, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    // Mark constraint voxels on the left edge (these are preserved)
    const constraintKeys = ['0,0,0,0', '0,1,0,0', '0,2,0,0', '0,3,0,0'];
    const config = {
        solver: '2d',
        volumeFraction: 0.4,
        maxIterations: 5,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        paintedConstraints: constraintKeys,
        useAMR: false,
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        useProjection: false,
        manufacturingConstraint: true,
        manufacturingAngle: 90,
    };

    const result = await solver.optimize(model, config, () => {});
    assert(result.iterations >= 1, `mfgPreserved: should complete at least 1 iteration, got ${result.iterations}`);

    // Verify constraint voxels remain solid despite manufacturing constraint (row-major 3D indexing)
    let constraintsSolid = true;
    for (const key of constraintKeys) {
        const parts = key.split(',');
        const vx = parseInt(parts[0], 10);
        const vy = parseInt(parts[1], 10);
        const idx = vx + vy * nx;
        if (result.densities[idx] < 0.99) {
            constraintsSolid = false;
            break;
        }
    }
    assert(constraintsSolid, 'mfgPreserved: constraint voxels should remain solid with manufacturing constraints enabled');
}

// ──────────────────────────────────────────────────
// Test 23: FEA-only mode – single analysis returns stress data
// ──────────────────────────────────────────────────
console.log('Test 23: FEA-only mode – single analysis returns stress data');
{
    const solver = new TopologySolver();
    const nx = 8, ny = 4, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: 'fea',
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
    };

    const result = await solver.optimize(model, config);

    assert(result.feaOnly === true, 'FEA-only result should have feaOnly=true');
    assert(result.iterations === 0, `FEA-only should have 0 iterations, got ${result.iterations}`);
    assert(typeof result.finalCompliance === 'number' && result.finalCompliance > 0, 'FEA-only: finalCompliance should be a positive number');
    assert(result.densities instanceof Float32Array, 'FEA-only: result.densities should be a Float32Array');
    assert(typeof result.maxStress === 'number' && result.maxStress > 0, 'FEA-only: maxStress should be a positive number');
    assert(result.elementStress instanceof Float32Array, 'FEA-only: elementStress should be a Float32Array');
    assert(result.elementStress.length === nx * ny, `FEA-only: elementStress length should be ${nx * ny}, got ${result.elementStress.length}`);
    assert(result.meshData != null, 'FEA-only: meshData should not be null');
}

// ──────────────────────────────────────────────────
// Test 24: analyzeFEA convenience method
// ──────────────────────────────────────────────────
console.log('Test 24: analyzeFEA convenience method');
{
    const solver = new TopologySolver();
    const nx = 6, ny = 3, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
    };

    const result = await solver.analyzeFEA(model, config);

    assert(result.feaOnly === true, 'analyzeFEA result should have feaOnly=true');
    assert(result.iterations === 0, `analyzeFEA should have 0 iterations, got ${result.iterations}`);
    assert(typeof result.finalCompliance === 'number' && result.finalCompliance > 0, 'analyzeFEA: finalCompliance should be a positive number');
    assert(result.elementStress instanceof Float32Array, 'analyzeFEA: elementStress should be a Float32Array');
}

// ──────────────────────────────────────────────────
// Test 25: Genetic optimization – completes and returns valid result
// ──────────────────────────────────────────────────
console.log('Test 25: Genetic optimization – completes and returns valid result');
{
    const solver = new TopologySolver();
    const nx = 6, ny = 3, nz = 1;
    const model = { nx, ny, nz, type: 'beam', elements: new Float32Array(nx * ny).fill(1) };
    const config = {
        solver: 'genetic',
        volumeFraction: 0.5,
        maxIterations: 3,
        penaltyFactor: 3,
        filterRadius: 0.9,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        youngsModulus: 2.3,
        poissonsRatio: 0.35,
        populationSize: 6,
        eliteCount: 2,
        mutationRate: 0.05,
        crossoverRate: 0.8,
        tournamentSize: 2,
    };

    let progressCalled = false;
    const result = await solver.optimize(model, config, (iter) => { progressCalled = true; });

    assert(progressCalled, 'Genetic: progress callback should be called');
    assert(result.geneticOptimization === true, 'Genetic: result should have geneticOptimization=true');
    assert(result.iterations === 3, `Genetic: should complete 3 generations, got ${result.iterations}`);
    assert(typeof result.finalCompliance === 'number' && result.finalCompliance > 0, 'Genetic: finalCompliance should be a positive number');
    assert(result.densities instanceof Float32Array, 'Genetic: result.densities should be a Float32Array');
    assert(result.densities.length === nx * ny * nz, `Genetic: densities length should be ${nx * ny * nz}, got ${result.densities.length}`);
    assert(result.meshData != null, 'Genetic: meshData should not be null');

    // Verify densities are binary (0 or 1) since genetic uses binary representation
    let allBinary = true;
    for (let i = 0; i < result.densities.length; i++) {
        if (result.densities[i] !== 0.0 && result.densities[i] !== 1.0) {
            allBinary = false;
            break;
        }
    }
    assert(allBinary, 'Genetic: densities should be binary (0 or 1)');
}

// ──────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
