// Voxelization tests for ModelImporter
// Run with: node tests/index.js

// Provide a minimal FileReader stub so ModelImporter can be instantiated in Node.js
globalThis.FileReader = class FileReader {};

import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toURL = (p) => pathToFileURL(p).href;

const { ModelImporter } = await import(toURL(join(__dirname, '..', 'js', 'importer.js')));
const { STEPParser } = await import(toURL(join(__dirname, '..', 'js', 'step-parser.js')));
const { TopologySolver } = await import(toURL(join(__dirname, '..', 'lib', 'topology-solver.js')));
const { NonlinearSolver } = await import(toURL(join(__dirname, '..', 'js', 'nonlinear-solver.js')));
const { createMaterial } = await import(toURL(join(__dirname, '..', 'js', 'material-models.js')));

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
// Test 26: 3D cube with top force, bottom Y-constraint, and manufacturing
//   constraint (90°) – no internal cavities
//   Simulates a CNC-milled box: force on +Y (top) faces, Y-only constraint
//   on -Y (bottom) faces, 80% volume fraction, 3-axis CNC at 90°.
//   All void elements must be reachable from the top surface (ey=nely-1).
// ──────────────────────────────────────────────────
console.log('Test 26: 3D cube – top force, bottom Y-constraint, mfg 90° – no internal cavities');
{
    const solver = new TopologySolver();
    const nx = 4, ny = 4, nz = 4;
    const nel = nx * ny * nz;
    const model = { nx, ny, nz, type: 'cube', elements: new Float32Array(nel).fill(1) };

    // Paint all top faces (+Y face = fi=3) on the top layer (ey=ny-1)
    const paintedForces = [];
    for (let ez = 0; ez < nz; ez++) {
        for (let ex = 0; ex < nx; ex++) {
            paintedForces.push(`${ex},${ny - 1},${ez},3`);
        }
    }

    // Paint all bottom faces (-Y face = fi=2) on the bottom layer (ey=0)
    const paintedConstraints = [];
    for (let ez = 0; ez < nz; ez++) {
        for (let ex = 0; ex < nx; ex++) {
            paintedConstraints.push(`${ex},0,${ez},2`);
        }
    }

    const config = {
        solver: '3d',
        volumeFraction: 0.8,
        maxIterations: 5,
        penaltyFactor: 3,
        filterRadius: 1.2,
        forceDirection: 'down',
        forceMagnitude: 1000,
        constraintPosition: 'bottom',
        constraintDOFs: 'y',
        paintedForces,
        paintedConstraints,
        useAMR: false,
        youngsModulus: 200,
        poissonsRatio: 0.3,
        useProjection: false,
        manufacturingConstraint: true,
        manufacturingAngle: 90,
    };

    const result = await solver.optimize(model, config, () => {});

    assert(result.iterations >= 1, `3dMfg: should complete at least 1 iteration, got ${result.iterations}`);
    assert(result.densities instanceof Float32Array, '3dMfg: result.densities should be a Float32Array');
    assert(result.densities.length === nel, `3dMfg: densities length should be ${nel}, got ${result.densities.length}`);

    // Verify: every void element must be reachable from the top surface (ey=ny-1)
    // using a top-to-bottom flood through void elements (strict vertical at 90°, span=0)
    // x-major indexing: idx = ex + ey * nx + ez * nx * ny
    const densities = result.densities;
    const threshold = 0.3;
    const accessible = new Uint8Array(nel);

    // Seed: void elements on top layer (ey=ny-1)
    for (let ez = 0; ez < nz; ez++) {
        for (let ex = 0; ex < nx; ex++) {
            const idx = ex + (ny - 1) * nx + ez * nx * ny;
            if (densities[idx] < threshold) accessible[idx] = 1;
        }
    }
    // Sweep top-to-bottom: at 90° span=0 so only the element directly above (ey+1) is checked
    for (let ey = ny - 2; ey >= 0; ey--) {
        for (let ez = 0; ez < nz; ez++) {
            for (let ex = 0; ex < nx; ex++) {
                const idx = ex + ey * nx + ez * nx * ny;
                if (densities[idx] >= threshold) continue;
                const aboveIdx = ex + (ey + 1) * nx + ez * nx * ny;
                if (accessible[aboveIdx]) accessible[idx] = 1;
            }
        }
    }
    let enclosedCavities = 0;
    for (let i = 0; i < nel; i++) {
        if (densities[i] < threshold && !accessible[i]) enclosedCavities++;
    }
    assert(enclosedCavities === 0, `3dMfg: should have no enclosed cavities, found ${enclosedCavities}`);
}

// ──────────────────────────────────────────────────
// Test 27: 3D manufacturing max depth constraint counts from top (ey=nely-1)
//   Verifies that the depth limit is measured from the physical top surface,
//   not from ey=0.
// ──────────────────────────────────────────────────
console.log('Test 27: 3D manufacturing max depth counts from top (ey=nely-1)');
{
    const solver = new TopologySolver();
    const nx = 4, ny = 6, nz = 4;
    const nel = nx * ny * nz;
    const model = { nx, ny, nz, type: 'cube', elements: new Float32Array(nel).fill(1) };
    const maxDepth = 3;

    // Paint all top faces (+Y, fi=3) for force
    const paintedForces = [];
    for (let ez = 0; ez < nz; ez++) {
        for (let ex = 0; ex < nx; ex++) {
            paintedForces.push(`${ex},${ny - 1},${ez},3`);
        }
    }

    // Paint all bottom faces (-Y, fi=2) for constraints
    const paintedConstraints = [];
    for (let ez = 0; ez < nz; ez++) {
        for (let ex = 0; ex < nx; ex++) {
            paintedConstraints.push(`${ex},0,${ez},2`);
        }
    }

    const config = {
        solver: '3d',
        volumeFraction: 0.5,
        maxIterations: 5,
        penaltyFactor: 3,
        filterRadius: 1.2,
        forceDirection: 'down',
        forceMagnitude: 1000,
        constraintPosition: 'bottom',
        constraintDOFs: 'y',
        paintedForces,
        paintedConstraints,
        useAMR: false,
        youngsModulus: 200,
        poissonsRatio: 0.3,
        useProjection: false,
        manufacturingConstraint: true,
        manufacturingAngle: 90,
        manufacturingMaxDepth: maxDepth,
    };

    const result = await solver.optimize(model, config, () => {});
    assert(result.iterations >= 1, `3dMaxDepth: should complete at least 1 iteration`);

    // Verify: no voids below the max depth from the top
    // Top = ey=ny-1. Tool reaches the top maxDepth layers.
    // Elements at ey < ny - maxDepth should all be solid
    const densities = result.densities;
    const threshold = 0.3;
    let deepVoids = 0;
    for (let ey = 0; ey < ny - maxDepth; ey++) {
        for (let ez = 0; ez < nz; ez++) {
            for (let ex = 0; ex < nx; ex++) {
                const idx = ex + ey * nx + ez * nx * ny;
                if (densities[idx] < threshold) deepVoids++;
            }
        }
    }
    assert(deepVoids === 0, `3dMaxDepth: no voids below depth ${maxDepth} from top, found ${deepVoids}`);
}

// ──────────────────────────────────────────────────
// Test 28: STEP parser – protocol detection (AP203)
// ──────────────────────────────────────────────────
console.log('Test 28: STEP parser – protocol detection (AP203)');
{
    const parser = new STEPParser();
    const stepText = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('test.stp', '2024-01-01', (''), (''), '', '', '');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;`;
    const result = parser._detectProtocol(stepText);
    assert(result === 'AP203', `AP203 protocol should be detected from CONFIG_CONTROL_DESIGN, got '${result}'`);
}

// ──────────────────────────────────────────────────
// Test 29: STEP parser – protocol detection (AP214)
// ──────────────────────────────────────────────────
console.log('Test 29: STEP parser – protocol detection (AP214)');
{
    const parser = new STEPParser();
    const stepText = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('test.stp', '2024-01-01', (''), (''), '', '', '');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;`;
    const result = parser._detectProtocol(stepText);
    assert(result === 'AP214', `AP214 protocol should be detected from AUTOMOTIVE_DESIGN, got '${result}'`);
}

// ──────────────────────────────────────────────────
// Test 30: STEP parser – parse cube B-Rep (AP203)
//   A 10×10×10 cube described as STEP B-Rep should produce triangles
// ──────────────────────────────────────────────────
console.log('Test 30: STEP parser – parse cube B-Rep (AP203)');
{
    // Minimal STEP file describing a cube [0,10]^3 with 6 planar faces
    const stepText = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('cube.stp', '2024-01-01', (''), (''), '', 'AP203', '');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('Origin', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = CARTESIAN_POINT('', (0.0, 0.0, 10.0));
#6 = CARTESIAN_POINT('', (10.0, 0.0, 10.0));
#7 = CARTESIAN_POINT('', (10.0, 10.0, 10.0));
#8 = CARTESIAN_POINT('', (0.0, 10.0, 10.0));
#10 = DIRECTION('', (0.0, 0.0, 1.0));
#11 = DIRECTION('', (1.0, 0.0, 0.0));
#12 = DIRECTION('', (0.0, 1.0, 0.0));
#13 = DIRECTION('', (0.0, 0.0, -1.0));
#14 = DIRECTION('', (-1.0, 0.0, 0.0));
#15 = DIRECTION('', (0.0, -1.0, 0.0));
#20 = VERTEX_POINT('', #1);
#21 = VERTEX_POINT('', #2);
#22 = VERTEX_POINT('', #3);
#23 = VERTEX_POINT('', #4);
#24 = VERTEX_POINT('', #5);
#25 = VERTEX_POINT('', #6);
#26 = VERTEX_POINT('', #7);
#27 = VERTEX_POINT('', #8);
#30 = LINE('', #1, #40);
#31 = LINE('', #2, #41);
#32 = LINE('', #3, #42);
#33 = LINE('', #4, #43);
#34 = LINE('', #1, #44);
#35 = LINE('', #2, #45);
#36 = LINE('', #3, #46);
#37 = LINE('', #4, #47);
#38 = LINE('', #5, #48);
#39 = LINE('', #6, #49);
#40 = VECTOR('', #11, 10.0);
#41 = VECTOR('', #12, 10.0);
#42 = VECTOR('', #14, 10.0);
#43 = VECTOR('', #15, 10.0);
#44 = VECTOR('', #10, 10.0);
#45 = VECTOR('', #10, 10.0);
#46 = VECTOR('', #10, 10.0);
#47 = VECTOR('', #10, 10.0);
#48 = VECTOR('', #11, 10.0);
#49 = VECTOR('', #12, 10.0);
#50 = EDGE_CURVE('', #20, #21, #30, .T.);
#51 = EDGE_CURVE('', #21, #22, #31, .T.);
#52 = EDGE_CURVE('', #22, #23, #32, .T.);
#53 = EDGE_CURVE('', #23, #20, #33, .T.);
#54 = EDGE_CURVE('', #20, #24, #34, .T.);
#55 = EDGE_CURVE('', #21, #25, #35, .T.);
#56 = EDGE_CURVE('', #22, #26, #36, .T.);
#57 = EDGE_CURVE('', #23, #27, #37, .T.);
#58 = EDGE_CURVE('', #24, #25, #38, .T.);
#59 = EDGE_CURVE('', #25, #26, #39, .T.);
#60 = EDGE_CURVE('', #26, #27, #32, .T.);
#61 = EDGE_CURVE('', #27, #24, #33, .T.);
#70 = ORIENTED_EDGE('', *, *, #50, .T.);
#71 = ORIENTED_EDGE('', *, *, #51, .T.);
#72 = ORIENTED_EDGE('', *, *, #52, .T.);
#73 = ORIENTED_EDGE('', *, *, #53, .T.);
#74 = ORIENTED_EDGE('', *, *, #50, .F.);
#75 = ORIENTED_EDGE('', *, *, #54, .T.);
#76 = ORIENTED_EDGE('', *, *, #58, .T.);
#77 = ORIENTED_EDGE('', *, *, #55, .F.);
#78 = ORIENTED_EDGE('', *, *, #51, .F.);
#79 = ORIENTED_EDGE('', *, *, #55, .T.);
#80 = ORIENTED_EDGE('', *, *, #59, .T.);
#81 = ORIENTED_EDGE('', *, *, #56, .F.);
#82 = ORIENTED_EDGE('', *, *, #52, .F.);
#83 = ORIENTED_EDGE('', *, *, #56, .T.);
#84 = ORIENTED_EDGE('', *, *, #60, .T.);
#85 = ORIENTED_EDGE('', *, *, #57, .F.);
#86 = ORIENTED_EDGE('', *, *, #53, .F.);
#87 = ORIENTED_EDGE('', *, *, #57, .T.);
#88 = ORIENTED_EDGE('', *, *, #61, .T.);
#89 = ORIENTED_EDGE('', *, *, #54, .F.);
#90 = ORIENTED_EDGE('', *, *, #58, .T.);
#91 = ORIENTED_EDGE('', *, *, #59, .T.);
#92 = ORIENTED_EDGE('', *, *, #60, .T.);
#93 = ORIENTED_EDGE('', *, *, #61, .T.);
#100 = EDGE_LOOP('', (#70, #71, #72, #73));
#101 = EDGE_LOOP('', (#74, #75, #76, #77));
#102 = EDGE_LOOP('', (#78, #79, #80, #81));
#103 = EDGE_LOOP('', (#82, #83, #84, #85));
#104 = EDGE_LOOP('', (#86, #87, #88, #89));
#105 = EDGE_LOOP('', (#90, #91, #92, #93));
#110 = FACE_OUTER_BOUND('', #100, .T.);
#111 = FACE_OUTER_BOUND('', #101, .T.);
#112 = FACE_OUTER_BOUND('', #102, .T.);
#113 = FACE_OUTER_BOUND('', #103, .T.);
#114 = FACE_OUTER_BOUND('', #104, .T.);
#115 = FACE_OUTER_BOUND('', #105, .T.);
#120 = AXIS2_PLACEMENT_3D('', #1, #13, #11);
#121 = AXIS2_PLACEMENT_3D('', #1, #15, #11);
#122 = AXIS2_PLACEMENT_3D('', #2, #11, #12);
#123 = AXIS2_PLACEMENT_3D('', #3, #12, #14);
#124 = AXIS2_PLACEMENT_3D('', #4, #14, #15);
#125 = AXIS2_PLACEMENT_3D('', #5, #10, #11);
#130 = PLANE('', #120);
#131 = PLANE('', #121);
#132 = PLANE('', #122);
#133 = PLANE('', #123);
#134 = PLANE('', #124);
#135 = PLANE('', #125);
#140 = ADVANCED_FACE('', (#110), #130, .T.);
#141 = ADVANCED_FACE('', (#111), #131, .T.);
#142 = ADVANCED_FACE('', (#112), #132, .T.);
#143 = ADVANCED_FACE('', (#113), #133, .T.);
#144 = ADVANCED_FACE('', (#114), #134, .T.);
#145 = ADVANCED_FACE('', (#115), #135, .T.);
#150 = CLOSED_SHELL('', (#140, #141, #142, #143, #144, #145));
ENDSEC;
END-ISO-10303-21;`;

    const parser = new STEPParser();
    const result = parser.parse(stepText);

    assert(result.protocol === 'AP203', `Protocol should be AP203, got '${result.protocol}'`);
    assert(result.vertices.length > 0, `Should produce triangles from cube, got ${result.vertices.length} vertices`);
    assert(result.vertices.length % 3 === 0, `Vertex count should be multiple of 3 (triangles), got ${result.vertices.length}`);

    const numTriangles = result.vertices.length / 3;
    // A cube has 6 faces × 4 vertices per face → 6 × 2 triangles = 12 triangles minimum
    assert(numTriangles >= 12, `Should have at least 12 triangles for a cube, got ${numTriangles}`);

    // Verify vertices are in the expected range [0, 10]
    let allInRange = true;
    for (const v of result.vertices) {
        if (v.x < -0.1 || v.x > 10.1 || v.y < -0.1 || v.y > 10.1 || v.z < -0.1 || v.z > 10.1) {
            allInRange = false;
            break;
        }
    }
    assert(allInRange, 'All vertices should be within [0, 10] range for cube');
}

// ──────────────────────────────────────────────────
// Test 31: STEP parser – isSTEP static method
// ──────────────────────────────────────────────────
console.log('Test 31: STEP parser – isSTEP detection');
{
    assert(STEPParser.isSTEP('ISO-10303-21; HEADER; DATA; ENDSEC;') === true, 'Should detect STEP from ISO-10303-21');
    assert(STEPParser.isSTEP('solid cube\nfacet normal') === false, 'Should not detect STL as STEP');
    assert(STEPParser.isSTEP('FILE_SCHEMA(("AP203")); DATA;') === true, 'Should detect STEP from FILE_SCHEMA + DATA');
}

// ──────────────────────────────────────────────────
// Test 32: STEP parser – argument splitting
// ──────────────────────────────────────────────────
console.log('Test 32: STEP parser – argument splitting');
{
    const parser = new STEPParser();
    const args1 = parser._splitArgs("'label', (1.0, 2.0, 3.0)");
    assert(args1.length === 2, `Should split into 2 args, got ${args1.length}`);
    assert(args1[0] === "'label'", `First arg should be the label, got '${args1[0]}'`);
    assert(args1[1] === '(1.0, 2.0, 3.0)', `Second arg should be the coord list, got '${args1[1]}'`);

    const args2 = parser._splitArgs("'', (#1, #2, #3), #4, .T.");
    assert(args2.length === 4, `Should split into 4 args, got ${args2.length}`);
    assert(args2[1] === '(#1, #2, #3)', `Second arg should be ref list, got '${args2[1]}'`);
    assert(args2[3] === '.T.', `Fourth arg should be .T., got '${args2[3]}'`);
}

// ──────────────────────────────────────────────────
// Test 33: STEP parser – CARTESIAN_POINT parsing
// ──────────────────────────────────────────────────
console.log('Test 33: STEP parser – CARTESIAN_POINT parsing');
{
    const parser = new STEPParser();
    parser.rawEntities[1] = { type: 'CARTESIAN_POINT', args: "'Origin', (5.5, 3.2, 7.1)" };
    const pt = parser._resolve(1);
    assert(pt.type === 'CARTESIAN_POINT', `Type should be CARTESIAN_POINT, got '${pt.type}'`);
    assert(Math.abs(pt.x - 5.5) < 1e-6, `x should be 5.5, got ${pt.x}`);
    assert(Math.abs(pt.y - 3.2) < 1e-6, `y should be 3.2, got ${pt.y}`);
    assert(Math.abs(pt.z - 7.1) < 1e-6, `z should be 7.1, got ${pt.z}`);
}

// ──────────────────────────────────────────────────
// Test 34: STEP parser – AP214 protocol detection with schema
// ──────────────────────────────────────────────────
console.log('Test 34: STEP parser – AP214 with explicit schema identifier');
{
    const stepText = `ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('test.stp', '2024-01-01', (''), (''), '', '', '');
FILE_SCHEMA(('AP214IS'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;`;

    const parser = new STEPParser();
    const protocol = parser._detectProtocol(stepText);
    assert(protocol === 'AP214', `Should detect AP214 from AP214IS, got '${protocol}'`);
}

// ──────────────────────────────────────────────────
// Test 35: ModelImporter.parseSTEP – voxelizes STEP cube
// ──────────────────────────────────────────────────
console.log('Test 35: ModelImporter.parseSTEP – voxelizes STEP cube');
{
    // Minimal STEP cube with a single planar face (a triangle)
    const stepText = `ISO-10303-21;
HEADER;
FILE_SCHEMA(('AP203'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = CARTESIAN_POINT('', (10.0, 0.0, 0.0));
#3 = CARTESIAN_POINT('', (10.0, 10.0, 0.0));
#4 = CARTESIAN_POINT('', (0.0, 10.0, 0.0));
#5 = CARTESIAN_POINT('', (0.0, 0.0, 10.0));
#6 = CARTESIAN_POINT('', (10.0, 0.0, 10.0));
#7 = CARTESIAN_POINT('', (10.0, 10.0, 10.0));
#8 = CARTESIAN_POINT('', (0.0, 10.0, 10.0));
#10 = DIRECTION('', (0.0, 0.0, 1.0));
#11 = DIRECTION('', (1.0, 0.0, 0.0));
#12 = DIRECTION('', (0.0, 1.0, 0.0));
#13 = DIRECTION('', (0.0, 0.0, -1.0));
#14 = DIRECTION('', (-1.0, 0.0, 0.0));
#15 = DIRECTION('', (0.0, -1.0, 0.0));
#20 = VERTEX_POINT('', #1);
#21 = VERTEX_POINT('', #2);
#22 = VERTEX_POINT('', #3);
#23 = VERTEX_POINT('', #4);
#24 = VERTEX_POINT('', #5);
#25 = VERTEX_POINT('', #6);
#26 = VERTEX_POINT('', #7);
#27 = VERTEX_POINT('', #8);
#30 = LINE('', #1, #40);
#31 = LINE('', #2, #41);
#32 = LINE('', #3, #42);
#33 = LINE('', #4, #43);
#34 = LINE('', #1, #44);
#35 = LINE('', #2, #45);
#36 = LINE('', #3, #46);
#37 = LINE('', #4, #47);
#38 = LINE('', #5, #48);
#39 = LINE('', #6, #49);
#40 = VECTOR('', #11, 10.0);
#41 = VECTOR('', #12, 10.0);
#42 = VECTOR('', #14, 10.0);
#43 = VECTOR('', #15, 10.0);
#44 = VECTOR('', #10, 10.0);
#45 = VECTOR('', #10, 10.0);
#46 = VECTOR('', #10, 10.0);
#47 = VECTOR('', #10, 10.0);
#48 = VECTOR('', #11, 10.0);
#49 = VECTOR('', #12, 10.0);
#50 = EDGE_CURVE('', #20, #21, #30, .T.);
#51 = EDGE_CURVE('', #21, #22, #31, .T.);
#52 = EDGE_CURVE('', #22, #23, #32, .T.);
#53 = EDGE_CURVE('', #23, #20, #33, .T.);
#54 = EDGE_CURVE('', #20, #24, #34, .T.);
#55 = EDGE_CURVE('', #21, #25, #35, .T.);
#56 = EDGE_CURVE('', #22, #26, #36, .T.);
#57 = EDGE_CURVE('', #23, #27, #37, .T.);
#58 = EDGE_CURVE('', #24, #25, #38, .T.);
#59 = EDGE_CURVE('', #25, #26, #39, .T.);
#60 = EDGE_CURVE('', #26, #27, #32, .T.);
#61 = EDGE_CURVE('', #27, #24, #33, .T.);
#70 = ORIENTED_EDGE('', *, *, #50, .T.);
#71 = ORIENTED_EDGE('', *, *, #51, .T.);
#72 = ORIENTED_EDGE('', *, *, #52, .T.);
#73 = ORIENTED_EDGE('', *, *, #53, .T.);
#74 = ORIENTED_EDGE('', *, *, #50, .F.);
#75 = ORIENTED_EDGE('', *, *, #54, .T.);
#76 = ORIENTED_EDGE('', *, *, #58, .T.);
#77 = ORIENTED_EDGE('', *, *, #55, .F.);
#78 = ORIENTED_EDGE('', *, *, #51, .F.);
#79 = ORIENTED_EDGE('', *, *, #55, .T.);
#80 = ORIENTED_EDGE('', *, *, #59, .T.);
#81 = ORIENTED_EDGE('', *, *, #56, .F.);
#82 = ORIENTED_EDGE('', *, *, #52, .F.);
#83 = ORIENTED_EDGE('', *, *, #56, .T.);
#84 = ORIENTED_EDGE('', *, *, #60, .T.);
#85 = ORIENTED_EDGE('', *, *, #57, .F.);
#86 = ORIENTED_EDGE('', *, *, #53, .F.);
#87 = ORIENTED_EDGE('', *, *, #57, .T.);
#88 = ORIENTED_EDGE('', *, *, #61, .T.);
#89 = ORIENTED_EDGE('', *, *, #54, .F.);
#90 = ORIENTED_EDGE('', *, *, #58, .T.);
#91 = ORIENTED_EDGE('', *, *, #59, .T.);
#92 = ORIENTED_EDGE('', *, *, #60, .T.);
#93 = ORIENTED_EDGE('', *, *, #61, .T.);
#100 = EDGE_LOOP('', (#70, #71, #72, #73));
#101 = EDGE_LOOP('', (#74, #75, #76, #77));
#102 = EDGE_LOOP('', (#78, #79, #80, #81));
#103 = EDGE_LOOP('', (#82, #83, #84, #85));
#104 = EDGE_LOOP('', (#86, #87, #88, #89));
#105 = EDGE_LOOP('', (#90, #91, #92, #93));
#110 = FACE_OUTER_BOUND('', #100, .T.);
#111 = FACE_OUTER_BOUND('', #101, .T.);
#112 = FACE_OUTER_BOUND('', #102, .T.);
#113 = FACE_OUTER_BOUND('', #103, .T.);
#114 = FACE_OUTER_BOUND('', #104, .T.);
#115 = FACE_OUTER_BOUND('', #105, .T.);
#120 = AXIS2_PLACEMENT_3D('', #1, #13, #11);
#121 = AXIS2_PLACEMENT_3D('', #1, #15, #11);
#122 = AXIS2_PLACEMENT_3D('', #2, #11, #12);
#123 = AXIS2_PLACEMENT_3D('', #3, #12, #14);
#124 = AXIS2_PLACEMENT_3D('', #4, #14, #15);
#125 = AXIS2_PLACEMENT_3D('', #5, #10, #11);
#130 = PLANE('', #120);
#131 = PLANE('', #121);
#132 = PLANE('', #122);
#133 = PLANE('', #123);
#134 = PLANE('', #124);
#135 = PLANE('', #125);
#140 = ADVANCED_FACE('', (#110), #130, .T.);
#141 = ADVANCED_FACE('', (#111), #131, .T.);
#142 = ADVANCED_FACE('', (#112), #132, .T.);
#143 = ADVANCED_FACE('', (#113), #133, .T.);
#144 = ADVANCED_FACE('', (#114), #134, .T.);
#145 = ADVANCED_FACE('', (#115), #135, .T.);
#150 = CLOSED_SHELL('', (#140, #141, #142, #143, #144, #145));
ENDSEC;
END-ISO-10303-21;`;

    const model = importer.parseSTEP(stepText);
    assert(model.sourceFormat === 'STEP', `sourceFormat should be 'STEP', got '${model.sourceFormat}'`);
    assert(model.protocol === 'AP203', `protocol should be 'AP203', got '${model.protocol}'`);
    assert(model.nx > 0 && model.ny > 0 && model.nz > 0, 'Voxelized model should have positive dimensions');
    assert(model.originalVertices.length > 0, 'Should have original vertices');

    // Verify some voxels are solid
    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0.5) solidCount++;
    }
    assert(solidCount > 0, `Should have some solid voxels from STEP cube, got ${solidCount}`);
}

// ──────────────────────────────────────────────────
// Test 36: STEP parser – DATA section error handling
// ──────────────────────────────────────────────────
console.log('Test 36: STEP parser – DATA section error handling');
{
    const parser = new STEPParser();
    let threwError = false;
    try {
        parser.parse('This is not a STEP file at all');
    } catch (e) {
        threwError = true;
        assert(e.message.includes('No DATA section'), `Error should mention DATA section, got '${e.message}'`);
    }
    assert(threwError, 'Should throw error for invalid STEP file');
}

// ──────────────────────────────────────────────────
// Test 37: STEP parser – empty geometry error
// ──────────────────────────────────────────────────
console.log('Test 37: STEP parser – empty geometry error');
{
    const stepText = `ISO-10303-21;
HEADER;
FILE_SCHEMA(('AP203'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;`;

    let threwError = false;
    try {
        importer.parseSTEP(stepText);
    } catch (e) {
        threwError = true;
        assert(e.message.includes('No geometry'), `Error should mention no geometry, got '${e.message}'`);
    }
    assert(threwError, 'Should throw error when no triangulable geometry is found');
}

// ──────────────────────────────────────────────────
// Test 38: STEP parser – vector math helpers
// ──────────────────────────────────────────────────
console.log('Test 38: STEP parser – vector math helpers');
{
    const parser = new STEPParser();
    const n = parser._normalize({ x: 3, y: 0, z: 4 });
    assert(Math.abs(n.x - 0.6) < 1e-6, `Normalized x should be 0.6, got ${n.x}`);
    assert(Math.abs(n.z - 0.8) < 1e-6, `Normalized z should be 0.8, got ${n.z}`);

    const c = parser._cross({ x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
    assert(Math.abs(c.x) < 1e-6, `Cross x should be 0, got ${c.x}`);
    assert(Math.abs(c.y) < 1e-6, `Cross y should be 0, got ${c.y}`);
    assert(Math.abs(c.z - 1) < 1e-6, `Cross z should be 1, got ${c.z}`);

    const d = parser._dot({ x: 1, y: 2, z: 3 }, { x: 4, y: 5, z: 6 });
    assert(Math.abs(d - 32) < 1e-6, `Dot product should be 32, got ${d}`);

    const dist = parser._distance({ x: 0, y: 0, z: 0 }, { x: 3, y: 4, z: 0 });
    assert(Math.abs(dist - 5) < 1e-6, `Distance should be 5, got ${dist}`);
}

// ──────────────────────────────────────────────────
// Test 39: Blended curvature mesh – cube produces correct grid
// ──────────────────────────────────────────────────
console.log('Test 39: Blended curvature mesh – cube');
{
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

    const result = importer.blendedCurvatureMesh(cubeVertices, 10);
    assert(result.meshType === 'blended-curvature', `meshType should be 'blended-curvature', got '${result.meshType}'`);
    assert(result.nx === 10, `Expected nx=10, got ${result.nx}`);
    assert(result.ny === 10, `Expected ny=10, got ${result.ny}`);
    assert(result.nz === 10, `Expected nz=10, got ${result.nz}`);

    let solidCount = 0;
    let blendedCount = 0;
    for (let i = 0; i < result.elements.length; i++) {
        if (result.elements[i] >= 1.0 - 1e-6) solidCount++;
        else if (result.elements[i] > 0) blendedCount++;
    }
    assert(solidCount > 0, `Should have some fully solid interior voxels, got ${solidCount}`);
    assert(result.originalVertices === cubeVertices, 'Should preserve original vertices');
    assert(result.bounds != null, 'Should have bounds');
}

// ──────────────────────────────────────────────────
// Test 40: Blended curvature mesh – meshType property on voxelizeVertices
// ──────────────────────────────────────────────────
console.log('Test 40: voxelizeVertices returns meshType="box"');
{
    const vertices = [
        { x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, { x: 5, y: 10, z: 0 },
    ];
    const result = importer.voxelizeVertices(vertices, 5);
    assert(result.meshType === 'box', `meshType should be 'box', got '${result.meshType}'`);
}

// ──────────────────────────────────────────────────
// Test 41: Blended curvature mesh – empty vertices
// ──────────────────────────────────────────────────
console.log('Test 41: Blended curvature mesh – no triangles fills all');
{
    const result = importer.blendedCurvatureMesh([], 5, 2);
    assert(result.meshType === 'blended-curvature', `meshType should be 'blended-curvature'`);
    let allOne = true;
    for (let i = 0; i < result.elements.length; i++) {
        if (result.elements[i] !== 1) { allOne = false; break; }
    }
    assert(allOne, 'With no triangles, all elements should be 1');
}

// ──────────────────────────────────────────────────
// AMR Surface Mesh Tests
// ──────────────────────────────────────────────────

const { generateUniformSurfaceMesh, generateAMRSurfaceMesh, checkWatertight, indexedMeshToTriangles } = await import(toURL(join(__dirname, '..', 'js', 'amr-surface-mesh.js')));

// ──────────────────────────────────────────────────
// Test 42: Uniform surface mesh – single voxel
// ──────────────────────────────────────────────────
console.log('Test 42: Uniform surface mesh – single voxel');
{
    const densities = new Float32Array([1.0]);
    const mesh = generateUniformSurfaceMesh({ densities, nx: 1, ny: 1, nz: 1 });

    // A single cube has 6 faces × 2 triangles = 12 triangles
    const numTriangles = mesh.indices.length / 3;
    assert(numTriangles === 12, `Single voxel should have 12 triangles, got ${numTriangles}`);
    assert(mesh.positions.length > 0, 'Should have positions');
    assert(mesh.normals.length === mesh.positions.length, 'Normals should match positions count');
    assert(mesh.watertight, 'Single voxel mesh should be watertight');
}

// ──────────────────────────────────────────────────
// Test 43: Uniform surface mesh – 2x2x2 solid cube
// ──────────────────────────────────────────────────
console.log('Test 43: Uniform surface mesh – 2x2x2 solid cube');
{
    const densities = new Float32Array(8).fill(1.0);
    const mesh = generateUniformSurfaceMesh({ densities, nx: 2, ny: 2, nz: 2 });

    // 2x2x2 cube has 24 outer faces × 2 triangles = 48 triangles
    const numTriangles = mesh.indices.length / 3;
    assert(numTriangles === 48, `2x2x2 cube should have 48 triangles, got ${numTriangles}`);
    assert(mesh.watertight, '2x2x2 cube mesh should be watertight');
}

// ──────────────────────────────────────────────────
// Test 44: Uniform surface mesh – vertex deduplication
// ──────────────────────────────────────────────────
console.log('Test 44: Uniform surface mesh – vertex deduplication');
{
    const densities = new Float32Array(8).fill(1.0);
    const mesh = generateUniformSurfaceMesh({ densities, nx: 2, ny: 2, nz: 2 });

    // Vertices should be fewer than non-deduped count (48 tris × 3 verts = 144)
    const numVerts = mesh.positions.length / 3;
    assert(numVerts < 144, `Deduplication should reduce vertex count from 144, got ${numVerts}`);
}

// ──────────────────────────────────────────────────
// Test 45: Uniform surface mesh – empty grid
// ──────────────────────────────────────────────────
console.log('Test 45: Uniform surface mesh – empty grid');
{
    const densities = new Float32Array(27).fill(0.0);
    const mesh = generateUniformSurfaceMesh({ densities, nx: 3, ny: 3, nz: 3 });

    assert(mesh.indices.length === 0, 'Empty grid should produce no triangles');
    assert(mesh.watertight, 'Empty mesh should be trivially watertight');
}

// ──────────────────────────────────────────────────
// Test 46: Uniform surface mesh – with stress field
// ──────────────────────────────────────────────────
console.log('Test 46: Uniform surface mesh – with stress field');
{
    const densities = new Float32Array([1.0]);
    const stress = new Float32Array([42.0]);
    const mesh = generateUniformSurfaceMesh({ densities, nx: 1, ny: 1, nz: 1, stress });

    assert(mesh.stress !== null, 'Should have stress output');
    assert(mesh.stress.length === mesh.positions.length / 3, 'Stress should be per-vertex');
    // All vertices belong to the same voxel, so stress should be ~42.0
    let allStress42 = true;
    for (let i = 0; i < mesh.stress.length; i++) {
        if (Math.abs(mesh.stress[i] - 42.0) > 0.01) { allStress42 = false; break; }
    }
    assert(allStress42, 'All vertex stress values should be ~42.0');
}

// ──────────────────────────────────────────────────
// Test 47: Uniform surface mesh – L-shaped model (watertight)
// ──────────────────────────────────────────────────
console.log('Test 47: Uniform surface mesh – L-shaped model watertight');
{
    // L-shape: 3×3×1 grid with top-right corner removed
    const densities = new Float32Array(9);
    // Fill all except (2,2)
    for (let y = 0; y < 3; y++) {
        for (let x = 0; x < 3; x++) {
            const idx = x + y * 3;
            densities[idx] = (x === 2 && y === 2) ? 0.0 : 1.0;
        }
    }
    const mesh = generateUniformSurfaceMesh({ densities, nx: 3, ny: 3, nz: 1 });
    assert(mesh.watertight, 'L-shaped model should be watertight');
    assert(mesh.indices.length > 0, 'L-shaped model should have triangles');
}

// ──────────────────────────────────────────────────
// Test 48: checkWatertight – valid closed mesh
// ──────────────────────────────────────────────────
console.log('Test 48: checkWatertight – valid closed mesh');
{
    // A tetrahedron with 4 faces (each edge shared by exactly 2 faces)
    const indices = new Uint32Array([
        0, 1, 2,
        0, 3, 1,
        0, 2, 3,
        1, 3, 2
    ]);
    assert(checkWatertight(indices), 'Tetrahedron should be watertight');
}

// ──────────────────────────────────────────────────
// Test 49: checkWatertight – open mesh
// ──────────────────────────────────────────────────
console.log('Test 49: checkWatertight – open mesh');
{
    // Single triangle – not closed
    const indices = new Uint32Array([0, 1, 2]);
    assert(!checkWatertight(indices), 'Single triangle should not be watertight');
}

// ──────────────────────────────────────────────────
// Test 50: indexedMeshToTriangles conversion
// ──────────────────────────────────────────────────
console.log('Test 50: indexedMeshToTriangles conversion');
{
    const densities = new Float32Array([1.0]);
    const mesh = generateUniformSurfaceMesh({ densities, nx: 1, ny: 1, nz: 1 });
    const triangles = indexedMeshToTriangles(mesh);

    assert(triangles.length === 12, `Should have 12 triangles, got ${triangles.length}`);
    for (const tri of triangles) {
        assert(tri.normal.length === 3, 'Each triangle should have a 3-component normal');
        assert(tri.vertices.length === 3, 'Each triangle should have 3 vertices');
        assert(tri.vertices[0].length === 3, 'Each vertex should have 3 components');
    }
}

// ──────────────────────────────────────────────────
// Test 51: AMR surface mesh – single cell
// ──────────────────────────────────────────────────
console.log('Test 51: AMR surface mesh – single cell');
{
    const cells = [{ x: 0, y: 0, z: 0, size: 1, density: 1.0 }];
    const mesh = generateAMRSurfaceMesh({ cells });

    const numTriangles = mesh.indices.length / 3;
    assert(numTriangles === 12, `Single AMR cell should have 12 triangles, got ${numTriangles}`);
    assert(mesh.watertight, 'Single AMR cell should be watertight');
}

// ──────────────────────────────────────────────────
// Test 52: AMR surface mesh – two adjacent cells
// ──────────────────────────────────────────────────
console.log('Test 52: AMR surface mesh – two adjacent cells');
{
    const cells = [
        { x: 0, y: 0, z: 0, size: 1, density: 1.0 },
        { x: 1, y: 0, z: 0, size: 1, density: 1.0 }
    ];
    const mesh = generateAMRSurfaceMesh({ cells });

    // Two adjacent cubes share one face → 10 outer faces × 2 tris = 20
    const numTriangles = mesh.indices.length / 3;
    assert(numTriangles === 20, `Two adjacent cells should have 20 triangles, got ${numTriangles}`);
    assert(mesh.watertight, 'Two adjacent cells should be watertight');
}

// ──────────────────────────────────────────────────
// Test 53: AMR surface mesh – empty cells
// ──────────────────────────────────────────────────
console.log('Test 53: AMR surface mesh – empty cells');
{
    const mesh = generateAMRSurfaceMesh({ cells: [] });
    assert(mesh.indices.length === 0, 'Empty cells should produce no triangles');
    assert(mesh.watertight, 'Empty mesh should be trivially watertight');
}

// ──────────────────────────────────────────────────
// Test 54: AMR surface mesh – inactive cells excluded
// ──────────────────────────────────────────────────
console.log('Test 54: AMR surface mesh – inactive cells excluded');
{
    const cells = [
        { x: 0, y: 0, z: 0, size: 1, density: 1.0 },
        { x: 1, y: 0, z: 0, size: 1, density: 0.01 } // below threshold
    ];
    const mesh = generateAMRSurfaceMesh({ cells });

    // Only one active cell → 12 triangles
    const numTriangles = mesh.indices.length / 3;
    assert(numTriangles === 12, `Only active cells should generate triangles, got ${numTriangles}`);
    assert(mesh.watertight, 'Mesh with inactive cells should still be watertight');
}

// ──────────────────────────────────────────────────
// Test 55: Exporter generates watertight mesh via AMR surface mesh
// ──────────────────────────────────────────────────
console.log('Test 55: Exporter generates watertight mesh via AMR surface mesh');
{
    // Test the surface mesh that the exporter delegates to
    const densities = new Float32Array(27);
    // Create an L-shape in 3x3x3
    for (let z = 0; z < 3; z++) {
        for (let y = 0; y < 3; y++) {
            for (let x = 0; x < 3; x++) {
                const idx = x + y * 3 + z * 9;
                densities[idx] = (x < 2 || y < 2) ? 1.0 : 0.0;
            }
        }
    }
    const mesh = generateUniformSurfaceMesh({ densities, nx: 3, ny: 3, nz: 3 });
    assert(mesh.watertight, 'Exporter-style L-shape mesh should be watertight');
    const triangles = indexedMeshToTriangles(mesh);
    assert(triangles.length > 0, 'Should have triangles for STL export');

    // Verify all normals are unit vectors
    let allUnit = true;
    for (const tri of triangles) {
        const len = Math.sqrt(tri.normal[0] ** 2 + tri.normal[1] ** 2 + tri.normal[2] ** 2);
        if (Math.abs(len - 1) > 0.01 && len > 0.001) { allUnit = false; break; }
    }
    assert(allUnit, 'All triangle normals should be unit vectors');
}

// ──────────────────────────────────────────────────
// Test 56: TopologySolver library – PETSc (KSP) solver with BDDC preconditioner
// ──────────────────────────────────────────────────
console.log('Test 56: TopologySolver library – PETSc KSP solver with BDDC');
{
    const solver = new TopologySolver();
    const nx = 4, ny = 4, nz = 4;
    const nel = nx * ny * nz;
    const model = { nx, ny, nz, type: 'cube', elements: new Float32Array(nel).fill(1) };
    const config = {
        solver: 'petsc',
        volumeFraction: 0.5,
        maxIterations: 3,
        penaltyFactor: 3,
        filterRadius: 1.2,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 1,
        poissonsRatio: 0.3,
        useProjection: false,
        petscPC: 'bddc',
    };
    let progressCalled = false;
    const result = await solver.optimize(model, config, (iter) => { progressCalled = true; });
    assert(typeof result.finalCompliance === 'number', 'PETSc KSP should produce a numeric compliance');
    assert(result.finalCompliance > 0, `PETSc KSP compliance should be > 0, got ${result.finalCompliance}`);
    assert(result.iterations >= 1, `PETSc KSP should complete at least 1 iteration, got ${result.iterations}`);
    assert(result.densities instanceof Float32Array, 'PETSc KSP should return Float32Array densities');
    assert(result.densities.length === nel, `PETSc KSP densities length should be ${nel}`);
    assert(progressCalled, 'PETSc KSP should call progress callback');
}

// ──────────────────────────────────────────────────
// Test 57: TopologySolver library – PETSc (KSP) solver with MG preconditioner
// ──────────────────────────────────────────────────
console.log('Test 57: TopologySolver library – PETSc KSP solver with MG preconditioner');
{
    const solver = new TopologySolver();
    const nx = 4, ny = 4, nz = 4;
    const nel = nx * ny * nz;
    const model = { nx, ny, nz, type: 'cube', elements: new Float32Array(nel).fill(1) };
    const config = {
        solver: 'petsc',
        volumeFraction: 0.5,
        maxIterations: 3,
        penaltyFactor: 3,
        filterRadius: 1.2,
        forceDirection: 'down',
        forceMagnitude: 100,
        constraintPosition: 'left',
        useAMR: false,
        youngsModulus: 1,
        poissonsRatio: 0.3,
        useProjection: false,
        petscPC: 'mg',
    };
    const result = await solver.optimize(model, config, () => {});
    assert(typeof result.finalCompliance === 'number', 'PETSc KSP+MG should produce a numeric compliance');
    assert(result.finalCompliance > 0, `PETSc KSP+MG compliance should be > 0, got ${result.finalCompliance}`);
    assert(result.iterations >= 1, `PETSc KSP+MG should complete at least 1 iteration, got ${result.iterations}`);
}

// ──────────────────────────────────────────────────
// Test 58: Marching Cubes – solid cube produces a closed surface
// ──────────────────────────────────────────────────
const { marchingCubes } = await import(toURL(join(__dirname, '..', 'js', 'marching-cubes.js')));

console.log('Test 58: Marching Cubes – solid cube produces a closed surface');
{
    // 2×2×2 grid, all solid (density = 1.0), threshold = 0.5
    const densities = new Float32Array(8).fill(1.0);
    const result = marchingCubes({ densities, nx: 2, ny: 2, nz: 2, threshold: 0.5 });
    assert(result.positions.length > 0, 'MC solid cube should produce vertices');
    assert(result.indices.length > 0, 'MC solid cube should produce triangles');
    assert(result.normals.length === result.positions.length, 'MC normals should match positions count');
    assert(result.indices.length % 3 === 0, 'MC indices should be a multiple of 3');
}

// ──────────────────────────────────────────────────
// Test 59: Marching Cubes – empty grid produces no surface
// ──────────────────────────────────────────────────
console.log('Test 59: Marching Cubes – empty grid produces no surface');
{
    const densities = new Float32Array(8).fill(0);
    const result = marchingCubes({ densities, nx: 2, ny: 2, nz: 2, threshold: 0.5 });
    assert(result.positions.length === 0, 'MC empty grid should produce no vertices');
    assert(result.indices.length === 0, 'MC empty grid should produce no triangles');
}

// ──────────────────────────────────────────────────
// Test 60: Marching Cubes – single voxel at center
// ──────────────────────────────────────────────────
console.log('Test 60: Marching Cubes – single voxel at center');
{
    // 3×3×3 grid, only center voxel is solid
    const densities = new Float32Array(27).fill(0);
    densities[1 + 1 * 3 + 1 * 9] = 1.0; // center voxel (1,1,1)
    const result = marchingCubes({ densities, nx: 3, ny: 3, nz: 3, threshold: 0.1 });
    assert(result.positions.length > 0, 'MC single voxel should produce vertices');
    assert(result.indices.length >= 12, `MC single voxel should produce at least 4 triangles, got ${result.indices.length / 3}`);

    // Verify all vertices are within the grid bounds
    let inBounds = true;
    for (let i = 0; i < result.positions.length; i += 3) {
        if (result.positions[i] < 0 || result.positions[i] > 3 ||
            result.positions[i+1] < 0 || result.positions[i+1] > 3 ||
            result.positions[i+2] < 0 || result.positions[i+2] > 3) {
            inBounds = false;
            break;
        }
    }
    assert(inBounds, 'MC vertices should be within grid bounds');
}

// ──────────────────────────────────────────────────
// Test 61: Marching Cubes – smooth normals are normalized
// ──────────────────────────────────────────────────
console.log('Test 61: Marching Cubes – smooth normals are normalized');
{
    const densities = new Float32Array(27).fill(0);
    densities[1 + 1 * 3 + 1 * 9] = 1.0;
    const result = marchingCubes({ densities, nx: 3, ny: 3, nz: 3, threshold: 0.1 });
    let allNormalized = true;
    for (let i = 0; i < result.normals.length; i += 3) {
        const len = Math.sqrt(result.normals[i]**2 + result.normals[i+1]**2 + result.normals[i+2]**2);
        if (Math.abs(len - 1.0) > 0.01) {
            allNormalized = false;
            break;
        }
    }
    assert(allNormalized, 'MC normals should be approximately unit length');
}

// ──────────────────────────────────────────────────
// Test 62: Marching Cubes – threshold varies surface position
// ──────────────────────────────────────────────────
console.log('Test 62: Marching Cubes – threshold varies surface position');
{
    // Full grid with half density → threshold below 0.5 should produce a surface
    const densities = new Float32Array(8).fill(0.5);
    const result1 = marchingCubes({ densities, nx: 2, ny: 2, nz: 2, threshold: 0.3 });
    const result2 = marchingCubes({ densities, nx: 2, ny: 2, nz: 2, threshold: 0.7 });
    assert(result1.indices.length > 0, 'MC with threshold below density should produce surface');
    assert(result2.indices.length === 0, 'MC with threshold above density should produce no surface');
}

// ──────────────────────────────────────────────────
// Test 63: Marching Cubes – gradient field produces smooth isosurface
// ──────────────────────────────────────────────────
console.log('Test 63: Marching Cubes – gradient field produces smooth isosurface');
{
    // 4×4×4 grid with linear density gradient along X
    const nx = 4, ny = 4, nz = 4;
    const densities = new Float32Array(nx * ny * nz);
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                densities[x + y * nx + z * nx * ny] = x / (nx - 1);
            }
        }
    }
    const result = marchingCubes({ densities, nx, ny, nz, threshold: 0.5 });
    assert(result.indices.length > 0, 'MC gradient field should produce surface');

    // Vertices should be near the middle of the X range (around x ≈ 2)
    let avgX = 0;
    const vertCount = result.positions.length / 3;
    for (let i = 0; i < result.positions.length; i += 3) {
        avgX += result.positions[i];
    }
    avgX /= vertCount;
    assert(Math.abs(avgX - 2.0) < 1.5, `MC gradient isosurface avg X should be near 2, got ${avgX.toFixed(2)}`);
}

// ──────────────────────────────────────────────────
// Test 64: DXF parsing – simple rectangle
// ──────────────────────────────────────────────────
console.log('Test 64: DXF parsing – rectangle LWPOLYLINE');
{
    const dxfText = `0
SECTION
2
ENTITIES
0
LWPOLYLINE
70
1
10
0
20
0
10
10
20
0
10
10
20
10
10
0
20
10
0
ENDSEC
0
EOF
`;
    const model = importer.parseDXF(dxfText);
    assert(model.nz === 1, `DXF rectangle should have nz=1, got ${model.nz}`);
    assert(model.nx > 0, `DXF rectangle should have positive nx, got ${model.nx}`);
    assert(model.ny > 0, `DXF rectangle should have positive ny, got ${model.ny}`);
    assert(model.sourceFormat === 'DXF', `sourceFormat should be DXF, got ${model.sourceFormat}`);

    // All voxels should be filled for a solid rectangle
    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    assert(solidCount === model.nx * model.ny, `DXF rectangle should fill all ${model.nx * model.ny} voxels, got ${solidCount}`);
}

// ──────────────────────────────────────────────────
// Test 65: DXF parsing – circle
// ──────────────────────────────────────────────────
console.log('Test 65: DXF parsing – circle');
{
    const dxfText = `0
SECTION
2
ENTITIES
0
CIRCLE
10
5
20
5
40
5
0
ENDSEC
0
EOF
`;
    const model = importer.parseDXF(dxfText);
    assert(model.nz === 1, `DXF circle should have nz=1, got ${model.nz}`);
    assert(model.nx > 0, `DXF circle should have positive nx`);
    assert(model.ny > 0, `DXF circle should have positive ny`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    const totalVoxels = model.nx * model.ny;
    const fillRatio = solidCount / totalVoxels;
    // Circle should fill roughly π/4 ≈ 0.785 of bounding box
    assert(fillRatio > 0.6 && fillRatio < 0.95, `DXF circle fill ratio should be ~0.785, got ${fillRatio.toFixed(3)}`);
}

// ──────────────────────────────────────────────────
// Test 66: DXF parsing – LINE entities
// ──────────────────────────────────────────────────
console.log('Test 66: DXF parsing – LINE entities');
{
    // Lines don't form closed polygon, so fallback to fill
    const dxfText = `0
SECTION
2
ENTITIES
0
LINE
10
0
20
0
11
10
21
0
0
LINE
10
10
20
0
11
10
21
10
0
ENDSEC
0
EOF
`;
    const model = importer.parseDXF(dxfText);
    assert(model.nz === 1, `DXF lines should have nz=1`);
    assert(model.sourceFormat === 'DXF', `sourceFormat should be DXF`);
}

// ──────────────────────────────────────────────────
// Test 67: SVG parsing – rectangle
// ──────────────────────────────────────────────────
console.log('Test 67: SVG parsing – rectangle');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <rect x="0" y="0" width="10" height="10" />
    </svg>`;
    const model = importer.parseSVG(svgText);
    assert(model.nz === 1, `SVG rect should have nz=1, got ${model.nz}`);
    assert(model.nx > 0, `SVG rect should have positive nx`);
    assert(model.ny > 0, `SVG rect should have positive ny`);
    assert(model.sourceFormat === 'SVG', `sourceFormat should be SVG, got ${model.sourceFormat}`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    assert(solidCount === model.nx * model.ny, `SVG rectangle should fill all voxels, got ${solidCount}/${model.nx * model.ny}`);
}

// ──────────────────────────────────────────────────
// Test 68: SVG parsing – circle
// ──────────────────────────────────────────────────
console.log('Test 68: SVG parsing – circle');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <circle cx="50" cy="50" r="50" />
    </svg>`;
    const model = importer.parseSVG(svgText);
    assert(model.nz === 1, `SVG circle should have nz=1`);
    assert(model.sourceFormat === 'SVG', `SVG circle sourceFormat should be SVG`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    const totalVoxels = model.nx * model.ny;
    const fillRatio = solidCount / totalVoxels;
    assert(fillRatio > 0.6 && fillRatio < 0.95, `SVG circle fill ratio should be ~0.785, got ${fillRatio.toFixed(3)}`);
}

// ──────────────────────────────────────────────────
// Test 69: SVG parsing – polygon
// ──────────────────────────────────────────────────
console.log('Test 69: SVG parsing – polygon');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <polygon points="0,0 20,0 20,20 0,20" />
    </svg>`;
    const model = importer.parseSVG(svgText);
    assert(model.nz === 1, `SVG polygon should have nz=1`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    assert(solidCount === model.nx * model.ny, `SVG polygon should fill all voxels, got ${solidCount}/${model.nx * model.ny}`);
}

// ──────────────────────────────────────────────────
// Test 70: SVG parsing – path with M L Z commands
// ──────────────────────────────────────────────────
console.log('Test 70: SVG parsing – path (M L Z)');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <path d="M 0 0 L 10 0 L 10 10 L 0 10 Z" />
    </svg>`;
    const model = importer.parseSVG(svgText);
    assert(model.nz === 1, `SVG path should have nz=1`);
    assert(model.sourceFormat === 'SVG', `SVG path sourceFormat should be SVG`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    assert(solidCount === model.nx * model.ny, `SVG path rect should fill all voxels`);
}

// ──────────────────────────────────────────────────
// Test 71: 2D voxelization – point-in-polygon
// ──────────────────────────────────────────────────
console.log('Test 71: 2D voxelization – point-in-polygon');
{
    // Test the internal _pointInPolygon method
    const square = [
        { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }
    ];
    assert(importer._pointInPolygon(5, 5, square), 'Point (5,5) should be inside square');
    assert(!importer._pointInPolygon(15, 5, square), 'Point (15,5) should be outside square');
    assert(!importer._pointInPolygon(-1, -1, square), 'Point (-1,-1) should be outside square');
    assert(importer._pointInPolygon(1, 1, square), 'Point (1,1) should be inside square');
}

// ──────────────────────────────────────────────────
// Test 72: 2D voxelization – voxelize2DPolygons directly
// ──────────────────────────────────────────────────
console.log('Test 72: 2D voxelization – voxelize2DPolygons');
{
    const polygons = [
        [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }]
    ];
    const model = importer.voxelize2DPolygons(polygons, 1);
    assert(model.nz === 1, `voxelize2DPolygons should produce nz=1, got ${model.nz}`);
    assert(model.nx === 10, `Expected nx=10, got ${model.nx}`);
    assert(model.ny === 10, `Expected ny=10, got ${model.ny}`);
    assert(model.is2D === true, `Model should have is2D=true`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    assert(solidCount === 100, `All 100 voxels should be solid, got ${solidCount}`);
}

// ──────────────────────────────────────────────────
// Test 73: Template defaults – beam has forcePosition and constraintPositions
// ──────────────────────────────────────────────────
console.log('Test 73: Template defaults – beam has boundary conditions');
{
    const beam = importer.createTemplate('beam', 30);
    assert(beam.forcePosition === 'right', `Beam forcePosition should be 'right', got '${beam.forcePosition}'`);
    assert(beam.forceDirection === 'down', `Beam forceDirection should be 'down', got '${beam.forceDirection}'`);
    assert(beam.constraintPositions === 'left', `Beam constraintPositions should be 'left', got '${beam.constraintPositions}'`);
}

// ──────────────────────────────────────────────────
// Test 74: Template defaults – bridge has forcePosition and constraintPositions
// ──────────────────────────────────────────────────
console.log('Test 74: Template defaults – bridge has boundary conditions');
{
    const bridge = importer.createTemplate('bridge', 40);
    assert(bridge.forcePosition === 'top-center', `Bridge forcePosition should be 'top-center', got '${bridge.forcePosition}'`);
    assert(bridge.forceDirection === 'down', `Bridge forceDirection should be 'down', got '${bridge.forceDirection}'`);
    assert(bridge.constraintPositions === 'both-ends', `Bridge constraintPositions should be 'both-ends', got '${bridge.constraintPositions}'`);
}

// ──────────────────────────────────────────────────
// Test 75: Template defaults – cube has forcePosition and constraintPositions
// ──────────────────────────────────────────────────
console.log('Test 75: Template defaults – cube has boundary conditions');
{
    const cube = importer.createTemplate('cube', 50);
    assert(cube.forcePosition === 'top-center', `Cube forcePosition should be 'top-center', got '${cube.forcePosition}'`);
    assert(cube.forceDirection === 'down', `Cube forceDirection should be 'down', got '${cube.forceDirection}'`);
    assert(cube.constraintPositions === 'bottom-corners', `Cube constraintPositions should be 'bottom-corners', got '${cube.constraintPositions}'`);
}

// ──────────────────────────────────────────────────
// Test 76: SVG parsing – ellipse
// ──────────────────────────────────────────────────
console.log('Test 76: SVG parsing – ellipse');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <ellipse cx="25" cy="15" rx="25" ry="15" />
    </svg>`;
    const model = importer.parseSVG(svgText);
    assert(model.nz === 1, `SVG ellipse should have nz=1`);
    assert(model.sourceFormat === 'SVG', `SVG ellipse sourceFormat should be SVG`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    const totalVoxels = model.nx * model.ny;
    const fillRatio = solidCount / totalVoxels;
    assert(fillRatio > 0.5 && fillRatio < 0.95, `SVG ellipse fill ratio should be ~0.785, got ${fillRatio.toFixed(3)}`);
}

// ──────────────────────────────────────────────────
// Test 77: SVG path – relative commands (m l h v z)
// ──────────────────────────────────────────────────
console.log('Test 77: SVG path – relative commands');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <path d="m 0 0 l 10 0 l 0 10 l -10 0 z" />
    </svg>`;
    const model = importer.parseSVG(svgText);
    assert(model.nz === 1, `SVG relative path should have nz=1`);

    let solidCount = 0;
    for (let i = 0; i < model.elements.length; i++) {
        if (model.elements[i] > 0) solidCount++;
    }
    assert(solidCount === model.nx * model.ny, `SVG relative path should fill all voxels`);
}

// ──────────────────────────────────────────────────
// Test 78: DXF parsing – no geometry throws error
// ──────────────────────────────────────────────────
console.log('Test 78: DXF parsing – no geometry throws error');
{
    const dxfText = `0
SECTION
2
ENTITIES
0
ENDSEC
0
EOF
`;
    let threw = false;
    try {
        importer.parseDXF(dxfText);
    } catch (e) {
        threw = true;
        assert(e.message.includes('No supported 2D geometry'), `Error message should mention geometry, got: ${e.message}`);
    }
    assert(threw, 'parseDXF should throw for empty entities');
}

// ──────────────────────────────────────────────────
// Test 79: SVG parsing – no geometry throws error
// ──────────────────────────────────────────────────
console.log('Test 79: SVG parsing – no geometry throws error');
{
    const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
    </svg>`;
    let threw = false;
    try {
        importer.parseSVG(svgText);
    } catch (e) {
        threw = true;
        assert(e.message.includes('No supported 2D geometry'), `Error message should mention geometry, got: ${e.message}`);
    }
    assert(threw, 'parseSVG should throw for empty SVG');
}

// ──────────────────────────────────────────────────
// Test 80: NonlinearSolver – small 2×1×1 mesh converges
// ──────────────────────────────────────────────────
console.log('Test 80: NonlinearSolver – small 2×1×1 mesh converges');
{
    const nelx = 2, nely = 1, nelz = 1;
    const nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    const nodeCount = nnx * nny * nnz;
    const elemCount = nelx * nely * nelz;

    const mesh = {
        nelx, nely, nelz,
        nodeCount,
        elemCount,
        getElementNodes: (e) => {
            const ez = Math.floor(e / (nelx * nely));
            const ey = Math.floor((e % (nelx * nely)) / nelx);
            const ex = e % nelx;
            const n = (iz, iy, ix) => iz * nny * nnx + iy * nnx + ix;
            return [
                n(ez,   ey,   ex),   n(ez,   ey,   ex+1),
                n(ez,   ey+1, ex+1), n(ez,   ey+1, ex),
                n(ez+1, ey,   ex),   n(ez+1, ey,   ex+1),
                n(ez+1, ey+1, ex+1), n(ez+1, ey+1, ex)
            ];
        },
        getNodeCoords: (n) => {
            const nz = Math.floor(n / (nny * nnx));
            const ny = Math.floor((n % (nny * nnx)) / nnx);
            const nx = n % nnx;
            return [nx, ny, nz];
        }
    };

    const material = createMaterial('neo-hookean', { E: 1.0, nu: 0.3 });

    // Fix left face (x=0): DOFs 0,1,2 per node
    const fixedNodes = [];
    for (let iz = 0; iz < nnz; iz++) {
        for (let iy = 0; iy < nny; iy++) {
            const ni = iz * nny * nnx + iy * nnx + 0;
            fixedNodes.push(ni * 3, ni * 3 + 1, ni * 3 + 2);
        }
    }
    const constraints = new Int32Array(fixedNodes);

    // Apply small downward force on right face (x=nelx)
    const loads = new Float64Array(nodeCount * 3);
    for (let iz = 0; iz < nnz; iz++) {
        for (let iy = 0; iy < nny; iy++) {
            const ni = iz * nny * nnx + iy * nnx + nelx;
            loads[ni * 3 + 1] = -0.001; // tiny downward force per node
        }
    }

    const solver = new NonlinearSolver({ numLoadSteps: 2, maxNewtonIter: 10, residualTol: 1e-4 });
    const result = solver.solve(mesh, material, constraints, loads);

    assert(result.displacement instanceof Float64Array, 'NonlinearSolver: displacement should be Float64Array');
    assert(result.displacement.length === nodeCount * 3, `NonlinearSolver: displacement length should be ${nodeCount * 3}, got ${result.displacement.length}`);
    assert(result.vonMisesStress instanceof Float64Array || result.vonMisesStress instanceof Float32Array || Array.isArray(result.vonMisesStress), 'NonlinearSolver: vonMisesStress should be array-like');
    assert(result.vonMisesStress.length === elemCount, `NonlinearSolver: vonMisesStress length should be ${elemCount}`);
    assert(typeof result.converged === 'boolean', 'NonlinearSolver: converged should be boolean');
    assert(typeof result.totalIterations === 'number', 'NonlinearSolver: totalIterations should be a number');
    assert(result.cauchyStress instanceof Float64Array || Array.isArray(result.cauchyStress), 'NonlinearSolver: cauchyStress should be array-like');
    assert(result.strainEnergy instanceof Float64Array || Array.isArray(result.strainEnergy), 'NonlinearSolver: strainEnergy should be array-like');
}

// ──────────────────────────────────────────────────
// Test 81: Gasket template – basic properties
// ──────────────────────────────────────────────────
console.log('Test 81: Gasket template – basic properties');
{
    const gasket = importer.createGasketTemplate(30);
    assert(gasket.type === 'gasket', `Gasket type should be 'gasket', got '${gasket.type}'`);
    assert(gasket.nz === 1, `Gasket should be 2D (nz=1), got nz=${gasket.nz}`);
    assert(gasket.nx >= 10, `Gasket nx should be >= 10, got ${gasket.nx}`);
    assert(gasket.ny >= 8, `Gasket ny should be >= 8, got ${gasket.ny}`);
    assert(gasket.nx > gasket.ny, `Gasket should be wider than tall (squished), nx=${gasket.nx}, ny=${gasket.ny}`);
    assert(gasket.elements.length === gasket.nx * gasket.ny, `Gasket elements length should be nx*ny=${gasket.nx * gasket.ny}, got ${gasket.elements.length}`);
}

// ──────────────────────────────────────────────────
// Test 82: Gasket template – has voids (not all-solid)
// ──────────────────────────────────────────────────
console.log('Test 82: Gasket template – has voids (not all-solid)');
{
    const gasket = importer.createGasketTemplate(30);
    let solidCount = 0;
    for (let i = 0; i < gasket.elements.length; i++) {
        if (gasket.elements[i] > 0.5) solidCount++;
    }
    assert(solidCount > 0, `Gasket should have some solid voxels, got ${solidCount}`);
    assert(solidCount < gasket.elements.length, `Gasket should have voids (not all-solid): ${solidCount}/${gasket.elements.length}`);
}

// ──────────────────────────────────────────────────
// Test 83: Gasket template – boundary conditions
// ──────────────────────────────────────────────────
console.log('Test 83: Gasket template – boundary conditions');
{
    const gasket = importer.createTemplate('gasket', 30);
    assert(gasket.forcePosition === 'top-center', `Gasket forcePosition should be 'top-center', got '${gasket.forcePosition}'`);
    assert(gasket.forceDirection === 'down', `Gasket forceDirection should be 'down', got '${gasket.forceDirection}'`);
    assert(gasket.constraintPositions === 'bottom', `Gasket constraintPositions should be 'bottom', got '${gasket.constraintPositions}'`);
    assert(gasket.recommendedSolver === 'nonlinear', `Gasket recommendedSolver should be 'nonlinear', got '${gasket.recommendedSolver}'`);
}

// ──────────────────────────────────────────────────
// Test 84: Gasket template – scaling responds to resolution
// ──────────────────────────────────────────────────
console.log('Test 84: Gasket template – scaling responds to resolution');
{
    const gasket1mm = importer.createGasketTemplate(30);
    const gasket2mm = importer.createGasketTemplate(15);
    assert(gasket1mm.nx === 30, `Gasket at resolution 30 should have nx=30, got ${gasket1mm.nx}`);
    assert(gasket2mm.nx === 15, `Gasket at resolution 15 should have nx=15, got ${gasket2mm.nx}`);
    assert(gasket1mm.nx > gasket2mm.nx, `Higher resolution gasket should have more voxels in x`);
}

// ──────────────────────────────────────────────────
// Test 85: Gasket template – getTemplateMaxDim returns 30
// ──────────────────────────────────────────────────
console.log('Test 85: Gasket template – getTemplateMaxDim returns 30');
{
    const maxDim = ModelImporter.getTemplateMaxDim('gasket');
    assert(maxDim === 30, `Gasket max dim should be 30, got ${maxDim}`);
}

// ──────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────
console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
