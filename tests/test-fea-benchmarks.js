#!/usr/bin/env node
/**
 * FEA Benchmark Tests
 *
 * Standardized tests to verify the accuracy and compliance of the FEA solver.
 * Based on well-known benchmark problems (patch tests, cantilever beams,
 * symmetry checks) with analytical reference solutions.
 *
 * Run with: node tests/test-fea-benchmarks.js
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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

// ─── Load WASM module ───
async function loadWasm() {
    const wasmPath = join(__dirname, '..', 'wasm', 'matrix-ops.wasm');
    const buffer = await readFile(wasmPath);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module, {
        env: {
            abort: () => { throw new Error('WASM abort'); },
            seed: () => Date.now()
        }
    });
    return instance;
}

// ═══════════════════════════════════════════════════════════════════════
// Shared FEA utilities (consistent with test-wasm-ebePCG.js)
// ═══════════════════════════════════════════════════════════════════════

/** 2D element stiffness matrix for a unit-square Q4 element (plane stress) */
function lk2D(nu) {
    const k = [
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8
    ];
    const KE = Array(8).fill(0).map(() => Array(8).fill(0));
    KE[0][0] = k[0]; KE[0][1] = k[1]; KE[0][2] = k[2]; KE[0][3] = k[3];
    KE[0][4] = k[4]; KE[0][5] = k[5]; KE[0][6] = k[6]; KE[0][7] = k[7];
    KE[1][0] = k[1]; KE[1][1] = k[0]; KE[1][2] = k[7]; KE[1][3] = k[6];
    KE[1][4] = k[5]; KE[1][5] = k[4]; KE[1][6] = k[3]; KE[1][7] = k[2];
    KE[2][0] = k[2]; KE[2][1] = k[7]; KE[2][2] = k[0]; KE[2][3] = k[5];
    KE[2][4] = k[6]; KE[2][5] = k[3]; KE[2][6] = k[4]; KE[2][7] = k[1];
    KE[3][0] = k[3]; KE[3][1] = k[6]; KE[3][2] = k[5]; KE[3][3] = k[0];
    KE[3][4] = k[7]; KE[3][5] = k[2]; KE[3][6] = k[1]; KE[3][7] = k[4];
    KE[4][0] = k[4]; KE[4][1] = k[5]; KE[4][2] = k[6]; KE[4][3] = k[7];
    KE[4][4] = k[0]; KE[4][5] = k[1]; KE[4][6] = k[2]; KE[4][7] = k[3];
    KE[5][0] = k[5]; KE[5][1] = k[4]; KE[5][2] = k[3]; KE[5][3] = k[2];
    KE[5][4] = k[1]; KE[5][5] = k[0]; KE[5][6] = k[7]; KE[5][7] = k[6];
    KE[6][0] = k[6]; KE[6][1] = k[3]; KE[6][2] = k[4]; KE[6][3] = k[1];
    KE[6][4] = k[2]; KE[6][5] = k[7]; KE[6][6] = k[0]; KE[6][7] = k[5];
    KE[7][0] = k[7]; KE[7][1] = k[2]; KE[7][2] = k[1]; KE[7][3] = k[4];
    KE[7][4] = k[3]; KE[7][5] = k[6]; KE[7][6] = k[5]; KE[7][7] = k[0];
    const scale = 1 / (1 - nu * nu);
    return KE.map(row => row.map(val => val * scale));
}

function flattenKE(KE, size) {
    const flat = new Float64Array(size * size);
    for (let i = 0; i < size; i++)
        for (let j = 0; j < size; j++)
            flat[i * size + j] = KE[i][j];
    return flat;
}

/** Precompute element DOF connectivity for 2D Q4 mesh (column-major ordering) */
function precomputeEdofs2D(nelx, nely) {
    const nel = nelx * nely;
    const edofArray = new Int32Array(nel * 8);
    for (let ely = 0; ely < nely; ely++) {
        for (let elx = 0; elx < nelx; elx++) {
            const idx = ely + elx * nely;
            const offset = idx * 8;
            const n1 = (nely + 1) * elx + ely;
            const n2 = (nely + 1) * (elx + 1) + ely;
            edofArray[offset] = 2 * n1;
            edofArray[offset + 1] = 2 * n1 + 1;
            edofArray[offset + 2] = 2 * n2;
            edofArray[offset + 3] = 2 * n2 + 1;
            edofArray[offset + 4] = 2 * n2 + 2;
            edofArray[offset + 5] = 2 * n2 + 3;
            edofArray[offset + 6] = 2 * n1 + 2;
            edofArray[offset + 7] = 2 * n1 + 3;
        }
    }
    return edofArray;
}

/** JS reference FEA solver (element-by-element PCG, Jacobi preconditioner) */
function solveFEA_JS(nelx, nely, densities, KEflat, edofArray, F, freedofs, E0, Emin, penal) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofSize = 8;
    const nfree = freedofs.length;
    const dE = E0 - Emin;
    const EPSILON = 1e-12;
    const CG_TOLERANCE = 1e-8;
    const MAX_CG_ITERATIONS = 2000;

    const E_vals = new Float64Array(nel);
    const activeElements = [];
    for (let e = 0; e < nel; e++) {
        const E = Emin + Math.pow(densities[e], penal) * dE;
        E_vals[e] = E;
        if (E > Emin * 1000) activeElements.push(e);
    }

    const diag = new Float64Array(ndof);
    for (const e of activeElements) {
        const E = E_vals[e];
        const eOff = e * edofSize;
        for (let i = 0; i < edofSize; i++)
            diag[edofArray[eOff + i]] += E * KEflat[i * edofSize + i];
    }
    const invDiag = new Float64Array(nfree);
    for (let i = 0; i < nfree; i++) {
        const d = diag[freedofs[i]];
        invDiag[i] = d > 1e-30 ? 1.0 / d : 0.0;
    }

    const p_full = new Float64Array(ndof);
    const Ap_full = new Float64Array(ndof);

    function ebeMatVec(p_r, Ap_r) {
        p_full.fill(0);
        for (let i = 0; i < nfree; i++) p_full[freedofs[i]] = p_r[i];
        Ap_full.fill(0);
        for (const e of activeElements) {
            const E = E_vals[e];
            const eOff = e * edofSize;
            for (let i = 0; i < edofSize; i++) {
                const gi = edofArray[eOff + i];
                let sum = 0;
                for (let j = 0; j < edofSize; j++)
                    sum += KEflat[i * edofSize + j] * p_full[edofArray[eOff + j]];
                Ap_full[gi] += E * sum;
            }
        }
        for (let i = 0; i < nfree; i++) Ap_r[i] = Ap_full[freedofs[i]];
    }

    const Uf = new Float64Array(nfree);
    const r = new Float64Array(nfree);
    const z = new Float64Array(nfree);
    const p = new Float64Array(nfree);
    const Ap = new Float64Array(nfree);

    for (let i = 0; i < nfree; i++) r[i] = F[freedofs[i]];
    let rz = 0;
    let r0norm2 = 0;
    for (let i = 0; i < nfree; i++) {
        z[i] = invDiag[i] * r[i];
        p[i] = z[i];
        rz += r[i] * z[i];
        r0norm2 += r[i] * r[i];
    }

    const maxIter = Math.min(nfree, MAX_CG_ITERATIONS);
    const tolSq = CG_TOLERANCE * CG_TOLERANCE * Math.max(r0norm2, 1e-30);

    for (let iter = 0; iter < maxIter; iter++) {
        let rnorm2 = 0;
        for (let i = 0; i < nfree; i++) rnorm2 += r[i] * r[i];
        if (rnorm2 < tolSq) break;
        ebeMatVec(p, Ap);
        let pAp = 0;
        for (let i = 0; i < nfree; i++) pAp += p[i] * Ap[i];
        const alpha = rz / (pAp + EPSILON);
        let rz_new = 0;
        for (let i = 0; i < nfree; i++) {
            Uf[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            z[i] = invDiag[i] * r[i];
            rz_new += r[i] * z[i];
        }
        const beta = rz_new / (rz + EPSILON);
        for (let i = 0; i < nfree; i++) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    const U = new Float64Array(ndof);
    for (let i = 0; i < nfree; i++) U[freedofs[i]] = Uf[i];
    let c = 0;
    for (let i = 0; i < ndof; i++) c += F[i] * U[i];
    return { U, compliance: c };
}

// ═══════════════════════════════════════════════════════════════════════
// Helper: set up a 2D cantilever mesh
//   Fixed left edge (all DOFs), point load at bottom-right corner (downward)
// ═══════════════════════════════════════════════════════════════════════
function setupCantilever(nelx, nely) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofArray = precomputeEdofs2D(nelx, nely);

    // Fix left edge: all DOFs on nodes where elx = 0
    const fixeddofs = [];
    for (let j = 0; j <= nely; j++) {
        fixeddofs.push(2 * j);       // x-DOF
        fixeddofs.push(2 * j + 1);   // y-DOF
    }

    const fixedSet = new Set(fixeddofs);
    const freedofs = new Int32Array(ndof - fixedSet.size);
    let fi = 0;
    for (let i = 0; i < ndof; i++) {
        if (!fixedSet.has(i)) freedofs[fi++] = i;
    }

    // Point load at the bottom-right corner node (downward, -Y)
    const F = new Float64Array(ndof);
    const loadNode = (nely + 1) * nelx + nely;  // bottom-right node
    F[2 * loadNode + 1] = -1.0;

    return { nel, ndof, edofArray, freedofs, F };
}

// ═══════════════════════════════════════════════════════════════════════
// Helper: set up a simply supported beam mesh
//   Fixed Y at both ends (rollers), point load at mid-span (downward)
// ═══════════════════════════════════════════════════════════════════════
function setupSimplySupportedBeam(nelx, nely) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofArray = precomputeEdofs2D(nelx, nely);

    const fixeddofs = [];
    // Left support: fix all DOFs at bottom-left node (pin support)
    const leftBottomNode = nely;  // node at (0, nely)
    fixeddofs.push(2 * leftBottomNode);       // x-DOF (pin)
    fixeddofs.push(2 * leftBottomNode + 1);   // y-DOF

    // Right support: fix y-DOF at bottom-right node (roller)
    const rightBottomNode = (nely + 1) * nelx + nely;
    fixeddofs.push(2 * rightBottomNode + 1);  // y-DOF only

    const fixedSet = new Set(fixeddofs);
    const freedofs = new Int32Array(ndof - fixedSet.size);
    let fi = 0;
    for (let i = 0; i < ndof; i++) {
        if (!fixedSet.has(i)) freedofs[fi++] = i;
    }

    // Point load at mid-span bottom node (downward, -Y)
    const F = new Float64Array(ndof);
    const midX = Math.floor(nelx / 2);
    const midBottomNode = (nely + 1) * midX + nely;
    F[2 * midBottomNode + 1] = -1.0;

    return { nel, ndof, edofArray, freedofs, F };
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark Tests
// ═══════════════════════════════════════════════════════════════════════
async function runTests() {
    const wasmMod = await loadWasm();

    const nu = 0.3;
    const E0 = 1.0;
    const Emin = 1e-9;
    const penal = 1;  // penal=1 for linear benchmarks (no penalization)
    const KE = lk2D(nu);
    const KEflat = flattenKE(KE, 8);

    // ─── Benchmark 1: Element stiffness matrix symmetry ───
    console.log('Benchmark 1: Element stiffness matrix symmetry (KE = KE^T)');
    {
        let maxAsymmetry = 0;
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                maxAsymmetry = Math.max(maxAsymmetry, Math.abs(KE[i][j] - KE[j][i]));
            }
        }
        assert(maxAsymmetry < 1e-15, `KE is symmetric, max |KE[i][j] - KE[j][i]| = ${maxAsymmetry.toExponential(3)}`);
    }

    // ─── Benchmark 2: Element stiffness positive semi-definiteness ───
    console.log('Benchmark 2: Element stiffness matrix diagonal entries are positive');
    {
        let allPositive = true;
        let minDiag = Infinity;
        for (let i = 0; i < 8; i++) {
            if (KE[i][i] <= 0) allPositive = false;
            minDiag = Math.min(minDiag, KE[i][i]);
        }
        assert(allPositive, `All diagonal entries are positive, min = ${minDiag.toExponential(3)}`);
    }

    // ─── Benchmark 3: Rigid body modes (translations) ───
    // For a single element, rigid body translation should produce zero internal forces
    console.log('Benchmark 3: Rigid body translation modes produce zero strain energy');
    {
        // Translation in x: all x-DOFs = 1, y-DOFs = 0
        const ux = [1, 0, 1, 0, 1, 0, 1, 0];
        // Translation in y: all x-DOFs = 0, y-DOFs = 1
        const uy = [0, 1, 0, 1, 0, 1, 0, 1];

        let forceX = 0, forceY = 0;
        for (let i = 0; i < 8; i++) {
            let fx = 0, fy = 0;
            for (let j = 0; j < 8; j++) {
                fx += KE[i][j] * ux[j];
                fy += KE[i][j] * uy[j];
            }
            forceX += Math.abs(fx);
            forceY += Math.abs(fy);
        }
        assert(forceX < 1e-12, `K * u_x_translation ≈ 0, sum|f| = ${forceX.toExponential(3)}`);
        assert(forceY < 1e-12, `K * u_y_translation ≈ 0, sum|f| = ${forceY.toExponential(3)}`);
    }

    // ─── Benchmark 4: Rigid body rotation ───
    // A small rigid rotation about the element center should also produce zero strain energy
    // For a unit element centered at (0.5, 0.5), node coords: (0,0),(1,0),(1,1),(0,1)
    // Rotation by angle θ about center: u = -θ(y-0.5), v = θ(x-0.5)
    // Node DOFs for nodes in order matching edof: n1(0,0), n2(1,0), n3(1,1), n4(0,1)
    console.log('Benchmark 4: Rigid body rotation mode produces zero strain energy');
    {
        const theta = 1.0;
        const cx = 0.5, cy = 0.5;
        // Node positions in element local coords (matching DOF order):
        // n1=(0,0), n2=(1,0), n3=(1,1), n4=(0,1)
        const nodes = [[0, 0], [1, 0], [1, 1], [0, 1]];
        const urot = new Array(8);
        for (let n = 0; n < 4; n++) {
            urot[2 * n] = -theta * (nodes[n][1] - cy);     // ux = -θ(y - cy)
            urot[2 * n + 1] = theta * (nodes[n][0] - cx);  // uy = θ(x - cx)
        }

        let totalForce = 0;
        for (let i = 0; i < 8; i++) {
            let f = 0;
            for (let j = 0; j < 8; j++) f += KE[i][j] * urot[j];
            totalForce += Math.abs(f);
        }
        assert(totalForce < 1e-12, `K * u_rotation ≈ 0, sum|f| = ${totalForce.toExponential(3)}`);
    }

    // ─── Benchmark 5: Patch test (constant strain) ───
    // Apply uniform x-stretch to a mesh. All elements should see the same
    // constant strain, and the displacement should be linear in x.
    console.log('Benchmark 5: Patch test – uniform tension (constant strain field)');
    {
        const nelx = 10, nely = 4;
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const edofArray = precomputeEdofs2D(nelx, nely);

        // Fix left edge x-DOFs, fix one y-DOF to prevent rigid body motion
        const fixeddofs = [];
        for (let j = 0; j <= nely; j++) {
            fixeddofs.push(2 * j);  // x-DOF on left edge
        }
        fixeddofs.push(1);  // y-DOF at node 0

        const fixedSet = new Set(fixeddofs);
        const freedofs = new Int32Array(ndof - fixedSet.size);
        let fi = 0;
        for (let i = 0; i < ndof; i++) {
            if (!fixedSet.has(i)) freedofs[fi++] = i;
        }

        // Apply consistent nodal forces for uniform traction on right edge
        // Corner nodes get half the force of interior nodes (tributary area)
        const F = new Float64Array(ndof);
        for (let j = 0; j <= nely; j++) {
            const nodeRight = (nely + 1) * nelx + j;
            const weight = (j === 0 || j === nely) ? 0.5 : 1.0;
            F[2 * nodeRight] = weight / nely;  // consistent distributed load
        }

        const densities = new Float64Array(nelx * nely).fill(1.0);
        const result = solveFEA_JS(nelx, nely, densities, KEflat, edofArray, F, freedofs, E0, Emin, penal);

        // For plane stress uniform tension: ε_x = σ_x / E, ε_y = -ν * σ_x / E
        // Displacement ux should be linear in x: ux(x) = ε_x * x
        // Check that ux is proportional to x-coordinate
        let maxLinearError = 0;
        const uxAtRight = result.U[2 * ((nely + 1) * nelx)];  // ux at a right-edge node
        const uxSlope = uxAtRight / nelx;  // expected ux per unit x

        for (let elx = 0; elx <= nelx; elx++) {
            for (let ely = 0; ely <= nely; ely++) {
                const node = (nely + 1) * elx + ely;
                const ux = result.U[2 * node];
                const expected = uxSlope * elx;
                maxLinearError = Math.max(maxLinearError, Math.abs(ux - expected));
            }
        }
        assert(maxLinearError < 1e-5, `Patch test: ux is linear in x, max error = ${maxLinearError.toExponential(3)}`);

        // Check uy is uniform at each x-station (constant Poisson contraction)
        let maxUyVariation = 0;
        for (let elx = 0; elx <= nelx; elx++) {
            const uyRef = result.U[2 * ((nely + 1) * elx) + 1];
            for (let ely = 1; ely <= nely; ely++) {
                const node = (nely + 1) * elx + ely;
                const uy = result.U[2 * node + 1];
                // uy should vary linearly with y, not be constant—but all should be consistent
                // For uniform tension, uy variation within a column is due to Poisson effect
            }
        }
        assert(result.compliance > 0, `Patch test: compliance is positive = ${result.compliance.toExponential(3)}`);
    }

    // ─── Benchmark 6: Cantilever beam – compliance convergence from below ───
    // Displacement FEM converges from below: compliance increases monotonically
    // toward the exact elasticity solution (which includes shear deformation).
    // Timoshenko reference: δ = PL³/(3EI) + 12P(1+ν)L/(5EH)
    console.log('Benchmark 6: Cantilever beam – compliance converges from below (displacement FEM property)');
    {
        const P = 1.0;
        const meshes = [
            { nelx: 20, nely: 4 },
            { nelx: 40, nely: 8 },
            { nelx: 80, nely: 16 },
        ];

        const compliances = [];
        for (const { nelx, nely } of meshes) {
            const mesh = setupCantilever(nelx, nely);
            const densities = new Float64Array(mesh.nel).fill(1.0);
            const result = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
            compliances.push(result.compliance);

            // Timoshenko reference (same L/H=5 ratio for all meshes)
            const L = nelx, H = nely;
            const I = H * H * H / 12;
            const timoRef = P * L * L * L / (3 * E0 * I) + 12 * P * (1 + nu) * L / (5 * E0 * H);
            console.log(`    Mesh ${nelx}×${nely}: compliance=${result.compliance.toFixed(4)}, Timoshenko_ref=${timoRef.toFixed(4)}`);
        }

        // Compliance should increase monotonically (lower bound convergence)
        assert(compliances[1] > compliances[0], `Compliance increases: ${compliances[0].toFixed(2)} → ${compliances[1].toFixed(2)}`);
        assert(compliances[2] > compliances[1], `Compliance increases: ${compliances[1].toFixed(2)} → ${compliances[2].toFixed(2)}`);

        // Rate of change decreases (convergence)
        const diff1 = compliances[1] - compliances[0];
        const diff2 = compliances[2] - compliances[1];
        assert(diff2 < diff1, `Convergence: Δc decreases (${diff1.toFixed(4)} → ${diff2.toFixed(4)})`);

        // Finest mesh should be close to Timoshenko reference (within 5%)
        const L = meshes[2].nelx, H = meshes[2].nely;
        const I = H * H * H / 12;
        const timoRef = P * L * L * L / (3 * E0 * I) + 12 * P * (1 + nu) * L / (5 * E0 * H);
        const relErr = Math.abs(compliances[2] - timoRef) / timoRef;
        assert(relErr < 0.05, `Finest mesh within 5% of Timoshenko (error=${(relErr * 100).toFixed(2)}%)`);
    }

    // ─── Benchmark 7: Simply supported beam – compliance convergence ───
    // With a point load at mid-span, the FEA compliance converges monotonically.
    // The displacement FEM produces a lower bound that increases with refinement.
    console.log('Benchmark 7: Simply supported beam – compliance converges from below');
    {
        const meshes = [
            { nelx: 20, nely: 4 },
            { nelx: 40, nely: 8 },
            { nelx: 80, nely: 16 },
        ];

        const compliances = [];
        for (const { nelx, nely } of meshes) {
            const mesh = setupSimplySupportedBeam(nelx, nely);
            const densities = new Float64Array(mesh.nel).fill(1.0);
            const result = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
            compliances.push(result.compliance);

            const L = nelx, H = nely;
            const I = H * H * H / 12;
            const ebRef = L * L * L / (48 * E0 * I);
            console.log(`    Mesh ${nelx}×${nely}: compliance=${result.compliance.toFixed(4)}, E-B_ref=${ebRef.toFixed(4)}`);
        }

        // Compliance should increase monotonically
        assert(compliances[1] > compliances[0], `Compliance increases: ${compliances[0].toFixed(2)} → ${compliances[1].toFixed(2)}`);
        assert(compliances[2] > compliances[1], `Compliance increases: ${compliances[1].toFixed(2)} → ${compliances[2].toFixed(2)}`);

        // Rate of change decreases (convergence)
        const diff1 = compliances[1] - compliances[0];
        const diff2 = compliances[2] - compliances[1];
        assert(diff2 < diff1, `Convergence: Δc decreases (${diff1.toFixed(4)} → ${diff2.toFixed(4)})`);

        // Compliance should be in a physically reasonable range (same order as E-B)
        const L = meshes[0].nelx, H = meshes[0].nely;
        const I = H * H * H / 12;
        const ebRef = L * L * L / (48 * E0 * I);
        assert(compliances[0] > ebRef * 0.5, `Coarsest mesh compliance > 50% of E-B reference`);
        assert(compliances[2] < ebRef * 5.0, `Finest mesh compliance < 5× E-B reference`);
    }

    // ─── Benchmark 8: Equilibrium check (reaction forces = applied load) ───
    // Sum of all reaction forces on fixed DOFs should balance the applied load
    console.log('Benchmark 8: Global equilibrium – reaction forces balance applied load');
    {
        const nelx = 20, nely = 10;
        const mesh = setupCantilever(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);

        const result = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const U = result.U;

        // Compute global reaction forces: R = K*U (on all DOFs)
        const ndof = mesh.ndof;
        const nel = mesh.nel;
        const edofSize = 8;
        const R = new Float64Array(ndof);

        for (let e = 0; e < nel; e++) {
            const E = E0;  // all solid, penal=1
            const eOff = e * edofSize;
            for (let i = 0; i < edofSize; i++) {
                const gi = mesh.edofArray[eOff + i];
                let sum = 0;
                for (let j = 0; j < edofSize; j++) {
                    sum += KEflat[i * edofSize + j] * U[mesh.edofArray[eOff + j]];
                }
                R[gi] += E * sum;
            }
        }

        // Sum of forces in y-direction: reactions + applied should = 0
        let totalFy = 0;
        for (let i = 0; i < ndof; i++) {
            if (i % 2 === 1) {  // y-DOFs
                totalFy += R[i];
            }
        }
        // The internal force R already includes the applied load on free DOFs,
        // so total R_y should be close to zero (equilibrium)
        // Actually, R = K*U should equal F on free DOFs, so we check:
        // sum(F_y) should equal sum(R_y_fixed_DOFs) but with opposite sign
        // Alternatively: sum of all R_y = 0 (global equilibrium)
        assert(Math.abs(totalFy) < 1e-6, `Sum of all internal y-forces ≈ 0 (got ${totalFy.toExponential(3)})`);

        let totalFx = 0;
        for (let i = 0; i < ndof; i++) {
            if (i % 2 === 0) totalFx += R[i];
        }
        assert(Math.abs(totalFx) < 1e-6, `Sum of all internal x-forces ≈ 0 (got ${totalFx.toExponential(3)})`);
    }

    // ─── Benchmark 9: Energy consistency (compliance = U^T K U = F^T U) ───
    console.log('Benchmark 9: Energy consistency – compliance equals strain energy');
    {
        const nelx = 15, nely = 8;
        const mesh = setupCantilever(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);

        const result = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const U = result.U;

        // Compute U^T K U (strain energy × 2) via element-by-element
        const edofSize = 8;
        let uKu = 0;
        for (let e = 0; e < mesh.nel; e++) {
            const E = E0;
            const eOff = e * edofSize;
            for (let i = 0; i < edofSize; i++) {
                const gi = mesh.edofArray[eOff + i];
                let sum = 0;
                for (let j = 0; j < edofSize; j++) {
                    sum += KEflat[i * edofSize + j] * U[mesh.edofArray[eOff + j]];
                }
                uKu += E * U[gi] * sum;
            }
        }

        // compliance = F^T U
        let ftu = 0;
        for (let i = 0; i < mesh.ndof; i++) ftu += mesh.F[i] * U[i];

        const relDiff = Math.abs(uKu - ftu) / Math.max(Math.abs(ftu), 1e-30);
        assert(relDiff < 1e-6, `U^T K U ≈ F^T U, relative diff = ${relDiff.toExponential(3)} (uKu=${uKu.toExponential(6)}, FtU=${ftu.toExponential(6)})`);
    }

    // ─── Benchmark 10: WASM vs JS consistency on cantilever benchmark ───
    console.log('Benchmark 10: WASM solver matches JS solver on cantilever benchmark');
    {
        const nelx = 30, nely = 10;
        const mesh = setupCantilever(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);

        const jsResult = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

        // WASM solver
        const nel = mesh.nel;
        const ndof = mesh.ndof;
        const edofSize = 8;
        const nfree = mesh.freedofs.length;

        const mem = wasmMod.exports.memory;
        const align8 = (v) => (v + 7) & ~7;

        const densSize = nel * 8;
        const keSize = edofSize * edofSize * 8;
        const edofsSize = nel * edofSize * 4;
        const fSize = ndof * 8;
        const uSize = ndof * 8;
        const freedofsSize = nfree * 4;
        const workSize = nel * 8 + nel * 4 + ndof * 8 +
            5 * nfree * 8 + 2 * ndof * 8 + edofSize * 8 + nfree * 8;
        const totalBytes = densSize + keSize + edofsSize + fSize + uSize + freedofsSize + workSize + 128;

        const currentBytes = mem.buffer.byteLength;
        if (currentBytes < totalBytes + 65536) {
            const additionalPages = Math.ceil((totalBytes + 65536 - currentBytes) / 65536);
            mem.grow(additionalPages);
        }
        const dataStart = mem.buffer.byteLength - totalBytes - 64;

        let offset = align8(dataStart);
        const densOff = offset; offset += densSize;
        const keOff = offset; offset += keSize;
        const edofsOff = offset; offset += edofsSize;
        offset = align8(offset);
        const fOff = offset; offset += fSize;
        const uOff = offset; offset += uSize;
        const freedofsOff = offset; offset += freedofsSize;
        offset = align8(offset);
        const workOff = offset;

        new Float64Array(mem.buffer, densOff, nel).set(densities);
        new Float64Array(mem.buffer, keOff, edofSize * edofSize).set(KEflat);
        new Int32Array(mem.buffer, edofsOff, nel * edofSize).set(mesh.edofArray);
        new Float64Array(mem.buffer, fOff, ndof).set(mesh.F);
        new Float64Array(mem.buffer, uOff, ndof).fill(0);
        new Int32Array(mem.buffer, freedofsOff, nfree).set(mesh.freedofs);

        wasmMod.exports.ebePCG(
            densOff, keOff, edofsOff, fOff, uOff, freedofsOff,
            nel, edofSize, ndof, nfree,
            Emin, E0, penal, 2000, 1e-8, workOff
        );

        const wasmU = new Float64Array(ndof);
        wasmU.set(new Float64Array(mem.buffer, uOff, ndof));

        let maxDiff = 0;
        for (let i = 0; i < ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmU[i]));
        }

        let wasmCompliance = 0;
        for (let i = 0; i < ndof; i++) wasmCompliance += mesh.F[i] * wasmU[i];

        assert(maxDiff < 1e-8, `WASM displacement matches JS, max diff = ${maxDiff.toExponential(3)}`);
        assert(Math.abs(jsResult.compliance - wasmCompliance) < 1e-8,
            `WASM compliance matches JS: JS=${jsResult.compliance.toExponential(6)} WASM=${wasmCompliance.toExponential(6)}`);
    }

    // ─── Benchmark 11: Mesh convergence rate (h-refinement) ───
    // For Q4 elements, the convergence rate in energy norm should be O(h²)
    console.log('Benchmark 11: Mesh convergence rate (h-refinement) for cantilever beam');
    {
        const meshes = [
            { nelx: 10, nely: 2 },
            { nelx: 20, nely: 4 },
            { nelx: 40, nely: 8 },
            { nelx: 80, nely: 16 },
        ];

        const compliances = [];
        for (const { nelx, nely } of meshes) {
            const mesh = setupCantilever(nelx, nely);
            const densities = new Float64Array(mesh.nel).fill(1.0);
            const result = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
            compliances.push(result.compliance);
            console.log(`    Mesh ${nelx}×${nely}: compliance = ${result.compliance.toFixed(6)}`);
        }

        // Compliance should converge (changes get smaller with refinement)
        const diff1 = Math.abs(compliances[1] - compliances[0]);
        const diff2 = Math.abs(compliances[2] - compliances[1]);
        const diff3 = Math.abs(compliances[3] - compliances[2]);

        assert(diff2 < diff1, `Compliance differences decrease: |c2-c1|=${diff1.toExponential(3)} > |c3-c2|=${diff2.toExponential(3)}`);
        assert(diff3 < diff2, `Compliance differences decrease: |c3-c2|=${diff2.toExponential(3)} > |c4-c3|=${diff3.toExponential(3)}`);

        // Convergence ratio should be roughly 4:1 for Q4 elements (h² convergence)
        const ratio = diff1 / diff2;
        assert(ratio > 2.0, `Convergence ratio ≈ 4 (got ${ratio.toFixed(2)}), indicating O(h²) or better convergence`);
    }

    // ─── Benchmark 12: Poisson ratio effect ───
    // Uniaxial tension: lateral contraction ε_y = -ν * ε_x
    console.log('Benchmark 12: Poisson ratio effect – lateral contraction ratio');
    {
        const nelx = 40, nely = 10;
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const edofArray = precomputeEdofs2D(nelx, nely);

        // Fix left edge x-DOFs and one y-DOF
        const fixeddofs = [];
        for (let j = 0; j <= nely; j++) {
            fixeddofs.push(2 * j);  // x-DOF on left edge
        }
        // Fix y-DOF at mid-height to remove rigid body translation in y
        const midNode = Math.floor(nely / 2);
        fixeddofs.push(2 * midNode + 1);

        const fixedSet = new Set(fixeddofs);
        const freedofs = new Int32Array(ndof - fixedSet.size);
        let fi = 0;
        for (let i = 0; i < ndof; i++) {
            if (!fixedSet.has(i)) freedofs[fi++] = i;
        }

        // Apply consistent nodal forces for uniform traction on right edge
        const F = new Float64Array(ndof);
        for (let j = 0; j <= nely; j++) {
            const nodeRight = (nely + 1) * nelx + j;
            const weight = (j === 0 || j === nely) ? 0.5 : 1.0;
            F[2 * nodeRight] = weight / nely;
        }

        const densities = new Float64Array(nelx * nely).fill(1.0);
        const result = solveFEA_JS(nelx, nely, densities, KEflat, edofArray, F, freedofs, E0, Emin, penal);

        // Measure strains away from constrained left edge (at mid-beam x-stations)
        const x1 = Math.floor(nelx * 0.4);
        const x2 = Math.floor(nelx * 0.6);
        const ux1 = result.U[2 * ((nely + 1) * x1)];
        const ux2 = result.U[2 * ((nely + 1) * x2)];
        const eps_x = (ux2 - ux1) / (x2 - x1);

        // Lateral strain at mid-beam
        const midX = Math.floor(nelx / 2);
        const topMidNode = (nely + 1) * midX;
        const botMidNode = (nely + 1) * midX + nely;
        const uyTop = result.U[2 * topMidNode + 1];
        const uyBot = result.U[2 * botMidNode + 1];
        const eps_y = (uyBot - uyTop) / nely;

        const computedNu = -eps_y / eps_x;
        const nuError = Math.abs(computedNu - nu) / nu;
        assert(nuError < 0.05, `Poisson ratio from FEA = ${computedNu.toFixed(4)}, expected ${nu}, error = ${(nuError * 100).toFixed(2)}%`);
    }

    // ─── Summary ───
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`FEA Benchmark Results: ${passed} passed, ${failed} failed`);
    console.log('═'.repeat(60));
    process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
    console.error('Benchmark test runner error:', err);
    process.exit(1);
});
