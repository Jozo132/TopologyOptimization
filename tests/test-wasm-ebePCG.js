#!/usr/bin/env node
/**
 * Tests for the WASM-accelerated ebePCG function.
 * Verifies that the full self-contained WASM FEA solver produces the same
 * results as the JavaScript reference implementation.
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

// ─── 2D element stiffness matrix ───
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

// ─── JS reference FEA solver ───
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
    for (let i = 0; i < nfree; i++) {
        z[i] = invDiag[i] * r[i];
        p[i] = z[i];
        rz += r[i] * z[i];
    }

    const maxIter = Math.min(nfree, MAX_CG_ITERATIONS);
    const tolSq = CG_TOLERANCE * CG_TOLERANCE;

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

// ─── WASM ebePCG wrapper ───
function solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, edofArray, F, freedofs, E0, Emin, penal) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofSize = 8;
    const nfree = freedofs.length;

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
    new Int32Array(mem.buffer, edofsOff, nel * edofSize).set(edofArray);
    new Float64Array(mem.buffer, fOff, ndof).set(F);
    new Int32Array(mem.buffer, freedofsOff, nfree).set(freedofs);

    const iterations = wasmMod.exports.ebePCG(
        densOff, keOff, edofsOff, fOff, uOff, freedofsOff,
        nel, edofSize, ndof, nfree,
        Emin, E0, penal, 2000, 1e-8, workOff
    );

    const U = new Float64Array(ndof);
    U.set(new Float64Array(mem.buffer, uOff, ndof));

    let c = 0;
    for (let i = 0; i < ndof; i++) c += F[i] * U[i];
    return { U, compliance: c, iterations };
}

// ─── Setup mesh ───
function setupMesh(nelx, nely) {
    const nel = nelx * nely;
    const ndof = 2 * (nelx + 1) * (nely + 1);
    const edofSize = 8;

    const edofArray = new Int32Array(nel * edofSize);
    for (let ely = 0; ely < nely; ely++) {
        for (let elx = 0; elx < nelx; elx++) {
            const idx = ely + elx * nely;
            const offset = idx * edofSize;
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

    const fixeddofs = [];
    for (let j = 0; j <= nely; j++) {
        fixeddofs.push(2 * j);
        fixeddofs.push(2 * j + 1);
    }

    const fixedSet = new Set(fixeddofs);
    const freedofs = new Int32Array(ndof - fixedSet.size);
    let fi = 0;
    for (let i = 0; i < ndof; i++) {
        if (!fixedSet.has(i)) freedofs[fi++] = i;
    }

    const F = new Float64Array(ndof);
    const loadNode = (nely + 1) * nelx + nely;
    F[2 * loadNode + 1] = -1.0;

    return { nel, ndof, edofSize, edofArray, freedofs, F };
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════
async function runTests() {
    const wasmMod = await loadWasm();

    // ─── Test 1: WASM module exports ebePCG ───
    console.log('Test 1: WASM module exports ebePCG');
    assert(typeof wasmMod.exports.ebePCG === 'function', 'ebePCG should be exported');

    const nu = 0.3;
    const E0 = 1.0;
    const Emin = 1e-9;
    const penal = 3;
    const KE = lk2D(nu);
    const KEflat = flattenKE(KE, 8);

    // ─── Test 2: Small mesh (all solid) ───
    console.log('Test 2: Small mesh 5×5 (all solid)');
    {
        const nelx = 5, nely = 5;
        const mesh = setupMesh(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);

        const jsResult = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const wasmResult = solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

        let maxDiff = 0;
        for (let i = 0; i < mesh.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        assert(maxDiff < 1e-10, `Displacement match, max diff = ${maxDiff.toExponential(3)}`);
        assert(Math.abs(jsResult.compliance - wasmResult.compliance) < 1e-10,
            `Compliance match: JS=${jsResult.compliance.toExponential(3)} WASM=${wasmResult.compliance.toExponential(3)}`);
    }

    // ─── Test 3: Medium mesh (all solid) ───
    console.log('Test 3: Medium mesh 20×10 (all solid)');
    {
        const nelx = 20, nely = 10;
        const mesh = setupMesh(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);

        const jsResult = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const wasmResult = solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

        let maxDiff = 0;
        for (let i = 0; i < mesh.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        assert(maxDiff < 1e-10, `Displacement match, max diff = ${maxDiff.toExponential(3)}`);
        assert(Math.abs(jsResult.compliance - wasmResult.compliance) < 1e-10,
            `Compliance match: JS=${jsResult.compliance.toExponential(3)} WASM=${wasmResult.compliance.toExponential(3)}`);
    }

    // ─── Test 4: Mixed densities ───
    console.log('Test 4: Mixed densities (10×10, varying density)');
    {
        const nelx = 10, nely = 10;
        const mesh = setupMesh(nelx, nely);
        const densities = new Float64Array(mesh.nel);
        for (let e = 0; e < mesh.nel; e++) {
            densities[e] = 0.2 + 0.8 * (e / mesh.nel);
        }

        const jsResult = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const wasmResult = solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

        let maxDiff = 0;
        for (let i = 0; i < mesh.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        assert(maxDiff < 1e-8, `Displacement match, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 5: Sparse densities (many near-zero) ───
    console.log('Test 5: Sparse densities (15×15, 50% near-zero)');
    {
        const nelx = 15, nely = 15;
        const mesh = setupMesh(nelx, nely);
        const densities = new Float64Array(mesh.nel);
        for (let e = 0; e < mesh.nel; e++) {
            densities[e] = e % 2 === 0 ? 1.0 : 0.001;
        }

        const jsResult = solveFEA_JS(nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);
        const wasmResult = solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

        let maxDiff = 0;
        for (let i = 0; i < mesh.ndof; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(jsResult.U[i] - wasmResult.U[i]));
        }
        assert(maxDiff < 1e-6, `Displacement match with sparse densities, max diff = ${maxDiff.toExponential(3)}`);
    }

    // ─── Test 6: Compliance should be non-zero ───
    console.log('Test 6: Compliance should be non-zero for loaded structure');
    {
        const nelx = 10, nely = 5;
        const mesh = setupMesh(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);

        const wasmResult = solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, mesh.edofArray, mesh.F, mesh.freedofs, E0, Emin, penal);

        assert(wasmResult.compliance !== 0, `Compliance is non-zero: ${wasmResult.compliance.toExponential(3)}`);
        assert(wasmResult.iterations > 0, `CG iterations > 0: ${wasmResult.iterations}`);
    }

    // ─── Test 7: Zero force produces zero displacement ───
    console.log('Test 7: Zero force produces zero displacement');
    {
        const nelx = 5, nely = 5;
        const mesh = setupMesh(nelx, nely);
        const densities = new Float64Array(mesh.nel).fill(1.0);
        const zeroF = new Float64Array(mesh.ndof);

        const wasmResult = solveFEA_WASM(wasmMod, nelx, nely, densities, KEflat, mesh.edofArray, zeroF, mesh.freedofs, E0, Emin, penal);

        let maxU = 0;
        for (let i = 0; i < mesh.ndof; i++) {
            maxU = Math.max(maxU, Math.abs(wasmResult.U[i]));
        }
        assert(maxU < 1e-15, `Zero force gives zero displacement, max |U| = ${maxU.toExponential(3)}`);
    }

    // ─── Summary ───
    console.log(`\nResults: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
    console.error('Test runner error:', err);
    process.exit(1);
});
