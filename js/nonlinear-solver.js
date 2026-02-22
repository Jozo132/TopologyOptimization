/**
 * Nonlinear FEA Solver for 8-node hexahedral elements.
 *
 * Implements a Total Lagrangian formulation with Newton-Raphson iteration,
 * element-by-element Jacobi-preconditioned CG for the linearized system,
 * and full stress recovery (Cauchy, von Mises, principal, triaxiality).
 *
 * @module nonlinear-solver
 */

import { createMaterial, MaterialState } from './material-models.js';
import { PhaseFieldFracture, ElementErosion } from './fracture-solver.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON = 1e-12;
const MAX_CG_ITERATIONS = 400;
const CG_TOLERANCE = 1e-8;

/** 2×2×2 Gauss quadrature: points and weights */
const GP = 1.0 / Math.sqrt(3.0);
const GAUSS_POINTS = [
    [-GP, -GP, -GP],
    [ GP, -GP, -GP],
    [ GP,  GP, -GP],
    [-GP,  GP, -GP],
    [-GP, -GP,  GP],
    [ GP, -GP,  GP],
    [ GP,  GP,  GP],
    [-GP,  GP,  GP]
];
const GAUSS_WEIGHT = 1.0; // each weight is 1×1×1 = 1

/** 3×3 identity (flat row-major) */
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// ---------------------------------------------------------------------------
// Small matrix / tensor helpers (flat row-major 3×3)
// ---------------------------------------------------------------------------

/**
 * Determinant of a flat 3×3 matrix.
 * @param {number[]} A - 9-element row-major array
 * @returns {number}
 */
function det3(A) {
    return (
        A[0] * (A[4] * A[8] - A[5] * A[7]) -
        A[1] * (A[3] * A[8] - A[5] * A[6]) +
        A[2] * (A[3] * A[7] - A[4] * A[6])
    );
}

/**
 * Multiply two flat 3×3 matrices C = A·B.
 * @param {number[]} A
 * @param {number[]} B
 * @returns {number[]} C (9-element)
 */
function mul33(A, B) {
    const C = new Array(9);
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            C[i * 3 + j] =
                A[i * 3]     * B[j] +
                A[i * 3 + 1] * B[3 + j] +
                A[i * 3 + 2] * B[6 + j];
        }
    }
    return C;
}

/**
 * Transpose of a flat 3×3 matrix.
 * @param {number[]} A
 * @returns {number[]}
 */
function transpose3(A) {
    return [
        A[0], A[3], A[6],
        A[1], A[4], A[7],
        A[2], A[5], A[8]
    ];
}

/**
 * Inverse of a flat 3×3 matrix.
 * @param {number[]} A
 * @returns {number[]}
 */
function inv3(A) {
    const d = det3(A);
    const invD = 1.0 / d;
    return [
        (A[4] * A[8] - A[5] * A[7]) * invD,
        (A[2] * A[7] - A[1] * A[8]) * invD,
        (A[1] * A[5] - A[2] * A[4]) * invD,
        (A[5] * A[6] - A[3] * A[8]) * invD,
        (A[0] * A[8] - A[2] * A[6]) * invD,
        (A[2] * A[3] - A[0] * A[5]) * invD,
        (A[3] * A[7] - A[4] * A[6]) * invD,
        (A[1] * A[6] - A[0] * A[7]) * invD,
        (A[0] * A[4] - A[1] * A[3]) * invD
    ];
}

// ---------------------------------------------------------------------------
// Shape functions for 8-node hex
// ---------------------------------------------------------------------------

/**
 * Evaluate trilinear shape functions at a natural coordinate point.
 * @param {number} xi
 * @param {number} eta
 * @param {number} zeta
 * @returns {number[]} N – 8 shape function values
 */
function shapeFunctions(xi, eta, zeta) {
    return [
        0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
        0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
        0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
        0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
        0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
        0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
        0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
        0.125 * (1 - xi) * (1 + eta) * (1 + zeta)
    ];
}

/**
 * Shape function derivatives w.r.t. natural coordinates (∂N/∂ξ, ∂N/∂η, ∂N/∂ζ).
 * Returns an 8×3 array (flat 24-element, row-major per node).
 * @param {number} xi
 * @param {number} eta
 * @param {number} zeta
 * @returns {Float64Array} dN – 24 values [dN0/dξ, dN0/dη, dN0/dζ, dN1/dξ, ...]
 */
function shapeFunctionDerivatives(xi, eta, zeta) {
    const dN = new Float64Array(24);
    // node 0: (1-xi)(1-eta)(1-zeta)/8
    dN[0]  = -0.125 * (1 - eta) * (1 - zeta);
    dN[1]  = -0.125 * (1 - xi)  * (1 - zeta);
    dN[2]  = -0.125 * (1 - xi)  * (1 - eta);
    // node 1: (1+xi)(1-eta)(1-zeta)/8
    dN[3]  =  0.125 * (1 - eta) * (1 - zeta);
    dN[4]  = -0.125 * (1 + xi)  * (1 - zeta);
    dN[5]  = -0.125 * (1 + xi)  * (1 - eta);
    // node 2: (1+xi)(1+eta)(1-zeta)/8
    dN[6]  =  0.125 * (1 + eta) * (1 - zeta);
    dN[7]  =  0.125 * (1 + xi)  * (1 - zeta);
    dN[8]  = -0.125 * (1 + xi)  * (1 + eta);
    // node 3: (1-xi)(1+eta)(1-zeta)/8
    dN[9]  = -0.125 * (1 + eta) * (1 - zeta);
    dN[10] =  0.125 * (1 - xi)  * (1 - zeta);
    dN[11] = -0.125 * (1 - xi)  * (1 + eta);
    // node 4: (1-xi)(1-eta)(1+zeta)/8
    dN[12] = -0.125 * (1 - eta) * (1 + zeta);
    dN[13] = -0.125 * (1 - xi)  * (1 + zeta);
    dN[14] =  0.125 * (1 - xi)  * (1 - eta);
    // node 5: (1+xi)(1-eta)(1+zeta)/8
    dN[15] =  0.125 * (1 - eta) * (1 + zeta);
    dN[16] = -0.125 * (1 + xi)  * (1 + zeta);
    dN[17] =  0.125 * (1 + xi)  * (1 - eta);
    // node 6: (1+xi)(1+eta)(1+zeta)/8
    dN[18] =  0.125 * (1 + eta) * (1 + zeta);
    dN[19] =  0.125 * (1 + xi)  * (1 + zeta);
    dN[20] =  0.125 * (1 + xi)  * (1 + eta);
    // node 7: (1-xi)(1+eta)(1+zeta)/8
    dN[21] = -0.125 * (1 + eta) * (1 + zeta);
    dN[22] =  0.125 * (1 - xi)  * (1 + zeta);
    dN[23] =  0.125 * (1 - xi)  * (1 + eta);
    return dN;
}

// ---------------------------------------------------------------------------
// Deformation gradient & strain helpers
// ---------------------------------------------------------------------------

/**
 * Compute the Jacobian matrix J = ∂x/∂ξ and its inverse for the reference
 * configuration at a given natural coordinate point.
 * @param {Float64Array} nodeCoords - 24 values [x0,y0,z0, x1,y1,z1, ...]
 * @param {Float64Array} dN - shape function derivatives (24 values)
 * @returns {{ J: number[], invJ: number[], detJ: number }}
 */
function jacobian(nodeCoords, dN) {
    const J = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let n = 0; n < 8; n++) {
        const x = nodeCoords[n * 3];
        const y = nodeCoords[n * 3 + 1];
        const z = nodeCoords[n * 3 + 2];
        const off = n * 3;
        // J[i][j] = Σ_n x_n_j · dN_n / dξ_i
        J[0] += dN[off]     * x;  J[1] += dN[off]     * y;  J[2] += dN[off]     * z;
        J[3] += dN[off + 1] * x;  J[4] += dN[off + 1] * y;  J[5] += dN[off + 1] * z;
        J[6] += dN[off + 2] * x;  J[7] += dN[off + 2] * y;  J[8] += dN[off + 2] * z;
    }
    const detJ = det3(J);
    const invJ = inv3(J);
    return { J, invJ, detJ };
}

/**
 * Compute the deformation gradient F = I + ∂u/∂X at a natural coordinate
 * point within an element.
 *
 * Uses the Total Lagrangian formulation: F is computed with respect to the
 * reference (undeformed) configuration.
 *
 * @param {Float64Array} nodeCoords - Reference coordinates (24 values)
 * @param {Float64Array} u_elem - Element displacements (24 values)
 * @param {number} xi
 * @param {number} eta
 * @param {number} zeta
 * @returns {{ F: number[], detF: number, dNdX: Float64Array }}
 */
export function computeDeformationGradient(nodeCoords, u_elem, xi, eta, zeta) {
    const dN = shapeFunctionDerivatives(xi, eta, zeta);
    const { invJ, detJ } = jacobian(nodeCoords, dN);

    // dN/dX = invJ · dN/dξ  (shape function derivatives in physical coords)
    const dNdX = new Float64Array(24);
    for (let n = 0; n < 8; n++) {
        const off = n * 3;
        const d0 = dN[off], d1 = dN[off + 1], d2 = dN[off + 2];
        dNdX[off]     = invJ[0] * d0 + invJ[1] * d1 + invJ[2] * d2;
        dNdX[off + 1] = invJ[3] * d0 + invJ[4] * d1 + invJ[5] * d2;
        dNdX[off + 2] = invJ[6] * d0 + invJ[7] * d1 + invJ[8] * d2;
    }

    // F_ij = δ_ij + Σ_n u_n_i · dN_n/dX_j
    const F = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (let n = 0; n < 8; n++) {
        const uOff = n * 3;
        const dOff = n * 3;
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                F[i * 3 + j] += u_elem[uOff + i] * dNdX[dOff + j];
            }
        }
    }

    const detF = det3(F);
    return { F, detF, dNdX, detJ };
}

/**
 * Compute Green-Lagrange strain E = 0.5(F^T·F - I).
 * @param {number[]} F - Deformation gradient (flat 3×3)
 * @returns {number[]} E in Voigt form [E11, E22, E33, 2E12, 2E23, 2E13]
 */
export function greenLagrangeStrain(F) {
    const FtF = mul33(transpose3(F), F);
    return [
        0.5 * (FtF[0] - 1),
        0.5 * (FtF[4] - 1),
        0.5 * (FtF[8] - 1),
        FtF[1],  // 2*E12 = C12
        FtF[5],  // 2*E23 = C23
        FtF[2]   // 2*E13 = C13
    ];
}

/**
 * Compute strain energy density Ψ = 0.5 * S : E (per unit reference volume).
 * @param {number[]} S_voigt - 2nd Piola-Kirchhoff stress in Voigt [S11,S22,S33,S12,S23,S13]
 * @param {number[]} E_voigt - Green-Lagrange strain in Voigt [E11,E22,E33,2E12,2E23,2E13]
 * @returns {number}
 */
export function strainEnergyDensity(S_voigt, E_voigt) {
    return 0.5 * (
        S_voigt[0] * E_voigt[0] +
        S_voigt[1] * E_voigt[1] +
        S_voigt[2] * E_voigt[2] +
        S_voigt[3] * E_voigt[3] +  // S12 * 2E12
        S_voigt[4] * E_voigt[4] +
        S_voigt[5] * E_voigt[5]
    );
}

// ---------------------------------------------------------------------------
// Stress recovery
// ---------------------------------------------------------------------------

/**
 * Push-forward 2nd Piola-Kirchhoff stress to Cauchy stress: σ = (1/J) F·S·F^T.
 * @param {number[]} F - Deformation gradient (flat 3×3)
 * @param {number[]} S_voigt - S in Voigt [S11,S22,S33,S12,S23,S13]
 * @returns {number[]} Cauchy stress in Voigt [σ11,σ22,σ33,σ12,σ23,σ13]
 */
export function cauchyStress(F, S_voigt) {
    // Expand Voigt to full 3×3 symmetric
    const S = [
        S_voigt[0], S_voigt[3], S_voigt[5],
        S_voigt[3], S_voigt[1], S_voigt[4],
        S_voigt[5], S_voigt[4], S_voigt[2]
    ];
    const J = det3(F);
    const invJ = 1.0 / J;
    const FS = mul33(F, S);
    const Ft = transpose3(F);
    const sig = mul33(FS, Ft);
    return [
        invJ * sig[0], invJ * sig[4], invJ * sig[8],
        invJ * sig[1], invJ * sig[5], invJ * sig[2]
    ];
}

/**
 * Von Mises equivalent stress from Voigt stress.
 * @param {number[]} s - Voigt [σ11,σ22,σ33,σ12,σ23,σ13]
 * @returns {number}
 */
export function vonMises(s) {
    const d01 = s[0] - s[1];
    const d12 = s[1] - s[2];
    const d20 = s[2] - s[0];
    return Math.sqrt(0.5 * (d01 * d01 + d12 * d12 + d20 * d20) +
        3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
}

/**
 * Principal stresses (eigenvalues of symmetric 3×3 stress tensor).
 * Uses the analytical cubic formula.
 * @param {number[]} s - Voigt [σ11,σ22,σ33,σ12,σ23,σ13]
 * @returns {number[]} [σ1, σ2, σ3] sorted descending (σ1 ≥ σ2 ≥ σ3)
 */
export function principalStresses(s) {
    const I1 = s[0] + s[1] + s[2];
    const I2 = s[0] * s[1] + s[1] * s[2] + s[2] * s[0] -
               s[3] * s[3] - s[4] * s[4] - s[5] * s[5];
    const I3 = s[0] * s[1] * s[2] +
               2.0 * s[3] * s[4] * s[5] -
               s[0] * s[4] * s[4] -
               s[1] * s[5] * s[5] -
               s[2] * s[3] * s[3];

    const p = I1 * I1 / 3.0 - I2;
    const q = 2.0 * I1 * I1 * I1 / 27.0 - I1 * I2 / 3.0 + I3;

    if (p < EPSILON) {
        const v = I1 / 3.0;
        return [v, v, v];
    }

    const sp = Math.sqrt(p / 3.0);
    let phi = Math.acos(Math.max(-1, Math.min(1, -q / (2.0 * sp * sp * sp))));
    phi /= 3.0;

    const s1 = I1 / 3.0 + 2.0 * sp * Math.cos(phi);
    const s2 = I1 / 3.0 + 2.0 * sp * Math.cos(phi - 2.0 * Math.PI / 3.0);
    const s3 = I1 / 3.0 + 2.0 * sp * Math.cos(phi + 2.0 * Math.PI / 3.0);

    const arr = [s1, s2, s3];
    arr.sort((a, b) => b - a);
    return arr;
}

/**
 * Stress triaxiality η = σ_h / σ_vm (hydrostatic / von Mises).
 * @param {number[]} s - Voigt stress
 * @returns {number}
 */
export function stressTriaxiality(s) {
    const hydrostat = (s[0] + s[1] + s[2]) / 3.0;
    const vm = vonMises(s);
    return vm > EPSILON ? hydrostat / vm : 0.0;
}

// ---------------------------------------------------------------------------
// Element computations (Total Lagrangian, 8-node hex)
// ---------------------------------------------------------------------------

/**
 * Compute the element internal force vector and tangent stiffness matrix.
 *
 * Total Lagrangian formulation:
 *   - Integrates over the reference configuration
 *   - Uses material.computeStress(F, state) → {stress, tangent, state}
 *     where stress is 2nd PK (Voigt) and tangent is 6×6 material tangent
 *   - Assembles the material tangent contribution and geometric (initial
 *     stress) stiffness contribution
 *
 * @param {Float64Array} nodeCoords - Reference node coordinates (24 values)
 * @param {Float64Array} u_elem - Element displacement vector (24 values)
 * @param {object} material - Material model with computeStress(F, state)
 * @param {MaterialState[]|null} state - Array of 8 MaterialState (one per GP), or null
 * @returns {{ fint: Float64Array, Kt: Float64Array, newState: MaterialState[],
 *             strainEnergy: number }}
 */
export function elementForceAndTangent(nodeCoords, u_elem, material, state) {
    const fint = new Float64Array(24);
    const Kt = new Float64Array(576); // 24×24 flat row-major
    const newState = new Array(8);
    let strainEnergy = 0.0;

    for (let gp = 0; gp < 8; gp++) {
        const [xi, eta, zeta] = GAUSS_POINTS[gp];
        const gpState = state ? state[gp] : null;

        // Deformation gradient and shape function derivatives in reference config
        const { F, detF, dNdX, detJ } = computeDeformationGradient(
            nodeCoords, u_elem, xi, eta, zeta
        );

        // Material response: 2nd Piola-Kirchhoff stress S and material tangent C
        const matResult = material.computeStress(F, gpState);
        const S_voigt = matResult.stress;   // [S11,S22,S33,S12,S23,S13]
        const C_flat = matResult.tangent;   // 6×6 flat row-major (36 values)
        newState[gp] = matResult.state || new MaterialState();

        // Green-Lagrange strain for energy
        const E_voigt = greenLagrangeStrain(F);
        strainEnergy += strainEnergyDensity(S_voigt, E_voigt) * detJ * GAUSS_WEIGHT;

        // Expand S to full 3×3
        const S = [
            S_voigt[0], S_voigt[3], S_voigt[5],
            S_voigt[3], S_voigt[1], S_voigt[4],
            S_voigt[5], S_voigt[4], S_voigt[2]
        ];

        // --- BL matrix (6×24): linear strain-displacement in reference config ---
        // BL relates δE (Voigt) to δu through F
        // BL[α][nI] where α = Voigt index, nI = node n, dof I
        // For Total Lagrangian: BL_ij = F_ki · dN_n/dX_j (contracted appropriately)
        const BL = new Float64Array(144); // 6×24

        for (let n = 0; n < 8; n++) {
            const dOff = n * 3;
            const col = n * 3;
            const dNx = dNdX[dOff];
            const dNy = dNdX[dOff + 1];
            const dNz = dNdX[dOff + 2];

            // BL row mapping:
            // 0:E11, 1:E22, 2:E33, 3:2E12, 4:2E23, 5:2E13
            // BL[0][col+i] = F[i][0] * dNx  (∂E11/∂u_i)
            // BL[1][col+i] = F[i][1] * dNy
            // BL[2][col+i] = F[i][2] * dNz
            // BL[3][col+i] = F[i][0] * dNy + F[i][1] * dNx
            // BL[4][col+i] = F[i][1] * dNz + F[i][2] * dNy
            // BL[5][col+i] = F[i][0] * dNz + F[i][2] * dNx
            for (let i = 0; i < 3; i++) {
                const Fi0 = F[i * 3];
                const Fi1 = F[i * 3 + 1];
                const Fi2 = F[i * 3 + 2];
                BL[0 * 24 + col + i] = Fi0 * dNx;
                BL[1 * 24 + col + i] = Fi1 * dNy;
                BL[2 * 24 + col + i] = Fi2 * dNz;
                BL[3 * 24 + col + i] = Fi0 * dNy + Fi1 * dNx;
                BL[4 * 24 + col + i] = Fi1 * dNz + Fi2 * dNy;
                BL[5 * 24 + col + i] = Fi0 * dNz + Fi2 * dNx;
            }
        }

        const w = detJ * GAUSS_WEIGHT;

        // --- Internal force: fint += BL^T · S_voigt · detJ · w ---
        for (let a = 0; a < 24; a++) {
            let f = 0.0;
            for (let alpha = 0; alpha < 6; alpha++) {
                f += BL[alpha * 24 + a] * S_voigt[alpha];
            }
            fint[a] += f * w;
        }

        // --- Material tangent stiffness: Kt_mat = BL^T · C · BL · detJ · w ---
        // C · BL (6×24)
        const CB = new Float64Array(144);
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 24; j++) {
                let s = 0.0;
                for (let k = 0; k < 6; k++) {
                    s += C_flat[i * 6 + k] * BL[k * 24 + j];
                }
                CB[i * 24 + j] = s;
            }
        }
        // BL^T · CB (24×24)
        for (let a = 0; a < 24; a++) {
            for (let b = 0; b < 24; b++) {
                let s = 0.0;
                for (let alpha = 0; alpha < 6; alpha++) {
                    s += BL[alpha * 24 + a] * CB[alpha * 24 + b];
                }
                Kt[a * 24 + b] += s * w;
            }
        }

        // --- Geometric (initial stress) stiffness: Kt_geo ---
        // K_geo[nI, mJ] = δ_IJ · Σ_kl S_kl · dN_n/dX_k · dN_m/dX_l · detJ · w
        for (let n = 0; n < 8; n++) {
            const nOff = n * 3;
            for (let m = 0; m < 8; m++) {
                const mOff = m * 3;
                // Scalar: Σ S_kl · dN_n/dX_k · dN_m/dX_l
                let sNN = 0.0;
                for (let k = 0; k < 3; k++) {
                    for (let l = 0; l < 3; l++) {
                        sNN += S[k * 3 + l] * dNdX[nOff + k] * dNdX[mOff + l];
                    }
                }
                sNN *= w;
                // Add to diagonal blocks (I = J only)
                for (let I = 0; I < 3; I++) {
                    Kt[(n * 3 + I) * 24 + (m * 3 + I)] += sNN;
                }
            }
        }
    }

    return { fint, Kt, newState, strainEnergy };
}

// ---------------------------------------------------------------------------
// CG solver (element-by-element, Jacobi preconditioned)
// ---------------------------------------------------------------------------

/**
 * Element-by-element matrix-vector product: y = Σ_e Kt_e · x_e.
 * Operates on full-DOF vectors; caller restricts to free DOFs.
 * @param {object} mesh
 * @param {Float64Array[]} elemTangents - Per-element 24×24 tangent (flat)
 * @param {Float64Array} x_full - Input vector (ndof)
 * @param {Float64Array} y_full - Output vector (ndof), zeroed on entry
 */
function ebeMatVec(mesh, elemTangents, x_full, y_full) {
    const elemCount = mesh.elemCount;
    for (let e = 0; e < elemCount; e++) {
        const Kt = elemTangents[e];
        if (!Kt) continue;
        const nodes = mesh.getElementNodes(e);
        // Gather element DOFs
        for (let i = 0; i < 8; i++) {
            const ni = nodes[i];
            const rowBase = i * 3;
            for (let di = 0; di < 3; di++) {
                const gi = ni * 3 + di;
                const row = rowBase + di;
                let sum = 0.0;
                for (let j = 0; j < 8; j++) {
                    const nj = nodes[j];
                    const colBase = j * 3;
                    for (let dj = 0; dj < 3; dj++) {
                        sum += Kt[row * 24 + colBase + dj] * x_full[nj * 3 + dj];
                    }
                }
                y_full[gi] += sum;
            }
        }
    }
}

/**
 * Compute Jacobi preconditioner (inverse diagonal) from element tangents.
 * @param {object} mesh
 * @param {Float64Array[]} elemTangents
 * @param {number} ndof
 * @returns {Float64Array} invDiag (ndof)
 */
function computeJacobiPreconditioner(mesh, elemTangents, ndof) {
    const diag = new Float64Array(ndof);
    const elemCount = mesh.elemCount;
    for (let e = 0; e < elemCount; e++) {
        const Kt = elemTangents[e];
        if (!Kt) continue;
        const nodes = mesh.getElementNodes(e);
        for (let i = 0; i < 8; i++) {
            for (let di = 0; di < 3; di++) {
                const local = i * 3 + di;
                diag[nodes[i] * 3 + di] += Kt[local * 24 + local];
            }
        }
    }
    const invDiag = new Float64Array(ndof);
    for (let i = 0; i < ndof; i++) {
        invDiag[i] = Math.abs(diag[i]) > EPSILON ? 1.0 / diag[i] : 0.0;
    }
    return invDiag;
}

/**
 * Solve K·δu = rhs using Jacobi-preconditioned CG with element-by-element
 * matrix-vector products.
 *
 * @param {object} mesh
 * @param {Float64Array[]} elemTangents - Per-element 24×24 tangent (flat)
 * @param {Float64Array} rhs - Right-hand side (ndof)
 * @param {Int32Array|Uint32Array} fixedDOFs - Fixed DOF indices
 * @param {number} ndof
 * @param {number} [tol=CG_TOLERANCE]
 * @param {number} [maxIter=MAX_CG_ITERATIONS]
 * @returns {{ x: Float64Array, iterations: number, residualNorm: number }}
 */
function solveCG(mesh, elemTangents, rhs, fixedDOFs, ndof, tol, maxIter) {
    tol = tol || CG_TOLERANCE;
    maxIter = maxIter || MAX_CG_ITERATIONS;

    // Build fixed-DOF mask
    const isFixed = new Uint8Array(ndof);
    for (let i = 0; i < fixedDOFs.length; i++) {
        isFixed[fixedDOFs[i]] = 1;
    }

    const invDiag = computeJacobiPreconditioner(mesh, elemTangents, ndof);
    // Zero preconditioner on fixed DOFs
    for (let i = 0; i < fixedDOFs.length; i++) {
        invDiag[fixedDOFs[i]] = 0.0;
    }

    const u = new Float64Array(ndof);
    const r = new Float64Array(ndof);
    const z = new Float64Array(ndof);
    const p = new Float64Array(ndof);
    const Ap = new Float64Array(ndof);

    // r = rhs (fixed DOFs zeroed)
    for (let i = 0; i < ndof; i++) {
        r[i] = isFixed[i] ? 0.0 : rhs[i];
        z[i] = invDiag[i] * r[i];
        p[i] = z[i];
    }

    let rz = 0.0;
    for (let i = 0; i < ndof; i++) rz += r[i] * z[i];

    const rhsNorm = Math.sqrt(rz > 0 ? rz : _dot(r, r, ndof));
    if (rhsNorm < EPSILON) {
        return { x: u, iterations: 0, residualNorm: 0 };
    }

    let iter;
    let rnorm = 0.0;
    for (iter = 0; iter < maxIter; iter++) {
        rnorm = Math.sqrt(_dot(r, r, ndof));
        if (rnorm < tol * rhsNorm + EPSILON) break;

        // Ap = K·p
        Ap.fill(0);
        ebeMatVec(mesh, elemTangents, p, Ap);
        // Zero fixed DOFs
        for (let i = 0; i < fixedDOFs.length; i++) Ap[fixedDOFs[i]] = 0.0;

        let pAp = _dot(p, Ap, ndof);
        if (Math.abs(pAp) < EPSILON) break;
        const alpha = rz / pAp;

        let rz_new = 0.0;
        for (let i = 0; i < ndof; i++) {
            u[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            z[i] = invDiag[i] * r[i];
            rz_new += r[i] * z[i];
        }

        const beta = rz_new / (rz + EPSILON);
        for (let i = 0; i < ndof; i++) {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }

    return { x: u, iterations: iter, residualNorm: rnorm };
}

/** @param {Float64Array} a @param {Float64Array} b @param {number} n */
function _dot(a, b, n) {
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// ---------------------------------------------------------------------------
// NonlinearSolver
// ---------------------------------------------------------------------------

/**
 * @typedef {object} NonlinearConfig
 * @property {number} [maxNewtonIter=20] - Maximum Newton iterations per load step
 * @property {number} [residualTol=1e-6] - Convergence tolerance on relative residual norm
 * @property {number} [incrementTol=1e-6] - Convergence tolerance on displacement increment norm
 * @property {number} [numLoadSteps=10] - Number of incremental load steps
 * @property {number} [lineSearchMaxIter=5] - Maximum line-search bisections
 * @property {number} [lineSearchBeta=0.5] - Line-search reduction factor
 * @property {number} [cgTolerance=1e-8] - CG solver tolerance
 * @property {number} [cgMaxIter=400] - CG maximum iterations
 */

/**
 * @typedef {object} NonlinearResult
 * @property {Float64Array} displacement - Global displacement vector (ndof)
 * @property {Float64Array} cauchyStress - Per-element average Cauchy stress (elemCount × 6, Voigt)
 * @property {Float64Array} vonMisesStress - Per-element average von Mises stress (elemCount)
 * @property {Float64Array} principalStressField - Per-element principal stresses (elemCount × 3)
 * @property {Float64Array} triaxiality - Per-element stress triaxiality (elemCount)
 * @property {Float64Array} strainEnergy - Per-element strain energy (elemCount)
 * @property {MaterialState[][]} materialStates - Per-element per-GP material state
 * @property {boolean} converged
 * @property {number} totalIterations
 */

/**
 * Nonlinear FEA solver using Total Lagrangian formulation and Newton-Raphson.
 *
 * Uses element-by-element Jacobi-preconditioned CG for the linearized system
 * at each Newton step. Supports arbitrary material models via the
 * material-models.js interface.
 */
export class NonlinearSolver {
    /**
     * @param {NonlinearConfig} [config]
     */
    constructor(config) {
        const c = config || {};
        /** @type {number} */ this.maxNewtonIter = c.maxNewtonIter || 20;
        /** @type {number} */ this.residualTol = c.residualTol || 1e-6;
        /** @type {number} */ this.incrementTol = c.incrementTol || 1e-6;
        /** @type {number} */ this.numLoadSteps = c.numLoadSteps || 10;
        /** @type {number} */ this.lineSearchMaxIter = c.lineSearchMaxIter || 5;
        /** @type {number} */ this.lineSearchBeta = c.lineSearchBeta || 0.5;
        /** @type {number} */ this.cgTolerance = c.cgTolerance || CG_TOLERANCE;
        /** @type {number} */ this.cgMaxIter = c.cgMaxIter || MAX_CG_ITERATIONS;

        // Fracture / damage configuration
        /** @type {boolean} */ this.enableDamage = c.enableDamage || false;
        /** @type {number} */ this.fractureToughness = c.fractureToughness || 2700;
        /** @type {number} */ this.fractureLengthScale = c.fractureLengthScale || 0.01;
        /** @type {number} */ this.erosionThreshold = c.erosionThreshold || 0.95;
    }

    /**
     * Run the full nonlinear analysis with incremental loading and Newton-Raphson.
     *
     * @param {object} mesh - Mesh descriptor
     * @param {number} mesh.nelx
     * @param {number} mesh.nely
     * @param {number} mesh.nelz
     * @param {number} mesh.nodeCount
     * @param {number} mesh.elemCount
     * @param {function(number): number[]} mesh.getElementNodes - Returns 8 node indices
     * @param {function(number): number[]} mesh.getNodeCoords - Returns [x, y, z]
     * @param {object} material - Material model with computeStress(F, state)
     * @param {Int32Array|Uint32Array|number[]} constraints - Fixed DOF indices
     * @param {Float64Array} loads - Full external force vector (ndof)
     * @param {function} [progressCallback] - Called with { step, iteration, residualNorm }
     * @returns {NonlinearResult}
     */
    solve(mesh, material, constraints, loads, progressCallback) {
        const ndof = mesh.nodeCount * 3;
        const fixedDOFs = constraints instanceof Int32Array || constraints instanceof Uint32Array
            ? constraints
            : new Int32Array(constraints);

        // Total displacement
        const u = new Float64Array(ndof);

        // Per-element per-GP material state
        let elemStates = new Array(mesh.elemCount);
        for (let e = 0; e < mesh.elemCount; e++) {
            elemStates[e] = null; // will be initialized on first use
        }

        let totalIterations = 0;
        let converged = true;
        let failedAtStep = -1;

        // Phase-field damage tracking
        let damageField = null;
        let damageHistory = null;
        let erodedSet = null;
        let phaseField = null;
        let erosion = null;

        if (this.enableDamage) {
            // Build flat connectivity and nodeCoords arrays for fracture solver
            const connectivity = new Int32Array(mesh.elemCount * 8);
            const nodeCoords = new Float64Array(mesh.nodeCount * 3);
            for (let e = 0; e < mesh.elemCount; e++) {
                const nodes = mesh.getElementNodes(e);
                for (let n = 0; n < 8; n++) connectivity[e * 8 + n] = nodes[n];
            }
            for (let n = 0; n < mesh.nodeCount; n++) {
                const c = mesh.getNodeCoords(n);
                nodeCoords[n * 3]     = c[0];
                nodeCoords[n * 3 + 1] = c[1];
                nodeCoords[n * 3 + 2] = c[2];
            }

            // Material properties for energy split
            const E  = material.E  || 210000;
            const nu = material.nu || 0.3;

            phaseField = new PhaseFieldFracture({
                Gc: this.fractureToughness,
                lengthScale: this.fractureLengthScale,
                E, nu
            });
            erosion = new ElementErosion({ threshold: this.erosionThreshold });

            const init = phaseField.initializeField(mesh.elemCount);
            damageField = init.d;
            damageHistory = init.H;
            erodedSet = erosion.erodedElements;

            // Attach flat arrays to mesh for fracture solver access
            mesh._fractureConnectivity = connectivity;
            mesh._fractureNodeCoords = nodeCoords;
        }

        // Per-step snapshots for time slider playback
        const stepSnapshots = [];

        // Incremental load stepping
        for (let step = 1; step <= this.numLoadSteps; step++) {
            const loadFraction = step / this.numLoadSteps;

            // Scaled external force for this step
            const f_ext = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) {
                f_ext[i] = loads[i] * loadFraction;
            }

            // Zero out forces on eroded element DOFs
            if (erodedSet && erodedSet.size > 0) {
                for (const e of erodedSet) {
                    const nodes = mesh.getElementNodes(e);
                    for (let n = 0; n < 8; n++) {
                        const nid = nodes[n];
                        f_ext[nid * 3]     = 0;
                        f_ext[nid * 3 + 1] = 0;
                        f_ext[nid * 3 + 2] = 0;
                    }
                }
            }

            // Newton-Raphson iterations
            const result = this._newtonLoop(
                mesh, material, u, f_ext, fixedDOFs, ndof, elemStates,
                step, progressCallback, damageField, phaseField, erodedSet
            );

            totalIterations += result.iterations;
            elemStates = result.elemStates;

            // Update damage field after mechanical equilibrium (staggered scheme)
            if (this.enableDamage && phaseField) {
                const fractureMesh = {
                    nElements: mesh.elemCount,
                    connectivity: mesh._fractureConnectivity,
                    nodeCoords: mesh._fractureNodeCoords
                };
                const updated = phaseField.updateDamageField(
                    fractureMesh, u, material, damageField, damageHistory
                );
                damageField = updated.d;
                damageHistory = updated.H;

                // Check for element erosion
                erosion.checkAndErode(damageField);
                erodedSet = erosion.erodedElements;
            }

            // Recover stresses at this step for the snapshot
            const stepStress = this._recoverStresses(mesh, material, u, elemStates);
            const snapData = {
                step,
                loadFraction,
                displacement: new Float64Array(u),
                vonMisesStress: stepStress.vonMisesStress,
                triaxiality: stepStress.triaxiality,
                converged: result.converged,
                residualNorm: result.residualNorm || 0
            };
            if (damageField) {
                snapData.damageField = new Float64Array(damageField);
            }
            if (erodedSet && erodedSet.size > 0) {
                snapData.erodedElements = Array.from(erodedSet);
            }
            stepSnapshots.push(snapData);

            if (!result.converged) {
                converged = false;
                failedAtStep = step;
                break;
            }
        }

        // Final stress recovery
        const stressResult = this._recoverStresses(mesh, material, u, elemStates);

        const result = {
            displacement: u,
            cauchyStress: stressResult.cauchyStress,
            vonMisesStress: stressResult.vonMisesStress,
            principalStressField: stressResult.principalStressField,
            triaxiality: stressResult.triaxiality,
            strainEnergy: stressResult.strainEnergy,
            materialStates: elemStates,
            converged,
            totalIterations,
            stepSnapshots,
            failedAtStep
        };

        if (damageField) {
            result.damageField = damageField;
        }
        if (erodedSet && erodedSet.size > 0) {
            result.erodedElements = Array.from(erodedSet);
        }

        return result;
    }

    /**
     * Perform a single Newton step: assemble residual, solve for increment,
     * apply line search.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Float64Array} u - Current displacement (ndof), modified in-place
     * @param {Float64Array} f_ext - External force (ndof)
     * @param {Int32Array|Uint32Array} fixedDOFs
     * @returns {{ du: Float64Array, residualNorm: number, elemTangents: Float64Array[],
     *             fint: Float64Array, elemStates: MaterialState[][] }}
     */
    newtonStep(mesh, material, u, f_ext, fixedDOFs) {
        const ndof = mesh.nodeCount * 3;
        return this._assembleAndSolve(mesh, material, u, f_ext, fixedDOFs, ndof, null);
    }

    // -----------------------------------------------------------------------
    // Internal methods
    // -----------------------------------------------------------------------

    /**
     * Newton-Raphson loop for a single load step.
     * @private
     */
    _newtonLoop(mesh, material, u, f_ext, fixedDOFs, ndof, elemStates, step, progressCallback, damageField, phaseField, erodedSet) {
        let converged = false;
        let iter = 0;
        let residualNorm = Infinity;

        for (iter = 0; iter < this.maxNewtonIter; iter++) {
            const assembled = this._assembleAndSolve(
                mesh, material, u, f_ext, fixedDOFs, ndof, elemStates,
                damageField, phaseField, erodedSet
            );

            residualNorm = assembled.residualNorm;
            const du = assembled.du;
            elemStates = assembled.elemStates;

            // Convergence check
            const uNorm = Math.sqrt(_dot(u, u, ndof)) + EPSILON;
            const duNorm = Math.sqrt(_dot(du, du, ndof));
            const relResidual = residualNorm / (Math.sqrt(_dot(f_ext, f_ext, ndof)) + EPSILON);

            if (progressCallback) {
                progressCallback({
                    step,
                    iteration: iter,
                    residualNorm,
                    relativeResidual: relResidual,
                    incrementNorm: duNorm / uNorm
                });
            }

            if (relResidual < this.residualTol && duNorm / uNorm < this.incrementTol) {
                converged = true;
                // Apply final increment (already applied in line search)
                break;
            }

            // Line search
            this._lineSearch(mesh, material, u, du, f_ext, fixedDOFs, ndof, elemStates, assembled.fint);
        }

        return { converged, iterations: iter, elemStates, residualNorm };
    }

    /**
     * Assemble global residual and tangent, solve for displacement increment.
     * @private
     */
    _assembleAndSolve(mesh, material, u, f_ext, fixedDOFs, ndof, elemStates, damageField, phaseField, erodedSet) {
        const elemCount = mesh.elemCount;
        const fint_global = new Float64Array(ndof);
        const elemTangents = new Array(elemCount);
        const newElemStates = new Array(elemCount);

        // Element loop: compute internal forces and tangent stiffness
        for (let e = 0; e < elemCount; e++) {
            // Skip eroded elements (fully damaged)
            if (erodedSet && erodedSet.has(e)) {
                elemTangents[e] = new Float64Array(576); // zero stiffness
                newElemStates[e] = elemStates ? elemStates[e] : null;
                continue;
            }

            const nodes = mesh.getElementNodes(e);
            const nodeCoords = new Float64Array(24);
            const u_elem = new Float64Array(24);

            for (let i = 0; i < 8; i++) {
                const c = mesh.getNodeCoords(nodes[i]);
                const off = i * 3;
                nodeCoords[off]     = c[0];
                nodeCoords[off + 1] = c[1];
                nodeCoords[off + 2] = c[2];
                u_elem[off]     = u[nodes[i] * 3];
                u_elem[off + 1] = u[nodes[i] * 3 + 1];
                u_elem[off + 2] = u[nodes[i] * 3 + 2];
            }

            const res = elementForceAndTangent(
                nodeCoords, u_elem, material, elemStates ? elemStates[e] : null
            );

            // Apply stiffness degradation from phase-field damage
            if (damageField && phaseField && damageField[e] > 0) {
                phaseField.degradeStiffness(res.Kt, damageField[e]);
                const g = phaseField.degradation(damageField[e]);
                for (let i = 0; i < 24; i++) res.fint[i] *= g;
            }

            elemTangents[e] = res.Kt;
            newElemStates[e] = res.newState;

            // Scatter internal force to global vector
            for (let i = 0; i < 8; i++) {
                const ni = nodes[i];
                fint_global[ni * 3]     += res.fint[i * 3];
                fint_global[ni * 3 + 1] += res.fint[i * 3 + 1];
                fint_global[ni * 3 + 2] += res.fint[i * 3 + 2];
            }
        }

        // Residual: R = f_ext - f_int
        const residual = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) {
            residual[i] = f_ext[i] - fint_global[i];
        }

        // Zero residual on fixed DOFs
        for (let i = 0; i < fixedDOFs.length; i++) {
            residual[fixedDOFs[i]] = 0.0;
        }

        const residualNorm = Math.sqrt(_dot(residual, residual, ndof));

        // Solve: Kt · δu = R
        const cgResult = solveCG(
            mesh, elemTangents, residual, fixedDOFs, ndof,
            this.cgTolerance, this.cgMaxIter
        );

        return {
            du: cgResult.x,
            residualNorm,
            elemTangents,
            fint: fint_global,
            elemStates: newElemStates
        };
    }

    /**
     * Backtracking line search on the residual norm.
     * Updates u in-place: u += α · du where α ∈ (0, 1].
     * @private
     */
    _lineSearch(mesh, material, u, du, f_ext, fixedDOFs, ndof, elemStates, fint0) {
        // Compute initial residual norm
        const r0 = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) r0[i] = f_ext[i] - fint0[i];
        for (let i = 0; i < fixedDOFs.length; i++) r0[fixedDOFs[i]] = 0.0;
        const norm0 = Math.sqrt(_dot(r0, r0, ndof));

        let alpha = 1.0;

        for (let ls = 0; ls < this.lineSearchMaxIter; ls++) {
            // Trial: u_trial = u + α · du
            const u_trial = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) u_trial[i] = u[i] + alpha * du[i];

            // Compute internal force at trial point
            const fint_trial = new Float64Array(ndof);
            for (let e = 0; e < mesh.elemCount; e++) {
                const nodes = mesh.getElementNodes(e);
                const nodeCoords = new Float64Array(24);
                const u_elem = new Float64Array(24);
                for (let i = 0; i < 8; i++) {
                    const c = mesh.getNodeCoords(nodes[i]);
                    const off = i * 3;
                    nodeCoords[off]     = c[0];
                    nodeCoords[off + 1] = c[1];
                    nodeCoords[off + 2] = c[2];
                    u_elem[off]     = u_trial[nodes[i] * 3];
                    u_elem[off + 1] = u_trial[nodes[i] * 3 + 1];
                    u_elem[off + 2] = u_trial[nodes[i] * 3 + 2];
                }
                // Only need internal force, not tangent
                const res = elementForceAndTangent(
                    nodeCoords, u_elem, material,
                    elemStates ? elemStates[e] : null
                );
                for (let i = 0; i < 8; i++) {
                    const ni = nodes[i];
                    fint_trial[ni * 3]     += res.fint[i * 3];
                    fint_trial[ni * 3 + 1] += res.fint[i * 3 + 1];
                    fint_trial[ni * 3 + 2] += res.fint[i * 3 + 2];
                }
            }

            // Trial residual
            const r_trial = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) r_trial[i] = f_ext[i] - fint_trial[i];
            for (let i = 0; i < fixedDOFs.length; i++) r_trial[fixedDOFs[i]] = 0.0;
            const normTrial = Math.sqrt(_dot(r_trial, r_trial, ndof));

            if (normTrial < norm0 || alpha < EPSILON) {
                // Accept this step
                for (let i = 0; i < ndof; i++) u[i] = u_trial[i];
                return alpha;
            }

            alpha *= this.lineSearchBeta;
        }

        // Fall back to full step if line search fails to improve
        for (let i = 0; i < ndof; i++) u[i] += alpha * du[i];
        return alpha;
    }

    /**
     * Recover stress fields from the converged displacement.
     * @private
     */
    _recoverStresses(mesh, material, u, elemStates) {
        const elemCount = mesh.elemCount;
        const cauchyStress = new Float64Array(elemCount * 6);
        const vonMisesStress = new Float64Array(elemCount);
        const principalStressField = new Float64Array(elemCount * 3);
        const triaxialityField = new Float64Array(elemCount);
        const strainEnergyField = new Float64Array(elemCount);

        for (let e = 0; e < elemCount; e++) {
            const nodes = mesh.getElementNodes(e);
            const nodeCoords = new Float64Array(24);
            const u_elem = new Float64Array(24);

            for (let i = 0; i < 8; i++) {
                const c = mesh.getNodeCoords(nodes[i]);
                const off = i * 3;
                nodeCoords[off]     = c[0];
                nodeCoords[off + 1] = c[1];
                nodeCoords[off + 2] = c[2];
                u_elem[off]     = u[nodes[i] * 3];
                u_elem[off + 1] = u[nodes[i] * 3 + 1];
                u_elem[off + 2] = u[nodes[i] * 3 + 2];
            }

            // Average over Gauss points
            const avgCauchy = [0, 0, 0, 0, 0, 0];
            let elemEnergy = 0.0;

            for (let gp = 0; gp < 8; gp++) {
                const [xi, eta, zeta] = GAUSS_POINTS[gp];
                const gpState = elemStates && elemStates[e] ? elemStates[e][gp] : null;

                const { F, detJ } = computeDeformationGradient(
                    nodeCoords, u_elem, xi, eta, zeta
                );

                const matResult = material.computeStress(F, gpState);
                const S_voigt = matResult.stress;
                const sig = cauchyStress_local(F, S_voigt);

                for (let i = 0; i < 6; i++) avgCauchy[i] += sig[i];

                const E_voigt = greenLagrangeStrain(F);
                elemEnergy += strainEnergyDensity(S_voigt, E_voigt) * detJ * GAUSS_WEIGHT;
            }

            // Average over 8 Gauss points
            const inv8 = 1.0 / 8.0;
            for (let i = 0; i < 6; i++) {
                avgCauchy[i] *= inv8;
                cauchyStress[e * 6 + i] = avgCauchy[i];
            }

            vonMisesStress[e] = vonMises(avgCauchy);

            const princ = principalStresses(avgCauchy);
            principalStressField[e * 3]     = princ[0];
            principalStressField[e * 3 + 1] = princ[1];
            principalStressField[e * 3 + 2] = princ[2];

            triaxialityField[e] = stressTriaxiality(avgCauchy);
            strainEnergyField[e] = elemEnergy;
        }

        return {
            cauchyStress,
            vonMisesStress,
            principalStressField,
            triaxiality: triaxialityField,
            strainEnergy: strainEnergyField
        };
    }
}

/**
 * Local Cauchy stress computation (avoids name clash with exported function).
 * @private
 */
function cauchyStress_local(F, S_voigt) {
    return cauchyStress(F, S_voigt);
}
