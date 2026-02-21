/**
 * Time integration module for dynamic and quasi-static FEA.
 *
 * Provides three solvers for 8-node hexahedral elements:
 *   1. ExplicitDynamics  – central difference with lumped mass (cutting/tearing)
 *   2. ImplicitQuasiStatic – Newton-Raphson with arc-length control
 *   3. NewmarkDynamics   – Newmark-β implicit dynamics (low-frequency)
 *
 * All vectors are flat Float64Arrays with DOF numbering: node i → [3i, 3i+1, 3i+2].
 *
 * @module dynamics-solver
 */

import { NonlinearSolver } from './nonlinear-solver.js';
import { createMaterial, MaterialState } from './material-models.js';

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
const GAUSS_WEIGHT = 1.0;

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

/** @param {Float64Array} a @param {Float64Array} b @param {number} n */
function _dot(a, b, n) {
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
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
 * Shape function derivatives w.r.t. natural coordinates.
 * Returns 8×3 flat array (row-major per node).
 * @param {number} xi
 * @param {number} eta
 * @param {number} zeta
 * @returns {Float64Array} dN – 24 values [dN0/dξ, dN0/dη, dN0/dζ, dN1/dξ, ...]
 */
function shapeFunctionDerivatives(xi, eta, zeta) {
    const dN = new Float64Array(24);
    dN[0]  = -0.125 * (1 - eta) * (1 - zeta);
    dN[1]  = -0.125 * (1 - xi)  * (1 - zeta);
    dN[2]  = -0.125 * (1 - xi)  * (1 - eta);
    dN[3]  =  0.125 * (1 - eta) * (1 - zeta);
    dN[4]  = -0.125 * (1 + xi)  * (1 - zeta);
    dN[5]  = -0.125 * (1 + xi)  * (1 - eta);
    dN[6]  =  0.125 * (1 + eta) * (1 - zeta);
    dN[7]  =  0.125 * (1 + xi)  * (1 - zeta);
    dN[8]  = -0.125 * (1 + xi)  * (1 + eta);
    dN[9]  = -0.125 * (1 + eta) * (1 - zeta);
    dN[10] =  0.125 * (1 - xi)  * (1 - zeta);
    dN[11] = -0.125 * (1 - xi)  * (1 + eta);
    dN[12] = -0.125 * (1 - eta) * (1 + zeta);
    dN[13] = -0.125 * (1 - xi)  * (1 + zeta);
    dN[14] =  0.125 * (1 - xi)  * (1 - eta);
    dN[15] =  0.125 * (1 - eta) * (1 + zeta);
    dN[16] = -0.125 * (1 + xi)  * (1 + zeta);
    dN[17] =  0.125 * (1 + xi)  * (1 - eta);
    dN[18] =  0.125 * (1 + eta) * (1 + zeta);
    dN[19] =  0.125 * (1 + xi)  * (1 + zeta);
    dN[20] =  0.125 * (1 + xi)  * (1 + eta);
    dN[21] = -0.125 * (1 + eta) * (1 + zeta);
    dN[22] =  0.125 * (1 - xi)  * (1 + zeta);
    dN[23] =  0.125 * (1 - xi)  * (1 + eta);
    return dN;
}

/**
 * Compute Jacobian matrix J = ∂x/∂ξ, its inverse, and determinant.
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
        J[0] += dN[off]     * x;  J[1] += dN[off]     * y;  J[2] += dN[off]     * z;
        J[3] += dN[off + 1] * x;  J[4] += dN[off + 1] * y;  J[5] += dN[off + 1] * z;
        J[6] += dN[off + 2] * x;  J[7] += dN[off + 2] * y;  J[8] += dN[off + 2] * z;
    }
    const detJ = det3(J);
    const invJ = inv3(J);
    return { J, invJ, detJ };
}

/**
 * Compute shape function derivatives in physical coordinates dN/dX.
 * @param {Float64Array} nodeCoords - 24 values
 * @param {number} xi
 * @param {number} eta
 * @param {number} zeta
 * @returns {{ dNdX: Float64Array, detJ: number }}
 */
function physicalDerivatives(nodeCoords, xi, eta, zeta) {
    const dN = shapeFunctionDerivatives(xi, eta, zeta);
    const { invJ, detJ } = jacobian(nodeCoords, dN);
    const dNdX = new Float64Array(24);
    for (let n = 0; n < 8; n++) {
        const off = n * 3;
        const d0 = dN[off], d1 = dN[off + 1], d2 = dN[off + 2];
        dNdX[off]     = invJ[0] * d0 + invJ[1] * d1 + invJ[2] * d2;
        dNdX[off + 1] = invJ[3] * d0 + invJ[4] * d1 + invJ[5] * d2;
        dNdX[off + 2] = invJ[6] * d0 + invJ[7] * d1 + invJ[8] * d2;
    }
    return { dNdX, detJ };
}

/**
 * Compute deformation gradient F = I + ∂u/∂X (Total Lagrangian).
 * @param {Float64Array} nodeCoords - Reference coordinates (24 values)
 * @param {Float64Array} u_elem - Element displacements (24 values)
 * @param {number} xi
 * @param {number} eta
 * @param {number} zeta
 * @returns {{ F: number[], detF: number, dNdX: Float64Array, detJ: number }}
 */
function computeDeformationGradient(nodeCoords, u_elem, xi, eta, zeta) {
    const { dNdX, detJ } = physicalDerivatives(nodeCoords, xi, eta, zeta);

    const F = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (let n = 0; n < 8; n++) {
        const uOff = n * 3;
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                F[i * 3 + j] += u_elem[uOff + i] * dNdX[uOff + j];
            }
        }
    }

    const detF = det3(F);
    return { F, detF, dNdX, detJ };
}

/**
 * Compute Green-Lagrange strain E = 0.5(F^T·F - I).
 * @param {number[]} F - Deformation gradient (flat 3×3)
 * @returns {number[]} Voigt [E11, E22, E33, 2E12, 2E23, 2E13]
 */
function greenLagrangeStrain(F) {
    const FtF = mul33(transpose3(F), F);
    return [
        0.5 * (FtF[0] - 1),
        0.5 * (FtF[4] - 1),
        0.5 * (FtF[8] - 1),
        FtF[1],
        FtF[5],
        FtF[2]
    ];
}

/**
 * Compute element internal force vector (Total Lagrangian, 8-node hex).
 * @param {Float64Array} nodeCoords - Reference node coordinates (24 values)
 * @param {Float64Array} u_elem - Element displacement vector (24 values)
 * @param {object} material - Material model with computeStress(F, state)
 * @param {MaterialState[]|null} state - 8 MaterialState (one per GP), or null
 * @returns {{ fint: Float64Array, newState: MaterialState[], strainEnergy: number }}
 */
function elementInternalForce(nodeCoords, u_elem, material, state) {
    const fint = new Float64Array(24);
    const newState = new Array(8);
    let strainEnergy = 0.0;

    for (let gp = 0; gp < 8; gp++) {
        const [xi, eta, zeta] = GAUSS_POINTS[gp];
        const gpState = state ? state[gp] : null;

        const { F, dNdX, detJ } = computeDeformationGradient(
            nodeCoords, u_elem, xi, eta, zeta
        );

        const matResult = material.computeStress(F, gpState);
        const S_voigt = matResult.stress;
        newState[gp] = matResult.state || new MaterialState();

        // Green-Lagrange strain for energy
        const E_voigt = greenLagrangeStrain(F);
        strainEnergy += 0.5 * (
            S_voigt[0] * E_voigt[0] + S_voigt[1] * E_voigt[1] +
            S_voigt[2] * E_voigt[2] + S_voigt[3] * E_voigt[3] +
            S_voigt[4] * E_voigt[4] + S_voigt[5] * E_voigt[5]
        ) * detJ * GAUSS_WEIGHT;

        // BL matrix (6×24) and internal force BL^T · S
        const w = detJ * GAUSS_WEIGHT;
        for (let n = 0; n < 8; n++) {
            const dOff = n * 3;
            const col = n * 3;
            const dNx = dNdX[dOff];
            const dNy = dNdX[dOff + 1];
            const dNz = dNdX[dOff + 2];

            for (let i = 0; i < 3; i++) {
                const Fi0 = F[i * 3];
                const Fi1 = F[i * 3 + 1];
                const Fi2 = F[i * 3 + 2];
                // BL rows dotted with S_voigt
                fint[col + i] += (
                    Fi0 * dNx * S_voigt[0] +
                    Fi1 * dNy * S_voigt[1] +
                    Fi2 * dNz * S_voigt[2] +
                    (Fi0 * dNy + Fi1 * dNx) * S_voigt[3] +
                    (Fi1 * dNz + Fi2 * dNy) * S_voigt[4] +
                    (Fi0 * dNz + Fi2 * dNx) * S_voigt[5]
                ) * w;
            }
        }
    }

    return { fint, newState, strainEnergy };
}

/**
 * Compute element internal force vector and tangent stiffness matrix.
 * @param {Float64Array} nodeCoords - Reference coordinates (24 values)
 * @param {Float64Array} u_elem - Element displacements (24 values)
 * @param {object} material - Material model with computeStress(F, state)
 * @param {MaterialState[]|null} state - 8 MaterialState (one per GP), or null
 * @returns {{ fint: Float64Array, Kt: Float64Array, newState: MaterialState[],
 *             strainEnergy: number }}
 */
function elementForceAndTangent(nodeCoords, u_elem, material, state) {
    const fint = new Float64Array(24);
    const Kt = new Float64Array(576); // 24×24 flat row-major
    const newState = new Array(8);
    let strainEnergy = 0.0;

    for (let gp = 0; gp < 8; gp++) {
        const [xi, eta, zeta] = GAUSS_POINTS[gp];
        const gpState = state ? state[gp] : null;

        const { F, dNdX, detJ } = computeDeformationGradient(
            nodeCoords, u_elem, xi, eta, zeta
        );

        const matResult = material.computeStress(F, gpState);
        const S_voigt = matResult.stress;
        const C_flat = matResult.tangent;
        newState[gp] = matResult.state || new MaterialState();

        const E_voigt = greenLagrangeStrain(F);
        strainEnergy += 0.5 * (
            S_voigt[0] * E_voigt[0] + S_voigt[1] * E_voigt[1] +
            S_voigt[2] * E_voigt[2] + S_voigt[3] * E_voigt[3] +
            S_voigt[4] * E_voigt[4] + S_voigt[5] * E_voigt[5]
        ) * detJ * GAUSS_WEIGHT;

        const S = [
            S_voigt[0], S_voigt[3], S_voigt[5],
            S_voigt[3], S_voigt[1], S_voigt[4],
            S_voigt[5], S_voigt[4], S_voigt[2]
        ];

        const BL = new Float64Array(144); // 6×24
        for (let n = 0; n < 8; n++) {
            const dOff = n * 3;
            const col = n * 3;
            const dNx = dNdX[dOff];
            const dNy = dNdX[dOff + 1];
            const dNz = dNdX[dOff + 2];
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

        // Internal force: fint += BL^T · S_voigt · w
        for (let a = 0; a < 24; a++) {
            let f = 0.0;
            for (let alpha = 0; alpha < 6; alpha++) {
                f += BL[alpha * 24 + a] * S_voigt[alpha];
            }
            fint[a] += f * w;
        }

        // Material tangent: Kt_mat = BL^T · C · BL · w
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
        for (let a = 0; a < 24; a++) {
            for (let b = 0; b < 24; b++) {
                let s = 0.0;
                for (let alpha = 0; alpha < 6; alpha++) {
                    s += BL[alpha * 24 + a] * CB[alpha * 24 + b];
                }
                Kt[a * 24 + b] += s * w;
            }
        }

        // Geometric stiffness
        for (let n = 0; n < 8; n++) {
            const nOff = n * 3;
            for (let m = 0; m < 8; m++) {
                const mOff = m * 3;
                let sNN = 0.0;
                for (let k = 0; k < 3; k++) {
                    for (let l = 0; l < 3; l++) {
                        sNN += S[k * 3 + l] * dNdX[nOff + k] * dNdX[mOff + l];
                    }
                }
                sNN *= w;
                for (let I = 0; I < 3; I++) {
                    Kt[(n * 3 + I) * 24 + (m * 3 + I)] += sNN;
                }
            }
        }
    }

    return { fint, Kt, newState, strainEnergy };
}

// ---------------------------------------------------------------------------
// Global assembly helpers
// ---------------------------------------------------------------------------

/**
 * Gather element node coordinates into a 24-element Float64Array.
 * @param {object} mesh
 * @param {number[]} nodes - 8 node indices
 * @returns {Float64Array}
 */
function gatherNodeCoords(mesh, nodes) {
    const coords = new Float64Array(24);
    for (let i = 0; i < 8; i++) {
        const c = mesh.getNodeCoords(nodes[i]);
        const off = i * 3;
        coords[off]     = c[0];
        coords[off + 1] = c[1];
        coords[off + 2] = c[2];
    }
    return coords;
}

/**
 * Gather element DOFs from a global vector.
 * @param {Float64Array} globalVec - Global vector (ndof)
 * @param {number[]} nodes - 8 node indices
 * @returns {Float64Array} 24-element local vector
 */
function gatherElementDOFs(globalVec, nodes) {
    const local = new Float64Array(24);
    for (let i = 0; i < 8; i++) {
        const gi = nodes[i] * 3;
        const li = i * 3;
        local[li]     = globalVec[gi];
        local[li + 1] = globalVec[gi + 1];
        local[li + 2] = globalVec[gi + 2];
    }
    return local;
}

/**
 * Scatter-add element vector into global vector.
 * @param {Float64Array} globalVec - Global vector (ndof), modified in-place
 * @param {Float64Array} elemVec - 24-element local vector
 * @param {number[]} nodes - 8 node indices
 */
function scatterAdd(globalVec, elemVec, nodes) {
    for (let i = 0; i < 8; i++) {
        const gi = nodes[i] * 3;
        const li = i * 3;
        globalVec[gi]     += elemVec[li];
        globalVec[gi + 1] += elemVec[li + 1];
        globalVec[gi + 2] += elemVec[li + 2];
    }
}

/**
 * Assemble global internal force vector.
 * @param {object} mesh
 * @param {object} material
 * @param {Float64Array} u - Global displacement (ndof)
 * @param {Array} elemStates - Per-element material states
 * @returns {{ fint: Float64Array, newStates: Array, totalStrainEnergy: number }}
 */
function assembleInternalForce(mesh, material, u, elemStates) {
    const ndof = mesh.nodeCount * 3;
    const fint = new Float64Array(ndof);
    const newStates = new Array(mesh.elemCount);
    let totalStrainEnergy = 0.0;

    for (let e = 0; e < mesh.elemCount; e++) {
        const nodes = mesh.getElementNodes(e);
        const nodeCoords = gatherNodeCoords(mesh, nodes);
        const u_elem = gatherElementDOFs(u, nodes);

        const res = elementInternalForce(
            nodeCoords, u_elem, material,
            elemStates ? elemStates[e] : null
        );

        scatterAdd(fint, res.fint, nodes);
        newStates[e] = res.newState;
        totalStrainEnergy += res.strainEnergy;
    }

    return { fint, newStates, totalStrainEnergy };
}

/**
 * Assemble global internal force and tangent stiffness (element-by-element).
 * @param {object} mesh
 * @param {object} material
 * @param {Float64Array} u
 * @param {Array} elemStates
 * @returns {{ fint: Float64Array, elemTangents: Float64Array[],
 *             newStates: Array, totalStrainEnergy: number }}
 */
function assembleForceAndTangent(mesh, material, u, elemStates) {
    const ndof = mesh.nodeCount * 3;
    const fint = new Float64Array(ndof);
    const elemTangents = new Array(mesh.elemCount);
    const newStates = new Array(mesh.elemCount);
    let totalStrainEnergy = 0.0;

    for (let e = 0; e < mesh.elemCount; e++) {
        const nodes = mesh.getElementNodes(e);
        const nodeCoords = gatherNodeCoords(mesh, nodes);
        const u_elem = gatherElementDOFs(u, nodes);

        const res = elementForceAndTangent(
            nodeCoords, u_elem, material,
            elemStates ? elemStates[e] : null
        );

        scatterAdd(fint, res.fint, nodes);
        elemTangents[e] = res.Kt;
        newStates[e] = res.newState;
        totalStrainEnergy += res.strainEnergy;
    }

    return { fint, elemTangents, newStates, totalStrainEnergy };
}

// ---------------------------------------------------------------------------
// CG solver (element-by-element, Jacobi preconditioned)
// ---------------------------------------------------------------------------

/**
 * Element-by-element matrix-vector product: y = Σ_e Kt_e · x_e.
 * @param {object} mesh
 * @param {Float64Array[]} elemTangents
 * @param {Float64Array} x_full - Input (ndof)
 * @param {Float64Array} y_full - Output (ndof), zeroed on entry
 */
function ebeMatVec(mesh, elemTangents, x_full, y_full) {
    for (let e = 0; e < mesh.elemCount; e++) {
        const Kt = elemTangents[e];
        if (!Kt) continue;
        const nodes = mesh.getElementNodes(e);
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
 * Compute Jacobi preconditioner from element tangents.
 * @param {object} mesh
 * @param {Float64Array[]} elemTangents
 * @param {number} ndof
 * @returns {Float64Array} invDiag (ndof)
 */
function computeJacobiPreconditioner(mesh, elemTangents, ndof) {
    const diag = new Float64Array(ndof);
    for (let e = 0; e < mesh.elemCount; e++) {
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
 * Solve K·δu = rhs using Jacobi-preconditioned CG (element-by-element).
 * @param {object} mesh
 * @param {Float64Array[]} elemTangents
 * @param {Float64Array} rhs
 * @param {Int32Array|Uint32Array} fixedDOFs
 * @param {number} ndof
 * @param {number} [tol]
 * @param {number} [maxIter]
 * @returns {{ x: Float64Array, iterations: number, residualNorm: number }}
 */
function solveCG(mesh, elemTangents, rhs, fixedDOFs, ndof, tol, maxIter) {
    tol = tol || CG_TOLERANCE;
    maxIter = maxIter || MAX_CG_ITERATIONS;

    const isFixed = new Uint8Array(ndof);
    for (let i = 0; i < fixedDOFs.length; i++) isFixed[fixedDOFs[i]] = 1;

    const invDiag = computeJacobiPreconditioner(mesh, elemTangents, ndof);
    for (let i = 0; i < fixedDOFs.length; i++) invDiag[fixedDOFs[i]] = 0.0;

    const u = new Float64Array(ndof);
    const r = new Float64Array(ndof);
    const z = new Float64Array(ndof);
    const p = new Float64Array(ndof);
    const Ap = new Float64Array(ndof);

    for (let i = 0; i < ndof; i++) {
        r[i] = isFixed[i] ? 0.0 : rhs[i];
        z[i] = invDiag[i] * r[i];
        p[i] = z[i];
    }

    let rz = 0.0;
    for (let i = 0; i < ndof; i++) rz += r[i] * z[i];

    const rhsNorm = Math.sqrt(rz > 0 ? rz : _dot(r, r, ndof));
    if (rhsNorm < EPSILON) return { x: u, iterations: 0, residualNorm: 0 };

    let iter;
    let rnorm = 0.0;
    for (iter = 0; iter < maxIter; iter++) {
        rnorm = Math.sqrt(_dot(r, r, ndof));
        if (rnorm < tol * rhsNorm + EPSILON) break;

        Ap.fill(0);
        ebeMatVec(mesh, elemTangents, p, Ap);
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
        for (let i = 0; i < ndof; i++) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    return { x: u, iterations: iter, residualNorm: rnorm };
}

// ===========================================================================
// 1. ExplicitDynamics – Central Difference Method
// ===========================================================================

/**
 * @typedef {object} ExplicitConfig
 * @property {number} [density=1.0] - Material density ρ (mass per unit volume)
 * @property {number} [safetyFactor=0.8] - Courant safety factor for Δt
 * @property {number} [linearBulkViscosity=0.06] - Linear bulk viscosity coefficient
 * @property {number} [quadraticBulkViscosity=1.2] - Quadratic bulk viscosity coefficient
 * @property {number} [totalTime=1.0] - Total simulation time
 * @property {number} [maxSteps=1000000] - Maximum number of time steps
 * @property {number} [outputInterval=100] - Steps between output snapshots
 */

/**
 * @typedef {object} EnergyBalance
 * @property {number} kinetic - Kinetic energy 0.5·v^T·M·v
 * @property {number} internal - Internal (strain) energy
 * @property {number} externalWork - Cumulative external work
 * @property {number} total - Kinetic + internal − externalWork (should be ~0)
 */

/**
 * @typedef {object} ExplicitStepResult
 * @property {Float64Array} u - Updated displacement
 * @property {Float64Array} v - Updated velocity (half-step)
 * @property {Float64Array} a - Acceleration at current step
 * @property {EnergyBalance} energy - Energy balance tracking
 * @property {number} dt - Time step used
 */

/**
 * Explicit dynamics solver using the central difference method.
 *
 * Ideal for cutting/tearing simulations with complex contact and severe
 * nonlinearity where implicit methods struggle to converge. Uses a lumped
 * (diagonal) mass matrix and conditionally stable time integration.
 */
export class ExplicitDynamics {
    /**
     * @param {ExplicitConfig} [config]
     */
    constructor(config) {
        const c = config || {};
        /** @type {number} */ this.density = c.density || 1.0;
        /** @type {number} */ this.safetyFactor = c.safetyFactor || 0.8;
        /** @type {number} */ this.linearBulkViscosity = c.linearBulkViscosity || 0.06;
        /** @type {number} */ this.quadraticBulkViscosity = c.quadraticBulkViscosity || 1.2;
        /** @type {number} */ this.totalTime = c.totalTime || 1.0;
        /** @type {number} */ this.maxSteps = c.maxSteps || 1000000;
        /** @type {number} */ this.outputInterval = c.outputInterval || 100;
    }

    /**
     * Compute lumped mass for a single hex element using row-sum lumping.
     *
     * Integrates ρ·N^T·N over the element volume with 2×2×2 Gauss quadrature,
     * then sums each row to produce a diagonal mass contribution.
     *
     * @param {Float64Array} nodeCoords - Reference coordinates (24 values)
     * @param {number} density - Element density ρ
     * @returns {Float64Array} elemMass – 8 nodal mass contributions
     */
    computeElementMass(nodeCoords, density) {
        const elemMass = new Float64Array(8);

        for (let gp = 0; gp < 8; gp++) {
            const [xi, eta, zeta] = GAUSS_POINTS[gp];
            const N = shapeFunctions(xi, eta, zeta);
            const dN = shapeFunctionDerivatives(xi, eta, zeta);
            const { detJ } = jacobian(nodeCoords, dN);

            const w = density * Math.abs(detJ) * GAUSS_WEIGHT;
            // Row-sum lumping: M_ii = Σ_j M_ij = ρ·N_i·(Σ_j N_j)·|J|·w
            // For trilinear hex, Σ_j N_j = 1, so M_ii = ρ·N_i·|J|·w
            for (let i = 0; i < 8; i++) {
                elemMass[i] += N[i] * w;
            }
        }

        return elemMass;
    }

    /**
     * Assemble the global lumped mass vector from all elements.
     *
     * Each entry stores the total mass associated with one DOF. Since mass is
     * isotropic, the same value is used for all 3 translational DOFs of a node.
     *
     * @param {object} mesh
     * @param {number} density - Uniform density (or use per-element density)
     * @returns {Float64Array} mass – lumped mass vector (ndof)
     */
    assembleLumpedMass(mesh, density) {
        const ndof = mesh.nodeCount * 3;
        const mass = new Float64Array(ndof);

        for (let e = 0; e < mesh.elemCount; e++) {
            const nodes = mesh.getElementNodes(e);
            const nodeCoords = gatherNodeCoords(mesh, nodes);
            const elemMass = this.computeElementMass(nodeCoords, density);

            for (let i = 0; i < 8; i++) {
                const gi = nodes[i] * 3;
                mass[gi]     += elemMass[i];
                mass[gi + 1] += elemMass[i];
                mass[gi + 2] += elemMass[i];
            }
        }

        return mass;
    }

    /**
     * Perform a single explicit time step using central difference integration.
     *
     * Central difference scheme:
     *   a_n     = M⁻¹ · (f_ext − f_int − f_visc)
     *   v_{n+½} = v_{n-½} + Δt · a_n
     *   u_{n+1} = u_n + Δt · v_{n+½}
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Float64Array} u - Current displacement (ndof), modified in-place
     * @param {Float64Array} v - Half-step velocity (ndof), modified in-place
     * @param {Float64Array} f_ext - External force vector (ndof)
     * @param {Float64Array} mass - Lumped mass vector (ndof)
     * @param {Int32Array|Uint32Array} fixedDOFs - Fixed DOF indices
     * @param {number} dt - Time step size
     * @param {Array} elemStates - Per-element material states
     * @param {number} externalWork - Cumulative external work from prior steps
     * @returns {ExplicitStepResult}
     */
    step(mesh, material, u, v, f_ext, mass, fixedDOFs, dt, elemStates, externalWork) {
        const ndof = mesh.nodeCount * 3;

        // Assemble internal forces
        const { fint, newStates, totalStrainEnergy } =
            assembleInternalForce(mesh, material, u, elemStates);

        // Compute bulk viscosity forces
        const f_visc = this._bulkViscosityForce(mesh, u, v, dt, material);

        // Acceleration: a = M⁻¹ · (f_ext − f_int − f_visc)
        const a = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) {
            if (mass[i] > EPSILON) {
                a[i] = (f_ext[i] - fint[i] - f_visc[i]) / mass[i];
            }
        }

        // Apply BCs: zero acceleration and velocity on fixed DOFs
        for (let i = 0; i < fixedDOFs.length; i++) {
            const dof = fixedDOFs[i];
            a[dof] = 0.0;
            v[dof] = 0.0;
        }

        // Update velocity: v_{n+1/2} = v_{n-1/2} + dt · a_n
        for (let i = 0; i < ndof; i++) {
            v[i] += dt * a[i];
        }

        // Zero velocity on fixed DOFs after update
        for (let i = 0; i < fixedDOFs.length; i++) {
            v[fixedDOFs[i]] = 0.0;
        }

        // External work increment: W += f_ext · Δu = f_ext · (dt · v_{n+1/2})
        let workIncrement = 0.0;
        for (let i = 0; i < ndof; i++) {
            workIncrement += f_ext[i] * dt * v[i];
        }
        externalWork += workIncrement;

        // Update displacement: u_{n+1} = u_n + dt · v_{n+1/2}
        for (let i = 0; i < ndof; i++) {
            u[i] += dt * v[i];
        }

        // Kinetic energy: 0.5 · v^T · M · v
        let kinetic = 0.0;
        for (let i = 0; i < ndof; i++) {
            kinetic += mass[i] * v[i] * v[i];
        }
        kinetic *= 0.5;

        const energy = {
            kinetic,
            internal: totalStrainEnergy,
            externalWork,
            total: kinetic + totalStrainEnergy - externalWork
        };

        return { u, v, a, energy, dt, elemStates: newStates };
    }

    /**
     * Compute the critical stable time step based on the CFL condition.
     *
     * Δt_crit = h_min / c_d, where c_d = √(E/ρ) is the dilatational wave speed
     * and h_min is the smallest element characteristic length.
     *
     * @param {object} mesh
     * @param {object} material - Must have E (Young's modulus) and nu (Poisson's ratio)
     * @param {number} [density] - Override density (uses this.density if omitted)
     * @returns {number} Critical time step with safety factor applied
     */
    criticalTimeStep(mesh, material, density) {
        const rho = density || this.density;
        const E = material.E || material.youngsModulus || 1.0;
        const nu = material.nu || material.poissonRatio || 0.3;

        // Dilatational wave speed (constrained modulus for 3D)
        const Ec = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const cd = Math.sqrt(Ec / rho);

        // Find minimum element characteristic length
        let hMin = Infinity;
        for (let e = 0; e < mesh.elemCount; e++) {
            const nodes = mesh.getElementNodes(e);
            const nodeCoords = gatherNodeCoords(mesh, nodes);
            const h = this._elementCharLength(nodeCoords);
            if (h < hMin) hMin = h;
        }

        return this.safetyFactor * hMin / cd;
    }

    /**
     * Run full explicit dynamics simulation.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Int32Array|Uint32Array|number[]} constraints - Fixed DOF indices
     * @param {Float64Array} loads - External force vector (ndof)
     * @param {ExplicitConfig} [config] - Override config
     * @param {function} [progressCallback] - Called with step info
     * @returns {{ snapshots: ExplicitStepResult[], finalDisplacement: Float64Array,
     *             totalSteps: number, finalTime: number, energyHistory: EnergyBalance[] }}
     */
    solve(mesh, material, constraints, loads, config, progressCallback) {
        const cfg = config || {};
        const totalTime = cfg.totalTime || this.totalTime;
        const maxSteps = cfg.maxSteps || this.maxSteps;
        const outputInterval = cfg.outputInterval || this.outputInterval;

        const fixedDOFs = constraints instanceof Int32Array || constraints instanceof Uint32Array
            ? constraints
            : new Int32Array(constraints);

        const ndof = mesh.nodeCount * 3;
        const density = cfg.density || this.density;

        // Assemble lumped mass
        const mass = this.assembleLumpedMass(mesh, density);

        // Compute stable time step
        const dt = this.criticalTimeStep(mesh, material, density);
        if (dt <= 0 || !isFinite(dt)) {
            throw new Error('Invalid critical time step: ' + dt);
        }

        // Initialize state
        const u = new Float64Array(ndof);
        const v = new Float64Array(ndof);
        let elemStates = new Array(mesh.elemCount).fill(null);
        let externalWork = 0.0;

        const snapshots = [];
        const energyHistory = [];
        let t = 0.0;
        let stepCount = 0;

        while (t < totalTime && stepCount < maxSteps) {
            // Clamp final step
            const dtStep = Math.min(dt, totalTime - t);

            // Scale loads for current time (linear ramp)
            const loadFactor = Math.min(t / totalTime, 1.0);
            const f_ext = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) f_ext[i] = loads[i] * loadFactor;

            const result = this.step(
                mesh, material, u, v, f_ext, mass, fixedDOFs,
                dtStep, elemStates, externalWork
            );

            elemStates = result.elemStates;
            externalWork = result.energy.externalWork;

            t += dtStep;
            stepCount++;

            energyHistory.push(result.energy);

            if (stepCount % outputInterval === 0 || t >= totalTime) {
                const snapshot = {
                    time: t,
                    step: stepCount,
                    u: new Float64Array(u),
                    v: new Float64Array(v),
                    energy: result.energy,
                    dt: dtStep
                };
                snapshots.push(snapshot);

                if (progressCallback) {
                    progressCallback({
                        step: stepCount,
                        time: t,
                        dt: dtStep,
                        energy: result.energy,
                        progress: t / totalTime
                    });
                }
            }
        }

        return {
            snapshots,
            finalDisplacement: new Float64Array(u),
            totalSteps: stepCount,
            finalTime: t,
            energyHistory
        };
    }

    /**
     * Compute element characteristic length (cube root of volume).
     * @private
     * @param {Float64Array} nodeCoords - 24 values
     * @returns {number}
     */
    _elementCharLength(nodeCoords) {
        let vol = 0.0;
        for (let gp = 0; gp < 8; gp++) {
            const [xi, eta, zeta] = GAUSS_POINTS[gp];
            const dN = shapeFunctionDerivatives(xi, eta, zeta);
            const { detJ } = jacobian(nodeCoords, dN);
            vol += Math.abs(detJ) * GAUSS_WEIGHT;
        }
        return Math.cbrt(vol);
    }

    /**
     * Compute bulk viscosity force to dampen shock oscillations.
     *
     * Uses both linear and quadratic bulk viscosity terms based on the
     * volumetric strain rate:
     *   q = c1 · ρ · cd · h · ε̇_vol + c2 · ρ · h² · ε̇_vol²  (compressive only)
     *
     * @private
     * @param {object} mesh
     * @param {Float64Array} u - Current displacement (ndof)
     * @param {Float64Array} v - Current velocity (ndof)
     * @param {number} dt - Current time step
     * @param {object} material
     * @returns {Float64Array} Viscous force vector (ndof)
     */
    _bulkViscosityForce(mesh, u, v, dt, material) {
        const ndof = mesh.nodeCount * 3;
        const f_visc = new Float64Array(ndof);

        const E = material.E || material.youngsModulus || 1.0;
        const nu = material.nu || material.poissonRatio || 0.3;
        const rho = this.density;
        const Ec = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const cd = Math.sqrt(Ec / rho);
        const c1 = this.linearBulkViscosity;
        const c2 = this.quadraticBulkViscosity;

        for (let e = 0; e < mesh.elemCount; e++) {
            const nodes = mesh.getElementNodes(e);
            const nodeCoords = gatherNodeCoords(mesh, nodes);
            const h = this._elementCharLength(nodeCoords);

            // Compute volumetric strain rate at element center (ξ=η=ζ=0)
            const { dNdX, detJ } = physicalDerivatives(nodeCoords, 0, 0, 0);

            // ε̇_vol = ∂v_x/∂x + ∂v_y/∂y + ∂v_z/∂z
            let epsVolRate = 0.0;
            for (let n = 0; n < 8; n++) {
                const gi = nodes[n] * 3;
                const dOff = n * 3;
                epsVolRate += dNdX[dOff]     * v[gi];
                epsVolRate += dNdX[dOff + 1] * v[gi + 1];
                epsVolRate += dNdX[dOff + 2] * v[gi + 2];
            }

            // Only apply in compression (negative volumetric strain rate)
            if (epsVolRate >= 0.0) continue;

            // Bulk viscosity pressure
            const qLinear = c1 * rho * cd * h * Math.abs(epsVolRate);
            const qQuad = c2 * rho * h * h * epsVolRate * epsVolRate;
            const q = qLinear + qQuad;

            // Distribute as volumetric force: f_visc += q · B_vol^T · V
            // Simplified: apply pressure q through shape function derivatives
            const vol = Math.abs(detJ) * 8.0; // approximate element volume
            for (let n = 0; n < 8; n++) {
                const gi = nodes[n] * 3;
                const dOff = n * 3;
                f_visc[gi]     += q * dNdX[dOff]     * vol;
                f_visc[gi + 1] += q * dNdX[dOff + 1] * vol;
                f_visc[gi + 2] += q * dNdX[dOff + 2] * vol;
            }
        }

        return f_visc;
    }
}

// ===========================================================================
// 2. ImplicitQuasiStatic – Newton-Raphson with arc-length control
// ===========================================================================

/**
 * @typedef {object} QuasiStaticConfig
 * @property {number} [loadSteps=20] - Number of load increments
 * @property {number} [tolerance=1e-6] - Newton-Raphson convergence tolerance
 * @property {number} [maxNewtonIter=20] - Maximum Newton iterations per step
 * @property {boolean} [arcLength=false] - Enable arc-length control
 * @property {number} [arcLengthDs=0.1] - Arc-length constraint radius
 * @property {number} [viscousRegParam=0.0] - Viscous regularization parameter
 * @property {number} [cgTolerance=1e-8] - CG solver tolerance
 * @property {number} [cgMaxIter=400] - CG maximum iterations
 */

/**
 * Implicit quasi-static solver with Newton-Raphson and optional arc-length control.
 *
 * Suitable for slow loading without inertial effects. Includes Riks/Crisfield
 * arc-length method for snap-through and post-peak softening, and optional
 * viscous regularization for damage localization.
 */
export class ImplicitQuasiStatic {
    /**
     * @param {QuasiStaticConfig} [config]
     */
    constructor(config) {
        const c = config || {};
        /** @type {number} */ this.loadSteps = c.loadSteps || 20;
        /** @type {number} */ this.tolerance = c.tolerance || 1e-6;
        /** @type {number} */ this.maxNewtonIter = c.maxNewtonIter || 20;
        /** @type {boolean} */ this.arcLength = c.arcLength || false;
        /** @type {number} */ this.arcLengthDs = c.arcLengthDs || 0.1;
        /** @type {number} */ this.viscousRegParam = c.viscousRegParam || 0.0;
        /** @type {number} */ this.cgTolerance = c.cgTolerance || CG_TOLERANCE;
        /** @type {number} */ this.cgMaxIter = c.cgMaxIter || MAX_CG_ITERATIONS;
    }

    /**
     * Perform a single load step with Newton-Raphson iteration.
     *
     * Solves the nonlinear equilibrium equation R(u) = λ·f_ext − f_int(u) = 0
     * at a fixed load factor λ.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Float64Array} u - Current displacement (ndof), modified in-place
     * @param {Float64Array} f_ext - Reference external force (ndof)
     * @param {Int32Array|Uint32Array} fixedDOFs
     * @param {number} loadFactor - Current load factor λ
     * @param {Array} elemStates - Per-element material states
     * @returns {{ converged: boolean, iterations: number, residualNorm: number,
     *             elemStates: Array, strainEnergy: number }}
     */
    solveStep(mesh, material, u, f_ext, fixedDOFs, loadFactor, elemStates) {
        const ndof = mesh.nodeCount * 3;

        // Scaled external force
        const f_scaled = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) f_scaled[i] = f_ext[i] * loadFactor;

        let converged = false;
        let iterations = 0;
        let residualNorm = Infinity;
        let currentStates = elemStates;
        let strainEnergy = 0.0;

        for (let iter = 0; iter < this.maxNewtonIter; iter++) {
            iterations = iter + 1;

            // Assemble internal force and tangent
            const assembled = assembleForceAndTangent(
                mesh, material, u, currentStates
            );
            strainEnergy = assembled.totalStrainEnergy;

            // Viscous regularization: add η/Δt · (u − u_prev) to internal force
            // Approximated here as additional stiffness on diagonal
            if (this.viscousRegParam > 0.0) {
                for (let e = 0; e < mesh.elemCount; e++) {
                    if (!assembled.elemTangents[e]) continue;
                    const Kt = assembled.elemTangents[e];
                    for (let d = 0; d < 24; d++) {
                        Kt[d * 24 + d] += this.viscousRegParam;
                    }
                }
            }

            // Residual: R = f_ext_scaled − f_int
            const residual = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) {
                residual[i] = f_scaled[i] - assembled.fint[i];
            }
            for (let i = 0; i < fixedDOFs.length; i++) residual[fixedDOFs[i]] = 0.0;

            residualNorm = Math.sqrt(_dot(residual, residual, ndof));
            const refNorm = Math.sqrt(_dot(f_scaled, f_scaled, ndof)) + EPSILON;

            if (residualNorm / refNorm < this.tolerance) {
                converged = true;
                currentStates = assembled.newStates;
                break;
            }

            // Solve: Kt · δu = R
            const cgResult = solveCG(
                mesh, assembled.elemTangents, residual, fixedDOFs, ndof,
                this.cgTolerance, this.cgMaxIter
            );

            // Update displacement
            for (let i = 0; i < ndof; i++) u[i] += cgResult.x[i];
            currentStates = assembled.newStates;
        }

        return { converged, iterations, residualNorm, elemStates: currentStates, strainEnergy };
    }

    /**
     * Perform an arc-length step using the Riks/Crisfield spherical method.
     *
     * Simultaneously solves for displacement increment δu and load factor
     * increment δλ subject to the constraint:
     *   ‖δu‖² + ψ²·(δλ)²·‖f_ext‖² = ds²
     *
     * Enables tracing of equilibrium paths through limit points and
     * snap-through/snap-back behavior.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Float64Array} u - Current displacement (ndof), modified in-place
     * @param {Float64Array} f_ext - Reference external force (ndof)
     * @param {Int32Array|Uint32Array} fixedDOFs
     * @param {number} ds - Arc-length constraint radius
     * @param {number} loadFactor - Current load factor λ
     * @param {Array} elemStates - Per-element material states
     * @returns {{ converged: boolean, deltaLambda: number, loadFactor: number,
     *             iterations: number, elemStates: Array }}
     */
    arcLengthStep(mesh, material, u, f_ext, fixedDOFs, ds, loadFactor, elemStates) {
        const ndof = mesh.nodeCount * 3;
        const fExtNorm = Math.sqrt(_dot(f_ext, f_ext, ndof));

        let converged = false;
        let iterations = 0;
        let deltaLambda = ds / (fExtNorm + EPSILON);
        let currentStates = elemStates;

        // Predictor: use previous direction or initial tangent
        const f_scaled = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) {
            f_scaled[i] = f_ext[i] * (loadFactor + deltaLambda);
        }

        for (let iter = 0; iter < this.maxNewtonIter; iter++) {
            iterations = iter + 1;

            const assembled = assembleForceAndTangent(
                mesh, material, u, currentStates
            );

            // Residual at current (u, λ)
            const currentLambda = loadFactor + deltaLambda;
            const residual = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) {
                residual[i] = f_ext[i] * currentLambda - assembled.fint[i];
            }
            for (let i = 0; i < fixedDOFs.length; i++) residual[fixedDOFs[i]] = 0.0;

            const residualNorm = Math.sqrt(_dot(residual, residual, ndof));
            const refNorm = Math.sqrt(_dot(f_ext, f_ext, ndof)) * Math.abs(currentLambda) + EPSILON;

            if (residualNorm / refNorm < this.tolerance) {
                converged = true;
                currentStates = assembled.newStates;
                break;
            }

            // Solve two systems: Kt · δu_r = R, Kt · δu_f = f_ext
            const cgR = solveCG(
                mesh, assembled.elemTangents, residual, fixedDOFs, ndof,
                this.cgTolerance, this.cgMaxIter
            );
            const cgF = solveCG(
                mesh, assembled.elemTangents, f_ext, fixedDOFs, ndof,
                this.cgTolerance, this.cgMaxIter
            );

            const du_r = cgR.x;
            const du_f = cgF.x;

            // Crisfield spherical constraint: solve quadratic for δδλ
            const a1 = _dot(du_f, du_f, ndof);
            const a2 = 2.0 * _dot(du_r, du_f, ndof);
            const a3 = _dot(du_r, du_r, ndof) - ds * ds;

            const disc = a2 * a2 - 4.0 * a1 * a3;
            let ddLambda;
            if (disc >= 0 && Math.abs(a1) > EPSILON) {
                const sqrtDisc = Math.sqrt(disc);
                const ddl1 = (-a2 + sqrtDisc) / (2.0 * a1);
                const ddl2 = (-a2 - sqrtDisc) / (2.0 * a1);

                // Choose root that gives positive work (forward progress)
                const work1 = _dot(du_r, du_r, ndof) + ddl1 * _dot(du_f, du_r, ndof);
                const work2 = _dot(du_r, du_r, ndof) + ddl2 * _dot(du_f, du_r, ndof);
                ddLambda = work1 >= work2 ? ddl1 : ddl2;
            } else {
                ddLambda = 0.0;
            }

            // Update
            deltaLambda += ddLambda;
            for (let i = 0; i < ndof; i++) {
                u[i] += du_r[i] + ddLambda * du_f[i];
            }
            currentStates = assembled.newStates;
        }

        return {
            converged,
            deltaLambda,
            loadFactor: loadFactor + deltaLambda,
            iterations,
            elemStates: currentStates
        };
    }

    /**
     * Run full quasi-static analysis with incremental loading.
     *
     * If arc-length control is enabled, uses the Riks/Crisfield method to trace
     * the equilibrium path. Otherwise uses standard Newton-Raphson with
     * proportional load stepping.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Int32Array|Uint32Array|number[]} constraints - Fixed DOF indices
     * @param {Float64Array} loads - Reference external force vector (ndof)
     * @param {QuasiStaticConfig} [config] - Override config
     * @param {function} [progressCallback] - Called with step info
     * @returns {{ displacement: Float64Array, loadFactors: number[],
     *             converged: boolean, totalIterations: number,
     *             strainEnergy: number[], equilibriumPath: Array }}
     */
    solve(mesh, material, constraints, loads, config, progressCallback) {
        const cfg = config || {};
        const loadSteps = cfg.loadSteps || this.loadSteps;
        const useArcLength = cfg.arcLength !== undefined ? cfg.arcLength : this.arcLength;
        const ds = cfg.arcLengthDs || this.arcLengthDs;

        const fixedDOFs = constraints instanceof Int32Array || constraints instanceof Uint32Array
            ? constraints
            : new Int32Array(constraints);

        const ndof = mesh.nodeCount * 3;
        const u = new Float64Array(ndof);
        let elemStates = new Array(mesh.elemCount).fill(null);

        const loadFactors = [];
        const strainEnergyHistory = [];
        const equilibriumPath = [];
        let totalIterations = 0;
        let converged = true;

        if (useArcLength) {
            // Arc-length controlled loading
            let lambda = 0.0;
            for (let step = 0; step < loadSteps; step++) {
                const result = this.arcLengthStep(
                    mesh, material, u, loads, fixedDOFs, ds, lambda, elemStates
                );

                totalIterations += result.iterations;
                elemStates = result.elemStates;
                lambda = result.loadFactor;

                loadFactors.push(lambda);
                equilibriumPath.push({
                    step,
                    loadFactor: lambda,
                    displacement: new Float64Array(u),
                    converged: result.converged
                });

                if (progressCallback) {
                    progressCallback({
                        step,
                        loadFactor: lambda,
                        converged: result.converged,
                        iterations: result.iterations,
                        progress: (step + 1) / loadSteps
                    });
                }

                if (!result.converged) {
                    converged = false;
                    break;
                }
            }
        } else {
            // Standard proportional load stepping
            for (let step = 1; step <= loadSteps; step++) {
                const loadFactor = step / loadSteps;

                const result = this.solveStep(
                    mesh, material, u, loads, fixedDOFs, loadFactor, elemStates
                );

                totalIterations += result.iterations;
                elemStates = result.elemStates;

                loadFactors.push(loadFactor);
                strainEnergyHistory.push(result.strainEnergy);
                equilibriumPath.push({
                    step,
                    loadFactor,
                    displacement: new Float64Array(u),
                    converged: result.converged
                });

                if (progressCallback) {
                    progressCallback({
                        step,
                        loadFactor,
                        residualNorm: result.residualNorm,
                        converged: result.converged,
                        iterations: result.iterations,
                        progress: step / loadSteps
                    });
                }

                if (!result.converged) {
                    converged = false;
                    break;
                }
            }
        }

        return {
            displacement: u,
            loadFactors,
            converged,
            totalIterations,
            strainEnergy: strainEnergyHistory,
            equilibriumPath
        };
    }
}

// ===========================================================================
// 3. NewmarkDynamics – Newmark-β Implicit Dynamics
// ===========================================================================

/**
 * @typedef {object} NewmarkConfig
 * @property {number} [beta=0.25] - Newmark β parameter (0.25 = average acceleration)
 * @property {number} [gamma=0.5] - Newmark γ parameter (0.5 = no numerical damping)
 * @property {number} [alphaDamping=0.0] - Rayleigh mass damping coefficient α
 * @property {number} [betaDamping=0.0] - Rayleigh stiffness damping coefficient β
 * @property {number} [dt=0.01] - Time step size
 * @property {number} [totalTime=1.0] - Total simulation time
 * @property {number} [maxSteps=10000] - Maximum time steps
 * @property {number} [newtonTol=1e-6] - Newton-Raphson tolerance
 * @property {number} [maxNewtonIter=20] - Max Newton iterations per step
 * @property {number} [outputInterval=10] - Steps between output snapshots
 * @property {number} [cgTolerance=1e-8] - CG solver tolerance
 * @property {number} [cgMaxIter=400] - CG max iterations
 */

/**
 * Newmark-β implicit dynamics solver.
 *
 * General Newmark family with default parameters β=0.25, γ=0.5
 * (unconditionally stable, second-order accurate). Includes Rayleigh
 * damping C = αM + βK for energy dissipation.
 *
 * Uses the effective stiffness approach:
 *   K* = K + a₀·M + a₁·C
 * where a₀ = 1/(β·Δt²), a₁ = γ/(β·Δt).
 */
export class NewmarkDynamics {
    /**
     * @param {NewmarkConfig} [config]
     */
    constructor(config) {
        const c = config || {};
        /** @type {number} */ this.beta = c.beta !== undefined ? c.beta : 0.25;
        /** @type {number} */ this.gamma = c.gamma !== undefined ? c.gamma : 0.5;
        /** @type {number} */ this.alphaDamping = c.alphaDamping || 0.0;
        /** @type {number} */ this.betaDamping = c.betaDamping || 0.0;
        /** @type {number} */ this.dt = c.dt || 0.01;
        /** @type {number} */ this.totalTime = c.totalTime || 1.0;
        /** @type {number} */ this.maxSteps = c.maxSteps || 10000;
        /** @type {number} */ this.newtonTol = c.newtonTol || 1e-6;
        /** @type {number} */ this.maxNewtonIter = c.maxNewtonIter || 20;
        /** @type {number} */ this.outputInterval = c.outputInterval || 10;
        /** @type {number} */ this.cgTolerance = c.cgTolerance || CG_TOLERANCE;
        /** @type {number} */ this.cgMaxIter = c.cgMaxIter || MAX_CG_ITERATIONS;
    }

    /**
     * Perform a single Newmark-β implicit time step.
     *
     * Uses Newton-Raphson to solve the nonlinear effective system:
     *   K*·Δu = f_ext − f_int − M·ã − C·ṽ
     * where ã and ṽ are the Newmark predictors for acceleration and velocity.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Float64Array} u - Displacement at t_n (ndof), modified in-place
     * @param {Float64Array} v - Velocity at t_n (ndof), modified in-place
     * @param {Float64Array} a - Acceleration at t_n (ndof), modified in-place
     * @param {Float64Array} f_ext - External force at t_{n+1} (ndof)
     * @param {Float64Array} mass - Lumped mass vector (ndof)
     * @param {Int32Array|Uint32Array} fixedDOFs
     * @param {number} dt - Time step
     * @param {Array} elemStates - Per-element material states
     * @returns {{ converged: boolean, iterations: number, elemStates: Array,
     *             energy: EnergyBalance }}
     */
    step(mesh, material, u, v, a, f_ext, mass, fixedDOFs, dt, elemStates) {
        const ndof = mesh.nodeCount * 3;
        const beta = this.beta;
        const gamma = this.gamma;

        // Newmark coefficients
        const a0 = 1.0 / (beta * dt * dt);
        const a1 = gamma / (beta * dt);
        const a2 = 1.0 / (beta * dt);
        const a3 = 1.0 / (2.0 * beta) - 1.0;
        const a4 = gamma / beta - 1.0;
        const a5 = dt * 0.5 * (gamma / beta - 2.0);

        // Predictor: estimate u_{n+1} from current state
        const u_pred = new Float64Array(ndof);
        const v_pred = new Float64Array(ndof);
        const a_pred = new Float64Array(ndof);
        for (let i = 0; i < ndof; i++) {
            u_pred[i] = u[i] + dt * v[i] + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a[i];
            v_pred[i] = v[i] + dt * (1.0 - gamma) * a[i];
            a_pred[i] = 0.0;
        }

        // Newton-Raphson on u_{n+1}
        const u_new = new Float64Array(u_pred);
        let converged = false;
        let iterations = 0;
        let currentStates = elemStates;
        let strainEnergy = 0.0;

        for (let iter = 0; iter < this.maxNewtonIter; iter++) {
            iterations = iter + 1;

            // Current acceleration and velocity from Newmark update
            const a_new = new Float64Array(ndof);
            const v_new = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) {
                a_new[i] = a0 * (u_new[i] - u_pred[i]);
                v_new[i] = v_pred[i] + gamma * dt * a_new[i];
            }

            // Assemble internal force and tangent
            const assembled = assembleForceAndTangent(
                mesh, material, u_new, currentStates
            );
            strainEnergy = assembled.totalStrainEnergy;

            // Effective residual:
            // R = f_ext − f_int − M·a_new − C·v_new
            // C = α·M + β·K (Rayleigh), applied via lumped mass and stiffness
            const residual = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) {
                const inertia = mass[i] * a_new[i];
                const dampingMass = this.alphaDamping * mass[i] * v_new[i];
                residual[i] = f_ext[i] - assembled.fint[i] - inertia - dampingMass;
            }

            // Stiffness damping: subtract β_damp · K · v_new (element-by-element)
            if (this.betaDamping > 0.0) {
                const Kv = new Float64Array(ndof);
                ebeMatVec(mesh, assembled.elemTangents, v_new, Kv);
                for (let i = 0; i < ndof; i++) {
                    residual[i] -= this.betaDamping * Kv[i];
                }
            }

            // Apply BCs
            for (let i = 0; i < fixedDOFs.length; i++) residual[fixedDOFs[i]] = 0.0;

            const residualNorm = Math.sqrt(_dot(residual, residual, ndof));
            const refNorm = Math.sqrt(_dot(f_ext, f_ext, ndof)) + EPSILON;

            if (residualNorm / refNorm < this.newtonTol) {
                converged = true;
                currentStates = assembled.newStates;

                // Finalize acceleration and velocity
                for (let i = 0; i < ndof; i++) {
                    a[i] = a_new[i];
                    v[i] = v_new[i];
                    u[i] = u_new[i];
                }
                break;
            }

            // Effective stiffness: K* = K + a0·M + a1·C
            // Add mass and damping contributions to element tangent diagonals
            // (lumped mass approximation for efficiency)
            const effTangents = new Array(mesh.elemCount);
            for (let e = 0; e < mesh.elemCount; e++) {
                const srcKt = assembled.elemTangents[e];
                if (!srcKt) { effTangents[e] = null; continue; }

                const Keff = new Float64Array(srcKt);
                const nodes = mesh.getElementNodes(e);

                // Add diagonal mass and damping terms
                for (let i = 0; i < 8; i++) {
                    const gi = nodes[i] * 3;
                    for (let di = 0; di < 3; di++) {
                        const local = i * 3 + di;
                        const diagIdx = local * 24 + local;
                        const mVal = mass[gi + di];
                        const kDiag = srcKt[diagIdx];
                        // a0·M + a1·(α·M + β·K)
                        Keff[diagIdx] +=
                            a0 * mVal +
                            a1 * this.alphaDamping * mVal +
                            a1 * this.betaDamping * kDiag;
                    }
                }
                effTangents[e] = Keff;
            }

            // Solve effective system: K* · δu = R
            const cgResult = solveCG(
                mesh, effTangents, residual, fixedDOFs, ndof,
                this.cgTolerance, this.cgMaxIter
            );

            // Update displacement
            for (let i = 0; i < ndof; i++) u_new[i] += cgResult.x[i];
            currentStates = assembled.newStates;
        }

        if (!converged) {
            // Accept last iterate even without convergence
            for (let i = 0; i < ndof; i++) {
                a[i] = a0 * (u_new[i] - u_pred[i]);
                v[i] = v_pred[i] + gamma * dt * a[i];
                u[i] = u_new[i];
            }
        }

        // Kinetic energy
        let kinetic = 0.0;
        for (let i = 0; i < ndof; i++) kinetic += mass[i] * v[i] * v[i];
        kinetic *= 0.5;

        const energy = {
            kinetic,
            internal: strainEnergy,
            externalWork: 0.0, // tracked externally
            total: kinetic + strainEnergy
        };

        return { converged, iterations, elemStates: currentStates, energy };
    }

    /**
     * Run full Newmark-β implicit dynamics simulation.
     *
     * @param {object} mesh
     * @param {object} material
     * @param {Int32Array|Uint32Array|number[]} constraints - Fixed DOF indices
     * @param {Float64Array} loads - External force vector (ndof)
     * @param {NewmarkConfig} [config] - Override config
     * @param {function} [progressCallback] - Called with step info
     * @returns {{ snapshots: Array, finalDisplacement: Float64Array,
     *             totalSteps: number, finalTime: number,
     *             energyHistory: EnergyBalance[], converged: boolean }}
     */
    solve(mesh, material, constraints, loads, config, progressCallback) {
        const cfg = config || {};
        const dt = cfg.dt || this.dt;
        const totalTime = cfg.totalTime || this.totalTime;
        const maxSteps = cfg.maxSteps || this.maxSteps;
        const outputInterval = cfg.outputInterval || this.outputInterval;
        const density = cfg.density || 1.0;

        const fixedDOFs = constraints instanceof Int32Array || constraints instanceof Uint32Array
            ? constraints
            : new Int32Array(constraints);

        const ndof = mesh.nodeCount * 3;

        // Assemble lumped mass using ExplicitDynamics helper
        const explicitHelper = new ExplicitDynamics({ density });
        const mass = explicitHelper.assembleLumpedMass(mesh, density);

        // Initialize state
        const u = new Float64Array(ndof);
        const v = new Float64Array(ndof);
        const a = new Float64Array(ndof);
        let elemStates = new Array(mesh.elemCount).fill(null);

        // Initial acceleration: a_0 = M⁻¹ · (f_ext − f_int(0))
        for (let i = 0; i < ndof; i++) {
            if (mass[i] > EPSILON) a[i] = loads[i] / mass[i];
        }
        for (let i = 0; i < fixedDOFs.length; i++) a[fixedDOFs[i]] = 0.0;

        const snapshots = [];
        const energyHistory = [];
        let t = 0.0;
        let stepCount = 0;
        let overallConverged = true;

        while (t < totalTime && stepCount < maxSteps) {
            const dtStep = Math.min(dt, totalTime - t);
            t += dtStep;
            stepCount++;

            // Scale loads for current time (linear ramp)
            const loadFactor = Math.min(t / totalTime, 1.0);
            const f_ext_t = new Float64Array(ndof);
            for (let i = 0; i < ndof; i++) f_ext_t[i] = loads[i] * loadFactor;

            const result = this.step(
                mesh, material, u, v, a, f_ext_t, mass, fixedDOFs,
                dtStep, elemStates
            );

            elemStates = result.elemStates;
            energyHistory.push(result.energy);

            if (!result.converged) overallConverged = false;

            if (stepCount % outputInterval === 0 || t >= totalTime) {
                snapshots.push({
                    time: t,
                    step: stepCount,
                    u: new Float64Array(u),
                    v: new Float64Array(v),
                    a: new Float64Array(a),
                    energy: result.energy,
                    converged: result.converged
                });

                if (progressCallback) {
                    progressCallback({
                        step: stepCount,
                        time: t,
                        dt: dtStep,
                        energy: result.energy,
                        converged: result.converged,
                        progress: t / totalTime
                    });
                }
            }
        }

        return {
            snapshots,
            finalDisplacement: new Float64Array(u),
            totalSteps: stepCount,
            finalTime: t,
            energyHistory,
            converged: overallConverged
        };
    }
}
