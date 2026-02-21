/**
 * @fileoverview Fracture and damage mechanics module for 8-node hex elements.
 *
 * Implements crack growth methods (phase-field, cohesive zone, element erosion)
 * and failure/damage models (Johnson-Cook, Gurson-Tvergaard-Needleman, Lemaitre
 * CDM, classical failure criteria).
 *
 * All tensors use flat row-major arrays for performance.
 *   - Voigt stress/strain order: [11, 22, 33, 12, 23, 13]
 *   - 3×3 matrix: flat 9-element array
 *   - 6×6 constitutive tangent: flat 36-element array (row-major)
 *
 * Uses the same hex mesh and 2×2×2 Gauss quadrature as the nonlinear solver.
 *
 * @module fracture-solver
 */

import { MaterialState } from './material-models.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EPSILON = 1e-12;

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

/** @param {number[]} A 3×3 flat → determinant */
function det3(A) {
    return (
        A[0] * (A[4] * A[8] - A[5] * A[7]) -
        A[1] * (A[3] * A[8] - A[5] * A[6]) +
        A[2] * (A[3] * A[7] - A[4] * A[6])
    );
}

/** @param {number[]} A 3×3 flat → trace */
function trace3(A) {
    return A[0] + A[4] + A[8];
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

/** Symmetric 3×3 tensor → Voigt vector [11,22,33,12,23,13] */
function symToVoigt(S) {
    return [S[0], S[4], S[8], S[1], S[5], S[2]];
}

/** Voigt stress [11,22,33,12,23,13] → symmetric 3×3 tensor (flat) */
function voigtToSym(v) {
    return [
        v[0], v[3], v[5],
        v[3], v[1], v[4],
        v[5], v[4], v[2]
    ];
}

/**
 * Von Mises equivalent stress from Voigt stress.
 * @param {number[]} s - Voigt [σ11,σ22,σ33,σ12,σ23,σ13]
 * @returns {number}
 */
function vonMises(s) {
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
function principalStresses(s) {
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
 * Principal strains (eigenvalues of symmetric 3×3 strain tensor).
 * Converts from engineering shear to tensor shear before computing.
 * @param {number[]} e - Voigt [ε11,ε22,ε33,2ε12,2ε23,2ε13]
 * @returns {number[]} [ε1, ε2, ε3] sorted descending (ε1 ≥ ε2 ≥ ε3)
 */
function principalStrains(e) {
    // Convert engineering shear to tensor shear for eigenvalue computation
    return principalStresses([e[0], e[1], e[2], 0.5 * e[3], 0.5 * e[4], 0.5 * e[5]]);
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
        J[0] += dN[off]     * x;  J[1] += dN[off]     * y;  J[2] += dN[off]     * z;
        J[3] += dN[off + 1] * x;  J[4] += dN[off + 1] * y;  J[5] += dN[off + 1] * z;
        J[6] += dN[off + 2] * x;  J[7] += dN[off + 2] * y;  J[8] += dN[off + 2] * z;
    }
    const detJ = det3(J);
    const d = 1.0 / detJ;
    const invJ = [
        (J[4] * J[8] - J[5] * J[7]) * d,
        (J[2] * J[7] - J[1] * J[8]) * d,
        (J[1] * J[5] - J[2] * J[4]) * d,
        (J[5] * J[6] - J[3] * J[8]) * d,
        (J[0] * J[8] - J[2] * J[6]) * d,
        (J[2] * J[3] - J[0] * J[5]) * d,
        (J[3] * J[7] - J[4] * J[6]) * d,
        (J[1] * J[6] - J[0] * J[7]) * d,
        (J[0] * J[4] - J[1] * J[3]) * d
    ];
    return { J, invJ, detJ };
}

// ---------------------------------------------------------------------------
// Strain energy density split (tension/compression decomposition)
// ---------------------------------------------------------------------------

/**
 * Volumetric-deviatoric split of strain energy density.
 *
 * Decomposes the small-strain energy density into a volumetric (mean-stress)
 * part and a deviatoric part, then assigns only the tensile volumetric part
 * as the crack-driving energy ψ⁺.
 *
 *   ψ⁺ = K/2 · ⟨tr ε⟩₊² + μ · (ε_dev : ε_dev)
 *   ψ⁻ = K/2 · ⟨tr ε⟩₋²
 *
 * @param {number[]} strain_voigt - Small strain [ε11,ε22,ε33,2ε12,2ε23,2ε13]
 * @param {number} K - Bulk modulus
 * @param {number} mu - Shear modulus
 * @returns {{ psiPlus: number, psiMinus: number }}
 */
export function volumetricDeviatoricSplit(strain_voigt, K, mu) {
    const trEps = strain_voigt[0] + strain_voigt[1] + strain_voigt[2];
    const trEpsPlus = Math.max(0.0, trEps);
    const trEpsMinus = Math.min(0.0, trEps);

    // Deviatoric strain
    const mean = trEps / 3.0;
    const ed0 = strain_voigt[0] - mean;
    const ed1 = strain_voigt[1] - mean;
    const ed2 = strain_voigt[2] - mean;
    const ed3 = 0.5 * strain_voigt[3]; // tensor shear
    const ed4 = 0.5 * strain_voigt[4];
    const ed5 = 0.5 * strain_voigt[5];

    const devNormSq = ed0 * ed0 + ed1 * ed1 + ed2 * ed2 +
                      2.0 * (ed3 * ed3 + ed4 * ed4 + ed5 * ed5);

    const psiPlus  = 0.5 * K * trEpsPlus * trEpsPlus + mu * devNormSq;
    const psiMinus = 0.5 * K * trEpsMinus * trEpsMinus;
    return { psiPlus, psiMinus };
}

/**
 * Spectral decomposition split of strain energy density.
 *
 * Decomposes strain into positive and negative parts using principal strains:
 *   ε⁺ = Σ ⟨εᵢ⟩₊ nᵢ⊗nᵢ,  ε⁻ = Σ ⟨εᵢ⟩₋ nᵢ⊗nᵢ
 *   ψ⁺ = λ/2 · ⟨tr ε⟩₊² + μ · (ε⁺ : ε⁺)
 *   ψ⁻ = λ/2 · ⟨tr ε⟩₋² + μ · (ε⁻ : ε⁻)
 *
 * @param {number[]} strain_voigt - Small strain [ε11,ε22,ε33,2ε12,2ε23,2ε13]
 * @param {number} lambda - Lamé first parameter
 * @param {number} mu - Lamé second parameter (shear modulus)
 * @returns {{ psiPlus: number, psiMinus: number }}
 */
export function spectralSplit(strain_voigt, lambda, mu) {
    const eps = principalStrains(strain_voigt);

    const trPlus  = Math.max(0.0, eps[0]) + Math.max(0.0, eps[1]) + Math.max(0.0, eps[2]);
    const trMinus = Math.min(0.0, eps[0]) + Math.min(0.0, eps[1]) + Math.min(0.0, eps[2]);

    const epsPlus2  = Math.max(0.0, eps[0]) ** 2 + Math.max(0.0, eps[1]) ** 2 + Math.max(0.0, eps[2]) ** 2;
    const epsMinus2 = Math.min(0.0, eps[0]) ** 2 + Math.min(0.0, eps[1]) ** 2 + Math.min(0.0, eps[2]) ** 2;

    const psiPlus  = 0.5 * lambda * trPlus * trPlus + mu * epsPlus2;
    const psiMinus = 0.5 * lambda * trMinus * trMinus + mu * epsMinus2;
    return { psiPlus, psiMinus };
}

// ---------------------------------------------------------------------------
// Phase-Field Fracture (AT2 model)
// ---------------------------------------------------------------------------

/**
 * Phase-field fracture model using the AT2 regularized crack functional.
 *
 * Represents cracks diffusely through an auxiliary scalar field d ∈ [0,1]
 * (0 = intact, 1 = fully broken). The crack surface energy is approximated as:
 *
 *   Γ(d) = Gc / (2cw) · ∫ (d²/ℓ + ℓ|∇d|²) dV
 *
 * where cw = 1/2 for AT2, Gc is the critical energy release rate, and ℓ is
 * the length-scale parameter controlling crack width.
 *
 * A staggered (alternate minimization) scheme solves the coupled displacement
 * and phase-field problems sequentially within each load step.
 */
export class PhaseFieldFracture {
    /**
     * @param {object} config
     * @param {number} config.Gc - Critical energy release rate (J/m²)
     * @param {number} config.lengthScale - Regularization length ℓ
     * @param {string} [config.splitType='volumetric-deviatoric'] - Energy split
     *   ('spectral' or 'volumetric-deviatoric')
     * @param {number} [config.kStab=1e-6] - Small residual stiffness for stability
     * @param {number} [config.E] - Young's modulus (needed for split)
     * @param {number} [config.nu] - Poisson's ratio (needed for split)
     */
    constructor(config) {
        const c = config || {};
        this.Gc = c.Gc || 2700.0;
        this.lengthScale = c.lengthScale || 0.01;
        this.splitType = c.splitType || 'volumetric-deviatoric';
        this.kStab = c.kStab != null ? c.kStab : 1e-6;

        // Elastic constants for energy split
        const E  = c.E || 210000.0;
        const nu = c.nu || 0.3;
        this.lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        this.mu = E / (2.0 * (1.0 + nu));
        this.K = E / (3.0 * (1.0 - 2.0 * nu));

        // AT2 constant: c_w = 1/2
        this.cw = 0.5;
    }

    /**
     * Degradation function g(d) = (1-d)² + k.
     * @param {number} d - Phase-field damage value
     * @returns {number}
     */
    degradation(d) {
        const omd = 1.0 - d;
        return omd * omd + this.kStab;
    }

    /**
     * Derivative of degradation function g'(d) = -2(1-d).
     * @param {number} d - Phase-field damage value
     * @returns {number}
     */
    degradationPrime(d) {
        return -2.0 * (1.0 - d);
    }

    /**
     * Initialize phase-field arrays for a given number of elements.
     * @param {number} nElements - Number of elements in the mesh
     * @returns {{ d: Float64Array, H: Float64Array }}
     *   d – damage field (per-element, initialized to 0)
     *   H – history variable max(ψ⁺) for irreversibility (per-element)
     */
    initializeField(nElements) {
        return {
            d: new Float64Array(nElements),
            H: new Float64Array(nElements)
        };
    }

    /**
     * Compute crack-driving energy ψ⁺ at a Gauss point from strain.
     * @param {number[]} strain_voigt - Small strain in Voigt notation
     * @returns {number} ψ⁺ (tensile strain energy density)
     */
    crackDrivingEnergy(strain_voigt) {
        if (this.splitType === 'spectral') {
            return spectralSplit(strain_voigt, this.lambda, this.mu).psiPlus;
        }
        return volumetricDeviatoricSplit(strain_voigt, this.K, this.mu).psiPlus;
    }

    /**
     * Update the phase-field damage and history variable for all elements.
     *
     * For each element, evaluates ψ⁺ at all Gauss points, takes the maximum,
     * updates the irreversibility history H, then solves the local phase-field
     * equation analytically (element-wise, explicit update):
     *
     *   d_new = H / (H + Gc/(2ℓcw))
     *
     * @param {object} mesh - Mesh with nodeCoords, connectivity, nElements
     * @param {Float64Array} u - Global displacement vector
     * @param {object} material - Material model (used for small-strain extraction)
     * @param {Float64Array} d - Current damage field (per-element)
     * @param {Float64Array} H - History variable (per-element)
     * @returns {{ d: Float64Array, H: Float64Array }}
     */
    updateDamageField(mesh, u, material, d, H) {
        const nElem = mesh.nElements || d.length;
        const threshold = this.Gc / (2.0 * this.lengthScale * this.cw);

        for (let e = 0; e < nElem; e++) {
            // Extract element displacements
            const u_elem = new Float64Array(24);
            const nodeCoords = new Float64Array(24);
            for (let n = 0; n < 8; n++) {
                const nid = mesh.connectivity[e * 8 + n];
                for (let j = 0; j < 3; j++) {
                    u_elem[n * 3 + j] = u[nid * 3 + j];
                    nodeCoords[n * 3 + j] = mesh.nodeCoords[nid * 3 + j];
                }
            }

            // Evaluate ψ⁺ at all Gauss points, take maximum
            let psiMax = 0.0;
            for (let gp = 0; gp < 8; gp++) {
                const [xi, eta, zeta] = GAUSS_POINTS[gp];
                const strain = this._computeSmallStrain(nodeCoords, u_elem, xi, eta, zeta);
                const psiPlus = this.crackDrivingEnergy(strain);
                if (psiPlus > psiMax) psiMax = psiPlus;
            }

            // Irreversibility: H = max(H_old, ψ⁺)
            if (psiMax > H[e]) H[e] = psiMax;

            // Analytical local phase-field update
            d[e] = H[e] / (H[e] + threshold);
        }

        return { d, H };
    }

    /**
     * Apply degradation to an element stiffness matrix.
     * Scales Ke by the degradation function g(d).
     * @param {Float64Array} Ke - 24×24 element stiffness (576 values), modified in-place
     * @param {number} d_elem - Element damage value
     * @returns {Float64Array} Ke (degraded)
     */
    degradeStiffness(Ke, d_elem) {
        const g = this.degradation(d_elem);
        for (let i = 0; i < 576; i++) {
            Ke[i] *= g;
        }
        return Ke;
    }

    /**
     * Compute the element-level phase-field stiffness matrix and residual
     * for the damage sub-problem.
     *
     * Phase-field weak form (AT2):
     *   Kd_ij = ∫ (Gc·ℓ/cw · ∇Nᵢ·∇Nⱼ + (Gc/(cw·ℓ) + 2H) · Nᵢ·Nⱼ) dV
     *   Rd_i  = ∫ (Gc·ℓ/cw · ∇d·∇Nᵢ + (Gc/(cw·ℓ)·d - 2H·(1-d)) · Nᵢ) dV
     *
     * @param {object} mesh - Mesh with nodeCoords, connectivity
     * @param {number} elemIdx - Element index
     * @param {Float64Array} d - Global damage field (per-node for this formulation)
     * @param {Float64Array} H - History variable (per-element, averaged to nodes)
     * @returns {{ Kd: Float64Array, Rd: Float64Array }}
     *   Kd – 8×8 phase-field stiffness (flat 64 values)
     *   Rd – 8-element phase-field residual
     */
    getElementPhaseFieldStiffness(mesh, elemIdx, d, H) {
        const Kd = new Float64Array(64);
        const Rd = new Float64Array(8);
        const ell = this.lengthScale;
        const gcOverCwEll = this.Gc / (this.cw * ell);
        const gcEllOverCw = this.Gc * ell / this.cw;
        const H_e = H[elemIdx];

        // Element node coordinates
        const nodeCoords = new Float64Array(24);
        const d_local = new Float64Array(8);
        for (let n = 0; n < 8; n++) {
            const nid = mesh.connectivity[elemIdx * 8 + n];
            for (let j = 0; j < 3; j++) {
                nodeCoords[n * 3 + j] = mesh.nodeCoords[nid * 3 + j];
            }
            d_local[n] = d[nid] != null ? d[nid] : d[elemIdx];
        }

        for (let gp = 0; gp < 8; gp++) {
            const [xi, eta, zeta] = GAUSS_POINTS[gp];
            const N = shapeFunctions(xi, eta, zeta);
            const dN = shapeFunctionDerivatives(xi, eta, zeta);
            const { invJ, detJ } = jacobian(nodeCoords, dN);
            const w = Math.abs(detJ) * GAUSS_WEIGHT;

            // Physical shape function gradients
            const dNdX = new Float64Array(24);
            for (let n = 0; n < 8; n++) {
                const off = n * 3;
                const d0 = dN[off], d1 = dN[off + 1], d2 = dN[off + 2];
                dNdX[off]     = invJ[0] * d0 + invJ[1] * d1 + invJ[2] * d2;
                dNdX[off + 1] = invJ[3] * d0 + invJ[4] * d1 + invJ[5] * d2;
                dNdX[off + 2] = invJ[6] * d0 + invJ[7] * d1 + invJ[8] * d2;
            }

            // Interpolate d and ∇d at this GP
            let d_gp = 0.0;
            const gradD = [0.0, 0.0, 0.0];
            for (let n = 0; n < 8; n++) {
                d_gp += N[n] * d_local[n];
                const off = n * 3;
                gradD[0] += dNdX[off]     * d_local[n];
                gradD[1] += dNdX[off + 1] * d_local[n];
                gradD[2] += dNdX[off + 2] * d_local[n];
            }

            // Source term coefficient
            const sourceCoeff = gcOverCwEll + 2.0 * H_e;

            for (let i = 0; i < 8; i++) {
                const iOff = i * 3;
                // Residual contribution
                let gradDotGradNi = dNdX[iOff] * gradD[0] +
                                    dNdX[iOff + 1] * gradD[1] +
                                    dNdX[iOff + 2] * gradD[2];
                Rd[i] += (gcEllOverCw * gradDotGradNi +
                          (gcOverCwEll * d_gp - 2.0 * H_e * (1.0 - d_gp)) * N[i]) * w;

                for (let j = 0; j < 8; j++) {
                    const jOff = j * 3;
                    // Gradient-gradient term
                    const gradNiGradNj = dNdX[iOff] * dNdX[jOff] +
                                         dNdX[iOff + 1] * dNdX[jOff + 1] +
                                         dNdX[iOff + 2] * dNdX[jOff + 2];
                    Kd[i * 8 + j] += (gcEllOverCw * gradNiGradNj +
                                      sourceCoeff * N[i] * N[j]) * w;
                }
            }
        }

        return { Kd, Rd };
    }

    /**
     * Compute small strain ε = 0.5(∂u/∂X + (∂u/∂X)^T) at a point.
     * @param {Float64Array} nodeCoords - Reference coordinates (24 values)
     * @param {Float64Array} u_elem - Element displacements (24 values)
     * @param {number} xi
     * @param {number} eta
     * @param {number} zeta
     * @returns {number[]} Voigt strain [ε11,ε22,ε33,2ε12,2ε23,2ε13]
     * @private
     */
    _computeSmallStrain(nodeCoords, u_elem, xi, eta, zeta) {
        const dN = shapeFunctionDerivatives(xi, eta, zeta);
        const { invJ } = jacobian(nodeCoords, dN);

        // dN/dX = invJ · dN/dξ
        const dNdX = new Float64Array(24);
        for (let n = 0; n < 8; n++) {
            const off = n * 3;
            const d0 = dN[off], d1 = dN[off + 1], d2 = dN[off + 2];
            dNdX[off]     = invJ[0] * d0 + invJ[1] * d1 + invJ[2] * d2;
            dNdX[off + 1] = invJ[3] * d0 + invJ[4] * d1 + invJ[5] * d2;
            dNdX[off + 2] = invJ[6] * d0 + invJ[7] * d1 + invJ[8] * d2;
        }

        // ∂u/∂X
        const dudX = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (let n = 0; n < 8; n++) {
            const uOff = n * 3;
            const dOff = n * 3;
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    dudX[i * 3 + j] += u_elem[uOff + i] * dNdX[dOff + j];
                }
            }
        }

        // ε = 0.5(∂u/∂X + (∂u/∂X)^T) in Voigt
        return [
            dudX[0],                          // ε11
            dudX[4],                          // ε22
            dudX[8],                          // ε33
            dudX[1] + dudX[3],                // 2ε12
            dudX[5] + dudX[7],                // 2ε23
            dudX[2] + dudX[6]                 // 2ε13
        ];
    }
}

// ---------------------------------------------------------------------------
// Cohesive Zone Model (CZM)
// ---------------------------------------------------------------------------

/**
 * Cohesive Zone Model with bilinear traction-separation law.
 *
 * Models fracture through interface elements between hex element faces.
 * Uses a bilinear law: linear elastic rise to peak traction σ_max at
 * separation δ_0, followed by linear softening to zero traction at δ_c.
 *
 * Mixed-mode coupling uses an effective displacement:
 *   δ_eff = √(δ_n² + β²·δ_t²)
 *
 * where β weights shear contribution relative to opening.
 */
export class CohesiveZoneModel {
    /**
     * @param {object} config
     * @param {number} config.sigmaMax - Peak cohesive traction (Mode I)
     * @param {number} config.deltaC - Critical separation (complete failure)
     * @param {number} config.GIc - Mode I fracture energy
     * @param {number} [config.GIIc] - Mode II fracture energy (defaults to GIc)
     * @param {number} [config.beta=1.0] - Mode-mixity parameter
     */
    constructor(config) {
        const c = config || {};
        this.sigmaMax = c.sigmaMax || 100.0;
        this.deltaC = c.deltaC || 0.01;
        this.GIc = c.GIc || (0.5 * this.sigmaMax * this.deltaC);
        this.GIIc = c.GIIc || this.GIc;
        this.beta = c.beta != null ? c.beta : 1.0;

        // Initial stiffness penalty: δ_0 = 2·GIc / σ_max
        this.delta0 = 2.0 * this.GIc / this.sigmaMax;
        // Ensure delta0 is on the ascending branch
        if (this.delta0 >= this.deltaC) {
            this.delta0 = this.deltaC * 0.01;
        }
        this.Kpenalty = this.sigmaMax / this.delta0;
    }

    /**
     * Compute traction, tangent stiffness, and damage state for a given
     * separation (opening and tangential).
     *
     * Bilinear law:
     *   δ_eff < δ_0 : T = K_penalty · δ_eff  (ascending, elastic)
     *   δ_0 ≤ δ_eff < δ_c : T = σ_max · (δ_c - δ_eff) / (δ_c - δ_0) (softening)
     *   δ_eff ≥ δ_c : T = 0 (fully separated)
     *
     * @param {number} delta_n - Normal (opening) separation
     * @param {number} delta_t - Tangential (shear) separation magnitude
     * @param {object} state - Cohesive state { maxDelta: number, damaged: boolean }
     * @returns {{ traction_n: number, traction_t: number, tangent_nn: number,
     *             tangent_tt: number, tangent_nt: number, damaged: boolean,
     *             state: object }}
     */
    computeTraction(delta_n, delta_t, state) {
        const st = state || { maxDelta: 0.0, damaged: false };

        // Effective displacement (mixed-mode)
        const deltaEff = Math.sqrt(
            Math.max(0.0, delta_n) * Math.max(0.0, delta_n) +
            this.beta * this.beta * delta_t * delta_t
        );

        // Irreversibility: track maximum effective displacement
        const maxDelta = Math.max(st.maxDelta, deltaEff);

        // Contact in compression: penalize negative opening
        if (delta_n < 0.0) {
            return {
                traction_n: this.Kpenalty * delta_n,
                traction_t: 0.0,
                tangent_nn: this.Kpenalty,
                tangent_tt: 0.0,
                tangent_nt: 0.0,
                damaged: st.damaged,
                state: { maxDelta, damaged: st.damaged }
            };
        }

        let T_eff, dTdDelta;

        if (maxDelta <= EPSILON) {
            // Zero separation
            T_eff = 0.0;
            dTdDelta = this.Kpenalty;
        } else if (maxDelta < this.delta0) {
            // Ascending branch (elastic)
            T_eff = this.Kpenalty * deltaEff;
            dTdDelta = this.Kpenalty;
        } else if (maxDelta < this.deltaC) {
            // Softening branch with irreversibility (unload/reload to origin)
            const softScale = this.sigmaMax * (this.deltaC - maxDelta) /
                              ((this.deltaC - this.delta0) * maxDelta);
            T_eff = softScale * deltaEff;
            if (deltaEff < maxDelta) {
                // Unloading/reloading: secant stiffness to origin
                dTdDelta = softScale;
            } else {
                // Continued softening: tangent of the softening curve
                dTdDelta = -this.sigmaMax / (this.deltaC - this.delta0);
            }
        } else {
            // Fully separated
            return {
                traction_n: 0.0,
                traction_t: 0.0,
                tangent_nn: 0.0,
                tangent_tt: 0.0,
                tangent_nt: 0.0,
                damaged: true,
                state: { maxDelta, damaged: true }
            };
        }

        // Decompose effective traction into normal and tangential components
        let traction_n, traction_t, tangent_nn, tangent_tt, tangent_nt;
        if (deltaEff > EPSILON) {
            const ratio_n = Math.max(0.0, delta_n) / deltaEff;
            const ratio_t = this.beta * delta_t / deltaEff;
            traction_n = T_eff * ratio_n;
            traction_t = T_eff * this.beta * ratio_t;
            const Tsec = T_eff / deltaEff;
            tangent_nn = dTdDelta * ratio_n * ratio_n + Tsec * (1.0 - ratio_n * ratio_n);
            tangent_tt = (dTdDelta * ratio_t * ratio_t + Tsec * (1.0 - ratio_t * ratio_t)) * this.beta * this.beta;
            tangent_nt = (dTdDelta - Tsec) * ratio_n * ratio_t * this.beta;
        } else {
            traction_n = 0.0;
            traction_t = 0.0;
            tangent_nn = this.Kpenalty;
            tangent_tt = this.Kpenalty * this.beta * this.beta;
            tangent_nt = 0.0;
        }

        const damaged = maxDelta >= this.delta0;
        return {
            traction_n,
            traction_t,
            tangent_nn,
            tangent_tt,
            tangent_nt,
            damaged,
            state: { maxDelta, damaged }
        };
    }
}

// ---------------------------------------------------------------------------
// Element Erosion / Deletion
// ---------------------------------------------------------------------------

/**
 * Element erosion (deletion) strategy for failure simulation.
 *
 * Removes elements from the active mesh when their damage exceeds a
 * critical threshold. Supports optional nonlocal regularization to
 * mitigate mesh dependence.
 */
export class ElementErosion {
    /**
     * @param {object} config
     * @param {number} [config.threshold=0.95] - Critical damage for erosion
     * @param {boolean} [config.useNonlocal=false] - Enable nonlocal averaging
     * @param {number} [config.nonlocalRadius=0.0] - Nonlocal interaction radius
     */
    constructor(config) {
        const c = config || {};
        this.threshold = c.threshold != null ? c.threshold : 0.95;
        this.useNonlocal = c.useNonlocal || false;
        this.nonlocalRadius = c.nonlocalRadius || 0.0;

        /** @type {Set<number>} Set of eroded element indices */
        this.erodedElements = new Set();
    }

    /**
     * Check element damage values and erode elements above threshold.
     *
     * @param {Float64Array|number[]} elementDamage - Damage value per element
     * @param {number} [threshold] - Override default threshold
     * @returns {{ eroded: Set<number>, newlyEroded: number[] }}
     */
    checkAndErode(elementDamage, threshold) {
        const thr = threshold != null ? threshold : this.threshold;
        const newlyEroded = [];

        for (let e = 0; e < elementDamage.length; e++) {
            if (this.erodedElements.has(e)) continue;
            if (elementDamage[e] >= thr) {
                this.erodedElements.add(e);
                newlyEroded.push(e);
            }
        }

        return { eroded: this.erodedElements, newlyEroded };
    }

    /**
     * Apply nonlocal averaging to element damage values.
     *
     * Replaces each element's damage with a weighted average over
     * neighboring elements within the nonlocal interaction radius.
     * Uses a Gaussian weighting function.
     *
     * @param {Float64Array|number[]} elementDamage - Raw damage per element
     * @param {Float64Array} elementCentroids - Centroid coordinates (nElem × 3 flat)
     * @returns {Float64Array} Smoothed (nonlocal) damage values
     */
    nonlocalAverage(elementDamage, elementCentroids) {
        const nElem = elementDamage.length;
        const smoothed = new Float64Array(nElem);
        const R = this.nonlocalRadius;
        const R2 = R * R;

        if (R < EPSILON) {
            smoothed.set(elementDamage);
            return smoothed;
        }

        for (let i = 0; i < nElem; i++) {
            if (this.erodedElements.has(i)) {
                smoothed[i] = elementDamage[i];
                continue;
            }
            const xi = elementCentroids[i * 3];
            const yi = elementCentroids[i * 3 + 1];
            const zi = elementCentroids[i * 3 + 2];

            let weightSum = 0.0;
            let damageSum = 0.0;

            for (let j = 0; j < nElem; j++) {
                if (this.erodedElements.has(j)) continue;
                const dx = elementCentroids[j * 3]     - xi;
                const dy = elementCentroids[j * 3 + 1] - yi;
                const dz = elementCentroids[j * 3 + 2] - zi;
                const dist2 = dx * dx + dy * dy + dz * dz;
                if (dist2 > R2) continue;

                // Gaussian weight: exp(-dist² / (2·(R/3)²))
                const w = Math.exp(-4.5 * dist2 / R2);
                weightSum += w;
                damageSum += w * elementDamage[j];
            }

            smoothed[i] = weightSum > EPSILON ? damageSum / weightSum : elementDamage[i];
        }

        return smoothed;
    }

    /**
     * Check whether an element has been eroded.
     * @param {number} elemIdx - Element index
     * @returns {boolean}
     */
    isEroded(elemIdx) {
        return this.erodedElements.has(elemIdx);
    }

    /**
     * Reset all erosion state.
     */
    reset() {
        this.erodedElements.clear();
    }
}

// ---------------------------------------------------------------------------
// Johnson-Cook Damage Model
// ---------------------------------------------------------------------------

/**
 * Johnson-Cook damage model for ductile fracture.
 *
 * Fracture strain:
 *   ε_f = [D1 + D2·exp(D3·η)] · [1 + D4·ln(ε̇*)] · [1 + D5·T*]
 *
 * Damage accumulation:
 *   D = Σ(Δε_p / ε_f)
 *
 * Element fails when D ≥ 1.
 *
 * Designed to couple with J2 plasticity from material-models.js through
 * the MaterialState plastic strain.
 */
export class JohnsonCookDamage {
    /**
     * @param {object} params
     * @param {number} params.D1 - Initial failure strain constant
     * @param {number} params.D2 - Exponential factor
     * @param {number} params.D3 - Triaxiality exponent
     * @param {number} params.D4 - Strain rate factor
     * @param {number} params.D5 - Temperature factor
     * @param {number} [params.epsDot0=1.0] - Reference strain rate
     * @param {number} [params.Tmelt=1500] - Melt temperature (K)
     * @param {number} [params.Troom=293] - Room temperature (K)
     */
    constructor(params) {
        const p = params || {};
        this.D1 = p.D1 != null ? p.D1 : 0.05;
        this.D2 = p.D2 != null ? p.D2 : 3.44;
        this.D3 = p.D3 != null ? p.D3 : -2.12;
        this.D4 = p.D4 != null ? p.D4 : 0.002;
        this.D5 = p.D5 != null ? p.D5 : 0.61;
        this.epsDot0 = p.epsDot0 || 1.0;
        this.Tmelt = p.Tmelt || 1500.0;
        this.Troom = p.Troom || 293.0;
    }

    /**
     * Compute the failure strain ε_f for given conditions.
     *
     * @param {number} triaxiality - Stress triaxiality η = σ_h / σ_vm
     * @param {number} [strainRate=1.0] - Equivalent plastic strain rate
     * @param {number} [temperature=293] - Current temperature (K)
     * @returns {number} ε_f – failure strain (clamped ≥ EPSILON)
     */
    computeFailureStrain(triaxiality, strainRate, temperature) {
        const sr = strainRate != null ? strainRate : 1.0;
        const T  = temperature != null ? temperature : this.Troom;

        // Triaxiality term
        const f1 = this.D1 + this.D2 * Math.exp(this.D3 * triaxiality);

        // Strain rate term: [1 + D4·ln(ε̇*)]
        const epsDotStar = Math.max(sr / this.epsDot0, EPSILON);
        const f2 = 1.0 + this.D4 * Math.log(epsDotStar);

        // Temperature term: [1 + D5·T*]  where T* = (T - Troom)/(Tmelt - Troom)
        const Tstar = Math.max(0.0, Math.min(1.0,
            (T - this.Troom) / (this.Tmelt - this.Troom)));
        const f3 = 1.0 + this.D5 * Tstar;

        return Math.max(EPSILON, f1 * f2 * f3);
    }

    /**
     * Update cumulative damage for a material point.
     *
     * @param {object} state - Damage state { D: number, epsPl: number }
     * @param {number} dEpsilonP - Incremental equivalent plastic strain
     * @param {number} triaxiality - Current stress triaxiality
     * @param {number} [strainRate=1.0] - Current strain rate
     * @param {number} [temperature=293] - Current temperature (K)
     * @returns {{ D: number, epsPl: number, failed: boolean }}
     */
    updateDamage(state, dEpsilonP, triaxiality, strainRate, temperature) {
        const st = state || { D: 0.0, epsPl: 0.0 };
        const epsF = this.computeFailureStrain(triaxiality, strainRate, temperature);
        const newD = st.D + dEpsilonP / epsF;
        const newEpsPl = st.epsPl + dEpsilonP;

        return {
            D: Math.min(1.0, newD),
            epsPl: newEpsPl,
            failed: newD >= 1.0
        };
    }
}

// ---------------------------------------------------------------------------
// Gurson-Tvergaard-Needleman (GTN) Model
// ---------------------------------------------------------------------------

/**
 * Gurson-Tvergaard-Needleman porous plasticity model.
 *
 * Yield function:
 *   Φ = (σ_eq/σ_y)² + 2·q1·f*·cosh(3·q2·σ_m/(2·σ_y)) - 1 - (q1·f*)² = 0
 *
 * Models void nucleation, growth, and coalescence in ductile materials.
 * The effective void volume fraction f* accelerates damage at coalescence:
 *   f* = f                                         for f ≤ f_c
 *   f* = f_c + (f_u* - f_c)/(f_F - f_c)·(f - f_c) for f > f_c
 *
 * where f_u* = 1/q1 (complete loss of stress-carrying capacity).
 */
export class GursonModel {
    /**
     * @param {object} params
     * @param {number} [params.q1=1.5] - Tvergaard fitting parameter
     * @param {number} [params.q2=1.0] - Tvergaard fitting parameter
     * @param {number} [params.f0=0.001] - Initial void volume fraction
     * @param {number} [params.fc=0.15] - Critical void fraction (coalescence onset)
     * @param {number} [params.fF=0.25] - Void fraction at final failure
     * @param {number} [params.fN=0.04] - Volume fraction of nucleating particles
     * @param {number} [params.epsN=0.3] - Mean nucleation strain
     * @param {number} [params.sN=0.1] - Standard deviation of nucleation strain
     * @param {number} [params.sigmaY=250] - Initial yield stress
     */
    constructor(params) {
        const p = params || {};
        this.q1 = p.q1 != null ? p.q1 : 1.5;
        this.q2 = p.q2 != null ? p.q2 : 1.0;
        this.f0 = p.f0 != null ? p.f0 : 0.001;
        this.fc = p.fc != null ? p.fc : 0.15;
        this.fF = p.fF != null ? p.fF : 0.25;
        this.fN = p.fN != null ? p.fN : 0.04;
        this.epsN = p.epsN != null ? p.epsN : 0.3;
        this.sN = p.sN != null ? p.sN : 0.1;
        this.sigmaY = p.sigmaY != null ? p.sigmaY : 250.0;

        // Ultimate void fraction f_u* = 1/q1
        this.fU = 1.0 / this.q1;
    }

    /**
     * Compute the effective void volume fraction f* accounting for
     * accelerated coalescence.
     *
     * @param {number} f - Current void volume fraction
     * @returns {number} f* – effective void fraction
     */
    effectiveVoidFraction(f) {
        if (f <= this.fc) {
            return f;
        }
        // Accelerated coalescence: linear interpolation to f_u*
        const kappa = (this.fU - this.fc) / (this.fF - this.fc);
        return this.fc + kappa * (f - this.fc);
    }

    /**
     * Evaluate the GTN yield function.
     *
     *   Φ = (σ_eq/σ_y)² + 2·q1·f*·cosh(3·q2·σ_m/(2·σ_y)) - 1 - (q1·f*)²
     *
     * @param {number} sigmaEq - Von Mises equivalent stress
     * @param {number} sigmaM - Mean (hydrostatic) stress σ_m = tr(σ)/3
     * @param {number} sigmaY - Current yield stress
     * @param {number} f - Current void volume fraction
     * @returns {number} Φ – yield function value (Φ ≤ 0 elastic, Φ = 0 yielding)
     */
    yieldFunction(sigmaEq, sigmaM, sigmaY, f) {
        const fStar = this.effectiveVoidFraction(f);
        const q1f = this.q1 * fStar;
        const ratio = sigmaEq / (sigmaY + EPSILON);
        const coshArg = 1.5 * this.q2 * sigmaM / (sigmaY + EPSILON);

        return ratio * ratio + 2.0 * q1f * Math.cosh(coshArg) - 1.0 - q1f * q1f;
    }

    /**
     * Compute void nucleation rate from strain-controlled nucleation
     * (Chu & Needleman Gaussian distribution).
     *
     *   A_N = f_N / (s_N·√(2π)) · exp(-0.5·((ε_p - ε_N)/s_N)²)
     *   ḟ_nucl = A_N · ε̇_p
     *
     * @param {number} epsPl - Current equivalent plastic strain
     * @returns {number} A_N – nucleation intensity
     */
    nucleationRate(epsPl) {
        const diff = (epsPl - this.epsN) / this.sN;
        const coeff = this.fN / (this.sN * Math.sqrt(2.0 * Math.PI));
        return coeff * Math.exp(-0.5 * diff * diff);
    }

    /**
     * Update void volume fraction for a load increment.
     *
     * Void evolution:
     *   ḟ = ḟ_growth + ḟ_nucleation
     *   ḟ_growth = (1-f)·tr(ε̇_p)            (volume change of matrix)
     *   ḟ_nucl = A_N · ε̇_p                  (strain-controlled nucleation)
     *
     * @param {object} state - Void state { f: number, epsPl: number }
     * @param {number} dEpsilonP - Incremental equivalent plastic strain
     * @param {number} [trDEpsP=0] - Trace of plastic strain increment
     * @returns {{ f: number, fStar: number, epsPl: number, failed: boolean }}
     */
    updateVoidFraction(state, dEpsilonP, trDEpsP) {
        const st = state || { f: this.f0, epsPl: 0.0 };
        const trDep = trDEpsP != null ? trDEpsP : 0.0;

        // Growth: ḟ_growth = (1 - f)·tr(Δε_p)
        const dfGrowth = (1.0 - st.f) * trDep;

        // Nucleation
        const AN = this.nucleationRate(st.epsPl);
        const dfNucl = AN * dEpsilonP;

        const newF = Math.max(0.0, Math.min(1.0, st.f + dfGrowth + dfNucl));
        const newEpsPl = st.epsPl + dEpsilonP;

        return {
            f: newF,
            fStar: this.effectiveVoidFraction(newF),
            epsPl: newEpsPl,
            failed: newF >= this.fF
        };
    }
}

// ---------------------------------------------------------------------------
// Lemaitre Continuum Damage Mechanics (CDM)
// ---------------------------------------------------------------------------

/**
 * Lemaitre CDM model for coupled damage-plasticity.
 *
 * Damage evolution law:
 *   Ḋ = (Y/S)^s · ṗ    for p ≥ p_D
 *
 * Energy release rate:
 *   Y = σ_eq² · R_ν / (2·E·(1-D)²)
 *
 * Triaxiality function:
 *   R_ν = 2/3·(1+ν) + 3·(1-2ν)·η²
 *
 * Damage activates only after a threshold plastic strain p_D is reached.
 */
export class LemaitreDamage {
    /**
     * @param {object} params
     * @param {number} params.E - Young's modulus
     * @param {number} params.nu - Poisson's ratio
     * @param {number} [params.S=2.0] - Damage strength parameter
     * @param {number} [params.s=1.0] - Damage exponent
     * @param {number} [params.pD=0.0] - Damage threshold plastic strain
     * @param {number} [params.Dc=0.99] - Critical damage at fracture
     */
    constructor(params) {
        const p = params || {};
        this.E = p.E || 210000.0;
        this.nu = p.nu != null ? p.nu : 0.3;
        this.S = p.S != null ? p.S : 2.0;
        this.s = p.s != null ? p.s : 1.0;
        this.pD = p.pD != null ? p.pD : 0.0;
        this.Dc = p.Dc != null ? p.Dc : 0.99;
    }

    /**
     * Compute the triaxiality function R_ν.
     *
     *   R_ν = 2/3·(1+ν) + 3·(1-2ν)·η²
     *
     * @param {number} triaxiality - Stress triaxiality η = σ_h / σ_vm
     * @returns {number}
     */
    triaxialityFunction(triaxiality) {
        const nu = this.nu;
        return (2.0 / 3.0) * (1.0 + nu) + 3.0 * (1.0 - 2.0 * nu) * triaxiality * triaxiality;
    }

    /**
     * Compute the damage energy release rate Y.
     *
     *   Y = σ_eq² · R_ν / (2·E·(1-D)²)
     *
     * @param {number} sigmaEq - Von Mises equivalent stress
     * @param {number} triaxiality - Stress triaxiality η
     * @param {number} D - Current damage variable
     * @returns {number}
     */
    energyReleaseRate(sigmaEq, triaxiality, D) {
        const Rv = this.triaxialityFunction(triaxiality);
        const omd = Math.max(EPSILON, 1.0 - D);
        return (sigmaEq * sigmaEq * Rv) / (2.0 * this.E * omd * omd);
    }

    /**
     * Compute the damage rate Ḋ.
     *
     *   Ḋ = (Y/S)^s · ṗ    for p ≥ p_D, D < D_c
     *
     * @param {number[]} sigma - Voigt stress [σ11,σ22,σ33,σ12,σ23,σ13]
     * @param {number} D - Current damage
     * @param {number} eqPlasticStrain - Accumulated equivalent plastic strain p
     * @param {number} eqPlasticStrainRate - Equivalent plastic strain rate ṗ
     * @returns {number} Ḋ – damage rate
     */
    computeDamageRate(sigma, D, eqPlasticStrain, eqPlasticStrainRate) {
        // Below threshold → no damage evolution
        if (eqPlasticStrain < this.pD || D >= this.Dc) {
            return 0.0;
        }

        const sigmaEq = vonMises(sigma);
        const hydrostat = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
        const eta = sigmaEq > EPSILON ? hydrostat / sigmaEq : 0.0;

        const Y = this.energyReleaseRate(sigmaEq, eta, D);
        const ratio = Y / this.S;

        return Math.pow(Math.max(0.0, ratio), this.s) * eqPlasticStrainRate;
    }

    /**
     * Update damage for a load increment.
     *
     * @param {object} state - { D: number, epsPl: number }
     * @param {number[]} sigma - Current Voigt stress
     * @param {number} dEpsilonP - Incremental equivalent plastic strain
     * @param {number} [dt=1.0] - Time step (for rate computation)
     * @returns {{ D: number, epsPl: number, failed: boolean }}
     */
    updateDamage(state, sigma, dEpsilonP, dt) {
        const st = state || { D: 0.0, epsPl: 0.0 };
        const timeStep = dt || 1.0;
        const strainRate = dEpsilonP / timeStep;

        const newEpsPl = st.epsPl + dEpsilonP;
        const dDot = this.computeDamageRate(sigma, st.D, newEpsPl, strainRate);
        const newD = Math.min(this.Dc, st.D + dDot * timeStep);

        return {
            D: newD,
            epsPl: newEpsPl,
            failed: newD >= this.Dc
        };
    }
}

// ---------------------------------------------------------------------------
// Failure Criteria
// ---------------------------------------------------------------------------

/**
 * Classical failure criteria for identifying fracture initiation.
 *
 * All methods are static — no instance state is needed.
 */
export class FailureCriteria {
    /**
     * Maximum principal stress criterion.
     *
     * Failure when the largest principal stress exceeds the threshold
     * (tensile strength).
     *
     * @param {number[]} stress - Voigt stress [σ11,σ22,σ33,σ12,σ23,σ13]
     * @param {number} threshold - Tensile strength σ_t
     * @returns {{ failed: boolean, sigma1: number, ratio: number }}
     */
    static maxPrincipalStress(stress, threshold) {
        const p = principalStresses(stress);
        const sigma1 = p[0];
        return {
            failed: sigma1 >= threshold,
            sigma1,
            ratio: sigma1 / (threshold + EPSILON)
        };
    }

    /**
     * Maximum principal strain criterion.
     *
     * Failure when the largest principal strain exceeds the threshold.
     *
     * @param {number[]} strain - Voigt strain [ε11,ε22,ε33,2ε12,2ε23,2ε13]
     * @param {number} threshold - Critical strain ε_c
     * @returns {{ failed: boolean, eps1: number, ratio: number }}
     */
    static maxPrincipalStrain(strain, threshold) {
        const p = principalStrains(strain);
        const eps1 = p[0];
        return {
            failed: eps1 >= threshold,
            eps1,
            ratio: eps1 / (threshold + EPSILON)
        };
    }

    /**
     * Mohr-Coulomb failure criterion for pressure-sensitive materials.
     *
     * Failure surface in terms of principal stresses:
     *   σ1 - σ3 + (σ1 + σ3)·sin(φ) ≥ 2c·cos(φ)
     *
     * @param {number[]} sigma - Voigt stress
     * @param {number} cohesion - Material cohesion c
     * @param {number} frictionAngle - Internal friction angle φ (radians)
     * @returns {{ failed: boolean, tau: number, strength: number, ratio: number }}
     */
    static mohrCoulomb(sigma, cohesion, frictionAngle) {
        const p = principalStresses(sigma);
        const sigma1 = p[0];
        const sigma3 = p[2];
        const sinPhi = Math.sin(frictionAngle);
        const cosPhi = Math.cos(frictionAngle);

        const tau = (sigma1 - sigma3) + (sigma1 + sigma3) * sinPhi;
        const strength = 2.0 * cohesion * cosPhi;

        return {
            failed: tau >= strength,
            tau,
            strength,
            ratio: tau / (strength + EPSILON)
        };
    }

    /**
     * Energy release rate (Griffith) criterion.
     *
     * Computes the total energy release rate G from mode I and mode II
     * stress intensity factors and checks against critical Gc:
     *
     *   G = (K_I² + K_II²) / E'
     *   E' = E (plane stress) or E/(1-ν²) (plane strain / 3D)
     *
     * @param {number} K_I - Mode I stress intensity factor
     * @param {number} K_II - Mode II stress intensity factor
     * @param {number} E - Young's modulus
     * @param {number} nu - Poisson's ratio
     * @param {number} [Gc] - Critical energy release rate (if provided, checks failure)
     * @param {boolean} [planeStress=false] - Use plane stress E' = E
     * @returns {{ G: number, failed: boolean }}
     */
    static energyReleaseRate(K_I, K_II, E, nu, Gc, planeStress) {
        // Effective modulus: plane strain → E' = E/(1-ν²), plane stress → E' = E
        const Eprime = planeStress ? E : E / (1.0 - nu * nu);
        const G = (K_I * K_I + K_II * K_II) / Eprime;

        return {
            G,
            failed: Gc != null ? G >= Gc : false
        };
    }
}
