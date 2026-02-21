/**
 * @fileoverview Comprehensive material law library for 3D FEA.
 *
 * All tensors use flat row-major arrays for performance.
 *   - Voigt stress/strain order: [11, 22, 33, 12, 23, 13]
 *   - Deformation gradient F: [F11,F12,F13, F21,F22,F23, F31,F32,F33]
 *   - Constitutive tangent C: 6×6 stored as flat array[36] (row-major)
 *
 * @module material-models
 */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** @param {number[]} A 3×3 flat → determinant */
function det3(A) {
    return (
        A[0] * (A[4] * A[8] - A[5] * A[7]) -
        A[1] * (A[3] * A[8] - A[5] * A[6]) +
        A[2] * (A[3] * A[7] - A[4] * A[6])
    );
}

/** @param {number[]} A 3×3 flat → inverse (new array) */
function inv3(A) {
    const d = det3(A);
    if (Math.abs(d) < 1e-30) throw new Error('Singular matrix in inv3');
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
        (A[0] * A[4] - A[1] * A[3]) * invD,
    ];
}

/** 3×3 matrix multiply (flat) C = A·B */
function mul33(A, B) {
    const C = new Array(9);
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            C[i * 3 + j] =
                A[i * 3] * B[j] +
                A[i * 3 + 1] * B[3 + j] +
                A[i * 3 + 2] * B[6 + j];
        }
    }
    return C;
}

/** Transpose 3×3 flat */
function transpose3(A) {
    return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
}

/** Trace of 3×3 flat */
function trace3(A) {
    return A[0] + A[4] + A[8];
}

/** Identity 3×3 flat */
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// Voigt scaling factors: normal components use 3/2, shear components use 3
// (arising from the tensor ↔ Voigt contraction with engineering shear strain)
const VOIGT_NORMAL = 1.5;
const VOIGT_SHEAR = 3.0;

/** 6×6 matrix-vector product (both flat) */
function mv6(M, v) {
    const r = new Array(6);
    for (let i = 0; i < 6; i++) {
        r[i] = 0;
        for (let j = 0; j < 6; j++) r[i] += M[i * 6 + j] * v[j];
    }
    return r;
}

/** Symmetric 3×3 tensor → Voigt vector [11,22,33,12,23,13] */
function symToVoigt(S) {
    return [S[0], S[4], S[8], S[1], S[5], S[2]];
}

// ---------------------------------------------------------------------------
// MaterialState
// ---------------------------------------------------------------------------

/**
 * Tracks per-integration-point state for history-dependent materials.
 */
export class MaterialState {
    constructor() {
        /** Equivalent plastic strain */
        this.epsPl = 0.0;
        /** Back-stress tensor (Voigt[6]) for kinematic hardening */
        this.backStress = [0, 0, 0, 0, 0, 0];
        /** Elastic left Cauchy-Green tensor (flat 3×3) for hyperelastic-plastic */
        this.be = [...I3];
        /** Scalar damage variable d ∈ [0,1] */
        this.damage = 0.0;
    }

    /** Deep-copy the state */
    clone() {
        const s = new MaterialState();
        s.epsPl = this.epsPl;
        s.backStress = [...this.backStress];
        s.be = [...this.be];
        s.damage = this.damage;
        return s;
    }
}

// ---------------------------------------------------------------------------
// LinearElastic
// ---------------------------------------------------------------------------

/**
 * Standard 3D isotropic linear elasticity.
 */
export class LinearElastic {
    /**
     * @param {object} params
     * @param {number} params.E  - Young's modulus
     * @param {number} params.nu - Poisson's ratio
     */
    constructor({ E = 1.0, nu = 0.3 } = {}) {
        this.E = E;
        this.nu = nu;
    }

    /**
     * Build the 6×6 constitutive matrix (flat array[36]).
     * @returns {number[]}
     */
    getConstitutiveMatrix() {
        const { E, nu } = this;
        const f = E / ((1 + nu) * (1 - 2 * nu));
        const d = f * (1 - nu);
        const o = f * nu;
        const g = f * (1 - 2 * nu) / 2;
        // prettier-ignore
        return [
            d, o, o, 0, 0, 0,
            o, d, o, 0, 0, 0,
            o, o, d, 0, 0, 0,
            0, 0, 0, g, 0, 0,
            0, 0, 0, 0, g, 0,
            0, 0, 0, 0, 0, g,
        ];
    }

    /**
     * Compute stress from deformation gradient (small-strain approximation).
     * @param {number[]} F  - Deformation gradient (flat 3×3)
     * @param {MaterialState} state
     * @returns {{ stress: number[], tangent: number[], state: MaterialState }}
     */
    computeStress(F, state) {
        const C = this.getConstitutiveMatrix();
        // Small strain: ε = 0.5*(F + F^T) - I
        const eps = [
            F[0] - 1,
            F[4] - 1,
            F[8] - 1,
            F[1] + F[3],       // γ12 = engineering shear strain
            F[5] + F[7],       // γ23
            F[2] + F[6],       // γ13
        ];
        const stress = mv6(C, eps);
        return { stress, tangent: C, state: state ? state.clone() : new MaterialState() };
    }
}

// ---------------------------------------------------------------------------
// NeoHookean
// ---------------------------------------------------------------------------

/**
 * Compressible Neo-Hookean hyperelastic material.
 *
 * W = μ/2 (I₁ - 3) - μ ln(J) + λ/2 (ln J)²
 */
export class NeoHookean {
    /**
     * @param {object} params
     * @param {number} params.E  - Young's modulus
     * @param {number} params.nu - Poisson's ratio
     */
    constructor({ E = 1.0, nu = 0.3 } = {}) {
        this.E = E;
        this.nu = nu;
        this.mu = E / (2 * (1 + nu));
        this.lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
    }

    /** Linear tangent (small-strain limit). */
    getConstitutiveMatrix() {
        return new LinearElastic({ E: this.E, nu: this.nu }).getConstitutiveMatrix();
    }

    /**
     * Cauchy stress from deformation gradient.
     * σ = (1/J)[μ(b - I) + λ ln(J) I]
     * @param {number[]} F  - Deformation gradient (flat 3×3)
     * @param {MaterialState} state
     * @returns {{ stress: number[], tangent: number[], state: MaterialState }}
     */
    computeStress(F, state) {
        const { mu, lambda } = this;
        const J = det3(F);
        const lnJ = Math.log(J);
        const Ft = transpose3(F);
        const b = mul33(F, Ft); // left Cauchy-Green b = F·Fᵀ

        // Cauchy stress (symmetric 3×3 → Voigt)
        const invJ = 1.0 / J;
        const sig = new Array(9);
        for (let i = 0; i < 9; i++) {
            sig[i] = invJ * (mu * (b[i] - I3[i]) + lambda * lnJ * I3[i]);
        }
        const stress = symToVoigt(sig);

        // Spatial tangent coefficients (already in current configuration)
        const c1 = (lambda * lnJ - mu) * invJ;
        const c2 = (lambda + mu) * invJ; // effective (λ+μ)/J for spatial tangent
        // Assemble tangent in current configuration
        const tangent = this._spatialTangent(c1, c2, mu * invJ);

        return { stress, tangent, state: state ? state.clone() : new MaterialState() };
    }

    /**
     * Build spatial tangent for Neo-Hookean.
     * c_ijkl = λ/J δ_ij δ_kl + 2(μ - λ ln J)/J  I_sym_ijkl
     * Coefficients c1, c2, muJ already incorporate the 1/J scaling.
     * Stored in Voigt 6×6.
     */
    _spatialTangent(c1, c2, muJ) {
        const lam = c2 - 2 * muJ; // effective λ/J
        const g = muJ;             // effective μ/J
        const d = lam + 2 * g;
        // prettier-ignore
        return [
            d,   lam, lam, 0, 0, 0,
            lam, d,   lam, 0, 0, 0,
            lam, lam, d,   0, 0, 0,
            0,   0,   0,   g, 0, 0,
            0,   0,   0,   0, g, 0,
            0,   0,   0,   0, 0, g,
        ];
    }
}

// ---------------------------------------------------------------------------
// J2Plasticity (von Mises)
// ---------------------------------------------------------------------------

/**
 * J2 (von Mises) elastoplastic model with combined isotropic + kinematic
 * hardening. Uses radial-return mapping in Voigt space.
 */
export class J2Plasticity {
    /**
     * @param {object} params
     * @param {number} params.E     - Young's modulus
     * @param {number} params.nu    - Poisson's ratio
     * @param {number} params.sigY  - Initial yield stress
     * @param {number} params.H     - Isotropic hardening modulus
     * @param {number} params.Hk    - Kinematic hardening modulus
     */
    constructor({ E = 210000, nu = 0.3, sigY = 250, H = 1000, Hk = 0 } = {}) {
        this.E = E;
        this.nu = nu;
        this.sigY = sigY;
        this.H = H;
        this.Hk = Hk;
        this.K = E / (3 * (1 - 2 * nu));
        this.G = E / (2 * (1 + nu));
    }

    /** Elastic constitutive matrix. */
    getConstitutiveMatrix() {
        return new LinearElastic({ E: this.E, nu: this.nu }).getConstitutiveMatrix();
    }

    /**
     * Radial-return stress update.
     * @param {number[]} F     - Deformation gradient (flat 3×3)
     * @param {MaterialState} state - Previous converged state
     * @returns {{ stress: number[], tangent: number[], state: MaterialState }}
     */
    computeStress(F, state) {
        const st = state ? state.clone() : new MaterialState();
        const { G, K, sigY, H, Hk } = this;

        // Small-strain assumption: ε = sym(F - I)
        const eps = [
            F[0] - 1,
            F[4] - 1,
            F[8] - 1,
            (F[1] + F[3]),
            (F[5] + F[7]),
            (F[2] + F[6]),
        ];

        // Elastic trial stress
        const Ce = this.getConstitutiveMatrix();
        const trialStress = mv6(Ce, eps);

        // Deviatoric trial stress (relative to back-stress for kinematic hardening)
        const p = (trialStress[0] + trialStress[1] + trialStress[2]) / 3;
        const sDev = [
            trialStress[0] - p - st.backStress[0],
            trialStress[1] - p - st.backStress[1],
            trialStress[2] - p - st.backStress[2],
            trialStress[3] - st.backStress[3],
            trialStress[4] - st.backStress[4],
            trialStress[5] - st.backStress[5],
        ];

        // Von Mises equivalent stress (with factor 2 on shear for tensor components)
        const sEq = Math.sqrt(
            1.5 * (sDev[0] * sDev[0] + sDev[1] * sDev[1] + sDev[2] * sDev[2]) +
            VOIGT_SHEAR * (sDev[3] * sDev[3] + sDev[4] * sDev[4] + sDev[5] * sDev[5])
        );

        // Current yield stress
        const sigYc = sigY + H * st.epsPl;
        const f = sEq - sigYc;

        let stress, tangent;

        if (f <= 0) {
            // Elastic step
            stress = [...trialStress];
            tangent = Ce;
        } else {
            // Plastic correction via radial return
            const dGamma = f / (3 * G + H + Hk);
            const ratio = 1 - (3 * G * dGamma) / sEq;

            // Updated deviatoric stress
            const sCorr = [
                sDev[0] * ratio, sDev[1] * ratio, sDev[2] * ratio,
                sDev[3] * ratio, sDev[4] * ratio, sDev[5] * ratio,
            ];

            stress = [
                sCorr[0] + p + st.backStress[0],
                sCorr[1] + p + st.backStress[1],
                sCorr[2] + p + st.backStress[2],
                sCorr[3] + st.backStress[3],
                sCorr[4] + st.backStress[4],
                sCorr[5] + st.backStress[5],
            ];

            // Update state
            st.epsPl += dGamma;
            // Update back-stress (kinematic hardening)
            const hkFac = Hk * dGamma / sEq;
            for (let i = 0; i < 6; i++) {
                st.backStress[i] += hkFac * sDev[i] * (i < 3 ? VOIGT_NORMAL : VOIGT_SHEAR);
            }

            // Consistent (algorithmic) tangent
            tangent = this._consistentTangent(G, K, dGamma, sEq, sDev, ratio);
        }

        return { stress, tangent, state: st };
    }

    /**
     * Build the consistent elastoplastic tangent (6×6 flat).
     * @private
     */
    _consistentTangent(G, K, dGamma, sEq, n, ratio) {
        const { H, Hk } = this;
        const theta = 1 - (3 * G * dGamma) / sEq;
        const thetaBar = 1 / (1 + (H + Hk) / (3 * G)) - (1 - theta);

        // Normal to yield surface (unit deviatoric direction)
        const nHat = new Array(6);
        const norm = sEq; // already equals sqrt(3/2 * s:s)
        for (let i = 0; i < 6; i++) {
            nHat[i] = (i < 3 ? VOIGT_NORMAL : VOIGT_SHEAR) * n[i] / norm;
        }

        const C = new Array(36);
        // Build: C = K (I_vol) + 2G*theta (I_dev) - 2G*thetaBar (nHat⊗nHat)
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                const idx = i * 6 + j;
                // Volumetric part
                const Ivol = (i < 3 && j < 3) ? 1.0 / 3.0 : 0;
                // Deviatoric identity (Isym - 1/3 I⊗I)
                let Idev = -1.0 / 3.0 * ((i < 3 && j < 3) ? 1 : 0);
                if (i === j) Idev += (i < 3 ? 1.0 : 0.5);

                C[idx] = 3 * K * Ivol + 2 * G * theta * Idev - 2 * G * thetaBar * nHat[i] * nHat[j];
            }
        }
        return C;
    }
}

// ---------------------------------------------------------------------------
// DruckerPrager
// ---------------------------------------------------------------------------

/**
 * Drucker-Prager elastoplastic model for pressure-dependent materials.
 *
 * Yield surface: f = √J₂ + α·I₁ - k = 0
 * where α, k are derived from cohesion c and friction angle φ.
 */
export class DruckerPrager {
    /**
     * @param {object} params
     * @param {number} params.E       - Young's modulus
     * @param {number} params.nu      - Poisson's ratio
     * @param {number} params.c       - Cohesion
     * @param {number} params.phi     - Friction angle (radians)
     * @param {number} params.psi     - Dilation angle (radians)
     * @param {number} params.H       - Hardening modulus
     */
    constructor({ E = 30000, nu = 0.2, c = 100, phi = Math.PI / 6, psi = 0, H = 0 } = {}) {
        this.E = E;
        this.nu = nu;
        this.c = c;
        this.phi = phi;
        this.psi = psi;
        this.H = H;
        this.G = E / (2 * (1 + nu));
        this.K = E / (3 * (1 - 2 * nu));
        // Drucker-Prager parameters matched to Mohr-Coulomb (outer cone)
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);
        this.alpha = 2 * sinPhi / (Math.sqrt(3) * (3 - sinPhi));
        this.kDP = 6 * c * cosPhi / (Math.sqrt(3) * (3 - sinPhi));
        // Flow rule parameter
        const sinPsi = Math.sin(psi);
        this.alphaQ = 2 * sinPsi / (Math.sqrt(3) * (3 - sinPsi));
    }

    getConstitutiveMatrix() {
        return new LinearElastic({ E: this.E, nu: this.nu }).getConstitutiveMatrix();
    }

    /**
     * Stress update with return mapping to the Drucker-Prager cone.
     * @param {number[]} F     - Deformation gradient (flat 3×3)
     * @param {MaterialState} state
     * @returns {{ stress: number[], tangent: number[], state: MaterialState }}
     */
    computeStress(F, state) {
        const st = state ? state.clone() : new MaterialState();
        const { G, K, alpha, kDP, alphaQ, H } = this;

        // Small strain
        const eps = [
            F[0] - 1, F[4] - 1, F[8] - 1,
            (F[1] + F[3]), (F[5] + F[7]), (F[2] + F[6]),
        ];

        const Ce = this.getConstitutiveMatrix();
        const trialStress = mv6(Ce, eps);

        // Hydrostatic pressure and deviatoric stress
        const I1 = trialStress[0] + trialStress[1] + trialStress[2];
        const p = I1 / 3;
        const s = [
            trialStress[0] - p, trialStress[1] - p, trialStress[2] - p,
            trialStress[3], trialStress[4], trialStress[5],
        ];

        // J2 = 0.5 * s:s  (tensor contraction, shears counted twice in Voigt)
        const J2 = 0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) +
                   s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
        const sqrtJ2 = Math.sqrt(Math.max(J2, 1e-30));

        // Yield function
        const kc = kDP + H * st.epsPl;
        const f = sqrtJ2 + alpha * I1 - kc;

        let stress, tangent;

        if (f <= 0) {
            stress = [...trialStress];
            tangent = Ce;
        } else {
            // Plastic multiplier
            const dGamma = f / (G + 9 * K * alpha * alphaQ + H);

            // Return to yield surface
            const ratio = 1 - G * dGamma / sqrtJ2;
            const dpVol = 3 * K * alphaQ * dGamma;

            stress = [
                s[0] * ratio + (p - dpVol), s[1] * ratio + (p - dpVol), s[2] * ratio + (p - dpVol),
                s[3] * ratio, s[4] * ratio, s[5] * ratio,
            ];

            st.epsPl += dGamma;

            // Algorithmic tangent
            tangent = this._algorithmicTangent(G, K, alpha, alphaQ, dGamma, sqrtJ2, s, ratio);
        }

        return { stress, tangent, state: st };
    }

    /** @private */
    _algorithmicTangent(G, K, alpha, alphaQ, dGamma, sqrtJ2, s, ratio) {
        const { H } = this;
        const denom = G + 9 * K * alpha * alphaQ + H;

        // Normal to yield surface (deviatoric direction)
        const n = new Array(6);
        for (let i = 0; i < 6; i++) {
            n[i] = (0.5 / sqrtJ2) * s[i] * (i < 3 ? 1 : 2);
        }

        const C = new Array(36);
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                const idx = i * 6 + j;
                const Ivol = (i < 3 && j < 3) ? 1.0 / 3.0 : 0;
                let Idev = -1.0 / 3.0 * ((i < 3 && j < 3) ? 1 : 0);
                if (i === j) Idev += (i < 3 ? 1.0 : 0.5);

                // Volumetric + deviatoric (scaled) + plastic correction terms
                const volPart = (9 * K * K * alpha * alphaQ / denom);
                const devPart = 2 * G * ratio;
                const nnPart = (G * G / (sqrtJ2 * denom));
                const crossPart = (3 * K * G / denom);

                C[idx] = (3 * K - volPart) * Ivol
                       + devPart * Idev
                       - 4 * nnPart * n[i] * n[j]
                       - crossPart * (alpha * ((i < 3) ? 1 : 0) * n[j] + alphaQ * n[i] * ((j < 3) ? 1 : 0));
            }
        }
        return C;
    }
}

// ---------------------------------------------------------------------------
// MooneyRivlin
// ---------------------------------------------------------------------------

/**
 * Mooney-Rivlin hyperelastic model.
 *
 * W = C10 (I₁ - 3) + C01 (I₂ - 3) + (1/D1)(J - 1)²
 */
export class MooneyRivlin {
    /**
     * @param {object} params
     * @param {number} params.C10 - First Mooney-Rivlin constant
     * @param {number} params.C01 - Second Mooney-Rivlin constant
     * @param {number} params.D1  - Volumetric penalty (≈ 2/K)
     */
    constructor({ C10 = 0.5, C01 = 0.1, D1 = 0.001 } = {}) {
        this.C10 = C10;
        this.C01 = C01;
        this.D1 = D1;
    }

    /** Small-strain tangent approximation via equivalent E, ν. */
    getConstitutiveMatrix() {
        const mu = 2 * (this.C10 + this.C01);
        const K = 2 / this.D1;
        const E = 9 * K * mu / (3 * K + mu);
        const nu = (3 * K - 2 * mu) / (2 * (3 * K + mu));
        return new LinearElastic({ E, nu }).getConstitutiveMatrix();
    }

    /**
     * Cauchy stress from deformation gradient.
     * @param {number[]} F
     * @param {MaterialState} state
     * @returns {{ stress: number[], tangent: number[], state: MaterialState }}
     */
    computeStress(F, state) {
        const { C10, C01, D1 } = this;
        const J = det3(F);
        const Jm23 = Math.pow(J, -2.0 / 3.0);
        const Ft = transpose3(F);
        const b = mul33(F, Ft); // left Cauchy-Green

        // Invariants of b
        const I1 = trace3(b);
        const bSq = mul33(b, b);
        const I2 = 0.5 * (I1 * I1 - trace3(bSq));

        // Deviatoric Cauchy stress
        // σ = (2/J) [ (C10 + C01*I1) b_dev - C01 b²_dev ] + (2/D1)(J-1) I
        const invJ = 1.0 / J;
        const pHydro = (2.0 / D1) * (J - 1);

        const sig = new Array(9);
        for (let i = 0; i < 9; i++) {
            const bBar = Jm23 * b[i];
            const bSqBar = Jm23 * Jm23 * bSq[i]; // = J^{-4/3} b²
            const bDev = bBar - (Jm23 * I1 / 3) * I3[i];
            const bSqDev = bSqBar - (Jm23 * Jm23 * trace3(bSq) / 3) * I3[i];
            sig[i] = 2 * invJ * ((C10 + C01 * Jm23 * I1) * bDev - C01 * bSqDev) + pHydro * I3[i];
        }

        const stress = symToVoigt(sig);
        const tangent = this.getConstitutiveMatrix(); // linearised tangent

        return { stress, tangent, state: state ? state.clone() : new MaterialState() };
    }
}

// ---------------------------------------------------------------------------
// Ogden (single-term)
// ---------------------------------------------------------------------------

/**
 * Single-term Ogden hyperelastic model.
 *
 * W = (μ/α)(λ₁^α + λ₂^α + λ₃^α - 3) + (1/D)(J - 1)²
 *
 * Reduces to Neo-Hookean when α = 2.
 */
export class Ogden {
    /**
     * @param {object} params
     * @param {number} params.mu    - Shear modulus parameter
     * @param {number} params.alpha - Ogden exponent
     * @param {number} params.D     - Volumetric penalty (≈ 2/K)
     */
    constructor({ mu = 1.0, alpha = 2.0, D = 0.001 } = {}) {
        this.mu = mu;
        this.alpha = alpha;
        this.D = D;
    }

    /** Small-strain tangent approximation. */
    getConstitutiveMatrix() {
        const muEff = this.mu;
        const K = 2 / this.D;
        const E = 9 * K * muEff / (3 * K + muEff);
        const nu = (3 * K - 2 * muEff) / (2 * (3 * K + muEff));
        return new LinearElastic({ E, nu }).getConstitutiveMatrix();
    }

    /**
     * Cauchy stress from principal stretches via eigenvalue decomposition of b.
     * @param {number[]} F
     * @param {MaterialState} state
     * @returns {{ stress: number[], tangent: number[], state: MaterialState }}
     */
    computeStress(F, state) {
        const { mu, alpha, D } = this;
        const J = det3(F);
        const Ft = transpose3(F);
        const b = mul33(F, Ft);

        // Eigenvalues of b (principal stretches squared) via characteristic polynomial
        const lambdaSq = this._eigenvalues3Sym(b);
        const lambdas = lambdaSq.map((l) => Math.sqrt(Math.max(l, 1e-30)));

        // Isochoric principal stretches
        const Jm13 = Math.pow(J, -1.0 / 3.0);
        const lamBar = lambdas.map((l) => l * Jm13);

        // Principal Cauchy stresses (deviatoric part)
        const sigPrinc = new Array(3);
        const pDev =
            (mu / alpha) *
            (Math.pow(lamBar[0], alpha) + Math.pow(lamBar[1], alpha) + Math.pow(lamBar[2], alpha)) /
            3;
        for (let i = 0; i < 3; i++) {
            sigPrinc[i] = (mu / J) * Math.pow(lamBar[i], alpha) - pDev / J;
        }

        // Volumetric
        const pHydro = (2.0 / D) * (J - 1);

        // Eigenvectors of b (for rotation back to spatial frame)
        const vecs = this._eigenvectors3Sym(b, lambdaSq);

        // Reconstruct Cauchy stress in spatial frame: σ = Σ σ_i n_i⊗n_i
        const sig = new Array(9).fill(0);
        for (let k = 0; k < 3; k++) {
            const sVal = sigPrinc[k] + pHydro;
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    sig[i * 3 + j] += sVal * vecs[k][i] * vecs[k][j];
                }
            }
        }

        const stress = symToVoigt(sig);
        const tangent = this.getConstitutiveMatrix();

        return { stress, tangent, state: state ? state.clone() : new MaterialState() };
    }

    /**
     * Eigenvalues of symmetric 3×3 matrix (Cardano's method).
     * @private
     * @param {number[]} A - Symmetric 3×3 flat
     * @returns {number[]} Three eigenvalues sorted descending
     */
    _eigenvalues3Sym(A) {
        const a11 = A[0], a22 = A[4], a33 = A[8];
        const a12 = A[1], a13 = A[2], a23 = A[5];

        const p1 = a12 * a12 + a13 * a13 + a23 * a23;
        if (p1 < 1e-30) {
            // Already diagonal
            const eigs = [a11, a22, a33];
            eigs.sort((a, b) => b - a);
            return eigs;
        }

        const q = (a11 + a22 + a33) / 3;
        const p2 =
            (a11 - q) * (a11 - q) +
            (a22 - q) * (a22 - q) +
            (a33 - q) * (a33 - q) +
            2 * p1;
        const p = Math.sqrt(p2 / 6);

        // B = (1/p)(A - q*I)
        const B = [
            (a11 - q) / p, a12 / p, a13 / p,
            a12 / p, (a22 - q) / p, a23 / p,
            a13 / p, a23 / p, (a33 - q) / p,
        ];
        const detB = det3(B);
        let phi = Math.acos(Math.min(1, Math.max(-1, detB / 2))) / 3;

        const eig1 = q + 2 * p * Math.cos(phi);
        const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
        const eig2 = 3 * q - eig1 - eig3;

        return [eig1, eig2, eig3];
    }

    /**
     * Eigenvectors of symmetric 3×3 via shifted inverse iteration.
     * @private
     * @param {number[]} A - Symmetric 3×3 flat
     * @param {number[]} eigs - Eigenvalues
     * @returns {number[][]} Three unit eigenvectors
     */
    _eigenvectors3Sym(A, eigs) {
        const vecs = [];
        for (let k = 0; k < 3; k++) {
            // (A - λI) v ≈ 0 → pick column of cofactor matrix
            const M = [...A];
            M[0] -= eigs[k];
            M[4] -= eigs[k];
            M[8] -= eigs[k];

            // Cofactor rows → any non-zero row is the eigenvector
            const r0 = [
                M[4] * M[8] - M[5] * M[7],
                M[5] * M[6] - M[3] * M[8],
                M[3] * M[7] - M[4] * M[6],
            ];
            const r1 = [
                M[2] * M[7] - M[1] * M[8],
                M[0] * M[8] - M[2] * M[6],
                M[1] * M[6] - M[0] * M[7],
            ];

            let v = r0;
            let len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (len < 1e-12) {
                v = r1;
                len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            }
            if (len < 1e-12) {
                // Fallback to coordinate axis
                v = [0, 0, 0];
                v[k] = 1;
                len = 1;
            }
            vecs.push([v[0] / len, v[1] / len, v[2] / len]);
        }
        return vecs;
    }
}

// ---------------------------------------------------------------------------
// Registry & Factory
// ---------------------------------------------------------------------------

/**
 * Maps material type names to their classes.
 * @type {Object<string, Function>}
 */
export const MaterialRegistry = {
    'linear-elastic': LinearElastic,
    'neo-hookean': NeoHookean,
    'j2-plasticity': J2Plasticity,
    'drucker-prager': DruckerPrager,
    'mooney-rivlin': MooneyRivlin,
    'ogden': Ogden,
};

/**
 * Factory function to create a material model by name.
 *
 * @param {string} type   - One of the keys in {@link MaterialRegistry}
 * @param {object} params - Constructor parameters forwarded to the material class
 * @returns {LinearElastic|NeoHookean|J2Plasticity|DruckerPrager|MooneyRivlin|Ogden}
 * @throws {Error} If the material type is not registered
 *
 * @example
 * const steel = createMaterial('j2-plasticity', { E: 210000, nu: 0.3, sigY: 250, H: 1000 });
 * const rubber = createMaterial('neo-hookean', { E: 1.0, nu: 0.45 });
 */
export function createMaterial(type, params = {}) {
    const Ctor = MaterialRegistry[type];
    if (!Ctor) {
        throw new Error(
            `Unknown material type "${type}". Available: ${Object.keys(MaterialRegistry).join(', ')}`
        );
    }
    return new Ctor(params);
}
