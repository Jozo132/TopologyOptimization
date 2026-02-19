// High-performance matrix math operations for topology optimization
// Compiled to WebAssembly for maximum efficiency

/**
 * Matrix-vector multiplication: result = K * x
 * @param K - stiffness matrix (sparse, stored as array)
 * @param x - input vector
 * @param result - output vector
 * @param n - dimension
 */
export function matVecMul(
  K: Float64Array,
  x: Float64Array,
  result: Float64Array,
  n: i32
): void {
  for (let i = 0; i < n; i++) {
    let sum: f64 = 0.0;
    for (let j = 0; j < n; j++) {
      sum += unchecked(K[i * n + j]) * unchecked(x[j]);
    }
    unchecked((result[i] = sum));
  }
}

/**
 * Conjugate Gradient solver for Ku = F
 * Solves sparse linear system iteratively
 */
export function conjugateGradient(
  K: Float64Array,
  F: Float64Array,
  U: Float64Array,
  n: i32,
  maxIter: i32,
  tolerance: f64
): i32 {
  // Allocate working arrays
  const r = new Float64Array(n);
  const p = new Float64Array(n);
  const Ap = new Float64Array(n);

  // r = F - K*U (initial residual)
  matVecMul(K, U, Ap, n);
  for (let i = 0; i < n; i++) {
    unchecked((r[i] = unchecked(F[i]) - unchecked(Ap[i])));
    unchecked((p[i] = unchecked(r[i])));
  }

  let rsold: f64 = 0.0;
  for (let i = 0; i < n; i++) {
    rsold += unchecked(r[i]) * unchecked(r[i]);
  }

  for (let iter = 0; iter < maxIter; iter++) {
    if (rsold < tolerance) {
      return iter;
    }

    // Ap = K * p
    matVecMul(K, p, Ap, n);

    // alpha = rsold / (p' * Ap)
    let pAp: f64 = 0.0;
    for (let i = 0; i < n; i++) {
      pAp += unchecked(p[i]) * unchecked(Ap[i]);
    }
    const alpha: f64 = rsold / pAp;

    // U = U + alpha * p
    // r = r - alpha * Ap
    for (let i = 0; i < n; i++) {
      unchecked((U[i] = unchecked(U[i]) + alpha * unchecked(p[i])));
      unchecked((r[i] = unchecked(r[i]) - alpha * unchecked(Ap[i])));
    }

    // rsnew = r' * r
    let rsnew: f64 = 0.0;
    for (let i = 0; i < n; i++) {
      rsnew += unchecked(r[i]) * unchecked(r[i]);
    }

    if (rsnew < tolerance) {
      return iter + 1;
    }

    // p = r + (rsnew/rsold) * p
    const beta: f64 = rsnew / rsold;
    for (let i = 0; i < n; i++) {
      unchecked((p[i] = unchecked(r[i]) + beta * unchecked(p[i])));
    }

    rsold = rsnew;
  }

  return maxIter;
}

/**
 * Dense matrix assembly for element stiffness
 * Assembles element stiffness matrices into global stiffness
 */
export function assembleElementStiffness(
  KE: Float64Array,
  globalK: Float64Array,
  edof: Int32Array,
  density: f64,
  E0: f64,
  Emin: f64,
  penal: f64,
  n: i32,
  edofSize: i32
): void {
  const stiffness: f64 = Emin + Math.pow(density, penal) * (E0 - Emin);

  for (let i = 0; i < edofSize; i++) {
    const gi: i32 = unchecked(edof[i]);
    if (gi < 0 || gi >= n) continue;

    for (let j = 0; j < edofSize; j++) {
      const gj: i32 = unchecked(edof[j]);
      if (gj < 0 || gj >= n) continue;

      const idx: i32 = gi * n + gj;
      unchecked(
        (globalK[idx] =
          unchecked(globalK[idx]) + stiffness * unchecked(KE[i * edofSize + j]))
      );
    }
  }
}

/**
 * 3D element stiffness matrix (8-node hexahedral element)
 * NOTE: This is a simplified placeholder for demonstration purposes.
 * The actual 3D FEA implementation in optimizer-worker-3d.js uses
 * proper 2×2×2 Gauss quadrature integration for accurate results.
 * This WASM function is provided for future integration but is not
 * currently used in production code.
 * 
 * For production use, this should be replaced with a full numerical
 * integration implementation matching the JavaScript version.
 */
export function get3DElementStiffness(
  KE: Float64Array,
  E: f64,
  nu: f64
): void {
  // Material stiffness matrix (D matrix for 3D)
  const k: f64 = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const G: f64 = E / (2.0 * (1.0 + nu));

  // Simplified single-point integration (for demonstration only)
  const a: f64 = k * (1.0 - nu);
  const b: f64 = k * nu;
  const c: f64 = G;

  // Zero out the matrix first
  for (let i = 0; i < 24 * 24; i++) {
    unchecked((KE[i] = 0.0));
  }

  // Diagonal blocks (simplified representative form)
  // Real implementation should use Gauss quadrature as in optimizer-worker-3d.js
  for (let node = 0; node < 8; node++) {
    const offset: i32 = node * 3;
    // Diagonal terms
    unchecked((KE[(offset + 0) * 24 + (offset + 0)] = a));
    unchecked((KE[(offset + 1) * 24 + (offset + 1)] = a));
    unchecked((KE[(offset + 2) * 24 + (offset + 2)] = a));
    // Coupling terms
    unchecked((KE[(offset + 0) * 24 + (offset + 1)] = b));
    unchecked((KE[(offset + 1) * 24 + (offset + 0)] = b));
    unchecked((KE[(offset + 0) * 24 + (offset + 2)] = b));
    unchecked((KE[(offset + 2) * 24 + (offset + 0)] = b));
    unchecked((KE[(offset + 1) * 24 + (offset + 2)] = b));
    unchecked((KE[(offset + 2) * 24 + (offset + 1)] = b));
  }
}

/**
 * Vector dot product
 */
export function dotProduct(a: Float64Array, b: Float64Array, n: i32): f64 {
  let sum: f64 = 0.0;
  for (let i = 0; i < n; i++) {
    sum += unchecked(a[i]) * unchecked(b[i]);
  }
  return sum;
}

/**
 * Vector scaling: result = a * x
 */
export function scaleVector(
  x: Float64Array,
  result: Float64Array,
  a: f64,
  n: i32
): void {
  for (let i = 0; i < n; i++) {
    unchecked((result[i] = a * unchecked(x[i])));
  }
}

/**
 * Vector addition: result = a + b
 */
export function addVectors(
  a: Float64Array,
  b: Float64Array,
  result: Float64Array,
  n: i32
): void {
  for (let i = 0; i < n; i++) {
    unchecked((result[i] = unchecked(a[i]) + unchecked(b[i])));
  }
}

/**
 * Apply density filter with radius
 * Smooths the density field based on neighboring elements
 */
export function applyDensityFilter(
  densities: Float64Array,
  filtered: Float64Array,
  nx: i32,
  ny: i32,
  nz: i32,
  radius: f64
): void {
  const rmin: i32 = i32(Math.ceil(radius));
  const radiusSq: f64 = radius * radius;

  for (let ez = 0; ez < nz; ez++) {
    for (let ey = 0; ey < ny; ey++) {
      for (let ex = 0; ex < nx; ex++) {
        let sum: f64 = 0.0;
        let sumWeight: f64 = 0.0;

        const minz: i32 = max(0, ez - rmin);
        const maxz: i32 = min(nz - 1, ez + rmin);
        const miny: i32 = max(0, ey - rmin);
        const maxy: i32 = min(ny - 1, ey + rmin);
        const minx: i32 = max(0, ex - rmin);
        const maxx: i32 = min(nx - 1, ex + rmin);

        for (let kz = minz; kz <= maxz; kz++) {
          for (let ky = miny; ky <= maxy; ky++) {
            for (let kx = minx; kx <= maxx; kx++) {
              const dx: f64 = f64(ex - kx);
              const dy: f64 = f64(ey - ky);
              const dz: f64 = f64(ez - kz);
              const distSq: f64 = dx * dx + dy * dy + dz * dz;

              // Early rejection using squared distance avoids sqrt for elements outside radius
              if (distSq <= radiusSq) {
                const dist: f64 = Math.sqrt(distSq);
                const weight: f64 = radius - dist;
                const idx: i32 = kx + ky * nx + kz * nx * ny;
                sum += weight * unchecked(densities[idx]);
                sumWeight += weight;
              }
            }
          }
        }

        const outIdx: i32 = ex + ey * nx + ez * nx * ny;
        unchecked((filtered[outIdx] = sum / sumWeight));
      }
    }
  }
}

/**
 * Compute element energies for sensitivity analysis
 */
export function computeElementEnergies(
  U: Float64Array,
  KE: Float64Array,
  energies: Float64Array,
  edofs: Int32Array,
  numElements: i32,
  edofSize: i32
): void {
  for (let el = 0; el < numElements; el++) {
    let energy: f64 = 0.0;
    const edofOffset: i32 = el * edofSize;

    for (let i = 0; i < edofSize; i++) {
      const dofI: i32 = unchecked(edofs[edofOffset + i]);
      if (dofI < 0) continue;
      const ui: f64 = unchecked(U[dofI]);

      for (let j = 0; j < edofSize; j++) {
        const dofJ: i32 = unchecked(edofs[edofOffset + j]);
        if (dofJ < 0) continue;
        const uj: f64 = unchecked(U[dofJ]);

        energy += ui * unchecked(KE[i * edofSize + j]) * uj;
      }
    }

    unchecked((energies[el] = energy));
  }
}

/**
 * Element-by-Element matrix-vector multiply: Ap = K(x) * p
 * Avoids assembling the global stiffness matrix.
 * Operates on full-space vectors (ndof-sized).
 * @param densities - element densities (nel)
 * @param KEflat - flat element stiffness matrix (edofSize x edofSize)
 * @param edofs - element DOF indices (nel * edofSize)
 * @param p - input vector (ndof)
 * @param Ap - output vector (ndof), must be pre-zeroed
 * @param nel - number of elements
 * @param edofSize - DOFs per element (8 for 2D, 24 for 3D)
 * @param Emin - minimum Young's modulus
 * @param E0 - base Young's modulus
 * @param penal - SIMP penalization exponent
 */
export function ebeMatVec(
  densities: Float64Array,
  KEflat: Float64Array,
  edofs: Int32Array,
  p: Float64Array,
  Ap: Float64Array,
  nel: i32,
  edofSize: i32,
  Emin: f64,
  E0: f64,
  penal: f64
): void {
  for (let e = 0; e < nel; e++) {
    const density: f64 = unchecked(densities[e]);
    const stiffness: f64 = Emin + Math.pow(density, penal) * (E0 - Emin);
    const eOff: i32 = e * edofSize;

    for (let i = 0; i < edofSize; i++) {
      const gi: i32 = unchecked(edofs[eOff + i]);
      let sum: f64 = 0.0;
      const keRow: i32 = i * edofSize;
      for (let j = 0; j < edofSize; j++) {
        sum += unchecked(KEflat[keRow + j]) * unchecked(p[unchecked(edofs[eOff + j])]);
      }
      unchecked(Ap[gi] = unchecked(Ap[gi]) + stiffness * sum);
    }
  }
}

/**
 * Compute diagonal of global stiffness matrix K(x) element-by-element.
 * Used for Jacobi preconditioning.
 * @param densities - element densities (nel)
 * @param KEflat - flat element stiffness matrix (edofSize x edofSize)
 * @param edofs - element DOF indices (nel * edofSize)
 * @param diag - output diagonal vector (ndof)
 * @param nel - number of elements
 * @param edofSize - DOFs per element
 * @param Emin - minimum Young's modulus
 * @param E0 - base Young's modulus
 * @param penal - SIMP penalization exponent
 */
export function computeDiagonal(
  densities: Float64Array,
  KEflat: Float64Array,
  edofs: Int32Array,
  diag: Float64Array,
  nel: i32,
  edofSize: i32,
  Emin: f64,
  E0: f64,
  penal: f64
): void {
  for (let e = 0; e < nel; e++) {
    const density: f64 = unchecked(densities[e]);
    const stiffness: f64 = Emin + Math.pow(density, penal) * (E0 - Emin);
    const eOff: i32 = e * edofSize;

    for (let i = 0; i < edofSize; i++) {
      const gi: i32 = unchecked(edofs[eOff + i]);
      unchecked(diag[gi] = unchecked(diag[gi]) + stiffness * unchecked(KEflat[i * edofSize + i]));
    }
  }
}

/**
 * Element-by-element matrix-vector multiply for 3D multigrid applyA.
 * Uses raw memory pointers (byte offsets into WASM linear memory) to avoid
 * AssemblyScript typed-array wrapper overhead on the hot path.
 *
 * @param kePtr       - KEflat: 24×24 f64 element stiffness (576 entries)
 * @param edofsPtr    - edofs: i32 element DOF indices (nel * 24)
 * @param evalsPtr    - E_vals: f64 per-element stiffness values
 * @param activePtr   - active: i32 indices of active elements
 * @param activeCount - number of active elements
 * @param pPtr        - p: f64 input vector (ndof)
 * @param apPtr       - Ap: f64 output vector (ndof), will be zeroed
 * @param ndof        - total degrees of freedom
 * @param scratchPtr  - scratch: f64 workspace (>=24 entries)
 */
export function applyAEbe3D(
  kePtr: usize,
  edofsPtr: usize,
  evalsPtr: usize,
  activePtr: usize,
  activeCount: i32,
  pPtr: usize,
  apPtr: usize,
  ndof: i32,
  scratchPtr: usize
): void {
  // Zero output vector
  memory.fill(apPtr, 0, <usize>ndof << 3);

  for (let ai: i32 = 0; ai < activeCount; ai++) {
    const e: i32 = load<i32>(activePtr + (<usize>ai << 2));
    const E: f64 = load<f64>(evalsPtr + (<usize>e << 3));
    const edofsBase: usize = edofsPtr + (<usize>(e * 24) << 2);

    // Gather 24 local DOFs into scratch buffer
    for (let j: i32 = 0; j < 24; j++) {
      const gj: i32 = load<i32>(edofsBase + (<usize>j << 2));
      store<f64>(scratchPtr + (<usize>j << 3), load<f64>(pPtr + (<usize>gj << 3)));
    }

    // Compute local matvec and scatter-add to Ap
    for (let i: i32 = 0; i < 24; i++) {
      const gi: i32 = load<i32>(edofsBase + (<usize>i << 2));
      let sum: f64 = 0.0;
      const rowBase: usize = kePtr + (<usize>(i * 24) << 3);
      for (let j: i32 = 0; j < 24; j++) {
        sum += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(scratchPtr + (<usize>j << 3));
      }
      store<f64>(apPtr + (<usize>gi << 3), load<f64>(apPtr + (<usize>gi << 3)) + E * sum);
    }
  }
}

/**
 * Dense matrix-vector multiply using raw memory pointers.
 * result[i] = sum_j( K[i*n + j] * p[j] )
 *
 * @param kPtr  - K: f64 dense matrix (n x n, row-major)
 * @param pPtr  - p: f64 input vector (n)
 * @param apPtr - Ap: f64 output vector (n)
 * @param n     - dimension
 */
export function denseMatVecRaw(
  kPtr: usize,
  pPtr: usize,
  apPtr: usize,
  n: i32
): void {
  for (let i: i32 = 0; i < n; i++) {
    let s: f64 = 0.0;
    const rowBase: usize = kPtr + (<usize>(i * n) << 3);
    for (let j: i32 = 0; j < n; j++) {
      s += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(pPtr + (<usize>j << 3));
    }
    store<f64>(apPtr + (<usize>i << 3), s);
  }
}

/**
 * Full self-contained Element-By-Element Preconditioned Conjugate Gradient
 * (EbE-PCG) FEA solver.
 *
 * Performs the entire Jacobi-preconditioned CG solve with element-by-element
 * matrix-vector multiplication internally, eliminating per-iteration JS↔WASM
 * boundary crossings.
 *
 * Steps performed internally:
 *   1. Compute per-element stiffness E_vals from densities
 *   2. Identify active elements (skip near-zero stiffness)
 *   3. Build Jacobi preconditioner (inverse diagonal of K)
 *   4. Run preconditioned CG with EbE matvec
 *   5. Write solution into U at free DOF positions
 *
 * All parameters use raw memory pointers (byte offsets) for maximum performance.
 *
 * @param densPtr    - densities: f64[nel] element densities
 * @param kePtr      - KEflat: f64[edofSize*edofSize] reference element stiffness
 * @param edofsPtr   - edofs: i32[nel*edofSize] element DOF connectivity
 * @param fPtr       - F: f64[ndof] global force vector
 * @param uPtr       - U: f64[ndof] displacement vector (input for warm-start, output with solution)
 * @param freedofsPtr - freedofs: i32[nfree] indices of free (unconstrained) DOFs
 * @param nel        - number of elements
 * @param edofSize   - DOFs per element (8 for 2D quad, 24 for 3D hex)
 * @param ndof       - total degrees of freedom
 * @param nfree      - number of free DOFs
 * @param Emin       - minimum Young's modulus
 * @param E0         - base Young's modulus
 * @param penal      - SIMP penalization exponent
 * @param maxIter    - maximum CG iterations
 * @param tolerance  - convergence tolerance (on squared residual norm)
 * @param workPtr    - workspace: must have at least (nel + nel + ndof + 5*nfree + ndof + edofSize) f64 entries
 *                     Layout: E_vals[nel] | activeF64[nel] | diag[ndof] |
 *                             Uf[nfree] | r[nfree] | z[nfree] | p[nfree] | Ap[nfree] |
 *                             p_full[ndof] | scratch[edofSize]
 * @returns number of CG iterations performed
 */
export function ebePCG(
  densPtr: usize,
  kePtr: usize,
  edofsPtr: usize,
  fPtr: usize,
  uPtr: usize,
  freedofsPtr: usize,
  nel: i32,
  edofSize: i32,
  ndof: i32,
  nfree: i32,
  Emin: f64,
  E0: f64,
  penal: f64,
  maxIter: i32,
  tolerance: f64,
  workPtr: usize
): i32 {
  const EPSILON: f64 = 1e-12;
  const dE: f64 = E0 - Emin;
  const skipThreshold: f64 = Emin * 1000.0;

  // ── Workspace layout ──────────────────────────────────────────────
  let off: usize = workPtr;
  const evalsPtr: usize = off;     off += <usize>nel << 3;       // f64[nel]
  const activePtr: usize = off;    off += <usize>nel << 2;       // i32[nel]
  const diagPtr: usize = off;      off += <usize>ndof << 3;      // f64[ndof]
  const ufPtr: usize = off;        off += <usize>nfree << 3;     // f64[nfree]
  const rPtr: usize = off;         off += <usize>nfree << 3;     // f64[nfree]
  const zPtr: usize = off;         off += <usize>nfree << 3;     // f64[nfree]
  const pPtr: usize = off;         off += <usize>nfree << 3;     // f64[nfree]
  const apPtr: usize = off;        off += <usize>nfree << 3;     // f64[nfree]
  const pfullPtr: usize = off;     off += <usize>ndof << 3;      // f64[ndof]
  const apfullPtr: usize = off;    off += <usize>ndof << 3;      // f64[ndof]
  const scratchPtr: usize = off;   // f64[edofSize]

  // ── Step 1: Compute per-element stiffness & identify active elements ──
  let activeCount: i32 = 0;
  for (let e: i32 = 0; e < nel; e++) {
    const density: f64 = load<f64>(densPtr + (<usize>e << 3));
    const stiffness: f64 = Emin + Math.pow(density, penal) * dE;
    store<f64>(evalsPtr + (<usize>e << 3), stiffness);
    if (stiffness > skipThreshold) {
      store<i32>(activePtr + (<usize>activeCount << 2), e);
      activeCount++;
    }
  }

  // ── Step 2: Compute Jacobi preconditioner (diagonal of K) ─────────
  memory.fill(diagPtr, 0, <usize>ndof << 3);
  for (let ai: i32 = 0; ai < activeCount; ai++) {
    const e: i32 = load<i32>(activePtr + (<usize>ai << 2));
    const E: f64 = load<f64>(evalsPtr + (<usize>e << 3));
    const eOff: usize = edofsPtr + (<usize>(e * edofSize) << 2);
    for (let i: i32 = 0; i < edofSize; i++) {
      const gi: i32 = load<i32>(eOff + (<usize>i << 2));
      const keDiag: f64 = load<f64>(kePtr + (<usize>(i * edofSize + i) << 3));
      store<f64>(diagPtr + (<usize>gi << 3),
        load<f64>(diagPtr + (<usize>gi << 3)) + E * keDiag);
    }
  }

  // Temporarily reuse zPtr to build invDiag (will be copied to safe location before CG)
  const invDiagTempPtr: usize = zPtr;
  for (let i: i32 = 0; i < nfree; i++) {
    const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
    const d: f64 = load<f64>(diagPtr + (<usize>fi << 3));
    const inv: f64 = d > 1e-30 ? 1.0 / d : 0.0;
    store<f64>(invDiagTempPtr + (<usize>i << 3), inv);
  }

  // Copy invDiag to safe location since zPtr will be overwritten by CG
  // Needs nfree * 8 bytes - use space after scratch buffer
  const invDiagSafePtr: usize = scratchPtr + (<usize>edofSize << 3);
  memory.copy(invDiagSafePtr, invDiagTempPtr, <usize>nfree << 3);

  // ── Step 3: Initialize CG (warm-start aware) ──────────────────────
  // Extract Uf from caller-provided U (supports warm-starting:
  // caller writes previous solution into uPtr before calling, or zeros it)
  for (let i: i32 = 0; i < nfree; i++) {
    const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
    store<f64>(ufPtr + (<usize>i << 3), load<f64>(uPtr + (<usize>fi << 3)));
  }

  // Compute initial residual: r = F_free - (K * U)_free
  // Scatter Uf → p_full (reuse buffer)
  memory.fill(pfullPtr, 0, <usize>ndof << 3);
  for (let i: i32 = 0; i < nfree; i++) {
    const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
    store<f64>(pfullPtr + (<usize>fi << 3), load<f64>(ufPtr + (<usize>i << 3)));
  }

  // EbE matvec: ap_full = K * p_full
  memory.fill(apfullPtr, 0, <usize>ndof << 3);
  for (let ai: i32 = 0; ai < activeCount; ai++) {
    const e0: i32 = load<i32>(activePtr + (<usize>ai << 2));
    const E0r: f64 = load<f64>(evalsPtr + (<usize>e0 << 3));
    const eOff0: usize = edofsPtr + (<usize>(e0 * edofSize) << 2);
    for (let j: i32 = 0; j < edofSize; j++) {
      const gj: i32 = load<i32>(eOff0 + (<usize>j << 2));
      store<f64>(scratchPtr + (<usize>j << 3), load<f64>(pfullPtr + (<usize>gj << 3)));
    }
    for (let i: i32 = 0; i < edofSize; i++) {
      const gi: i32 = load<i32>(eOff0 + (<usize>i << 2));
      let sum: f64 = 0.0;
      const rowBase: usize = kePtr + (<usize>(i * edofSize) << 3);
      for (let j: i32 = 0; j < edofSize; j++) {
        sum += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(scratchPtr + (<usize>j << 3));
      }
      store<f64>(apfullPtr + (<usize>gi << 3),
        load<f64>(apfullPtr + (<usize>gi << 3)) + E0r * sum);
    }
  }

  // r = F_free - Ap_free
  for (let i: i32 = 0; i < nfree; i++) {
    const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
    store<f64>(rPtr + (<usize>i << 3),
      load<f64>(fPtr + (<usize>fi << 3)) - load<f64>(apfullPtr + (<usize>fi << 3)));
  }

  // z = M^{-1} r; p = z; rz = r^T z
  let rz: f64 = 0.0;
  let r0norm2: f64 = 0.0;
  for (let i: i32 = 0; i < nfree; i++) {
    const invD: f64 = load<f64>(invDiagSafePtr + (<usize>i << 3));
    const ri: f64 = load<f64>(rPtr + (<usize>i << 3));
    const zi: f64 = invD * ri;
    store<f64>(zPtr + (<usize>i << 3), zi);
    store<f64>(pPtr + (<usize>i << 3), zi);
    rz += ri * zi;
    r0norm2 += ri * ri;
  }

  // Relative tolerance: ||r||² < tol² · ||r₀||²
  const tolSq: f64 = tolerance * tolerance * (r0norm2 > 1e-30 ? r0norm2 : <f64>1e-30);
  const cgMax: i32 = maxIter < nfree ? maxIter : nfree;

  // ── Step 4: CG iterations ─────────────────────────────────────────
  for (let iter: i32 = 0; iter < cgMax; iter++) {
    // Check convergence: ||r||^2 < tolSq
    let rnorm2: f64 = 0.0;
    for (let i: i32 = 0; i < nfree; i++) {
      const ri: f64 = load<f64>(rPtr + (<usize>i << 3));
      rnorm2 += ri * ri;
    }
    if (rnorm2 < tolSq) return iter;

    // ── EbE matvec: Ap = K * p (in full space, then restrict) ────
    // Scatter p_reduced → p_full
    memory.fill(pfullPtr, 0, <usize>ndof << 3);
    for (let i: i32 = 0; i < nfree; i++) {
      const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
      store<f64>(pfullPtr + (<usize>fi << 3), load<f64>(pPtr + (<usize>i << 3)));
    }

    // Ap_full = 0
    memory.fill(apfullPtr, 0, <usize>ndof << 3);

    // Element-by-element matvec on active elements
    for (let ai: i32 = 0; ai < activeCount; ai++) {
      const e: i32 = load<i32>(activePtr + (<usize>ai << 2));
      const E: f64 = load<f64>(evalsPtr + (<usize>e << 3));
      const eOff: usize = edofsPtr + (<usize>(e * edofSize) << 2);

      // Gather local DOFs into scratch
      for (let j: i32 = 0; j < edofSize; j++) {
        const gj: i32 = load<i32>(eOff + (<usize>j << 2));
        store<f64>(scratchPtr + (<usize>j << 3), load<f64>(pfullPtr + (<usize>gj << 3)));
      }

      // Local matvec and scatter-add
      for (let i: i32 = 0; i < edofSize; i++) {
        const gi: i32 = load<i32>(eOff + (<usize>i << 2));
        let sum: f64 = 0.0;
        const rowBase: usize = kePtr + (<usize>(i * edofSize) << 3);
        for (let j: i32 = 0; j < edofSize; j++) {
          sum += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(scratchPtr + (<usize>j << 3));
        }
        store<f64>(apfullPtr + (<usize>gi << 3),
          load<f64>(apfullPtr + (<usize>gi << 3)) + E * sum);
      }
    }

    // Gather Ap_full → Ap_reduced
    for (let i: i32 = 0; i < nfree; i++) {
      const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
      store<f64>(apPtr + (<usize>i << 3), load<f64>(apfullPtr + (<usize>fi << 3)));
    }

    // ── CG update ────────────────────────────────────────────────
    let pAp: f64 = 0.0;
    for (let i: i32 = 0; i < nfree; i++) {
      pAp += load<f64>(pPtr + (<usize>i << 3)) * load<f64>(apPtr + (<usize>i << 3));
    }
    const alpha: f64 = rz / (pAp + EPSILON);

    let rz_new: f64 = 0.0;
    for (let i: i32 = 0; i < nfree; i++) {
      const pi: f64 = load<f64>(pPtr + (<usize>i << 3));
      const api: f64 = load<f64>(apPtr + (<usize>i << 3));
      const ri: f64 = load<f64>(rPtr + (<usize>i << 3)) - alpha * api;
      const invD: f64 = load<f64>(invDiagSafePtr + (<usize>i << 3));
      const zi: f64 = invD * ri;
      store<f64>(ufPtr + (<usize>i << 3),
        load<f64>(ufPtr + (<usize>i << 3)) + alpha * pi);
      store<f64>(rPtr + (<usize>i << 3), ri);
      store<f64>(zPtr + (<usize>i << 3), zi);
      rz_new += ri * zi;
    }

    const beta: f64 = rz_new / (rz + EPSILON);
    for (let i: i32 = 0; i < nfree; i++) {
      store<f64>(pPtr + (<usize>i << 3),
        load<f64>(zPtr + (<usize>i << 3)) + beta * load<f64>(pPtr + (<usize>i << 3)));
    }

    rz = rz_new;
  }

  // ── Step 5: Write solution to U ───────────────────────────────────
  // Zero full U first (ensures fixed DOFs remain zero)
  memory.fill(uPtr, 0, <usize>ndof << 3);
  for (let i: i32 = 0; i < nfree; i++) {
    const fi: i32 = load<i32>(freedofsPtr + (<usize>i << 2));
    store<f64>(uPtr + (<usize>fi << 3), load<f64>(ufPtr + (<usize>i << 3)));
  }

  return cgMax;
}

