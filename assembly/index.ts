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

// ═══════════════════════════════════════════════════════════════════════
// MGPCG — Full self-contained Multigrid-Preconditioned Conjugate Gradient
// ═══════════════════════════════════════════════════════════════════════
//
// Performs the entire MGPCG solve inside WASM, including V-cycle
// preconditioner with EbE/dense matvec, trilinear restriction &
// prolongation, and damped-Jacobi smoothing.
//
// The caller lays out a level descriptor table in WASM linear memory.
// Each level descriptor is 80 bytes (20 × i32) at stride 80.
//
// Level descriptor layout (byte offsets):
//   0:  nelx (i32)            4:  nely (i32)
//   8:  nelz (i32)           12:  ndof (i32)
//  16:  nel (i32)            20:  nfree (i32)
//  24:  activeCount (i32)    28:  hasDenseK (i32)  0 or 1
//  32:  edofsPtr (u32)       36:  evalsPtr (u32)
//  40:  activePtr (u32)      44:  fixedMaskPtr (u32)
//  48:  freeDofsPtr (u32)    52:  invDiagPtr (u32)
//  56:  denseKPtr (u32)      60:  AuPtr (u32)
//  64:  resPtr (u32)         68:  bPtr (u32)
//  72:  xPtr (u32)           76:  scratchPtr (u32)  [24 f64]
// ═══════════════════════════════════════════════════════════════════════

const MG_NU1: i32 = 2;
const MG_NU2: i32 = 2;
const MG_OMEGA: f64 = 0.5;
const MG_COARSE_ITERS: i32 = 30;
const MG_LEVEL_STRIDE: usize = 80;

// ─── Helpers for level descriptor access ───
@inline function lvlDesc(base: usize, li: i32): usize {
  return base + <usize>li * MG_LEVEL_STRIDE;
}

// ─── applyA: matrix-vector product at a given level ───
// Supports dense Galerkin (hasDenseK=1) or EbE with active elements.
// Writes result to apPtr; for EbE, apPtr is zeroed internally.
function mgApplyA(lvl: usize, kePtr: usize, pPtr: usize, apPtr: usize): void {
  const ndof: i32 = load<i32>(lvl + 12);
  const hasDense: i32 = load<i32>(lvl + 28);

  if (hasDense) {
    const dkPtr: usize = <usize>load<u32>(lvl + 56);
    for (let i: i32 = 0; i < ndof; i++) {
      let s: f64 = 0.0;
      const rowBase: usize = dkPtr + (<usize>(i * ndof) << 3);
      for (let j: i32 = 0; j < ndof; j++) {
        s += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(pPtr + (<usize>j << 3));
      }
      store<f64>(apPtr + (<usize>i << 3), s);
    }
    return;
  }

  // EbE matvec
  memory.fill(apPtr, 0, <usize>ndof << 3);
  const activeCount: i32 = load<i32>(lvl + 24);
  const activeArr: usize = <usize>load<u32>(lvl + 40);
  const edofsArr: usize = <usize>load<u32>(lvl + 32);
  const evalsArr: usize = <usize>load<u32>(lvl + 36);
  const scratch: usize = <usize>load<u32>(lvl + 76);

  for (let ai: i32 = 0; ai < activeCount; ai++) {
    const e: i32 = load<i32>(activeArr + (<usize>ai << 2));
    const E: f64 = load<f64>(evalsArr + (<usize>e << 3));
    const edBase: usize = edofsArr + (<usize>(e * 24) << 2);

    for (let j: i32 = 0; j < 24; j++) {
      const gj: i32 = load<i32>(edBase + (<usize>j << 2));
      store<f64>(scratch + (<usize>j << 3), load<f64>(pPtr + (<usize>gj << 3)));
    }
    for (let i: i32 = 0; i < 24; i++) {
      const gi: i32 = load<i32>(edBase + (<usize>i << 2));
      let sum: f64 = 0.0;
      const rowBase: usize = kePtr + (<usize>(i * 24) << 3);
      for (let j: i32 = 0; j < 24; j++) {
        sum += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(scratch + (<usize>j << 3));
      }
      store<f64>(apPtr + (<usize>gi << 3), load<f64>(apPtr + (<usize>gi << 3)) + E * sum);
    }
  }
}

// ─── Damped Jacobi smoothing: x += ω·D⁻¹·(b − A·x) ───
function mgSmooth(lvl: usize, kePtr: usize, bPtr: usize, xPtr: usize, auPtr: usize): void {
  const nfree: i32 = load<i32>(lvl + 20);
  const freeDofs: usize = <usize>load<u32>(lvl + 48);
  const invDiag: usize = <usize>load<u32>(lvl + 52);

  mgApplyA(lvl, kePtr, xPtr, auPtr);

  for (let ii: i32 = 0; ii < nfree; ii++) {
    const d: i32 = load<i32>(freeDofs + (<usize>ii << 2));
    const dOff: usize = <usize>d << 3;
    const r: f64 = load<f64>(bPtr + dOff) - load<f64>(auPtr + dOff);
    store<f64>(xPtr + dOff,
      load<f64>(xPtr + dOff) + MG_OMEGA * load<f64>(invDiag + dOff) * r);
  }
}

// ─── V-cycle: recursive multigrid preconditioner ───
function mgVcycle(levelsPtr: usize, li: i32, numLevels: i32, kePtr: usize,
                  bPtr: usize, xPtr: usize): void {
  const lvl: usize = lvlDesc(levelsPtr, li);
  const ndof: i32 = load<i32>(lvl + 12);
  const nfree: i32 = load<i32>(lvl + 20);
  const freeDofs: usize = <usize>load<u32>(lvl + 48);
  const auPtr: usize = <usize>load<u32>(lvl + 60);
  const resPtr: usize = <usize>load<u32>(lvl + 64);

  // ── Pre-smoothing ──
  for (let s: i32 = 0; s < MG_NU1; s++) {
    mgSmooth(lvl, kePtr, bPtr, xPtr, auPtr);
  }

  // ── Compute residual: res = b − A·x ──
  mgApplyA(lvl, kePtr, xPtr, auPtr);
  memory.fill(resPtr, 0, <usize>ndof << 3);
  for (let ii: i32 = 0; ii < nfree; ii++) {
    const d: i32 = load<i32>(freeDofs + (<usize>ii << 2));
    const dOff: usize = <usize>d << 3;
    store<f64>(resPtr + dOff, load<f64>(bPtr + dOff) - load<f64>(auPtr + dOff));
  }

  // ── Coarsest level: extra smoothing then return ──
  if (li === numLevels - 1) {
    for (let s: i32 = 0; s < MG_COARSE_ITERS; s++) {
      mgSmooth(lvl, kePtr, bPtr, xPtr, auPtr);
    }
    return;
  }

  // ── Restriction: bC = P^T · res  (trilinear full-weighting) ──
  const clvl: usize = lvlDesc(levelsPtr, li + 1);
  const ndofC: i32 = load<i32>(clvl + 12);
  const bCPtr: usize = <usize>load<u32>(clvl + 68);
  const xCPtr: usize = <usize>load<u32>(clvl + 72);

  const nxF: i32 = load<i32>(lvl + 0) + 1;   // nelx+1
  const nyF: i32 = load<i32>(lvl + 4) + 1;   // nely+1
  const nzF: i32 = load<i32>(lvl + 8) + 1;   // nelz+1
  const nxC: i32 = load<i32>(clvl + 0) + 1;
  const nyC: i32 = load<i32>(clvl + 4) + 1;
  const nzC: i32 = load<i32>(clvl + 8) + 1;

  memory.fill(bCPtr, 0, <usize>ndofC << 3);

  for (let fz: i32 = 0; fz < nzF; fz++) {
    const cz0: i32 = min(fz >> 1, nzC - 1);
    const cz1: i32 = min(cz0 + 1, nzC - 1);
    const tz: f64 = (cz0 !== cz1 && (fz & 1) !== 0) ? 0.5 : 0.0;
    const wz0: f64 = 1.0 - tz;
    const wz1: f64 = tz;

    for (let fy: i32 = 0; fy < nyF; fy++) {
      const cy0: i32 = min(fy >> 1, nyC - 1);
      const cy1: i32 = min(cy0 + 1, nyC - 1);
      const ty: f64 = (cy0 !== cy1 && (fy & 1) !== 0) ? 0.5 : 0.0;
      const wy0: f64 = 1.0 - ty;
      const wy1: f64 = ty;

      for (let fx: i32 = 0; fx < nxF; fx++) {
        const cx0: i32 = min(fx >> 1, nxC - 1);
        const cx1: i32 = min(cx0 + 1, nxC - 1);
        const tx: f64 = (cx0 !== cx1 && (fx & 1) !== 0) ? 0.5 : 0.0;
        const wx0: f64 = 1.0 - tx;
        const wx1: f64 = tx;

        const nF: i32 = fx * nyF * nzF + fy * nzF + fz;
        const rBase: usize = <usize>(3 * nF) << 3;
        const r0: f64 = load<f64>(resPtr + rBase);
        const r1: f64 = load<f64>(resPtr + rBase + 8);
        const r2: f64 = load<f64>(resPtr + rBase + 16);

        // Accumulate to 8 coarse corners
        const w000: f64 = wx0 * wy0 * wz0;
        if (w000 > 0) {
          const c3: usize = <usize>(3 * (cx0 * nyC * nzC + cy0 * nzC + cz0)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w000 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w000 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w000 * r2);
        }
        const w100: f64 = wx1 * wy0 * wz0;
        if (w100 > 0) {
          const c3: usize = <usize>(3 * (cx1 * nyC * nzC + cy0 * nzC + cz0)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w100 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w100 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w100 * r2);
        }
        const w010: f64 = wx0 * wy1 * wz0;
        if (w010 > 0) {
          const c3: usize = <usize>(3 * (cx0 * nyC * nzC + cy1 * nzC + cz0)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w010 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w010 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w010 * r2);
        }
        const w110: f64 = wx1 * wy1 * wz0;
        if (w110 > 0) {
          const c3: usize = <usize>(3 * (cx1 * nyC * nzC + cy1 * nzC + cz0)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w110 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w110 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w110 * r2);
        }
        const w001: f64 = wx0 * wy0 * wz1;
        if (w001 > 0) {
          const c3: usize = <usize>(3 * (cx0 * nyC * nzC + cy0 * nzC + cz1)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w001 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w001 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w001 * r2);
        }
        const w101: f64 = wx1 * wy0 * wz1;
        if (w101 > 0) {
          const c3: usize = <usize>(3 * (cx1 * nyC * nzC + cy0 * nzC + cz1)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w101 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w101 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w101 * r2);
        }
        const w011: f64 = wx0 * wy1 * wz1;
        if (w011 > 0) {
          const c3: usize = <usize>(3 * (cx0 * nyC * nzC + cy1 * nzC + cz1)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w011 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w011 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w011 * r2);
        }
        const w111: f64 = wx1 * wy1 * wz1;
        if (w111 > 0) {
          const c3: usize = <usize>(3 * (cx1 * nyC * nzC + cy1 * nzC + cz1)) << 3;
          store<f64>(bCPtr + c3,      load<f64>(bCPtr + c3)      + w111 * r0);
          store<f64>(bCPtr + c3 + 8,  load<f64>(bCPtr + c3 + 8)  + w111 * r1);
          store<f64>(bCPtr + c3 + 16, load<f64>(bCPtr + c3 + 16) + w111 * r2);
        }
      }
    }
  }

  // Zero fixed DOFs on coarse level
  const fixedC: usize = <usize>load<u32>(clvl + 44);
  for (let i: i32 = 0; i < ndofC; i++) {
    if (load<u8>(fixedC + <usize>i)) store<f64>(bCPtr + (<usize>i << 3), 0.0);
  }

  // ── Recursive coarse solve ──
  memory.fill(xCPtr, 0, <usize>ndofC << 3);
  mgVcycle(levelsPtr, li + 1, numLevels, kePtr, bCPtr, xCPtr);

  // ── Prolongation: x += P · xC  (trilinear interpolation) ──
  const fixedF: usize = <usize>load<u32>(lvl + 44);
  for (let fz: i32 = 0; fz < nzF; fz++) {
    const cz0: i32 = min(fz >> 1, nzC - 1);
    const cz1: i32 = min(cz0 + 1, nzC - 1);
    const tz: f64 = (cz0 !== cz1 && (fz & 1) !== 0) ? 0.5 : 0.0;
    const wz0: f64 = 1.0 - tz;
    const wz1: f64 = tz;

    for (let fy: i32 = 0; fy < nyF; fy++) {
      const cy0: i32 = min(fy >> 1, nyC - 1);
      const cy1: i32 = min(cy0 + 1, nyC - 1);
      const ty: f64 = (cy0 !== cy1 && (fy & 1) !== 0) ? 0.5 : 0.0;
      const wy0: f64 = 1.0 - ty;
      const wy1: f64 = ty;

      for (let fx: i32 = 0; fx < nxF; fx++) {
        const cx0: i32 = min(fx >> 1, nxC - 1);
        const cx1: i32 = min(cx0 + 1, nxC - 1);
        const tx: f64 = (cx0 !== cx1 && (fx & 1) !== 0) ? 0.5 : 0.0;
        const wx0: f64 = 1.0 - tx;
        const wx1: f64 = tx;

        const nF: i32 = fx * nyF * nzF + fy * nzF + fz;
        const n000: i32 = cx0 * nyC * nzC + cy0 * nzC + cz0;
        const n100: i32 = cx1 * nyC * nzC + cy0 * nzC + cz0;
        const n010: i32 = cx0 * nyC * nzC + cy1 * nzC + cz0;
        const n110: i32 = cx1 * nyC * nzC + cy1 * nzC + cz0;
        const n001: i32 = cx0 * nyC * nzC + cy0 * nzC + cz1;
        const n101: i32 = cx1 * nyC * nzC + cy0 * nzC + cz1;
        const n011: i32 = cx0 * nyC * nzC + cy1 * nzC + cz1;
        const n111: i32 = cx1 * nyC * nzC + cy1 * nzC + cz1;

        const pw000: f64 = wx0 * wy0 * wz0;
        const pw100: f64 = wx1 * wy0 * wz0;
        const pw010: f64 = wx0 * wy1 * wz0;
        const pw110: f64 = wx1 * wy1 * wz0;
        const pw001: f64 = wx0 * wy0 * wz1;
        const pw101: f64 = wx1 * wy0 * wz1;
        const pw011: f64 = wx0 * wy1 * wz1;
        const pw111: f64 = wx1 * wy1 * wz1;

        const bF: i32 = 3 * nF;
        for (let d: i32 = 0; d < 3; d++) {
          const fIdx: i32 = bF + d;
          if (!load<u8>(fixedF + <usize>fIdx)) {
            const val: f64 =
              load<f64>(xCPtr + (<usize>(3 * n000 + d) << 3)) * pw000 +
              load<f64>(xCPtr + (<usize>(3 * n100 + d) << 3)) * pw100 +
              load<f64>(xCPtr + (<usize>(3 * n010 + d) << 3)) * pw010 +
              load<f64>(xCPtr + (<usize>(3 * n110 + d) << 3)) * pw110 +
              load<f64>(xCPtr + (<usize>(3 * n001 + d) << 3)) * pw001 +
              load<f64>(xCPtr + (<usize>(3 * n101 + d) << 3)) * pw101 +
              load<f64>(xCPtr + (<usize>(3 * n011 + d) << 3)) * pw011 +
              load<f64>(xCPtr + (<usize>(3 * n111 + d) << 3)) * pw111;
            store<f64>(xPtr + (<usize>fIdx << 3),
              load<f64>(xPtr + (<usize>fIdx << 3)) + val);
          }
        }
      }
    }
  }

  // ── Post-smoothing ──
  for (let s: i32 = 0; s < MG_NU2; s++) {
    mgSmooth(lvl, kePtr, bPtr, xPtr, auPtr);
  }
}

/**
 * Full self-contained MGPCG (Multigrid-Preconditioned Conjugate Gradient)
 * FEA solver running entirely in WASM.
 *
 * Eliminates per-CG-iteration JS↔WASM boundary crossings by performing
 * the complete PCG solve, including multigrid V-cycle preconditioning,
 * inside a single WASM call.
 *
 * The caller must:
 *   1. Build the multigrid hierarchy on the JS side
 *   2. Lay out level descriptors and data in WASM linear memory
 *   3. Pre-compute E_vals, active elements, invDiag for each level
 *   4. Optionally assemble dense Galerkin matrices for coarse levels
 *
 * @param levelsPtr   - Base pointer for level descriptor table (numLevels × 80 bytes)
 * @param numLevels   - Number of multigrid levels
 * @param kePtr       - KEflat: f64[576] shared reference element stiffness
 * @param fPtr        - F: f64[ndof0] force vector at finest level
 * @param uPtr        - U: f64[ndof0] displacement output at finest level
 * @param fixedMask0  - fixedMask: u8[ndof0] at finest level
 * @param ndof0       - Total DOFs at finest level
 * @param maxIter     - Maximum CG iterations
 * @param tolerance   - Convergence tolerance (relative)
 * @param cgWorkPtr   - CG workspace: r[ndof0] + z[ndof0] + p[ndof0] + Ap[ndof0]
 * @returns number of CG iterations performed
 */
export function ebeMGPCG(
  levelsPtr: usize,
  numLevels: i32,
  kePtr: usize,
  fPtr: usize,
  uPtr: usize,
  fixedMask0: usize,
  ndof0: i32,
  maxIter: i32,
  tolerance: f64,
  cgWorkPtr: usize
): i32 {
  const EPSILON: f64 = 1e-12;
  const ndofBytes: usize = <usize>ndof0 << 3;

  // CG workspace layout
  const rPtr: usize = cgWorkPtr;
  const zPtr: usize = rPtr + ndofBytes;
  const pPtr: usize = zPtr + ndofBytes;
  const apPtr: usize = pPtr + ndofBytes;

  const lvl0: usize = lvlDesc(levelsPtr, 0);

  // ── Initial residual: r = F − A·U (with fixed DOFs zeroed) ──
  // Zero U first (cold start)
  memory.fill(uPtr, 0, ndofBytes);

  // r = F, zeroing fixed DOFs
  for (let i: i32 = 0; i < ndof0; i++) {
    const iOff: usize = <usize>i << 3;
    if (load<u8>(fixedMask0 + <usize>i)) {
      store<f64>(rPtr + iOff, 0.0);
    } else {
      store<f64>(rPtr + iOff, load<f64>(fPtr + iOff));
    }
  }

  // ── Apply preconditioner: z = M⁻¹·r  (V-cycle) ──
  memory.fill(zPtr, 0, ndofBytes);
  mgVcycle(levelsPtr, 0, numLevels, kePtr, rPtr, zPtr);
  // Zero fixed DOFs in z
  for (let i: i32 = 0; i < ndof0; i++) {
    if (load<u8>(fixedMask0 + <usize>i)) store<f64>(zPtr + (<usize>i << 3), 0.0);
  }

  // p = z
  memory.copy(pPtr, zPtr, ndofBytes);

  // rz = r^T · z,  r0norm2 = ||r||^2
  let rz: f64 = 0.0;
  let r0norm2: f64 = 0.0;
  for (let i: i32 = 0; i < ndof0; i++) {
    const iOff: usize = <usize>i << 3;
    const ri: f64 = load<f64>(rPtr + iOff);
    rz += ri * load<f64>(zPtr + iOff);
    r0norm2 += ri * ri;
  }

  const tolSq: f64 = tolerance * tolerance * (r0norm2 > 1e-30 ? r0norm2 : <f64>1e-30);

  // ── PCG iteration loop ──
  for (let iter: i32 = 0; iter < maxIter; iter++) {
    // Check convergence
    let rnorm2: f64 = 0.0;
    for (let i: i32 = 0; i < ndof0; i++) {
      const ri: f64 = load<f64>(rPtr + (<usize>i << 3));
      rnorm2 += ri * ri;
    }
    if (rnorm2 < tolSq) return iter;

    // Ap = A · p
    mgApplyA(lvl0, kePtr, pPtr, apPtr);
    // Zero fixed DOFs in Ap
    for (let i: i32 = 0; i < ndof0; i++) {
      if (load<u8>(fixedMask0 + <usize>i)) store<f64>(apPtr + (<usize>i << 3), 0.0);
    }

    // α = rz / (p^T · Ap)
    let pAp: f64 = 0.0;
    for (let i: i32 = 0; i < ndof0; i++) {
      const iOff: usize = <usize>i << 3;
      pAp += load<f64>(pPtr + iOff) * load<f64>(apPtr + iOff);
    }
    const alpha: f64 = rz / (pAp + EPSILON);

    // U += α·p,  r -= α·Ap
    for (let i: i32 = 0; i < ndof0; i++) {
      const iOff: usize = <usize>i << 3;
      store<f64>(uPtr + iOff, load<f64>(uPtr + iOff) + alpha * load<f64>(pPtr + iOff));
      store<f64>(rPtr + iOff, load<f64>(rPtr + iOff) - alpha * load<f64>(apPtr + iOff));
    }

    // z = M⁻¹ · r  (V-cycle preconditioner)
    memory.fill(zPtr, 0, ndofBytes);
    mgVcycle(levelsPtr, 0, numLevels, kePtr, rPtr, zPtr);
    for (let i: i32 = 0; i < ndof0; i++) {
      if (load<u8>(fixedMask0 + <usize>i)) store<f64>(zPtr + (<usize>i << 3), 0.0);
    }

    // β = rz_new / rz
    let rz_new: f64 = 0.0;
    for (let i: i32 = 0; i < ndof0; i++) {
      const iOff: usize = <usize>i << 3;
      rz_new += load<f64>(rPtr + iOff) * load<f64>(zPtr + iOff);
    }
    const beta: f64 = rz_new / (rz + EPSILON);

    // p = z + β·p
    for (let i: i32 = 0; i < ndof0; i++) {
      const iOff: usize = <usize>i << 3;
      store<f64>(pPtr + iOff, load<f64>(zPtr + iOff) + beta * load<f64>(pPtr + iOff));
    }

    rz = rz_new;
  }

  return maxIter;
}

// ═══════════════════════════════════════════════════════════════════════
// KSP CG + BDDC — PETSc-style KSP solver with Balancing Domain
// Decomposition by Constraints preconditioner, fully self-contained.
//
// The caller partitions the mesh into subdomains and describes each via
// a 32-byte descriptor (see BDDC_SUB_STRIDE).  Corner (primal) DOFs
// form the coarse space that couples the subdomains.
// ═══════════════════════════════════════════════════════════════════════

const BDDC_SUB_STRIDE: usize = 32;

// ─── Helper: subdomain descriptor field access ───
@inline function bddcSubNumElem(base: usize, si: i32): i32 {
  return load<i32>(base + <usize>si * BDDC_SUB_STRIDE);
}
@inline function bddcSubNumDofs(base: usize, si: i32): i32 {
  return load<i32>(base + <usize>si * BDDC_SUB_STRIDE + 4);
}
@inline function bddcSubElemPtr(base: usize, si: i32): usize {
  return <usize>load<u32>(base + <usize>si * BDDC_SUB_STRIDE + 8);
}
@inline function bddcSubDofsPtr(base: usize, si: i32): usize {
  return <usize>load<u32>(base + <usize>si * BDDC_SUB_STRIDE + 12);
}
@inline function bddcSubInvDiagPtr(base: usize, si: i32): usize {
  return <usize>load<u32>(base + <usize>si * BDDC_SUB_STRIDE + 16);
}

// ─── bddcApplyA: EbE matrix-vector product y = A·x (active elements) ───
// Zeros y before accumulation. Uses scratchPtr for 24-f64 gather buffer.
function bddcApplyA(
  kePtr: usize, edofsPtr: usize, evalsPtr: usize,
  activePtr: usize, activeCount: i32,
  xPtr: usize, yPtr: usize, ndof: i32, scratchPtr: usize
): void {
  memory.fill(yPtr, 0, <usize>ndof << 3);

  for (let ai: i32 = 0; ai < activeCount; ai++) {
    const e: i32 = load<i32>(activePtr + (<usize>ai << 2));
    const E: f64 = load<f64>(evalsPtr + (<usize>e << 3));
    const edBase: usize = edofsPtr + (<usize>(e * 24) << 2);

    // Gather local DOFs into scratch
    for (let j: i32 = 0; j < 24; j++) {
      const gj: i32 = load<i32>(edBase + (<usize>j << 2));
      store<f64>(scratchPtr + (<usize>j << 3), load<f64>(xPtr + (<usize>gj << 3)));
    }
    // Local matvec and scatter-add
    for (let i: i32 = 0; i < 24; i++) {
      const gi: i32 = load<i32>(edBase + (<usize>i << 2));
      let sum: f64 = 0.0;
      const rowBase: usize = kePtr + (<usize>(i * 24) << 3);
      for (let j: i32 = 0; j < 24; j++) {
        sum += load<f64>(rowBase + (<usize>j << 3)) * load<f64>(scratchPtr + (<usize>j << 3));
      }
      store<f64>(yPtr + (<usize>gi << 3),
        load<f64>(yPtr + (<usize>gi << 3)) + E * sum);
    }
  }
}

// ─── bddcApply: z = M⁻¹·r  (BDDC preconditioner) ───
// Steps:
//   1. Additive Schwarz: local Jacobi smoothing per subdomain
//   2. Additional smoother sweeps using global matvec
//   3. Coarse-level correction via primal (corner) DOFs
//   4. Combine local + coarse corrections; zero fixed DOFs
function bddcApply(
  subsPtr: usize, numSubs: i32,
  coarsePtr: usize, numCoarse: i32,
  kePtr: usize, edofsPtr: usize, evalsPtr: usize,
  activePtr: usize, activeCount: i32,
  fixedMask: usize, invDiagPtr: usize,
  rPtr: usize, zPtr: usize, ndof: i32,
  smootherIters: i32,
  localCorrPtr: usize, weightPtr: usize, coarseCorrPtr: usize,
  scratchPtr: usize
): void {
  const ndofBytes: usize = <usize>ndof << 3;

  // ── Step 1: Additive Schwarz — local Jacobi per subdomain ──
  memory.fill(zPtr, 0, ndofBytes);
  memory.fill(weightPtr, 0, ndofBytes);

  for (let si: i32 = 0; si < numSubs; si++) {
    const subNDofs: i32 = bddcSubNumDofs(subsPtr, si);
    const subDofsArr: usize = bddcSubDofsPtr(subsPtr, si);
    const subInvDiag: usize = bddcSubInvDiagPtr(subsPtr, si);

    for (let li: i32 = 0; li < subNDofs; li++) {
      const gi: i32 = load<i32>(subDofsArr + (<usize>li << 2));
      const giOff: usize = <usize>gi << 3;
      const invD: f64 = load<f64>(subInvDiag + giOff);
      const ri: f64 = load<f64>(rPtr + giOff);
      store<f64>(zPtr + giOff, load<f64>(zPtr + giOff) + invD * ri);
      store<f64>(weightPtr + giOff, load<f64>(weightPtr + giOff) + 1.0);
    }
  }

  // Average overlapping contributions
  for (let i: i32 = 0; i < ndof; i++) {
    const iOff: usize = <usize>i << 3;
    const w: f64 = load<f64>(weightPtr + iOff);
    if (w > 1e-12) {
      store<f64>(zPtr + iOff, load<f64>(zPtr + iOff) / w);
    }
  }

  // ── Step 2: Additional Jacobi smoother sweeps using global matvec ──
  // Step 1 already performed one sweep; loop adds (smootherIters - 1) more
  for (let s: i32 = 1; s < smootherIters; s++) {
    // localCorr = A · z
    bddcApplyA(kePtr, edofsPtr, evalsPtr,
               activePtr, activeCount,
               zPtr, localCorrPtr, ndof, scratchPtr);

    // z += invDiag * (r - A·z)
    for (let i: i32 = 0; i < ndof; i++) {
      const iOff: usize = <usize>i << 3;
      const res: f64 = load<f64>(rPtr + iOff) - load<f64>(localCorrPtr + iOff);
      const invD: f64 = load<f64>(invDiagPtr + iOff);
      store<f64>(zPtr + iOff, load<f64>(zPtr + iOff) + invD * res);
    }
  }

  // ── Step 3: Coarse-level correction via primal DOFs ──
  memory.fill(coarseCorrPtr, 0, ndofBytes);

  if (numCoarse > 0) {
    // Compute coarse residual: r_c = r - A·z at coarse DOFs
    bddcApplyA(kePtr, edofsPtr, evalsPtr,
               activePtr, activeCount,
               zPtr, localCorrPtr, ndof, scratchPtr);

    for (let ci: i32 = 0; ci < numCoarse; ci++) {
      const gi: i32 = load<i32>(coarsePtr + (<usize>ci << 2));
      const giOff: usize = <usize>gi << 3;
      const rc: f64 = load<f64>(rPtr + giOff) - load<f64>(localCorrPtr + giOff);
      const invD: f64 = load<f64>(invDiagPtr + giOff);
      store<f64>(coarseCorrPtr + giOff, invD * rc);
    }
  }

  // ── Step 4: Combine local + coarse corrections ──
  for (let i: i32 = 0; i < ndof; i++) {
    const iOff: usize = <usize>i << 3;
    store<f64>(zPtr + iOff,
      load<f64>(zPtr + iOff) + load<f64>(coarseCorrPtr + iOff));
  }

  // Zero fixed DOFs in z
  for (let i: i32 = 0; i < ndof; i++) {
    if (load<u8>(fixedMask + <usize>i)) store<f64>(zPtr + (<usize>i << 3), 0.0);
  }
}

/**
 * Full self-contained KSP CG solver with BDDC domain decomposition preconditioner.
 * Runs the entire PCG solve inside a single WASM call.
 *
 * Memory layout for subdomain descriptor (BDDC_SUB_STRIDE = 32 bytes each):
 *   0: numElements (i32)       - number of elements in this subdomain
 *   4: numDofs (i32)           - number of DOFs touched by this subdomain
 *   8: elementsPtr (u32)       - pointer to i32[numElements] element indices
 *  12: dofsPtr (u32)           - pointer to i32[numDofs] DOF indices
 *  16: localInvDiagPtr (u32)   - pointer to f64[ndof] local inverse diagonal
 *  20: padding (i32)
 *  24: padding (i32)
 *  28: padding (i32)
 *
 * @param subsPtr     - Base pointer for subdomain descriptor table
 * @param numSubs     - Number of subdomains
 * @param coarsePtr   - Pointer to i32[numCoarse] coarse (primal) DOF indices
 * @param numCoarse   - Number of coarse DOFs
 * @param kePtr       - KEflat: f64[576] reference element stiffness
 * @param edofsPtr    - Element DOF connectivity: i32[nel*24]
 * @param evalsPtr    - Per-element stiffness: f64[nel]
 * @param activePtr   - Active element indices: i32[activeCount]
 * @param activeCount - Number of active elements
 * @param fixedMask   - Fixed DOF mask: u8[ndof]
 * @param invDiagPtr  - Global inverse diagonal: f64[ndof]
 * @param fPtr        - F: f64[ndof] force vector
 * @param uPtr        - U: f64[ndof] displacement output
 * @param ndof        - Total DOFs
 * @param maxIter     - Maximum CG iterations
 * @param tolerance   - Convergence tolerance (relative)
 * @param smootherIters - Number of local Jacobi smoother iterations (typically 3)
 * @param workPtr     - Workspace: r[ndof] + z[ndof] + p[ndof] + Ap[ndof] + localCorr[ndof] + weight[ndof] + coarseCorr[ndof]
 * @returns number of CG iterations performed
 */
export function ebeKSP_BDDC(
  subsPtr: usize,
  numSubs: i32,
  coarsePtr: usize,
  numCoarse: i32,
  kePtr: usize,
  edofsPtr: usize,
  evalsPtr: usize,
  activePtr: usize,
  activeCount: i32,
  fixedMask: usize,
  invDiagPtr: usize,
  fPtr: usize,
  uPtr: usize,
  ndof: i32,
  maxIter: i32,
  tolerance: f64,
  smootherIters: i32,
  workPtr: usize
): i32 {
  const EPSILON: f64 = 1e-12;
  const ndofBytes: usize = <usize>ndof << 3;

  // ── Workspace layout ──────────────────────────────────────────────
  let off: usize = workPtr;
  const rPtr: usize = off;           off += ndofBytes;        // f64[ndof]
  const zPtr: usize = off;           off += ndofBytes;        // f64[ndof]
  const pPtr: usize = off;           off += ndofBytes;        // f64[ndof]
  const apPtr: usize = off;          off += ndofBytes;        // f64[ndof]
  const localCorrPtr: usize = off;   off += ndofBytes;        // f64[ndof]
  const weightPtr: usize = off;      off += ndofBytes;        // f64[ndof]
  const coarseCorrPtr: usize = off;  off += ndofBytes;        // f64[ndof]
  const scratchPtr: usize = off;     // f64[24] = 192 bytes

  // ── Zero U (cold start) ──
  memory.fill(uPtr, 0, ndofBytes);

  // ── r = F, zeroing fixed DOFs ──
  for (let i: i32 = 0; i < ndof; i++) {
    const iOff: usize = <usize>i << 3;
    if (load<u8>(fixedMask + <usize>i)) {
      store<f64>(rPtr + iOff, 0.0);
    } else {
      store<f64>(rPtr + iOff, load<f64>(fPtr + iOff));
    }
  }

  // ── Apply preconditioner: z = BDDC(r) ──
  memory.fill(zPtr, 0, ndofBytes);
  bddcApply(subsPtr, numSubs, coarsePtr, numCoarse,
            kePtr, edofsPtr, evalsPtr,
            activePtr, activeCount,
            fixedMask, invDiagPtr,
            rPtr, zPtr, ndof, smootherIters,
            localCorrPtr, weightPtr, coarseCorrPtr, scratchPtr);

  // p = z
  memory.copy(pPtr, zPtr, ndofBytes);

  // rz = r^T · z,  r0norm2 = ||r||^2
  let rz: f64 = 0.0;
  let r0norm2: f64 = 0.0;
  for (let i: i32 = 0; i < ndof; i++) {
    const iOff: usize = <usize>i << 3;
    const ri: f64 = load<f64>(rPtr + iOff);
    rz += ri * load<f64>(zPtr + iOff);
    r0norm2 += ri * ri;
  }

  const tolSq: f64 = tolerance * tolerance * (r0norm2 > 1e-30 ? r0norm2 : <f64>1e-30);

  // ── PCG iteration loop ──
  for (let iter: i32 = 0; iter < maxIter; iter++) {
    // Check convergence
    let rnorm2: f64 = 0.0;
    for (let i: i32 = 0; i < ndof; i++) {
      const ri: f64 = load<f64>(rPtr + (<usize>i << 3));
      rnorm2 += ri * ri;
    }
    if (rnorm2 < tolSq) return iter;

    // Ap = A · p
    bddcApplyA(kePtr, edofsPtr, evalsPtr,
               activePtr, activeCount,
               pPtr, apPtr, ndof, scratchPtr);
    // Zero fixed DOFs in Ap
    for (let i: i32 = 0; i < ndof; i++) {
      if (load<u8>(fixedMask + <usize>i)) store<f64>(apPtr + (<usize>i << 3), 0.0);
    }

    // α = rz / (p^T · Ap)
    let pAp: f64 = 0.0;
    for (let i: i32 = 0; i < ndof; i++) {
      const iOff: usize = <usize>i << 3;
      pAp += load<f64>(pPtr + iOff) * load<f64>(apPtr + iOff);
    }
    const alpha: f64 = rz / (pAp + EPSILON);

    // U += α·p,  r -= α·Ap
    for (let i: i32 = 0; i < ndof; i++) {
      const iOff: usize = <usize>i << 3;
      store<f64>(uPtr + iOff, load<f64>(uPtr + iOff) + alpha * load<f64>(pPtr + iOff));
      store<f64>(rPtr + iOff, load<f64>(rPtr + iOff) - alpha * load<f64>(apPtr + iOff));
    }

    // z = BDDC(r)
    memory.fill(zPtr, 0, ndofBytes);
    bddcApply(subsPtr, numSubs, coarsePtr, numCoarse,
              kePtr, edofsPtr, evalsPtr,
              activePtr, activeCount,
              fixedMask, invDiagPtr,
              rPtr, zPtr, ndof, smootherIters,
              localCorrPtr, weightPtr, coarseCorrPtr, scratchPtr);

    // β = rz_new / rz
    let rz_new: f64 = 0.0;
    for (let i: i32 = 0; i < ndof; i++) {
      const iOff: usize = <usize>i << 3;
      rz_new += load<f64>(rPtr + iOff) * load<f64>(zPtr + iOff);
    }
    const beta: f64 = rz_new / (rz + EPSILON);

    // p = z + β·p
    for (let i: i32 = 0; i < ndof; i++) {
      const iOff: usize = <usize>i << 3;
      store<f64>(pPtr + iOff, load<f64>(zPtr + iOff) + beta * load<f64>(pPtr + iOff));
    }

    rz = rz_new;
  }

  return maxIter;
}

