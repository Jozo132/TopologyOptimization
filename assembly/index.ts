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

