// Web Worker for topology optimization using SIMP algorithm
// This runs in a separate thread so the UI stays responsive.

const EPSILON = 1e-12;
const CG_TOLERANCE = 1e-8;

class TopologyOptimizerWorker {
    constructor() {
        this.rmin = 1.5;
        this.penal = 3;
        this.E0 = 1;
        this.Emin = 1e-9;
        this.nu = 0.3;
        this.cancelled = false;
    }

    optimize(model, config) {
        const { nx, ny, nz } = model;
        const nelx = nx;
        const nely = ny;
        const nelz = nz;
        const volfrac = config.volumeFraction;
        const maxIterations = config.maxIterations;

        this.penal = config.penaltyFactor;
        this.rmin = config.filterRadius;
        this.cancelled = false;

        const nel = nelx * nely;

        let x = new Float32Array(nel).fill(volfrac);
        let xnew = new Float32Array(nel);
        let xold = new Float32Array(nel).fill(1);

        const { H, Hs } = this.prepareFilter(nelx, nely, this.rmin);
        let fixeddofs = this.getFixedDOFs(nelx, nely, config.constraintPosition);
        let F = this.getLoadVector(nelx, nely, config.forceDirection, config.forceMagnitude);

        // Apply painted constraints (override dropdown if painted faces exist)
        if (config.paintedConstraints && config.paintedConstraints.length > 0) {
            fixeddofs = this.getFixedDOFsFromPaint(nelx, nely, config.paintedConstraints);
        }

        // Apply painted forces (override dropdown if painted faces exist)
        if (config.paintedForces && config.paintedForces.length > 0) {
            F = this.getLoadVectorFromPaint(nelx, nely, config.paintedForces, config.forceDirection, config.forceMagnitude);
        }

        // Build set of element indices that must stay solid (constraint/force surfaces)
        const preservedElements = new Set();
        const allPaintedKeys = [
            ...(config.paintedConstraints || []),
            ...(config.paintedForces || [])
        ];
        for (const key of allPaintedKeys) {
            const parts = key.split(',');
            if (parts.length < 2) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            if (!isNaN(vx) && !isNaN(vy) && vx >= 0 && vx < nelx && vy >= 0 && vy < nely) {
                preservedElements.add(vy + vx * nely);
            }
        }
        // Also preserve elements along dropdown-selected constraint/force positions
        if ((!config.paintedConstraints || config.paintedConstraints.length === 0)) {
            const constraintElems = this.getConstraintElements(nelx, nely, config.constraintPosition);
            for (const idx of constraintElems) preservedElements.add(idx);
        }
        if ((!config.paintedForces || config.paintedForces.length === 0)) {
            const forceElems = this.getForceElements(nelx, nely, config.forceDirection);
            for (const idx of forceElems) preservedElements.add(idx);
        }

        // Initialize preserved elements to full density
        for (const idx of preservedElements) {
            x[idx] = 1.0;
        }

        const ndof = 2 * (nelx + 1) * (nely + 1);
        const alldofs = Array.from({ length: ndof }, (_, i) => i);
        const fixedSet = new Set(fixeddofs);
        const freedofs = alldofs.filter(dof => !fixedSet.has(dof));

        const KE = this.lk();

        // Compute per-element force magnitudes for adaptive mesh info
        const elementForces = this.computeElementForces(nelx, nely, F);

        let loop = 0;
        let change = 1;
        let c = 0;
        let lastElementEnergies = null;

        while (change > 0.01 && loop < maxIterations) {
            if (this.cancelled) {
                postMessage({ type: 'cancelled', iteration: loop });
                return;
            }

            loop++;
            xold = Float32Array.from(x);

            const { U, c: compliance } = this.FE(nelx, nely, x, this.penal, KE, F, freedofs, fixeddofs);
            c = compliance;

            const dc = new Float32Array(nel);
            const elementEnergies = new Float32Array(nel);
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const n1 = (nely + 1) * elx + ely;
                    const n2 = (nely + 1) * (elx + 1) + ely;
                    const edof = [
                        2 * n1, 2 * n1 + 1,
                        2 * n2, 2 * n2 + 1,
                        2 * n2 + 2, 2 * n2 + 3,
                        2 * n1 + 2, 2 * n1 + 3
                    ];

                    const Ue = edof.map(dof => U[dof] || 0);
                    const idx = ely + elx * nely;
                    const energy = this.computeElementEnergy(KE, Ue);
                    elementEnergies[idx] = energy;

                    dc[idx] = -this.penal * Math.pow(x[idx], this.penal - 1) *
                              this.E0 * energy;
                }
            }

            const dcn = this.filterSensitivities(dc, x, H, Hs, nelx, nely);
            xnew = this.OC(nelx, nely, x, volfrac, dcn, preservedElements);

            change = 0;
            for (let i = 0; i < nel; i++) {
                change = Math.max(change, Math.abs(xnew[i] - xold[i]));
            }

            x = Float32Array.from(xnew);
            lastElementEnergies = elementEnergies;

            // Build adaptive mesh data for this iteration
            const meshData = this.buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces);

            postMessage({
                type: 'progress',
                iteration: loop,
                compliance: c,
                meshData: meshData
            });
        }

        if (this.cancelled) {
            postMessage({ type: 'cancelled', iteration: loop });
            return;
        }

        const finalMesh = this.buildAdaptiveMesh(nelx, nely, nelz, x, lastElementEnergies, elementForces);

        // Also build the flat densities3D for export compatibility
        const densities3D = new Float32Array(nelx * nely * nelz);
        for (let z = 0; z < nelz; z++) {
            for (let y = 0; y < nely; y++) {
                for (let xpos = 0; xpos < nelx; xpos++) {
                    const idx2D = y + xpos * nely;
                    const idx3D = xpos + y * nelx + z * nelx * nely;
                    densities3D[idx3D] = x[idx2D];
                }
            }
        }

        postMessage({
            type: 'complete',
            result: {
                densities: densities3D,
                finalCompliance: c,
                iterations: loop,
                volumeFraction: volfrac,
                nx: nelx,
                ny: nely,
                nz: nelz,
                meshData: finalMesh
            }
        });
    }

    // Compute per-element force magnitude based on the global load vector
    computeElementForces(nelx, nely, F) {
        const nel = nelx * nely;
        const forces = new Float32Array(nel);
        for (let ely = 0; ely < nely; ely++) {
            for (let elx = 0; elx < nelx; elx++) {
                const n1 = (nely + 1) * elx + ely;
                const n2 = (nely + 1) * (elx + 1) + ely;
                const edof = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n2 + 2, 2 * n2 + 3,
                    2 * n1 + 2, 2 * n1 + 3
                ];
                let mag = 0;
                for (let d = 0; d < edof.length; d++) {
                    mag += F[edof[d]] * F[edof[d]];
                }
                forces[ely + elx * nely] = Math.sqrt(mag);
            }
        }
        return forces;
    }

    /**
     * Build adaptive mesh data.
     * Elements with higher stress or near applied forces get subdivided
     * into smaller triangles, while low-stress regions use coarser triangles.
     * Uses stiffness-weighted strain energy (stiffness × raw energy) for auto-resizing.
     * Returns an array of { vertices, density } triangle objects for rendering.
     */
    buildAdaptiveMesh(nelx, nely, nelz, x, elementEnergies, elementForces) {
    // Duplicated from constants.js since workers cannot use ES module imports
    const DENSITY_THRESHOLD = 0.3;
        const triangles = [];

        // Compute stress-based metric: scale energy by element stiffness (density^penal)
        let maxStress = 0;
        let maxForce = 0;
        const elementStress = elementEnergies ? new Float32Array(elementEnergies.length) : null;
        if (elementEnergies) {
            for (let i = 0; i < elementEnergies.length; i++) {
                const stiffness = this.Emin + Math.pow(x[i], this.penal) * (this.E0 - this.Emin);
                elementStress[i] = stiffness * elementEnergies[i];
                if (elementStress[i] > maxStress) maxStress = elementStress[i];
            }
        }
        if (elementForces) {
            for (let i = 0; i < elementForces.length; i++) {
                if (elementForces[i] > maxForce) maxForce = elementForces[i];
            }
        }

        for (let z = 0; z < nelz; z++) {
            for (let ely = 0; ely < nely; ely++) {
                for (let elx = 0; elx < nelx; elx++) {
                    const idx2D = ely + elx * nely;
                    const density = x[idx2D];

                    if (density <= DENSITY_THRESHOLD) continue;

                    // Determine subdivision level based on stress / force ratio
                    let subdivLevel = 1; // default: 2 triangles per face (1x1 subdivision)
                    if (maxStress > 0 && elementStress) {
                        const stressRatio = elementStress[idx2D] / maxStress;
                        const forceRatio = maxForce > 0 ? elementForces[idx2D] / maxForce : 0;
                        const ratio = Math.max(stressRatio, forceRatio);
                        if (ratio > 0.6) {
                            subdivLevel = 4; // 4x4 subdivision for high-stress areas
                        } else if (ratio > 0.3) {
                            subdivLevel = 2; // 2x2 subdivision for medium areas
                        }
                    }

                    // Generate subdivided mesh for this element
                    this.addSubdividedElement(triangles, elx, ely, z, density, subdivLevel);
                }
            }
        }

        return triangles;
    }

    /**
     * Add an element as triangulated quads with adaptive subdivision.
     * subdivLevel=1 means 2 triangles per visible face (standard).
     * subdivLevel=2 means 4x2=8 triangles per face (2x2 grid).
     * subdivLevel=4 means 16x2=32 triangles per face (4x4 grid).
     */
    addSubdividedElement(triangles, ex, ey, ez, density, subdivLevel) {
        const n = subdivLevel;
        const step = 1.0 / n;

        // For a 2D-extruded element, generate top face (Z) triangles for each z-layer
        // We generate the front face (XY plane) as the primary visible face
        const baseX = ex;
        const baseY = ey;
        const baseZ = ez;

        // Front face (z = baseZ) and back face (z = baseZ + 1)
        for (let sy = 0; sy < n; sy++) {
            for (let sx = 0; sx < n; sx++) {
                const x0 = baseX + sx * step;
                const y0 = baseY + sy * step;
                const x1 = x0 + step;
                const y1 = y0 + step;

                // Front face (z = baseZ)
                triangles.push({
                    vertices: [[x0, y0, baseZ], [x1, y0, baseZ], [x1, y1, baseZ]],
                    normal: [0, 0, -1],
                    density
                });
                triangles.push({
                    vertices: [[x0, y0, baseZ], [x1, y1, baseZ], [x0, y1, baseZ]],
                    normal: [0, 0, -1],
                    density
                });

                // Back face (z = baseZ + 1)
                triangles.push({
                    vertices: [[x1, y0, baseZ + 1], [x0, y0, baseZ + 1], [x0, y1, baseZ + 1]],
                    normal: [0, 0, 1],
                    density
                });
                triangles.push({
                    vertices: [[x1, y0, baseZ + 1], [x0, y1, baseZ + 1], [x1, y1, baseZ + 1]],
                    normal: [0, 0, 1],
                    density
                });
            }
        }

        // Side faces (along X and Y edges)
        for (let s = 0; s < n; s++) {
            const t0 = s * step;
            const t1 = t0 + step;

            // Bottom face (y = baseY)
            triangles.push({
                vertices: [[baseX + t0, baseY, baseZ], [baseX + t1, baseY, baseZ], [baseX + t1, baseY, baseZ + 1]],
                normal: [0, -1, 0],
                density
            });
            triangles.push({
                vertices: [[baseX + t0, baseY, baseZ], [baseX + t1, baseY, baseZ + 1], [baseX + t0, baseY, baseZ + 1]],
                normal: [0, -1, 0],
                density
            });

            // Top face (y = baseY + 1)
            triangles.push({
                vertices: [[baseX + t0, baseY + 1, baseZ + 1], [baseX + t1, baseY + 1, baseZ + 1], [baseX + t1, baseY + 1, baseZ]],
                normal: [0, 1, 0],
                density
            });
            triangles.push({
                vertices: [[baseX + t0, baseY + 1, baseZ + 1], [baseX + t1, baseY + 1, baseZ], [baseX + t0, baseY + 1, baseZ]],
                normal: [0, 1, 0],
                density
            });

            // Left face (x = baseX)
            triangles.push({
                vertices: [[baseX, baseY + t0, baseZ + 1], [baseX, baseY + t1, baseZ + 1], [baseX, baseY + t1, baseZ]],
                normal: [-1, 0, 0],
                density
            });
            triangles.push({
                vertices: [[baseX, baseY + t0, baseZ + 1], [baseX, baseY + t1, baseZ], [baseX, baseY + t0, baseZ]],
                normal: [-1, 0, 0],
                density
            });

            // Right face (x = baseX + 1)
            triangles.push({
                vertices: [[baseX + 1, baseY + t0, baseZ], [baseX + 1, baseY + t1, baseZ], [baseX + 1, baseY + t1, baseZ + 1]],
                normal: [1, 0, 0],
                density
            });
            triangles.push({
                vertices: [[baseX + 1, baseY + t0, baseZ], [baseX + 1, baseY + t1, baseZ + 1], [baseX + 1, baseY + t0, baseZ + 1]],
                normal: [1, 0, 0],
                density
            });
        }
    }

    prepareFilter(nelx, nely, rmin) {
        const iH = [];
        const jH = [];
        const sH = [];
        let k = 0;

        for (let i = 0; i < nelx; i++) {
            for (let j = 0; j < nely; j++) {
                const e1 = i * nely + j;

                for (let k_iter = Math.max(i - Math.floor(rmin), 0);
                     k_iter <= Math.min(i + Math.floor(rmin), nelx - 1);
                     k_iter++) {
                    for (let l = Math.max(j - Math.floor(rmin), 0);
                         l <= Math.min(j + Math.floor(rmin), nely - 1);
                         l++) {
                        const e2 = k_iter * nely + l;
                        const dist = Math.sqrt((i - k_iter) ** 2 + (j - l) ** 2);

                        if (dist <= rmin) {
                            iH[k] = e1;
                            jH[k] = e2;
                            sH[k] = Math.max(0, rmin - dist);
                            k++;
                        }
                    }
                }
            }
        }

        const H = { i: iH, j: jH, s: sH };
        const Hs = new Float32Array(nelx * nely);

        for (let i = 0; i < k; i++) {
            Hs[iH[i]] += sH[i];
        }

        return { H, Hs };
    }

    filterSensitivities(dc, x, H, Hs, nelx, nely) {
        const dcn = new Float32Array(nelx * nely);

        for (let i = 0; i < H.i.length; i++) {
            dcn[H.i[i]] += H.s[i] * x[H.j[i]] * dc[H.j[i]];
        }

        for (let i = 0; i < nelx * nely; i++) {
            dcn[i] = dcn[i] / (Hs[i] * Math.max(1e-3, x[i]));
        }

        return dcn;
    }

    OC(nelx, nely, x, volfrac, dc, preservedElements) {
        const nel = nelx * nely;
        const xnew = new Float32Array(nel);
        const move = 0.2;

        let l1 = 0;
        let l2 = 1e9;

        while ((l2 - l1) / (l2 + l1) > 1e-3) {
            const lmid = 0.5 * (l2 + l1);

            for (let i = 0; i < nel; i++) {
                if (preservedElements && preservedElements.has(i)) {
                    xnew[i] = 1.0;
                } else {
                    const Be = -dc[i] / lmid;
                    xnew[i] = Math.max(0.0,
                              Math.max(x[i] - move,
                              Math.min(1.0,
                              Math.min(x[i] + move, x[i] * Math.sqrt(Be)))));
                }
            }

            let sumXnew = 0;
            for (let i = 0; i < nel; i++) {
                sumXnew += xnew[i];
            }

            if (sumXnew > volfrac * nel) {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }

        return xnew;
    }

    lk() {
        const nu = 0.3;
        const k = [
            1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
            -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8
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

        return KE.map(row => row.map(val => val * 1 / (1 - nu * nu)));
    }

    FE(nelx, nely, x, penal, KE, F, freedofs, fixeddofs) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const U = new Float32Array(ndof);

        const K = this.assembleK(nelx, nely, x, penal, KE);
        const Uf = this.solveCG(K, F, freedofs, fixeddofs);

        freedofs.forEach((dof, i) => {
            U[dof] = Uf[i];
        });

        let c = 0;
        for (let i = 0; i < ndof; i++) {
            c += F[i] * U[i];
        }

        return { U, c };
    }

    assembleK(nelx, nely, x, penal, KE) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const K = Array(ndof).fill(0).map(() => Array(ndof).fill(0));

        for (let ely = 0; ely < nely; ely++) {
            for (let elx = 0; elx < nelx; elx++) {
                const n1 = (nely + 1) * elx + ely;
                const n2 = (nely + 1) * (elx + 1) + ely;
                const edof = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n2 + 2, 2 * n2 + 3,
                    2 * n1 + 2, 2 * n1 + 3
                ];

                const idx = ely + elx * nely;
                const E = this.Emin + Math.pow(x[idx], penal) * (this.E0 - this.Emin);

                for (let i = 0; i < 8; i++) {
                    for (let j = 0; j < 8; j++) {
                        K[edof[i]][edof[j]] += E * KE[i][j];
                    }
                }
            }
        }

        return K;
    }

    solveCG(K, F, freedofs, fixeddofs) {
        const n = freedofs.length;
        const Uf = new Float32Array(n);
        const r = new Float32Array(n);
        const p = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            r[i] = F[freedofs[i]];
        }

        const maxIter = Math.min(n, 1000);

        let rho = 0;
        for (let i = 0; i < n; i++) {
            rho += r[i] * r[i];
            p[i] = r[i];
        }

        for (let iter = 0; iter < maxIter; iter++) {
            if (Math.sqrt(rho) < CG_TOLERANCE) break;

            const Ap = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Ap[i] += K[freedofs[i]][freedofs[j]] * p[j];
                }
            }

            let pAp = 0;
            for (let i = 0; i < n; i++) {
                pAp += p[i] * Ap[i];
            }
            const alpha = rho / (pAp + EPSILON);

            let rho_new = 0;
            for (let i = 0; i < n; i++) {
                Uf[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rho_new += r[i] * r[i];
            }

            const beta = rho_new / (rho + EPSILON);
            for (let i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
            }

            rho = rho_new;
        }

        return Uf;
    }

    computeElementEnergy(KE, Ue) {
        let energy = 0;
        for (let i = 0; i < 8; i++) {
            for (let j = 0; j < 8; j++) {
                energy += Ue[i] * KE[i][j] * Ue[j];
            }
        }
        return energy;
    }

    getFixedDOFs(nelx, nely, position) {
        const fixeddofs = [];

        switch (position) {
            case 'left':
                for (let j = 0; j <= nely; j++) {
                    fixeddofs.push(2 * j, 2 * j + 1);
                }
                break;
            case 'right':
                for (let j = 0; j <= nely; j++) {
                    const n = (nely + 1) * nelx + j;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
            case 'bottom':
                for (let i = 0; i <= nelx; i++) {
                    const n = (nely + 1) * i;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
            case 'top':
                for (let i = 0; i <= nelx; i++) {
                    const n = (nely + 1) * i + nely;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
        }

        return fixeddofs;
    }

    getLoadVector(nelx, nely, direction, magnitude) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const F = new Float32Array(ndof);

        switch (direction) {
            case 'down': {
                const n_down = (nely + 1) * nelx + nely;
                F[2 * n_down + 1] = -magnitude;
                break;
            }
            case 'up': {
                const n_up = (nely + 1) * nelx;
                F[2 * n_up + 1] = magnitude;
                break;
            }
            case 'left': {
                const n_left = (nely + 1) * nelx + Math.floor(nely / 2);
                F[2 * n_left] = -magnitude;
                break;
            }
            case 'right': {
                const n_right = Math.floor(nely / 2);
                F[2 * n_right] = magnitude;
                break;
            }
        }

        return F;
    }

    /**
     * Get element indices along the constraint edge.
     */
    getConstraintElements(nelx, nely, position) {
        const elems = [];
        switch (position) {
            case 'left':
                for (let ey = 0; ey < nely; ey++) elems.push(ey); // elx=0
                break;
            case 'right':
                for (let ey = 0; ey < nely; ey++) elems.push(ey + (nelx - 1) * nely);
                break;
            case 'bottom':
                for (let ex = 0; ex < nelx; ex++) elems.push(ex * nely); // ely=0
                break;
            case 'top':
                for (let ex = 0; ex < nelx; ex++) elems.push((nely - 1) + ex * nely);
                break;
        }
        return elems;
    }

    /**
     * Get element indices where the default force is applied.
     */
    getForceElements(nelx, nely, direction) {
        const elems = [];
        switch (direction) {
            case 'down':
                // Force applied downward at right edge bottom node — nearest element (nelx-1, nely-1)
                elems.push((nely - 1) + (nelx - 1) * nely);
                break;
            case 'up':
                // Force applied upward at right edge top node — nearest element (nelx-1, 0)
                elems.push((nelx - 1) * nely);
                break;
            case 'left':
                // Force at right-middle — element at (nelx-1, floor(nely/2))
                elems.push(Math.floor(nely / 2) + (nelx - 1) * nely);
                break;
            case 'right':
                // Force at left-middle — element at (0, floor(nely/2))
                elems.push(Math.floor(nely / 2));
                break;
        }
        return elems;
    }

    /**
     * Convert painted constraint face keys to fixed DOFs.
     * Face keys are "x,y,z,faceIndex" where x,y are voxel coordinates.
     * Maps to 2D nodes at the voxel corners.
     */
    getFixedDOFsFromPaint(nelx, nely, paintedKeys) {
        const dofSet = new Set();
        for (const key of paintedKeys) {
            const parts = key.split(',');
            if (parts.length < 2) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            if (isNaN(vx) || isNaN(vy)) continue;
            // Map voxel (vx, vy) to its 4 corner nodes in 2D
            const n1 = (nely + 1) * vx + vy;
            const n2 = (nely + 1) * (vx + 1) + vy;
            const nodes = [n1, n2, n2 + 1, n1 + 1];
            for (const n of nodes) {
                if (n >= 0 && n < (nelx + 1) * (nely + 1)) {
                    dofSet.add(2 * n);
                    dofSet.add(2 * n + 1);
                }
            }
        }
        return Array.from(dofSet);
    }

    /**
     * Convert painted force face keys to a load vector.
     * Distributes force evenly across all painted face nodes.
     */
    getLoadVectorFromPaint(nelx, nely, paintedKeys, direction, magnitude) {
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const F = new Float32Array(ndof);

        // Determine force direction components
        let fx = 0, fy = 0;
        switch (direction) {
            case 'down':  fy = -1; break;
            case 'up':    fy = 1; break;
            case 'left':  fx = -1; break;
            case 'right': fx = 1; break;
            default:      fy = -1;
        }

        // Collect unique nodes from painted faces
        const nodeSet = new Set();
        for (const key of paintedKeys) {
            const parts = key.split(',');
            if (parts.length < 2) continue;
            const vx = parseInt(parts[0], 10);
            const vy = parseInt(parts[1], 10);
            if (isNaN(vx) || isNaN(vy)) continue;
            const n1 = (nely + 1) * vx + vy;
            const n2 = (nely + 1) * (vx + 1) + vy;
            const nodes = [n1, n2, n2 + 1, n1 + 1];
            for (const n of nodes) {
                if (n >= 0 && n < (nelx + 1) * (nely + 1)) {
                    nodeSet.add(n);
                }
            }
        }

        // Distribute force evenly across all unique nodes
        const nodeCount = nodeSet.size;
        if (nodeCount > 0) {
            const forcePerNode = magnitude / nodeCount;
            for (const n of nodeSet) {
                F[2 * n] += fx * forcePerNode;
                F[2 * n + 1] += fy * forcePerNode;
            }
        }

        return F;
    }
}

// Worker message handler
const optimizer = new TopologyOptimizerWorker();

self.onmessage = function(e) {
    const { type, model, config } = e.data;

    if (type === 'start') {
        optimizer.cancelled = false;
        optimizer.optimize(model, config);
    } else if (type === 'cancel') {
        optimizer.cancelled = true;
    }
};
