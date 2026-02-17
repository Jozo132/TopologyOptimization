// Topology Optimizer using SIMP (Solid Isotropic Material with Penalization)
export class TopologyOptimizer {
    constructor() {
        this.rmin = 1.5; // Filter radius
        this.penal = 3; // Penalization factor
        this.E0 = 1; // Young's modulus for solid material
        this.Emin = 1e-9; // Young's modulus for void
        this.nu = 0.3; // Poisson's ratio
    }

    async optimize(model, config, progressCallback) {
        const { nx, ny, nz } = model;
        const nelx = nx;
        const nely = ny;
        const nelz = nz;
        const volfrac = config.volumeFraction;
        const maxIterations = config.maxIterations;
        
        this.penal = config.penaltyFactor;
        this.rmin = config.filterRadius;
        
        console.log('Starting SIMP optimization...');
        console.log(`Domain: ${nelx}x${nely}x${nelz}, Target volume: ${volfrac * 100}%`);
        
        // For simplicity, we'll use 2D optimization (treat as 2D problem in XY plane)
        const nel = nelx * nely;
        
        // Initialize design variables (densities)
        let x = new Float32Array(nel).fill(volfrac);
        let xnew = new Float32Array(nel);
        let xold = new Float32Array(nel).fill(1);
        let low = new Float32Array(nel).fill(0);
        let upp = new Float32Array(nel).fill(1);
        
        // Prepare filter
        const { H, Hs } = this.prepareFilter(nelx, nely, this.rmin);
        
        // Fixed DOFs based on constraint position
        const fixeddofs = this.getFixedDOFs(nelx, nely, config.constraintPosition);
        
        // Load vector
        const F = this.getLoadVector(nelx, nely, config.forceDirection, config.forceMagnitude);
        
        // Free DOFs
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const alldofs = Array.from({ length: ndof }, (_, i) => i);
        const freedofs = alldofs.filter(dof => !fixeddofs.includes(dof));
        
        // Prepare FE (finite element) stiffness matrix
        const KE = this.lk();
        
        let loop = 0;
        let change = 1;
        let c = 0;
        
        // Optimization loop
        while (change > 0.01 && loop < maxIterations) {
            loop++;
            xold = Float32Array.from(x);
            
            // FE-Analysis
            const { U, c: compliance } = this.FE(nelx, nely, x, this.penal, KE, F, freedofs, fixeddofs);
            c = compliance;
            
            // Objective function and sensitivity analysis
            const dc = new Float32Array(nel);
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
                    
                    dc[idx] = -this.penal * Math.pow(x[idx], this.penal - 1) * 
                              this.E0 * this.computeElementEnergy(KE, Ue);
                }
            }
            
            // Filtering/modification of sensitivities
            const dcn = this.filterSensitivities(dc, x, H, Hs, nelx, nely);
            
            // Optimality criteria update
            xnew = this.OC(nelx, nely, x, volfrac, dcn);
            
            // Compute change
            change = 0;
            for (let i = 0; i < nel; i++) {
                change = Math.max(change, Math.abs(xnew[i] - xold[i]));
            }
            
            x = Float32Array.from(xnew);
            
            // Progress callback
            if (progressCallback) {
                progressCallback(loop, c);
            }
            
            // Small delay to allow UI updates
            await this.sleep(10);
        }
        
        console.log(`Optimization finished after ${loop} iterations`);
        
        // Convert 2D result to 3D for visualization
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
        
        return {
            densities: densities3D,
            finalCompliance: c,
            iterations: loop,
            volumeFraction: volfrac,
            nx: nelx,
            ny: nely,
            nz: nelz
        };
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

    OC(nelx, nely, x, volfrac, dc) {
        const nel = nelx * nely;
        const xnew = new Float32Array(nel);
        const move = 0.2;
        
        let l1 = 0;
        let l2 = 1e9;
        
        while ((l2 - l1) / (l2 + l1) > 1e-3) {
            const lmid = 0.5 * (l2 + l1);
            
            for (let i = 0; i < nel; i++) {
                const Be = -dc[i] / lmid;
                xnew[i] = Math.max(0.0,
                          Math.max(x[i] - move,
                          Math.min(1.0,
                          Math.min(x[i] + move, x[i] * Math.sqrt(Be)))));
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
        // Element stiffness matrix for plane stress
        const E = 1;
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
        
        return KE.map(row => row.map(val => val * E / (1 - nu * nu)));
    }

    FE(nelx, nely, x, penal, KE, F, freedofs, fixeddofs) {
        // Simplified FE analysis using conjugate gradient method
        const ndof = 2 * (nelx + 1) * (nely + 1);
        const U = new Float32Array(ndof);
        
        // Build global stiffness matrix (sparse representation)
        const K = this.assembleK(nelx, nely, x, penal, KE);
        
        // Solve KU = F (only for free DOFs)
        const Uf = this.solveCG(K, F, freedofs, fixeddofs);
        
        // Place solution in full vector
        freedofs.forEach((dof, i) => {
            U[dof] = Uf[i];
        });
        
        // Compute compliance
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
        // Conjugate gradient solver for Kf * Uf = Ff
        const n = freedofs.length;
        const Uf = new Float32Array(n);
        const r = new Float32Array(n);
        const p = new Float32Array(n);
        
        // Extract submatrix and subvector
        for (let i = 0; i < n; i++) {
            r[i] = F[freedofs[i]];
        }
        
        // CG iterations (simplified)
        const maxIter = Math.min(n, 1000);
        const tol = 1e-8;
        
        let rho = 0;
        for (let i = 0; i < n; i++) {
            rho += r[i] * r[i];
            p[i] = r[i];
        }
        
        for (let iter = 0; iter < maxIter; iter++) {
            if (Math.sqrt(rho) < tol) break;
            
            // Compute Ap
            const Ap = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Ap[i] += K[freedofs[i]][freedofs[j]] * p[j];
                }
            }
            
            // alpha = rho / (p' * Ap)
            let pAp = 0;
            for (let i = 0; i < n; i++) {
                pAp += p[i] * Ap[i];
            }
            const alpha = rho / (pAp + 1e-12);
            
            // Update solution and residual
            let rho_new = 0;
            for (let i = 0; i < n; i++) {
                Uf[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rho_new += r[i] * r[i];
            }
            
            // Update search direction
            const beta = rho_new / (rho + 1e-12);
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
                // Fix left edge
                for (let j = 0; j <= nely; j++) {
                    fixeddofs.push(2 * j, 2 * j + 1);
                }
                break;
            case 'right':
                // Fix right edge
                for (let j = 0; j <= nely; j++) {
                    const n = (nely + 1) * nelx + j;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
            case 'bottom':
                // Fix bottom edge
                for (let i = 0; i <= nelx; i++) {
                    const n = (nely + 1) * i;
                    fixeddofs.push(2 * n, 2 * n + 1);
                }
                break;
            case 'top':
                // Fix top edge
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
        
        // Apply load at specific location based on problem type
        // For cantilever: load at tip, opposite to fixed edge
        
        switch (direction) {
            case 'down':
                // Apply downward force at top-right corner
                const n_down = (nely + 1) * nelx + nely;
                F[2 * n_down + 1] = -magnitude;
                break;
            case 'up':
                // Apply upward force at bottom-right corner
                const n_up = (nely + 1) * nelx;
                F[2 * n_up + 1] = magnitude;
                break;
            case 'left':
                // Apply leftward force
                const n_left = (nely + 1) * nelx + Math.floor(nely / 2);
                F[2 * n_left] = -magnitude;
                break;
            case 'right':
                // Apply rightward force
                const n_right = Math.floor(nely / 2);
                F[2 * n_right] = magnitude;
                break;
        }
        
        return F;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
