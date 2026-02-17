#!/usr/bin/env node
/**
 * Benchmark Test: WASM vs Pure JavaScript Performance Comparison
 * 
 * This script tests the performance difference between WASM-accelerated
 * and pure JavaScript implementations of the conjugate gradient solver.
 */

import fs from 'fs';
import { performance } from 'perf_hooks';

const CG_TOLERANCE = 1e-8;
const EPSILON = 1e-12;

// Load WASM module
let wasmModule = null;
let wasmLoaded = false;

async function loadWasmModule() {
    try {
        const buffer = fs.readFileSync('./wasm/matrix-ops.wasm');
        const module = await WebAssembly.compile(buffer);
        
        wasmModule = await WebAssembly.instantiate(module, {
            env: {
                abort: () => console.error('WASM abort called'),
                seed: () => Date.now()
            }
        });
        
        wasmLoaded = true;
        return true;
    } catch (error) {
        console.error('Failed to load WASM:', error.message);
        wasmLoaded = false;
        return false;
    }
}

// Pure JavaScript CG Solver
function solveCG_JS(K, F, n, maxIter) {
    const U = new Float64Array(n);
    const r = new Float64Array(n);
    const p = new Float64Array(n);

    for (let i = 0; i < n; i++) {
        r[i] = F[i];
        p[i] = r[i];
    }

    let rho = 0;
    for (let i = 0; i < n; i++) {
        rho += r[i] * r[i];
    }

    for (let iter = 0; iter < maxIter; iter++) {
        if (Math.sqrt(rho) < CG_TOLERANCE) break;

        const Ap = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                Ap[i] += K[i * n + j] * p[j];
            }
        }

        let pAp = 0;
        for (let i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        const alpha = rho / (pAp + EPSILON);

        let rho_new = 0;
        for (let i = 0; i < n; i++) {
            U[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rho_new += r[i] * r[i];
        }

        const beta = rho_new / (rho + EPSILON);
        for (let i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rho = rho_new;
    }

    return U;
}

// WASM CG Solver
function solveCG_WASM(K, F, n, maxIter) {
    if (!wasmModule) throw new Error('WASM not loaded');

    // Allocate TypedArrays in WASM memory
    function copyToWasm(arr) {
        const byteLength = arr.length * 8;
        const buffer = wasmModule.exports.__pin(wasmModule.exports.__new(byteLength, 1));
        const header = wasmModule.exports.__pin(wasmModule.exports.__new(12, 4));
        
        const memory = wasmModule.exports.memory;
        const view = new DataView(memory.buffer);
        view.setUint32(header, buffer, true);
        view.setUint32(header + 4, buffer, true);
        view.setUint32(header + 8, byteLength, true);
        
        new Float64Array(memory.buffer, buffer, arr.length).set(arr);
        wasmModule.exports.__unpin(buffer);
        
        return header;
    }

    function readFromWasm(ptr, length) {
        const memory = wasmModule.exports.memory;
        const view = new DataView(memory.buffer);
        const buffer = view.getUint32(ptr, true);
        const result = new Float64Array(length);
        result.set(new Float64Array(memory.buffer, buffer, length));
        return result;
    }

    const U = new Float64Array(n);
    const ptrK = copyToWasm(K);
    const ptrF = copyToWasm(F);
    const ptrU = copyToWasm(U);

    wasmModule.exports.conjugateGradient(ptrK, ptrF, ptrU, n, maxIter, CG_TOLERANCE);

    const result = readFromWasm(ptrU, n);

    wasmModule.exports.__unpin(ptrK);
    wasmModule.exports.__unpin(ptrF);
    wasmModule.exports.__unpin(ptrU);

    return result;
}

// Generate a symmetric positive definite matrix for testing
function generateSPDMatrix(n) {
    const A = new Float64Array(n * n);
    
    // Create a random matrix
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A[i * n + j] = Math.random();
        }
    }
    
    // Make it symmetric positive definite: K = A^T * A + nI
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            K[i * n + j] = sum;
            if (i === j) K[i * n + j] += n; // Add diagonal dominance
        }
    }
    
    return K;
}

// Run benchmark for a specific matrix size
async function runBenchmark(size, iterations = 10) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Benchmarking ${size}×${size} system (${iterations} runs)`);
    console.log('='.repeat(60));
    
    // Generate test data
    const K = generateSPDMatrix(size);
    const F = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        F[i] = Math.random();
    }
    
    const maxIter = Math.min(size, 1000);
    
    // Warm-up runs
    console.log('Warming up...');
    solveCG_JS(K, F, size, maxIter);
    if (wasmLoaded) {
        try {
            solveCG_WASM(K, F, size, maxIter);
        } catch (err) {
            console.warn('WASM warm-up failed:', err.message);
        }
    }
    
    // Benchmark Pure JavaScript
    const jsResults = [];
    console.log('\nRunning JavaScript benchmarks...');
    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        const result = solveCG_JS(K, F, size, maxIter);
        const end = performance.now();
        jsResults.push(end - start);
        process.stdout.write(`  Run ${i + 1}/${iterations}: ${(end - start).toFixed(2)}ms\r`);
    }
    console.log('');
    
    const jsAvg = jsResults.reduce((a, b) => a + b) / jsResults.length;
    const jsMin = Math.min(...jsResults);
    const jsMax = Math.max(...jsResults);
    const jsStdDev = Math.sqrt(jsResults.reduce((sq, n) => sq + Math.pow(n - jsAvg, 2), 0) / jsResults.length);
    
    // Benchmark WASM
    let wasmAvg = null, wasmMin = null, wasmMax = null, wasmStdDev = null, speedup = null;
    
    if (wasmLoaded) {
        const wasmResults = [];
        console.log('Running WASM benchmarks...');
        
        try {
            for (let i = 0; i < iterations; i++) {
                const start = performance.now();
                const result = solveCG_WASM(K, F, size, maxIter);
                const end = performance.now();
                wasmResults.push(end - start);
                process.stdout.write(`  Run ${i + 1}/${iterations}: ${(end - start).toFixed(2)}ms\r`);
            }
            console.log('');
            
            wasmAvg = wasmResults.reduce((a, b) => a + b) / wasmResults.length;
            wasmMin = Math.min(...wasmResults);
            wasmMax = Math.max(...wasmResults);
            wasmStdDev = Math.sqrt(wasmResults.reduce((sq, n) => sq + Math.pow(n - wasmAvg, 2), 0) / wasmResults.length);
            speedup = ((jsAvg - wasmAvg) / jsAvg * 100);
            
            // Verify correctness
            const jsResult = solveCG_JS(K, F, size, maxIter);
            const wasmResult = solveCG_WASM(K, F, size, maxIter);
            let maxDiff = 0;
            for (let i = 0; i < size; i++) {
                const diff = Math.abs(jsResult[i] - wasmResult[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            
            console.log(`\nCorrectness check: Max difference = ${maxDiff.toExponential(2)}`);
            if (maxDiff < 1e-6) {
                console.log('✓ Results match within tolerance');
            } else {
                console.log('✗ Warning: Results differ significantly');
            }
        } catch (err) {
            console.error('\nWASM benchmark failed:', err.message);
        }
    }
    
    // Print results
    console.log('\n' + '─'.repeat(60));
    console.log('RESULTS:');
    console.log('─'.repeat(60));
    console.log('JavaScript (Pure):');
    console.log(`  Average: ${jsAvg.toFixed(2)}ms`);
    console.log(`  Min:     ${jsMin.toFixed(2)}ms`);
    console.log(`  Max:     ${jsMax.toFixed(2)}ms`);
    console.log(`  StdDev:  ${jsStdDev.toFixed(2)}ms`);
    
    if (wasmLoaded && wasmAvg !== null) {
        console.log('\nWebAssembly:');
        console.log(`  Average: ${wasmAvg.toFixed(2)}ms`);
        console.log(`  Min:     ${wasmMin.toFixed(2)}ms`);
        console.log(`  Max:     ${wasmMax.toFixed(2)}ms`);
        console.log(`  StdDev:  ${wasmStdDev.toFixed(2)}ms`);
        
        console.log('\nPerformance:');
        if (speedup > 0) {
            console.log(`  🚀 WASM is ${speedup.toFixed(1)}% FASTER`);
        } else {
            console.log(`  ⚠️  WASM is ${Math.abs(speedup).toFixed(1)}% SLOWER`);
        }
        console.log(`  Speedup factor: ${(jsAvg / wasmAvg).toFixed(2)}x`);
    } else {
        console.log('\nWASM: Not available');
    }
    
    return {
        size,
        jsAvg,
        wasmAvg,
        speedup
    };
}

// Main benchmark suite
async function main() {
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║  WASM vs JavaScript CG Solver Performance Benchmark       ║');
    console.log('╚════════════════════════════════════════════════════════════╝');
    
    // Load WASM
    console.log('\nLoading WASM module...');
    const loaded = await loadWasmModule();
    if (loaded) {
        console.log('✓ WASM module loaded successfully');
    } else {
        console.log('✗ WASM module failed to load, will only benchmark JavaScript');
    }
    
    // Test different matrix sizes
    const testSizes = [
        { size: 10, iterations: 50 },
        { size: 50, iterations: 20 },
        { size: 100, iterations: 10 },
        { size: 200, iterations: 5 },
        { size: 500, iterations: 3 }
    ];
    
    const results = [];
    for (const test of testSizes) {
        const result = await runBenchmark(test.size, test.iterations);
        results.push(result);
    }
    
    // Summary
    console.log('\n\n' + '═'.repeat(60));
    console.log('SUMMARY');
    console.log('═'.repeat(60));
    console.log('Size    | JS (ms) | WASM (ms) | Speedup');
    console.log('─'.repeat(60));
    
    for (const r of results) {
        const jsStr = r.jsAvg.toFixed(2).padStart(7);
        const wasmStr = r.wasmAvg !== null ? r.wasmAvg.toFixed(2).padStart(9) : '    N/A  ';
        const speedupStr = r.speedup !== null ? `${r.speedup > 0 ? '+' : ''}${r.speedup.toFixed(1)}%` : 'N/A';
        console.log(`${String(r.size).padEnd(7)} | ${jsStr} | ${wasmStr} | ${speedupStr}`);
    }
    
    if (wasmLoaded) {
        const avgSpeedup = results
            .filter(r => r.speedup !== null)
            .reduce((sum, r) => sum + r.speedup, 0) / results.filter(r => r.speedup !== null).length;
        
        console.log('─'.repeat(60));
        console.log(`Average speedup: ${avgSpeedup > 0 ? '+' : ''}${avgSpeedup.toFixed(1)}%`);
        console.log('═'.repeat(60));
        
        if (avgSpeedup > 0) {
            console.log('✓ WASM provides performance improvement!');
        } else {
            console.log('⚠ WASM overhead is greater than benefit for these sizes');
            console.log('  (WASM shines with larger matrices > 500×500)');
        }
    }
}

main().catch(err => {
    console.error('Benchmark failed:', err);
    process.exit(1);
});
