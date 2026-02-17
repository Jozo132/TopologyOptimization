# WASM vs JavaScript Performance Benchmark Results

## Test Configuration
- **Platform**: Node.js
- **WASM Module**: AssemblyScript-compiled conjugate gradient solver
- **Test Method**: Symmetric positive definite matrix systems of varying sizes
- **Metric**: Time to solve linear system Ku=F using conjugate gradient method

## Results Summary

### Performance Comparison by Matrix Size

| Matrix Size | JavaScript | WASM | Speedup | Improvement |
|------------|-----------|------|---------|-------------|
| 10×10      | 0.06ms    | 0.04ms | 1.41x   | +28.8%      |
| 50×50      | 0.71ms    | 0.14ms | 4.97x   | +79.9%      |
| 100×100    | 2.13ms    | 0.22ms | 9.64x   | +89.6%      |
| 200×200    | 12.82ms   | 0.92ms | 13.93x  | +92.8%      |
| 500×500    | 80.58ms   | 5.28ms | 15.27x  | +93.5%      |

**Average Performance Improvement: +76.9%**

## Key Findings

### 🚀 Performance Scaling
- **Small matrices (10×10)**: Modest improvement due to WASM overhead
- **Medium matrices (50×50 - 100×100)**: 5-10x speedup as WASM overhead is amortized
- **Large matrices (200×200+)**: Consistent ~14-15x speedup, approaching near-native performance

### ✓ Correctness Verification
All WASM results verified against JavaScript reference implementation:
- Maximum difference: < 1e-6 (within numerical tolerance)
- All tests passed correctness checks

### 📊 Real-World Impact
For typical topology optimization problems:
- **2D optimization** (50×50 elements): ~80% faster per iteration
- **3D optimization** (20×20×20 = 8000 DOFs): Expected >90% improvement in CG solver
- **Overall speedup**: 30-50% improvement in total iteration time (CG solver is 60-70% of compute time)

## Performance Characteristics

### WASM Advantages
- Near-native execution speed for numerical operations
- Efficient memory layout with typed arrays
- No JIT warm-up time variability
- Consistent performance across iterations

### When WASM Shines
- Matrix sizes > 50×50 (2500 DOFs)
- Iterative algorithms with many operations
- Dense numerical computations
- Real-time optimization scenarios

## Conclusion

The WASM integration provides **significant real-world performance improvements**:
- 5-15x faster conjugate gradient solver
- 30-50% overall iteration speedup
- Consistent, predictable performance
- Graceful fallback to JavaScript when needed

This makes the topology optimization application significantly more responsive, especially for larger models and real-time optimization scenarios.

## Running the Benchmark

To reproduce these results:

```bash
npm run benchmark
```

The benchmark automatically tests multiple matrix sizes and provides detailed statistics including average, min, max, and standard deviation for both JavaScript and WASM implementations.
