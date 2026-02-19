# Topology Optimization Web App

A minimalistic but fully functional web-based topology optimization application using vanilla JavaScript and ES6 modules.

**[Try it out on GitHub Pages](https://jozo132.github.io/TopologyOptimization)**

## Features

- 🚀 **4-Step Workflow**: Import → Assign → Solve → Export
- 🎨 **3D Visualization**: Real-time 3D rendering using Canvas 2D with interactive controls
- 🔧 **SIMP Algorithm**: Solid Isotropic Material with Penalization
- 🎯 **Full 3D Optimization**: True 3D FEA with 8-node hexahedral elements
- 📦 **STL Support**: Import and export STL files
- 📐 **STEP Support**: Import STEP files (AP203 and AP214 protocols)
- 🎯 **Template Models**: Quick start with cantilever beam, bridge, or cube test templates
- 💾 **Export Options**: Download optimized STL or JSON data
- ⚡ **Web Worker Optimization**: Runs optimization in a background thread, keeping the UI responsive
- ⚡ **WASM Acceleration**: High-performance AssemblyScript WASM module for matrix operations with automatic fallback to pure JS
- 🖥️ **GPU Acceleration**: WebGL2 GPGPU and WebGPU compute shader APIs for large-scale matrix operations
- ❌ **Cancellable Optimization**: Cancel a running optimization at any point
- 🔬 **Adaptive Mesh**: Dynamically refines mesh resolution in high-force / high-energy regions
- 🔺 **Triangle Mesh Rendering**: Displays elements as proper triangulated mesh, not just squares
- 🎚️ **Min Cross-Section Control**: Optional minimum feature size enforcement
- 📊 **Performance Benchmarking**: Real-time timing metrics and benchmark history tracking
- 🏁 **Baseline Comparisons**: Compare optimization performance against baseline cube pyramid benchmark

## Technology Stack

- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: No frameworks, pure ES6 modules
- **Web Workers**: Background thread for heavy computation
- **WebAssembly**: High-performance matrix operations via AssemblyScript
- **WebGL2 GPGPU**: Render-to-texture compute for GPU-accelerated matrix ops
- **WebGPU Compute**: Native compute shaders for modern GPU acceleration
- **Canvas 2D**: Interactive 3D visualization with no external dependencies
- **SIMP Algorithm**: Industry-standard topology optimization

## New in This Version

### 📊 Performance Benchmarking System
- **Real-time Metrics**: Live display of iteration time, average time, and elapsed time during optimization
- **Benchmark History**: Automatic tracking of optimization performance in local storage
- **Baseline Comparisons**: Compare current runs against the baseline cube pyramid benchmark
- **Engine Indicators**: Visual badges showing whether WASM or pure JS is being used
- **Performance Tracking**: See iteration throughput and timing improvements over multiple runs

### 🎯 True 3D Optimization
- Full 3D finite element analysis with 8-node hexahedral (brick) elements
- 3 degrees of freedom per node (x, y, z displacements)
- 24×24 element stiffness matrix with Gauss quadrature integration
- Automatic selection between 2D and 3D optimizers based on model type

### 🧊 Cube Test Template
- 5×5×5 element cube with predefined boundary conditions
- Force applied at top center
- Constraints at bottom 4 corners
- Produces pyramid-like structure demonstrating 3D optimization capabilities
- Ideal for performance benchmarking

### ⚡ WASM Acceleration (Fully Integrated & Benchmarked)
- **Pre-compiled AssemblyScript WASM library** (only 15KB)
- **Fully integrated** into optimization pipeline - WASM functions are actively used
- **Automatic loading** with graceful fallback to pure JavaScript
- **Accelerated operations**:
  - Conjugate gradient solver (linear system Ku=F)
  - Matrix-vector multiplication
  - Density filtering operations
  - Element energy computation
- **Measured performance gains** (see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)):
  - 5-15x faster CG solver (depending on problem size)
  - 30-50% overall iteration speedup
  - Average 77% improvement across tested sizes
- **Zero-copy memory operations** for maximum efficiency

### 🎚️ Minimum Cross-Section Control
- UI control for setting minimum feature size
- Prevents thin, unbuildable members
- Configurable from 0 (disabled) to 5 elements

## Usage

1. **Import Model**
   - Upload an STL or STEP file (AP203/AP214), or
   - Use a pre-defined template:
     - **Cantilever Beam**: Classic 2D beam optimization
     - **Bridge**: 2D bridge span optimization
     - **Cube Test (3D)**: Full 3D optimization demo

2. **Assign Loads & Constraints**
   - Set volume fraction (target material percentage)
   - Choose force direction and magnitude
   - Select constraint position (fixed boundary)
   - **Or** use paint tools to manually mark constraints and forces on the model

3. **Solve Optimization**
   - Configure optimization parameters:
     - Max iterations (default: 100)
     - Penalty factor (default: 3)
     - Filter radius (default: 1.5)
     - Min cross-section (default: 0 = disabled)
   - Run the optimization and watch progress
   - Cancel at any time using the Cancel button

4. **Export Result**
   - Download optimized model as STL file
   - Export optimization data as JSON
   - Start over for a new optimization

## Performance Benchmarking

The application includes a comprehensive benchmarking system to track and compare optimization performance:

### Benchmark Features
- **Real-time Metrics**: During optimization, see live updates of:
  - Current iteration time
  - Average iteration time across all iterations
  - Total elapsed time
  - Engine type (WASM 🚀 or JS)
  
- **Final Statistics**: After optimization completes:
  - Total optimization time
  - Average time per iteration
  - Throughput (iterations per second)
  - Engine used

- **Benchmark History**: 
  - Automatically stores up to 10 recent optimization runs
  - Compares against baseline (typically the first cube test)
  - Shows performance improvements or regressions as percentages
  - Persists between sessions using localStorage

### Typical Performance (Cube Pyramid 5×5×5, 20 iterations)
- **Pure JavaScript**: ~1,466ms per iteration (baseline)
- **With WASM (integrated)**: 30-50% improvement in overall iteration time
- **CG Solver Speedup**: 5-15x faster (see [benchmark results](BENCHMARK_RESULTS.md))
- **WASM Status**: ✅ Active - conjugate gradient solver runs in WASM

The cube pyramid test serves as the standard benchmark for comparing performance improvements. With full WASM integration, the most computationally expensive part (solving the linear system Ku=F) now runs with near-native performance. Run `npm run benchmark` to see detailed performance comparisons.

## Getting Started

Simply open `index.html` in a modern web browser served via a local HTTP server. ES modules require a server context.

```bash
# Clone the repository
git clone https://github.com/Jozo132/TopologyOptimization.git
cd TopologyOptimization

# Serve with any static HTTP server, for example:
npx http-server .          # Node.js (requires npm/npx)
# or
python3 -m http.server     # Python 3
# Then open http://localhost:8080 (or :8000 for Python) in your browser
```

### Optional: Build WASM Module

If you want to rebuild the WASM module:

```bash
npm install
npm run asbuild
```

### WASM Integration

The application now features **full WASM integration** for high-performance computation:

#### What's Accelerated with WASM

- **Conjugate Gradient Solver**: The iterative linear system solver (Ku=F) runs in WASM, providing significant speedup for large systems
- **Matrix Operations**: Dense matrix-vector multiplications use WASM's optimized implementations
- **Automatic Fallback**: If WASM fails to load, the application seamlessly falls back to pure JavaScript

#### Performance Benchmarks

Measured performance improvements (see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for details):

| Problem Size | JS Time | WASM Time | Speedup |
|--------------|---------|-----------|---------|
| 50×50 DOFs   | 0.71ms  | 0.14ms    | **4.97x** |
| 100×100 DOFs | 2.13ms  | 0.22ms    | **9.64x** |
| 500×500 DOFs | 80.58ms | 5.28ms    | **15.27x** |

**Average improvement: 77%** across all tested sizes

Run your own benchmarks:
```bash
npm run benchmark
```

#### How It Works

1. The WASM module (`wasm/matrix-ops.wasm`) is compiled from AssemblyScript source code
2. Web workers automatically load the WASM module when starting an optimization
3. Performance-critical functions use WASM when available, with zero-copy memory operations
4. The UI displays a WASM 🚀 badge when WASM acceleration is active

#### Building the WASM Module

The WASM module is pre-compiled and included in the repository. To rebuild from source:

```bash
# Install dependencies (includes AssemblyScript compiler)
npm install

# Build both debug and release versions
npm run asbuild

# The release build will be copied to wasm/matrix-ops.wasm
cp build/optimized.release.wasm wasm/matrix-ops.wasm
```

The source code is in `assembly/index.ts` and includes:
- `conjugateGradient()` - Iterative linear solver
- `matVecMul()` - Matrix-vector multiplication
- `applyDensityFilter()` - Density smoothing with spatial radius
- `computeElementEnergies()` - Strain energy computation
- And more utility functions

## Technology Stack

- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: No frameworks, pure ES6 modules
- **Web Workers**: Background thread for heavy computation
- **WebAssembly (WASM)**: High-performance math operations via AssemblyScript
- **Canvas 2D**: Interactive 3D visualization with no external dependencies
- **SIMP Algorithm**: Industry-standard topology optimization

## Algorithm

This application implements the SIMP (Solid Isotropic Material with Penalization) method for topology optimization:

1. **Finite Element Analysis**: Solves structural mechanics equations
   - 2D: 4-node quadrilateral elements (plane stress)
   - 3D: 8-node hexahedral elements (brick elements)
2. **Sensitivity Analysis**: Computes design sensitivities
3. **Density Filtering**: Smooths density field in 2D or 3D neighborhoods
4. **Optimality Criteria**: Updates design variables
5. **Convergence Check**: Iterates until convergence

### 3D Finite Element Analysis

The 3D optimizer uses:
- **8-node hexahedral elements**: Each element has 8 corner nodes
- **3 DOF per node**: x, y, z displacements
- **24×24 stiffness matrix**: Full 3D element formulation
- **Gauss quadrature**: 2×2×2 integration for accuracy
- **3D constitutive matrix**: Properly accounts for 3D stress-strain relationships

### Adaptive Mesh Resolution

Elements with higher strain energy or proximity to applied forces are subdivided into finer triangles during visualization. Low-energy regions use coarser triangles. This provides more visual detail where it matters most without uniformly increasing the mesh resolution.

## Development

### Building WASM Module

The project includes a fully integrated AssemblyScript WASM module for high-performance matrix operations:

```bash
# Install dependencies
npm install

# Build WASM module (debug and release)
npm run asbuild

# Copy the release build to the wasm directory
cp build/optimized.release.wasm wasm/matrix-ops.wasm

# Or build individually
npm run asbuild:debug   # Debuggable version with source maps
npm run asbuild:release # Optimized version (15KB)
```

The compiled WASM file is located at `wasm/matrix-ops.wasm` and is **actively used** by the optimizer workers:

**Integrated Functions:**
- ✅ **Conjugate gradient solver** - Solves Ku=F iteratively (primary performance boost)
- ✅ **Matrix-vector multiplication** - Dense matrix operations
- ✅ **Density filtering** - Spatial smoothing with radius
- ✅ **Element energy computation** - Strain energy calculations

**Memory Management:**
- Uses AssemblyScript's managed memory model with typed array headers
- Automatic memory allocation and deallocation via `__new`, `__pin`, `__unpin`
- Zero-copy data transfers between JavaScript and WASM where possible

**Source Code:**
- `assembly/index.ts` - WASM implementation
- `js/optimizer-worker.js` - 2D optimizer with WASM integration
- `js/optimizer-worker-3d.js` - 3D optimizer with WASM integration

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern mobile browsers

## License

See LICENSE file for details.
