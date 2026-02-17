# Topology Optimization Web App

A minimalistic but fully functional web-based topology optimization application using vanilla JavaScript and ES6 modules.

## Features

- 🚀 **4-Step Workflow**: Import → Assign → Solve → Export
- 🎨 **3D Visualization**: Real-time 3D rendering using Canvas 2D with interactive controls
- 🔧 **SIMP Algorithm**: Solid Isotropic Material with Penalization
- 🎯 **Full 3D Optimization**: True 3D FEA with 8-node hexahedral elements
- 📦 **STL Support**: Import and export STL files
- 🎯 **Template Models**: Quick start with cantilever beam, bridge, or cube test templates
- 💾 **Export Options**: Download optimized STL or JSON data
- ⚡ **Web Worker Optimization**: Runs optimization in a background thread, keeping the UI responsive
- ⚡ **WASM Acceleration**: High-performance AssemblyScript WASM module for matrix operations (15KB)
- ❌ **Cancellable Optimization**: Cancel a running optimization at any point
- 🔬 **Adaptive Mesh**: Dynamically refines mesh resolution in high-force / high-energy regions
- 🔺 **Triangle Mesh Rendering**: Displays elements as proper triangulated mesh, not just squares
- 🎚️ **Min Cross-Section Control**: Optional minimum feature size enforcement

## Technology Stack

- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: No frameworks, pure ES6 modules
- **Web Workers**: Background thread for heavy computation
- **WebAssembly**: High-performance matrix operations via AssemblyScript
- **Canvas 2D**: Interactive 3D visualization with no external dependencies
- **SIMP Algorithm**: Industry-standard topology optimization

## New in This Version

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

### ⚡ WASM Performance Module
- Pre-compiled AssemblyScript WASM library (only 15KB)
- Optimized matrix-vector multiplication
- Conjugate gradient solver
- Density filtering operations
- Element energy computation
- Ready for integration (currently using pure JS for compatibility)

### 🎚️ Minimum Cross-Section Control
- UI control for setting minimum feature size
- Prevents thin, unbuildable members
- Configurable from 0 (disabled) to 5 elements

## Usage

1. **Import Model**
   - Upload an STL file, or
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

## Technology Stack

- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: No frameworks, pure ES6 modules
- **Web Workers**: Background thread for heavy computation
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

The project includes an AssemblyScript WASM module for high-performance matrix operations:

```bash
# Install dependencies
npm install

# Build WASM module (debug and release)
npm run asbuild

# Or build individually
npm run asbuild:debug   # Debuggable version
npm run asbuild:release # Optimized version (15KB)
```

The compiled WASM file is located at `wasm/matrix-ops.wasm` and includes:
- Matrix-vector multiplication
- Conjugate gradient solver
- Density filtering
- Element energy computation

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern mobile browsers

## License

See LICENSE file for details.
