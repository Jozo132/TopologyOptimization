# Topology Optimization Web App

A minimalistic but fully functional web-based topology optimization application using vanilla JavaScript and ES6 modules.

## Features

- 🚀 **4-Step Workflow**: Import → Assign → Solve → Export
- 🎨 **3D Visualization**: Real-time 3D rendering using Canvas 2D with interactive controls
- 🔧 **SIMP Algorithm**: Solid Isotropic Material with Penalization
- 📦 **STL Support**: Import and export STL files
- 🎯 **Template Models**: Quick start with cantilever beam or bridge templates
- 💾 **Export Options**: Download optimized STL or JSON data

## Usage

1. **Import Model**
   - Upload an STL file, or
   - Use a pre-defined template (Cantilever Beam or Bridge)

2. **Assign Loads & Constraints**
   - Set volume fraction (target material percentage)
   - Choose force direction and magnitude
   - Select constraint position (fixed boundary)

3. **Solve Optimization**
   - Configure optimization parameters:
     - Max iterations (default: 100)
     - Penalty factor (default: 3)
     - Filter radius (default: 1.5)
   - Run the optimization and watch progress

4. **Export Result**
   - Download optimized model as STL file
   - Export optimization data as JSON
   - Start over for a new optimization

## Getting Started

Simply open `index.html` in a modern web browser. No build step or server required!

```bash
# Clone the repository
git clone https://github.com/Jozo132/TopologyOptimization.git
cd TopologyOptimization

# Open in browser
open index.html  # macOS
# or
start index.html  # Windows
# or
xdg-open index.html  # Linux
```

## Technology Stack

- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: No frameworks, pure ES6 modules
- **Canvas 2D**: Interactive 3D visualization with no external dependencies
- **SIMP Algorithm**: Industry-standard topology optimization

## Algorithm

This application implements the SIMP (Solid Isotropic Material with Penalization) method for topology optimization:

1. **Finite Element Analysis**: Solves structural mechanics equations
2. **Sensitivity Analysis**: Computes design sensitivities
3. **Density Filtering**: Smooths density field
4. **Optimality Criteria**: Updates design variables
5. **Convergence Check**: Iterates until convergence

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern mobile browsers

## License

See LICENSE file for details.
