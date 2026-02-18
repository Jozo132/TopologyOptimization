// Topology Optimizer - thin wrapper around the unified TopologySolver library.
// Re-exports TopologySolver as TopologyOptimizer for backward compatibility.
// New code should import TopologySolver from '../lib/topology-solver.js' directly.
import { TopologySolver } from '../lib/topology-solver.js';

export { TopologySolver as TopologyOptimizer };
export { TopologySolver };
