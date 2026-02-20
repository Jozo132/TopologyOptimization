// Marching Cubes isosurface extraction for smooth voxel rendering
// Produces a smooth triangle mesh from a 3D density field at a given threshold.
//
// Algorithm:
//   For each cell in the (nx-1)×(ny-1)×(nz-1) dual grid of voxel corners,
//   look up the 8 corner density values, classify them as inside/outside
//   the isosurface, then emit interpolated triangles from the MC lookup tables.
//
// The density field is defined at voxel CENTERS. We sample at voxel CORNERS
// by averaging the densities of the up-to-8 voxels sharing that corner, giving
// a smooth continuous field suitable for isosurface extraction.
//
// Output: { positions: Float32Array, normals: Float32Array, indices: Uint32Array }

import { DENSITY_THRESHOLD } from './constants.js';

// ─── Marching Cubes lookup tables ───────────────────────────────────────────
// Edge table: for each of the 256 cube configurations, a 12-bit mask of which
// edges are intersected by the isosurface.
// prettier-ignore
const EDGE_TABLE = [
    0x0,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x99,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x33,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0xaa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x66,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x55,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0
];

// Triangle table: for each configuration, up to 5 triangles (15 edge indices),
// terminated by -1.
// prettier-ignore
const TRI_TABLE = [
    [-1],
    [0,8,3,-1],
    [0,1,9,-1],
    [1,8,3,9,8,1,-1],
    [1,2,10,-1],
    [0,8,3,1,2,10,-1],
    [9,2,10,0,2,9,-1],
    [2,8,3,2,10,8,10,9,8,-1],
    [3,11,2,-1],
    [0,11,2,8,11,0,-1],
    [1,9,0,2,3,11,-1],
    [1,11,2,1,9,11,9,8,11,-1],
    [3,10,1,11,10,3,-1],
    [0,10,1,0,8,10,8,11,10,-1],
    [3,9,0,3,11,9,11,10,9,-1],
    [9,8,10,10,8,11,-1],
    [4,7,8,-1],
    [4,3,0,7,3,4,-1],
    [0,1,9,8,4,7,-1],
    [4,1,9,4,7,1,7,3,1,-1],
    [1,2,10,8,4,7,-1],
    [3,4,7,3,0,4,1,2,10,-1],
    [9,2,10,9,0,2,8,4,7,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1],
    [8,4,7,3,11,2,-1],
    [11,4,7,11,2,4,2,0,4,-1],
    [9,0,1,8,4,7,2,3,11,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1],
    [3,10,1,3,11,10,7,8,4,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1],
    [4,7,11,4,11,9,9,11,10,-1],
    [9,5,4,-1],
    [9,5,4,0,8,3,-1],
    [0,5,4,1,5,0,-1],
    [8,5,4,8,3,5,3,1,5,-1],
    [1,2,10,9,5,4,-1],
    [3,0,8,1,2,10,4,9,5,-1],
    [5,2,10,5,4,2,4,0,2,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1],
    [9,5,4,2,3,11,-1],
    [0,11,2,0,8,11,4,9,5,-1],
    [0,5,4,0,1,5,2,3,11,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1],
    [10,3,11,10,1,3,9,5,4,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1],
    [5,4,8,5,8,10,10,8,11,-1],
    [9,7,8,5,7,9,-1],
    [9,3,0,9,5,3,5,7,3,-1],
    [0,7,8,0,1,7,1,5,7,-1],
    [1,5,3,3,5,7,-1],
    [9,7,8,9,5,7,10,1,2,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1],
    [2,10,5,2,5,3,3,5,7,-1],
    [7,9,5,7,8,9,3,11,2,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1],
    [11,2,1,11,1,7,7,1,5,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1],
    [10,6,5,-1],
    [0,8,3,5,10,6,-1],
    [9,0,1,5,10,6,-1],
    [1,8,3,1,9,8,5,10,6,-1],
    [1,6,5,2,6,1,-1],
    [1,6,5,1,2,6,3,0,8,-1],
    [9,6,5,9,0,6,0,2,6,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1],
    [2,3,11,10,6,5,-1],
    [11,0,8,11,2,0,10,6,5,-1],
    [0,1,9,2,3,11,5,10,6,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1],
    [6,3,11,6,5,3,5,1,3,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1],
    [6,5,9,6,9,11,11,9,8,-1],
    [5,10,6,4,7,8,-1],
    [4,3,0,4,7,3,6,5,10,-1],
    [1,9,0,5,10,6,8,4,7,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1],
    [6,1,2,6,5,1,4,7,8,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1],
    [10,4,9,6,4,10,-1],
    [4,10,6,4,9,10,0,8,3,-1],
    [10,0,1,10,6,0,6,4,0,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1],
    [1,4,9,1,2,4,2,6,4,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1],
    [0,2,4,4,2,6,-1],
    [8,3,2,8,2,4,4,2,6,-1],
    [10,4,9,10,6,4,11,2,3,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1],
    [6,4,8,11,6,8,-1],
    [7,10,6,7,8,10,8,9,10,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1],
    [10,6,7,10,7,1,1,7,3,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1],
    [7,3,2,6,7,2,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1],
    [7,11,6,-1],
    [7,6,11,-1],
    [3,0,8,11,7,6,-1],
    [0,1,9,11,7,6,-1],
    [8,1,9,8,3,1,11,7,6,-1],
    [10,1,2,6,11,7,-1],
    [1,2,10,3,0,8,6,11,7,-1],
    [2,9,0,2,10,9,6,11,7,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1],
    [7,2,3,6,2,7,-1],
    [7,0,8,7,6,0,6,2,0,-1],
    [2,7,6,2,3,7,0,1,9,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1],
    [10,7,6,10,1,7,1,3,7,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1],
    [7,6,10,7,10,8,8,10,9,-1],
    [6,8,4,11,8,6,-1],
    [3,6,11,3,0,6,0,4,6,-1],
    [8,6,11,8,4,6,9,0,1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1],
    [6,8,4,6,11,8,2,10,1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1],
    [0,4,2,4,6,2,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1],
    [1,9,4,1,4,2,2,4,6,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1],
    [10,1,0,10,0,6,6,0,4,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1],
    [4,9,5,7,6,11,-1],
    [0,8,3,4,9,5,11,7,6,-1],
    [5,0,1,5,4,0,7,6,11,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1],
    [9,5,4,10,1,2,7,6,11,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1],
    [6,9,5,6,11,9,11,8,9,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1],
    [6,11,3,6,3,5,5,3,1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1],
    [9,5,6,9,6,0,0,6,2,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1],
    [0,3,8,5,6,10,-1],
    [10,5,6,-1],
    [11,5,10,7,5,11,-1],
    [11,5,10,11,7,5,8,3,0,-1],
    [5,11,7,5,10,11,1,9,0,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1],
    [11,1,2,11,7,1,7,5,1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1],
    [0,8,7,0,7,1,1,7,5,-1],
    [9,0,3,9,3,5,5,3,7,-1],
    [9,8,7,5,9,7,-1],
    [5,8,4,5,10,8,10,11,8,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1],
    [5,10,2,5,2,4,4,2,0,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1],
    [8,4,5,8,5,3,3,5,1,-1],
    [0,4,5,1,0,5,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1],
    [9,4,5,-1],
    [4,11,7,4,9,11,9,10,11,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1],
    [4,9,1,4,1,7,7,1,3,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1],
    [4,0,3,7,4,3,-1],
    [4,8,7,-1],
    [9,10,8,10,11,8,-1],
    [3,0,9,3,9,11,11,9,10,-1],
    [0,1,10,0,10,8,8,10,11,-1],
    [3,1,10,11,3,10,-1],
    [1,2,11,1,11,9,9,11,8,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1],
    [0,2,11,8,0,11,-1],
    [3,2,11,-1],
    [2,3,8,2,8,10,10,8,9,-1],
    [9,10,2,0,9,2,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1],
    [1,10,2,-1],
    [1,3,8,9,1,8,-1],
    [0,9,1,-1],
    [0,3,8,-1],
    [-1]
];

// Edge vertex pairs: each of the 12 edges connects two of the 8 cube corners
const EDGE_VERTICES = [
    [0,1],[1,2],[2,3],[3,0], // bottom face edges
    [4,5],[5,6],[6,7],[7,4], // top face edges
    [0,4],[1,5],[2,6],[3,7]  // vertical edges
];

// Cube corner offsets: 8 corners of a unit cube
const CORNER_OFFSETS = [
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]
];

/**
 * Sample the density field at a voxel corner by averaging the densities
 * of the (up to 8) voxels sharing that corner. This produces a smooth
 * continuous scalar field suitable for isosurface extraction.
 *
 * Corner (cx, cy, cz) is shared by voxels (cx-1..cx, cy-1..cy, cz-1..cz).
 */
function buildCornerField(densities, nx, ny, nz) {
    // Build corner field with 1-cell zero padding on each side.
    // Interior corners use divide-by-count for accurate averaging.
    // Padding creates natural isosurface transitions at domain boundaries.
    //
    // Padded grid: (nx+3) × (ny+3) × (nz+3)
    // Padded index (px,py,pz) → original corner (px-1, py-1, pz-1)
    const cnx = nx + 3, cny = ny + 3, cnz = nz + 3;
    const field = new Float32Array(cnx * cny * cnz); // zeros = padding

    // Fill original corners at padded positions (1..nx+1, 1..ny+1, 1..nz+1)
    for (let cz = 0; cz <= nz; cz++) {
        for (let cy = 0; cy <= ny; cy++) {
            for (let cx = 0; cx <= nx; cx++) {
                let sum = 0, count = 0;
                for (let dz = -1; dz <= 0; dz++) {
                    for (let dy = -1; dy <= 0; dy++) {
                        for (let dx = -1; dx <= 0; dx++) {
                            const vx = cx + dx, vy = cy + dy, vz = cz + dz;
                            if (vx >= 0 && vx < nx && vy >= 0 && vy < ny && vz >= 0 && vz < nz) {
                                sum += densities[vx + vy * nx + vz * nx * ny];
                                count++;
                            }
                        }
                    }
                }
                field[(cx + 1) + (cy + 1) * cnx + (cz + 1) * cnx * cny] = count > 0 ? sum / count : 0;
            }
        }
    }
    return field;
}

/**
 * Compute a gradient-based normal at a corner position by central differences
 * on the corner field. Returns normalized vector.
 */
function cornerNormal(field, cx, cy, cz, cnx, cny, cnz) {
    const idx = (x, y, z) => {
        x = Math.max(0, Math.min(cnx - 1, x));
        y = Math.max(0, Math.min(cny - 1, y));
        z = Math.max(0, Math.min(cnz - 1, z));
        return field[x + y * cnx + z * cnx * cny];
    };
    // Central differences for gradient
    const gx = idx(cx + 1, cy, cz) - idx(cx - 1, cy, cz);
    const gy = idx(cx, cy + 1, cz) - idx(cx, cy - 1, cz);
    const gz = idx(cx, cy, cz + 1) - idx(cx, cy, cz - 1);
    const len = Math.sqrt(gx * gx + gy * gy + gz * gz);
    if (len < 1e-12) return [0, 1, 0];
    // Negate: gradient points from low→high, normal should point from high→low (outward)
    return [-gx / len, -gy / len, -gz / len];
}

/**
 * Generate a smooth isosurface mesh from a 3D density field using Marching Cubes.
 *
 * @param {object} options
 * @param {Float32Array} options.densities - Density per voxel (nx*ny*nz), voxel-center values
 * @param {number} options.nx - Grid size X
 * @param {number} options.ny - Grid size Y
 * @param {number} options.nz - Grid size Z
 * @param {number} [options.threshold] - Isosurface threshold (default DENSITY_THRESHOLD)
 * @param {Uint8Array} [options.visible] - Pre-built visibility mask (if provided, used as binary field: 1→density=1, 0→density=0)
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint32Array }}
 */
export function marchingCubes(options) {
    const { densities, nx, ny, nz } = options;
    const threshold = options.threshold !== undefined ? options.threshold : DENSITY_THRESHOLD;
    const visibleMask = options.visible || null;

    // Build the scalar field at voxel corners by averaging neighbor densities
    let srcDensities = densities;
    if (visibleMask) {
        // Convert binary mask to a synthetic density field
        srcDensities = new Float32Array(nx * ny * nz);
        for (let i = 0; i < srcDensities.length; i++) {
            srcDensities[i] = visibleMask[i] ? (densities ? densities[i] : 1.0) : 0.0;
        }
    }
    const field = buildCornerField(srcDensities, nx, ny, nz);
    // Padded corner grid dimensions
    const cnx = nx + 3, cny = ny + 3, cnz = nz + 3;
    // MC cells cover the padded grid: (cnx-1) × (cny-1) × (cnz-1) cells
    const numCellsX = cnx - 1, numCellsY = cny - 1, numCellsZ = cnz - 1;

    // Vertex dedup via edge-based hash (each vertex lies on exactly one edge)
    const vertexMap = new Map();
    const vertPositions = [];
    const vertNormals = [];
    const triIndices = [];

    function getEdgeVertex(cx0, cy0, cz0, cx1, cy1, cz1, v0, v1) {
        // Canonical edge key (smaller corner first) - all coords are non-negative in padded space
        const k0 = cx0 + cy0 * cnx + cz0 * cnx * cny;
        const k1 = cx1 + cy1 * cnx + cz1 * cnx * cny;
        const key = k0 < k1 ? k0 * cnx * cny * cnz + k1 : k1 * cnx * cny * cnz + k0;

        let idx = vertexMap.get(key);
        if (idx !== undefined) return idx;

        // Interpolate position along edge, then offset by -1 to convert
        // from padded coordinates back to original voxel-grid coordinates
        const t = (threshold - v0) / (v1 - v0);
        const px = cx0 + t * (cx1 - cx0) - 1;
        const py = cy0 + t * (cy1 - cy0) - 1;
        const pz = cz0 + t * (cz1 - cz0) - 1;

        // Interpolate normal from corner gradients
        const n0 = cornerNormal(field, cx0, cy0, cz0, cnx, cny, cnz);
        const n1 = cornerNormal(field, cx1, cy1, cz1, cnx, cny, cnz);
        let nnx = n0[0] + t * (n1[0] - n0[0]);
        let nny = n0[1] + t * (n1[1] - n0[1]);
        let nnz = n0[2] + t * (n1[2] - n0[2]);
        const nlen = Math.sqrt(nnx * nnx + nny * nny + nnz * nnz);
        if (nlen > 1e-12) { nnx /= nlen; nny /= nlen; nnz /= nlen; }

        idx = vertPositions.length / 3;
        vertexMap.set(key, idx);
        vertPositions.push(px, py, pz);
        vertNormals.push(nnx, nny, nnz);
        return idx;
    }

    // March through each cell in the padded grid
    for (let z = 0; z < numCellsZ; z++) {
        for (let y = 0; y < numCellsY; y++) {
            for (let x = 0; x < numCellsX; x++) {
                // 8 corner values from padded field
                const vals = new Float64Array(8);
                for (let c = 0; c < 8; c++) {
                    const ox = CORNER_OFFSETS[c][0], oy = CORNER_OFFSETS[c][1], oz = CORNER_OFFSETS[c][2];
                    vals[c] = field[(x + ox) + (y + oy) * cnx + (z + oz) * cnx * cny];
                }

                // Compute cube index (strict > so zero-padding stays "outside" at threshold 0)
                let cubeIndex = 0;
                for (let c = 0; c < 8; c++) {
                    if (vals[c] > threshold) cubeIndex |= (1 << c);
                }

                const edgeMask = EDGE_TABLE[cubeIndex];
                if (edgeMask === 0) continue;

                // Compute edge vertex indices (only for edges that are crossed)
                const edgeVerts = new Int32Array(12).fill(-1);
                for (let e = 0; e < 12; e++) {
                    if (!(edgeMask & (1 << e))) continue;
                    const [c0, c1] = EDGE_VERTICES[e];
                    const cx0 = x + CORNER_OFFSETS[c0][0], cy0 = y + CORNER_OFFSETS[c0][1], cz0 = z + CORNER_OFFSETS[c0][2];
                    const cx1 = x + CORNER_OFFSETS[c1][0], cy1 = y + CORNER_OFFSETS[c1][1], cz1 = z + CORNER_OFFSETS[c1][2];
                    edgeVerts[e] = getEdgeVertex(cx0, cy0, cz0, cx1, cy1, cz1, vals[c0], vals[c1]);
                }

                // Emit triangles (swap v1/v2 to flip winding from CW to CCW for WebGL front-face)
                const triList = TRI_TABLE[cubeIndex];
                for (let i = 0; i < triList.length && triList[i] !== -1; i += 3) {
                    triIndices.push(edgeVerts[triList[i]], edgeVerts[triList[i + 2]], edgeVerts[triList[i + 1]]);
                }
            }
        }
    }

    const positions = new Float32Array(vertPositions);
    const normals = new Float32Array(vertNormals);
    const indices = new Uint32Array(triIndices);

    return { positions, normals, indices };
}
