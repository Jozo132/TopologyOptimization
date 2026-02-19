// WebGL2 GPGPU Compute Module for GPU-accelerated matrix operations
// Uses render-to-texture with floating-point textures for general-purpose GPU computing
// Falls back gracefully when WebGL2 or required extensions are not available

export class WebGLCompute {
    constructor() {
        this.gl = null;
        this.available = false;
        this._initPromise = null;
        this._programs = {};
    }

    /**
     * Initialize WebGL2 context and check availability
     * @param {HTMLCanvasElement} [canvas] - Optional canvas element; one is created if omitted
     * @returns {Promise<boolean>} Whether WebGL2 compute is available
     */
    async init(canvas) {
        if (this._initPromise) return this._initPromise;
        this._initPromise = this._doInit(canvas);
        return this._initPromise;
    }

    async _doInit(canvas) {
        try {
            if (typeof document === 'undefined') {
                console.log('WebGL2 requires a browser environment');
                return false;
            }

            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.width = 1;
                canvas.height = 1;
            }

            const gl = canvas.getContext('webgl2');
            if (!gl) {
                console.log('WebGL2 not supported');
                return false;
            }

            // Require float textures for GPGPU
            const extColorFloat = gl.getExtension('EXT_color_buffer_float');
            if (!extColorFloat) {
                console.log('EXT_color_buffer_float not supported');
                return false;
            }

            this.gl = gl;
            this._buildPrograms();
            this.available = true;
            console.log('WebGL2 GPGPU compute initialized successfully');
            return true;
        } catch (error) {
            console.log('WebGL2 initialization failed:', error.message);
            this.available = false;
            return false;
        }
    }

    /** Check if WebGL compute is available */
    isAvailable() {
        return this.available && this.gl !== null;
    }

    // ── internal helpers ─────────────────────────────────────────────

    _compileShader(src, type) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, src);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error('Shader compile error: ' + info);
        }
        return shader;
    }

    _linkProgram(vsSrc, fsSrc, tfVaryings) {
        const gl = this.gl;
        const vs = this._compileShader(vsSrc, gl.VERTEX_SHADER);
        const fs = this._compileShader(fsSrc, gl.FRAGMENT_SHADER);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        if (tfVaryings) {
            gl.transformFeedbackVaryings(prog, tfVaryings, gl.SEPARATE_ATTRIBS);
        }
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(prog);
            gl.deleteProgram(prog);
            throw new Error('Program link error: ' + info);
        }
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        return prog;
    }

    /** Full-screen quad vertex shader used by all render-to-texture passes */
    static _quadVS() {
        return `#version 300 es
        in vec2 a_pos;
        out vec2 v_uv;
        void main() {
            v_uv = a_pos * 0.5 + 0.5;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }`;
    }

    _createQuadVAO() {
        const gl = this.gl;
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
        return vao;
    }

    _createFloatTexture(width, height, data) {
        const gl = this.gl;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
        return tex;
    }

    _createFBO(tex) {
        const gl = this.gl;
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        return fbo;
    }

    _drawQuad() {
        const gl = this.gl;
        gl.bindVertexArray(this._quadVAO);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindVertexArray(null);
    }

    _readPixelsFloat(fbo, width, height) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        const out = new Float32Array(width * height);
        gl.readPixels(0, 0, width, height, gl.RED, gl.FLOAT, out);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return out;
    }

    // ── program compilation ──────────────────────────────────────────

    _buildPrograms() {
        const gl = this.gl;
        this._quadVAO = this._createQuadVAO();

        // ── Matrix-vector multiply (one row per fragment) ─────────
        this._programs.matVecMul = this._linkProgram(
            WebGLCompute._quadVS(),
            `#version 300 es
            precision highp float;
            uniform sampler2D u_matrix;
            uniform sampler2D u_vec;
            uniform int u_n;
            in vec2 v_uv;
            out float outVal;
            void main() {
                int row = int(gl_FragCoord.y) * int(textureSize(u_matrix, 0).x / u_n) + int(gl_FragCoord.x);
                if (row >= u_n) { outVal = 0.0; return; }
                float sum = 0.0;
                // GLSL requires a compile-time loop bound; 4096 is the max
                // supported n. The inner break exits early for smaller matrices.
                for (int j = 0; j < 4096; j++) {
                    if (j >= u_n) break;
                    int idx = row * u_n + j;
                    int my = idx / textureSize(u_matrix, 0).x;
                    int mx = idx - my * textureSize(u_matrix, 0).x;
                    float a = texelFetch(u_matrix, ivec2(mx, my), 0).r;
                    float b = texelFetch(u_vec, ivec2(j, 0), 0).r;
                    sum += a * b;
                }
                outVal = sum;
            }`
        );

        // ── Dot-product (parallel partial sums) ───────────────────
        this._programs.dotPartial = this._linkProgram(
            WebGLCompute._quadVS(),
            `#version 300 es
            precision highp float;
            uniform sampler2D u_a;
            uniform sampler2D u_b;
            uniform int u_n;
            in vec2 v_uv;
            out float outVal;
            void main() {
                int i = int(gl_FragCoord.y) * int(textureSize(u_a, 0).x) + int(gl_FragCoord.x);
                if (i >= u_n) { outVal = 0.0; return; }
                float a = texelFetch(u_a, ivec2(i, 0), 0).r;
                float b = texelFetch(u_b, ivec2(i, 0), 0).r;
                outVal = a * b;
            }`
        );

        // ── AXPY: result[i] = alpha * x[i] + y[i] ────────────────
        this._programs.axpy = this._linkProgram(
            WebGLCompute._quadVS(),
            `#version 300 es
            precision highp float;
            uniform sampler2D u_x;
            uniform sampler2D u_y;
            uniform float u_alpha;
            uniform int u_n;
            in vec2 v_uv;
            out float outVal;
            void main() {
                int i = int(gl_FragCoord.y) * int(textureSize(u_x, 0).x) + int(gl_FragCoord.x);
                if (i >= u_n) { outVal = 0.0; return; }
                float x = texelFetch(u_x, ivec2(i, 0), 0).r;
                float y = texelFetch(u_y, ivec2(i, 0), 0).r;
                outVal = u_alpha * x + y;
            }`
        );

        // ── Element energies ──────────────────────────────────────
        this._programs.elementEnergies = this._linkProgram(
            WebGLCompute._quadVS(),
            `#version 300 es
            precision highp float;
            uniform sampler2D u_U;
            uniform sampler2D u_KE;
            uniform sampler2D u_edofs;
            uniform int u_nel;
            uniform int u_edofSize;
            in vec2 v_uv;
            out float outVal;
            void main() {
                int el = int(gl_FragCoord.y) * int(textureSize(u_U, 0).x) + int(gl_FragCoord.x);
                if (el >= u_nel) { outVal = 0.0; return; }
                float energy = 0.0;
                // Max edofSize is 24 (8-node hex element × 3 DOFs).
                // The inner break exits early for smaller element types (e.g. 8 for 2D).
                for (int i = 0; i < 24; i++) {
                    if (i >= u_edofSize) break;
                    int edofIdx = el * u_edofSize + i;
                    int ey = edofIdx / textureSize(u_edofs, 0).x;
                    int ex = edofIdx - ey * textureSize(u_edofs, 0).x;
                    int dofI = int(texelFetch(u_edofs, ivec2(ex, ey), 0).r);
                    if (dofI < 0) continue;
                    float ui = texelFetch(u_U, ivec2(dofI, 0), 0).r;
                    for (int j = 0; j < 24; j++) {
                        if (j >= u_edofSize) break;
                        int edofIdx2 = el * u_edofSize + j;
                        int ey2 = edofIdx2 / textureSize(u_edofs, 0).x;
                        int ex2 = edofIdx2 - ey2 * textureSize(u_edofs, 0).x;
                        int dofJ = int(texelFetch(u_edofs, ivec2(ex2, ey2), 0).r);
                        if (dofJ < 0) continue;
                        float uj = texelFetch(u_U, ivec2(dofJ, 0), 0).r;
                        int keIdx = i * u_edofSize + j;
                        float ke = texelFetch(u_KE, ivec2(keIdx, 0), 0).r;
                        energy += ui * ke * uj;
                    }
                }
                outVal = energy;
            }`
        );
    }

    // ── public API ───────────────────────────────────────────────────

    /**
     * Dense matrix-vector multiply: result = A * x
     * @param {Float64Array} A - Dense matrix (n × n, row-major)
     * @param {Float64Array} x - Input vector (n)
     * @param {number} n - Dimension
     * @returns {Float64Array} Result vector
     */
    matVecMul(A, x, n) {
        if (!this.isAvailable()) throw new Error('WebGL compute not available');
        const gl = this.gl;
        const A32 = new Float32Array(A);
        const x32 = new Float32Array(x);

        // Pack matrix into a texture of width=n, height=n
        const matTex = this._createFloatTexture(n, n, A32);
        const vecTex = this._createFloatTexture(n, 1, x32);

        // Result texture (one row of n pixels)
        const resTex = this._createFloatTexture(n, 1, null);
        const fbo = this._createFBO(resTex);

        const prog = this._programs.matVecMul;
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, matTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_matrix'), 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, vecTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_vec'), 1);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_n'), n);

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.viewport(0, 0, n, 1);
        this._drawQuad();

        const res32 = this._readPixelsFloat(fbo, n, 1);
        const result = new Float64Array(res32);

        // Cleanup
        gl.deleteTexture(matTex);
        gl.deleteTexture(vecTex);
        gl.deleteTexture(resTex);
        gl.deleteFramebuffer(fbo);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return result;
    }

    /**
     * Dot product: returns a · b
     * Computed as parallel element-wise multiply + CPU reduction
     * @param {Float64Array} a - Vector a (n)
     * @param {Float64Array} b - Vector b (n)
     * @param {number} n - Length
     * @returns {number}
     */
    dotProduct(a, b, n) {
        if (!this.isAvailable()) throw new Error('WebGL compute not available');
        const gl = this.gl;
        const a32 = new Float32Array(a);
        const b32 = new Float32Array(b);

        const aTex = this._createFloatTexture(n, 1, a32);
        const bTex = this._createFloatTexture(n, 1, b32);
        const resTex = this._createFloatTexture(n, 1, null);
        const fbo = this._createFBO(resTex);

        const prog = this._programs.dotPartial;
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, aTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_a'), 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, bTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_b'), 1);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_n'), n);

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.viewport(0, 0, n, 1);
        this._drawQuad();

        const partial = this._readPixelsFloat(fbo, n, 1);

        // CPU reduction
        let sum = 0;
        for (let i = 0; i < n; i++) sum += partial[i];

        gl.deleteTexture(aTex);
        gl.deleteTexture(bTex);
        gl.deleteTexture(resTex);
        gl.deleteFramebuffer(fbo);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return sum;
    }

    /**
     * AXPY: result = alpha * x + y
     * @param {Float64Array} x - Vector x (n)
     * @param {Float64Array} y - Vector y (n)
     * @param {number} alpha - Scalar
     * @param {number} n - Length
     * @returns {Float64Array}
     */
    axpy(x, y, alpha, n) {
        if (!this.isAvailable()) throw new Error('WebGL compute not available');
        const gl = this.gl;
        const x32 = new Float32Array(x);
        const y32 = new Float32Array(y);

        const xTex = this._createFloatTexture(n, 1, x32);
        const yTex = this._createFloatTexture(n, 1, y32);
        const resTex = this._createFloatTexture(n, 1, null);
        const fbo = this._createFBO(resTex);

        const prog = this._programs.axpy;
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, xTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_x'), 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, yTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_y'), 1);
        gl.uniform1f(gl.getUniformLocation(prog, 'u_alpha'), alpha);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_n'), n);

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.viewport(0, 0, n, 1);
        this._drawQuad();

        const res32 = this._readPixelsFloat(fbo, n, 1);
        const result = new Float64Array(res32);

        gl.deleteTexture(xTex);
        gl.deleteTexture(yTex);
        gl.deleteTexture(resTex);
        gl.deleteFramebuffer(fbo);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return result;
    }

    /**
     * GPU-accelerated element energy computation
     * Computes u^T * KE * u for all elements in parallel
     * @param {Float32Array} U - Global displacement vector
     * @param {Float32Array} KEflat - Flattened element stiffness matrix
     * @param {Int32Array} edofs - Element DOF indices (nel * edofSize)
     * @param {number} nel - Number of elements
     * @param {number} edofSize - DOFs per element (8 for 2D, 24 for 3D)
     * @returns {Float32Array} Element energies
     */
    computeElementEnergies(U, KEflat, edofs, nel, edofSize) {
        if (!this.isAvailable()) throw new Error('WebGL compute not available');
        const gl = this.gl;

        const ndof = U.length;
        const uTex = this._createFloatTexture(ndof, 1, U);
        const keTex = this._createFloatTexture(edofSize * edofSize, 1, KEflat);
        // Store edofs as floats for texture sampling
        const edofsF = new Float32Array(edofs);
        const edofTex = this._createFloatTexture(edofs.length, 1, edofsF);

        const resTex = this._createFloatTexture(nel, 1, null);
        const fbo = this._createFBO(resTex);

        const prog = this._programs.elementEnergies;
        gl.useProgram(prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, uTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_U'), 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, keTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_KE'), 1);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, edofTex);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_edofs'), 2);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_nel'), nel);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_edofSize'), edofSize);

        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.viewport(0, 0, nel, 1);
        this._drawQuad();

        const result = this._readPixelsFloat(fbo, nel, 1);

        gl.deleteTexture(uTex);
        gl.deleteTexture(keTex);
        gl.deleteTexture(edofTex);
        gl.deleteTexture(resTex);
        gl.deleteFramebuffer(fbo);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return result;
    }

    /** Destroy WebGL resources */
    destroy() {
        if (this.gl) {
            const ext = this.gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
            this.gl = null;
        }
        this.available = false;
    }
}
