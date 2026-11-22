#include "inkling2.cuh"
#include "vecdotq.cuh"
#include "quantize.cuh"
#include "/home/admin/.t3/projects/inkling/kernels/inkling2_rtx3090_kernels.cuh"

// ============================================================================
// Inkling-2-Mini Custom RTX 3090 (sm_86) CUDA Kernel Dispatch + Fused MoE
// ============================================================================

// Generic fallbacks for non-2304 / non-128 / non-16 shapes
static __global__ void inkling2_attn_res_generic_kernel(
        const inkling2_cuda::AttnResPtrs states,
        const float * __restrict__ query,
        const float * __restrict__ post_norm_w,
        float * __restrict__ dst,
        const int n_embd,
        const int n_states,
        const int n_queries,
        const int n_tokens,
        const bool has_post_norm,
        const float eps) {
    const int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarps = blockDim.x >> 5;

    __shared__ float smem_sq[8][8];
    __shared__ float smem_dot[4][8][8];
    __shared__ float smem_w[4][8];

    float sq[8] = {0};
    float dot[4][8] = {{0}};

    for (int d = tid; d < n_embd; d += blockDim.x) {
        float qv[4];
        for (int q = 0; q < n_queries; ++q) qv[q] = query[q * n_embd + d];
        for (int s = 0; s < n_states; ++s) {
            const float xv = states.s[s][int64_t(tok) * n_embd + d];
            sq[s] = fmaf(xv, xv, sq[s]);
            for (int q = 0; q < n_queries; ++q) dot[q][s] = fmaf(xv, qv[q], dot[q][s]);
        }
    }
    for (int s = 0; s < n_states; ++s) {
        float wsq = inkling2_cuda::warp_sum32(sq[s]);
        if (lane == 0) smem_sq[s][warp] = wsq;
        for (int q = 0; q < n_queries; ++q) {
            float wd = inkling2_cuda::warp_sum32(dot[q][s]);
            if (lane == 0) smem_dot[q][s][warp] = wd;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float inv_rms[8];
        const float inv_d = 1.0f / float(n_embd);
        const float inv_sqrt_d = rsqrtf(float(n_embd));
        for (int s = 0; s < n_states; ++s) {
            float tot = 0.0f;
            for (int w = 0; w < nwarps; ++w) tot += smem_sq[s][w];
            inv_rms[s] = rsqrtf(tot * inv_d + eps);
        }
        for (int q = 0; q < n_queries; ++q) {
            float sc[8], mx = -1e30f;
            for (int s = 0; s < n_states; ++s) {
                float td = 0.0f;
                for (int w = 0; w < nwarps; ++w) td += smem_dot[q][s][w];
                sc[s] = td * inv_rms[s] * inv_sqrt_d;
                mx = fmaxf(mx, sc[s]);
            }
            float se = 0.0f;
            for (int s = 0; s < n_states; ++s) { sc[s] = expf(sc[s] - mx); se += sc[s]; }
            const float inv_se = 1.0f / se;
            for (int s = 0; s < n_states; ++s) smem_w[q][s] = sc[s] * inv_se * inv_rms[s];
        }
    }
    __syncthreads();

    if (has_post_norm && n_queries == 1) {
        float osq = 0.0f;
        for (int d = tid; d < n_embd; d += blockDim.x) {
            float acc = 0.0f;
            for (int s = 0; s < n_states; ++s) acc = fmaf(smem_w[0][s], states.s[s][int64_t(tok) * n_embd + d], acc);
            dst[int64_t(tok) * n_embd + d] = acc;
            osq = fmaf(acc, acc, osq);
        }
        float wosq = inkling2_cuda::warp_sum32(osq);
        if (lane == 0) smem_sq[0][warp] = wosq;
        __syncthreads();
        __shared__ float post_ir;
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < nwarps; ++w) tot += smem_sq[0][w];
            post_ir = rsqrtf(tot / float(n_embd) + eps);
        }
        __syncthreads();
        for (int d = tid; d < n_embd; d += blockDim.x) {
            dst[int64_t(tok) * n_embd + d] *= post_ir * post_norm_w[d];
        }
    } else {
        for (int q = 0; q < n_queries; ++q) {
            float * out = dst + (int64_t(tok) * n_queries + q) * n_embd;
            for (int d = tid; d < n_embd; d += blockDim.x) {
                float acc = 0.0f;
                for (int s = 0; s < n_states; ++s) acc = fmaf(smem_w[q][s], states.s[s][int64_t(tok) * n_embd + d], acc);
                out[d] = acc;
            }
        }
    }
}

void ggml_cuda_op_inkling_attn_res(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n_states      = ggml_get_op_params_i32(dst, 0);
    const bool has_post_norm = ggml_get_op_params_i32(dst, 1) != 0;
    const float eps         = ggml_get_op_params_f32(dst, 2);

    const ggml_tensor * query = dst->src[0];
    const int n_embd    = (int) query->ne[0];
    const int n_queries = (int) query->ne[1];
    const int n_tokens  = (int) dst->src[1]->ne[1];

    inkling2_cuda::AttnResPtrs ptrs = {};
    for (int i = 0; i < n_states; ++i) {
        ptrs.s[i] = (const float *) dst->src[1 + i]->data;
    }
    const float * post_norm_w = has_post_norm ? (const float *) dst->src[1 + n_states]->data : nullptr;
    const float * q_d   = (const float *) query->data;
    float *       dst_d = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

#define DISPATCH_ATTN_RES(NS, NQ) \
    if (has_post_norm && (NQ) == 1) { \
        inkling2_cuda::inkling2_attn_res_d2304_kernel<NS, 1, true><<<n_tokens, 192, 0, stream>>>(ptrs, q_d, post_norm_w, dst_d, n_tokens, eps); \
    } else { \
        inkling2_cuda::inkling2_attn_res_d2304_kernel<NS, NQ, false><<<n_tokens, 192, 0, stream>>>(ptrs, q_d, nullptr, dst_d, n_tokens, eps); \
    }

    if (n_embd == 2304 && n_states >= 1 && n_states <= 5 && (n_queries == 1 || n_queries == 3 || n_queries == 4)) {
        if (n_queries == 1) {
            switch (n_states) {
                case 1: DISPATCH_ATTN_RES(1, 1); return;
                case 2: DISPATCH_ATTN_RES(2, 1); return;
                case 3: DISPATCH_ATTN_RES(3, 1); return;
                case 4: DISPATCH_ATTN_RES(4, 1); return;
                case 5: DISPATCH_ATTN_RES(5, 1); return;
            }
        } else if (n_queries == 3) {
            switch (n_states) {
                case 1: DISPATCH_ATTN_RES(1, 3); return;
                case 2: DISPATCH_ATTN_RES(2, 3); return;
                case 3: DISPATCH_ATTN_RES(3, 3); return;
                case 4: DISPATCH_ATTN_RES(4, 3); return;
                case 5: DISPATCH_ATTN_RES(5, 3); return;
            }
        } else if (n_queries == 4) {
            switch (n_states) {
                case 1: DISPATCH_ATTN_RES(1, 4); return;
                case 2: DISPATCH_ATTN_RES(2, 4); return;
                case 3: DISPATCH_ATTN_RES(3, 4); return;
                case 4: DISPATCH_ATTN_RES(4, 4); return;
                case 5: DISPATCH_ATTN_RES(5, 4); return;
            }
        }
    }
#undef DISPATCH_ATTN_RES

    inkling2_attn_res_generic_kernel<<<n_tokens, 256, 0, stream>>>(
        ptrs, q_d, post_norm_w, dst_d, n_embd, n_states, n_queries, n_tokens, has_post_norm, eps);
}

static __global__ void inkling2_l2_clip_generic_kernel(
        const float * __restrict__ src, const float * __restrict__ scale, const float * __restrict__ tau,
        float * __restrict__ dst, const int D, const int n_heads, const int n_tokens, const float threshold) {
    const int row = blockIdx.x;
    if (row >= n_heads * n_tokens) return;
    const int head = row % n_heads;
    const int tok  = row / n_heads;
    const int lane = threadIdx.x;
    float sq = 0.0f;
    for (int d = lane; d < D; d += 32) {
        float v = src[int64_t(row) * D + d];
        if (scale) v *= scale[head * D + d];
        sq = fmaf(v, v, sq);
    }
    sq = inkling2_cuda::warp_sum32(sq);
    float f = fminf(1.0f, threshold / (sqrtf(sq) + 1e-6f));
    if (tau) f *= tau[tok];
    for (int d = lane; d < D; d += 32) {
        float v = src[int64_t(row) * D + d];
        if (scale) v *= scale[head * D + d];
        dst[int64_t(row) * D + d] = v * f;
    }
}

void ggml_cuda_op_inkling_l2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int flags     = ggml_get_op_params_i32(dst, 0);
    const float thresh  = ggml_get_op_params_f32(dst, 1);
    const bool has_sc   = (flags & 1) != 0;
    const bool has_tau  = (flags & 2) != 0;

    const ggml_tensor * x = dst->src[0];
    int s = 1;
    const float * sc_d  = has_sc  ? (const float *) dst->src[s++]->data : nullptr;
    const float * tau_d = has_tau ? (const float *) dst->src[s++]->data : nullptr;
    const float * x_d   = (const float *) x->data;
    float *       dst_d = (float *) dst->data;

    const int D       = (int) x->ne[0];
    const int n_heads = (int) x->ne[1];
    const int n_tok   = (int) x->ne[2];
    const int nrows   = n_heads * n_tok;
    cudaStream_t stream = ctx.stream();

    if (D == 128) {
        const int blocks = (nrows + 3) / 4;
        if (has_sc && has_tau)      inkling2_cuda::inkling2_l2_clip_d128_kernel<true,  true ><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
        else if (has_sc)            inkling2_cuda::inkling2_l2_clip_d128_kernel<true,  false><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
        else if (has_tau)           inkling2_cuda::inkling2_l2_clip_d128_kernel<false, true ><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
        else                        inkling2_cuda::inkling2_l2_clip_d128_kernel<false, false><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
    } else if (D == 16) {
        const int blocks = (nrows + 15) / 16;
        if (has_sc && has_tau)      inkling2_cuda::inkling2_l2_clip_d16_kernel<true,  true ><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
        else if (has_sc)            inkling2_cuda::inkling2_l2_clip_d16_kernel<true,  false><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
        else if (has_tau)           inkling2_cuda::inkling2_l2_clip_d16_kernel<false, true ><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
        else                        inkling2_cuda::inkling2_l2_clip_d16_kernel<false, false><<<blocks, 128, 0, stream>>>(x_d, sc_d, tau_d, dst_d, n_heads, n_tok, thresh);
    } else {
        inkling2_l2_clip_generic_kernel<<<nrows, 32, 0, stream>>>(x_d, sc_d, tau_d, dst_d, D, n_heads, n_tok, thresh);
    }
}

void ggml_cuda_op_inkling_short_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n_seq_tokens     = ggml_get_op_params_i32(dst, 0);
    const int n_seqs           = ggml_get_op_params_i32(dst, 1);
    const int hist_chan_offset = ggml_get_op_params_i32(dst, 2);
    const int state_head       = ggml_get_op_params_i32(dst, 3);
    const int state_size       = ggml_get_op_params_i32(dst, 4);
    const int n_rs_seq         = ggml_get_op_params_i32(dst, 5);
    const bool has_dyn         = ggml_get_op_params_i32(dst, 6) != 0;
    const float inv_scale      = ggml_get_op_params_f32(dst, 7);

    const ggml_tensor * cur     = dst->src[0];
    const ggml_tensor * history = dst->src[1];
    const ggml_tensor * weight  = dst->src[2];
    const ggml_tensor * state   = dst->src[3];
    const ggml_tensor * dynamic = has_dyn ? dst->src[4] : nullptr;

    const int n_channels = (int) cur->ne[0];
    const int n_heads    = has_dyn ? (int) (dynamic->ne[0] / 4) : 1;
    const int hist_stride_chan = (int) (history->nb[1] / sizeof(float));
    const int hist_stride_seq  = (int) (history->nb[2] / sizeof(float));
    const int64_t state_row_stride_f  = (int64_t) (state->nb[1] / sizeof(float));
    const int64_t state_slot_stride_f = int64_t(state_size) * state_row_stride_f;
    const int64_t state_cell_offs_f   = int64_t(state_head) * state_row_stride_f + int64_t(hist_chan_offset) * 3;

    const dim3 blocks((n_channels + 255) / 256, n_seqs);
    cudaStream_t stream = ctx.stream();

    if (has_dyn) {
        inkling2_cuda::inkling2_short_conv_f32_kernel<true><<<blocks, 256, 0, stream>>>(
            (const float *) cur->data, (const float *) history->data, (const float *) weight->data,
            (const float *) dynamic->data, (float *) dst->data, (float *) state->data,
            n_channels, n_seq_tokens, n_seqs, n_heads,
            hist_stride_chan, hist_stride_seq, hist_chan_offset,
            state_row_stride_f, state_slot_stride_f, state_cell_offs_f, n_rs_seq, inv_scale);
    } else {
        inkling2_cuda::inkling2_short_conv_f32_kernel<false><<<blocks, 256, 0, stream>>>(
            (const float *) cur->data, (const float *) history->data, (const float *) weight->data,
            nullptr, (float *) dst->data, (float *) state->data,
            n_channels, n_seq_tokens, n_seqs, 1,
            hist_stride_chan, hist_stride_seq, hist_chan_offset,
            state_row_stride_f, state_slot_stride_f, state_cell_offs_f, n_rs_seq, inv_scale);
    }
}

void ggml_cuda_op_inkling_loop_ctrl_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float eps = ggml_get_op_params_f32(dst, 0);
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * msg = dst->src[1];
    const int n_tokens = (int) cur->ne[1];
    GGML_ASSERT(cur->ne[0] == 2304);
    inkling2_cuda::inkling2_dual_rms_concat_d2304_kernel<<<dim3(n_tokens, 2), 192, 0, ctx.stream()>>>(
        (const float *) cur->data, (const float *) msg->data, (float *) dst->data, n_tokens, eps);
}

void ggml_cuda_op_inkling_loop_ctrl_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * h        = dst->src[0];
    const ggml_tensor * w_gain   = dst->src[1];
    const ggml_tensor * w_forget = dst->src[2];
    const ggml_tensor * cur      = dst->src[3];
    const ggml_tensor * msg      = dst->src[4];
    const int D        = (int) cur->ne[0];
    const int n_tokens = (int) cur->ne[1];
    GGML_ASSERT(h->ne[0] == 64);
    inkling2_cuda::inkling2_loop_ctrl_apply_decode_kernel<<<dim3((D + 15) / 16, n_tokens), 256, 0, ctx.stream()>>>(
        (const float *) h->data, (const float *) w_gain->data, (const float *) w_forget->data,
        (const float *) cur->data, (const float *) msg->data, (float *) dst->data, D, n_tokens);
}

void ggml_cuda_op_inkling_loop_gate_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const bool has_norm = ggml_get_op_params_i32(dst, 0) != 0;
    const float eps     = ggml_get_op_params_f32(dst, 1);
    const ggml_tensor * cur = dst->src[0];
    const int n_tokens = (int) cur->ne[1];
    GGML_ASSERT(cur->ne[0] == 2304);
    if (has_norm) {
        inkling2_cuda::inkling2_loop_gate_norm_d2304_kernel<true><<<n_tokens, 192, 0, ctx.stream()>>>(
            (const float *) cur->data, (const float *) dst->src[1]->data,
            (const float *) dst->src[2]->data, (const float *) dst->src[3]->data,
            (const float *) dst->src[4]->data, (float *) dst->data, n_tokens, eps);
    } else {
        inkling2_cuda::inkling2_loop_gate_norm_d2304_kernel<false><<<n_tokens, 192, 0, ctx.stream()>>>(
            (const float *) cur->data, (const float *) dst->src[1]->data,
            (const float *) dst->src[2]->data, (const float *) dst->src[3]->data,
            nullptr, (float *) dst->data, n_tokens, eps);
    }
}

void ggml_cuda_op_inkling_mul_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(dst);
    GGML_ASSERT(ne % 4 == 0);
    const int64_t n4 = ne / 4;
    const int64_t blocks = (n4 + 255) / 256;
    inkling2_cuda::inkling2_mul_sigmoid_f32_kernel<<<blocks, 256, 0, ctx.stream()>>>(
        (const float *) dst->src[0]->data, (const float *) dst->src[1]->data, (float *) dst->data, n4);
}

// ============================================================================
// CROWN JEWEL: Custom RTX 3090 (sm_86) Fused Inkling-2-Mini MoE FFN Kernels
// Supports BOTH GGML_TYPE_NVFP4 (layers 0-19) AND GGML_TYPE_Q8_0 (layers 20-23)
// Fuses: quantize_x -> [gate_up + expert_scale + SiTU + Q8_1] -> [down + scale + top-8 reduce]
// Eliminates ~21 kernels/layer x 48 layers = ~1,008 kernel launches/token!
// ============================================================================

// Fast warp-level quantization of F32 [D=2304, T] -> block_q8_1 [72, T]
// 1 block per 8 blocks (256 threads = 8 warps, 1 warp per 32-elem block_q8_1!)
static __launch_bounds__(256, 4)
__global__ void inkling2_quantize_x_q8_1_kernel(
        const float * __restrict__ x,
        block_q8_1 * __restrict__ dst,
        const int n_blocks_per_tok, // 72 for D=2304
        const int n_tokens) {
    const int tok = blockIdx.y;
    const int b   = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tok >= n_tokens || b >= n_blocks_per_tok) return;

    const int lane = threadIdx.x & 31;
    const float v = x[(int64_t(tok) * n_blocks_per_tok + b) * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    const float d = amax / 127.0f;
    const int q = amax > 0.0f ? __float2int_rn(v * (127.0f / amax)) : 0;

    block_q8_1 * out = dst + int64_t(tok) * n_blocks_per_tok + b;
    out->qs[lane] = (int8_t) q;
    if (lane == 0) {
        out->ds = make_half2(__float2half(d), __float2half(0.0f));
    }
}

// Pure-ALU branchless NVFP4 nibble unpacker using PTX prmt (__byte_perm).
// Bit-exact match to get_int_from_table_16(q, kvalues_mxfp4) with ZERO __constant__ bank serialization!
static __device__ __forceinline__ int2 inkling_nvfp4_unpack(const uint32_t q) {
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t lo  = __byte_perm(__byte_perm(0x03020100u, 0x0c080604u, q),
                                     __byte_perm(0xfdfeff00u, 0xf4f8fafcu, q), sel);
    const uint32_t hi  = __byte_perm(__byte_perm(0x03020100u, 0x0c080604u, q >> 16),
                                     __byte_perm(0xfdfeff00u, 0xf4f8fafcu, q >> 16), sel >> 16);
    return make_int2((int) __byte_perm(lo, hi, 0x6420u), (int) __byte_perm(lo, hi, 0x7531u));
}

// Helper: 16-element NVFP4 subblock dot product against Q8_1 (in smem/regs)
static __device__ __forceinline__ float nvfp4_subblock_dot(
        const block_nvfp4 * __restrict__ bq4,
        const int is,
        const int q8_0, const int q8_1, const int q8_2, const int q8_3,
        const float d8) {
    const int32_t iqs0 = is << 1;
    const int2 v0 = inkling_nvfp4_unpack((uint32_t) get_int_b4(bq4->qs, iqs0 + 0));
    const int2 v1 = inkling_nvfp4_unpack((uint32_t) get_int_b4(bq4->qs, iqs0 + 1));
    int sumi = ggml_cuda_dp4a(v0.x, q8_0, 0);
    sumi = ggml_cuda_dp4a(v0.y, q8_2, sumi);
    sumi = ggml_cuda_dp4a(v1.x, q8_1, sumi);
    sumi = ggml_cuda_dp4a(v1.y, q8_3, sumi);
    return ggml_cuda_ue4m3_to_fp32(bq4->d[is]) * d8 * float(sumi);
}

// Helper: decode 16-element NVFP4 subblock weights once for reuse across multiple tokens sharing an expert
struct DecodedSub16 {
    int w0, w1, w2, w3;
    float d;
};

static __device__ __forceinline__ DecodedSub16 decode_nvfp4_sub16(const block_nvfp4 * __restrict__ bq4, const int is) {
    const int32_t iqs0 = is << 1;
    const int2 v0 = inkling_nvfp4_unpack((uint32_t) get_int_b4(bq4->qs, iqs0 + 0));
    const int2 v1 = inkling_nvfp4_unpack((uint32_t) get_int_b4(bq4->qs, iqs0 + 1));
    return { v0.x, v1.x, v0.y, v1.y, ggml_cuda_ue4m3_to_fp32(bq4->d[is]) };
}

static __device__ __forceinline__ DecodedSub16 decode_q8_0_sub16(const block_q8_0 * __restrict__ bq, const int ih) {
    const int base = ih << 2;
    return {
        get_int_b2(bq->qs, base + 0),
        get_int_b2(bq->qs, base + 1),
        get_int_b2(bq->qs, base + 2),
        get_int_b2(bq->qs, base + 3),
        __half2float(bq->d)
    };
}

static __device__ __forceinline__ float dot_decoded_sub16(
        const DecodedSub16 & w,
        const int q8_0, const int q8_1, const int q8_2, const int q8_3,
        const float d8) {
    int sumi = ggml_cuda_dp4a(w.w0, q8_0, 0);
    sumi = ggml_cuda_dp4a(w.w1, q8_1, sumi);
    sumi = ggml_cuda_dp4a(w.w2, q8_2, sumi);
    sumi = ggml_cuda_dp4a(w.w3, q8_3, sumi);
    return w.d * d8 * float(sumi);
}

// Helper: 16-element Q8_0 half-block dot product (handles 2-byte aligned block_q8_0 safely!)
static __device__ __forceinline__ float q8_0_halfblock_dot(
        const block_q8_0 * __restrict__ bq,
        const int ih,
        const int q8_0, const int q8_1, const int q8_2, const int q8_3,
        const float d8) {
    const int base = ih << 2;
    int sumi = ggml_cuda_dp4a(get_int_b2(bq->qs, base + 0), q8_0, 0);
    sumi = ggml_cuda_dp4a(get_int_b2(bq->qs, base + 1), q8_1, sumi);
    sumi = ggml_cuda_dp4a(get_int_b2(bq->qs, base + 2), q8_2, sumi);
    sumi = ggml_cuda_dp4a(get_int_b2(bq->qs, base + 3), q8_3, sumi);
    return __half2float(bq->d) * d8 * float(sumi);
}

// FUSED MoE Stage 1: gate_up_exps (NVFP4 or Q8_0) + expert_scale + SiTU + Q8_1 quantization!
// Exact dimensions: D = 2304 (36 NVFP4 / 72 Q8_0 blocks -> 144 subblocks = 16 lanes x 9 iters!), n_ff = 1152 (36 blocks of 32)
// Grid: (n_ff / 32 = 36, n_exp_used = 8, n_tokens), Block: 256 threads (16 half-warps -> 2 paired rows/half-warp = 32 rows/block!)
template <ggml_type WTYPE>
__launch_bounds__(256, 3)
static __global__ void inkling2_moe_gate_up_situ_q8_1_kernel(
        const char * __restrict__ gate_up_w,
        const block_q8_1 * __restrict__ x_q8_1,
        const int32_t * __restrict__ ids,
        const float * __restrict__ gate_up_s,
        block_q8_1 * __restrict__ h_q8_1,
        const int64_t nb01,         // bytes per row of gate_up_w
        const int64_t nb02,         // bytes per expert of gate_up_w
        const int n_ff,             // 1152
        const int n_exp_used,       // 8
        const int ids_stride,
        const float beta,
        const float inv_beta,
        const float lb,
        const float inv_lb) {
    constexpr int D = 2304;
    constexpr int NQ8 = D / 32; // 72

    // Load x_q8_1 for this token ONCE into shared memory (72 * 36 = 2592 bytes)
    __shared__ block_q8_1 s_x[NQ8];
    const int tok = blockIdx.z;
    const int exp_slot = blockIdx.y;
    const int tid = threadIdx.x;

    // Cooperative copy of 72 * 9 = 648 uint32s
    const uint32_t * src_u32 = reinterpret_cast<const uint32_t *>(x_q8_1 + int64_t(tok) * NQ8);
    uint32_t * dst_u32       = reinterpret_cast<uint32_t *>(s_x);
#pragma unroll
    for (int i = tid; i < NQ8 * 9; i += 256) {
        dst_u32[i] = src_u32[i];
    }
    __syncthreads();

    const int expert_id = ids[int64_t(tok) * ids_stride + exp_slot];
    const char * exp_base = gate_up_w + int64_t(expert_id) * nb02;

    const int hw     = tid >> 4; // 0..15 half-warp index
    const int lane16 = tid & 15; // 0..15 lane within half-warp
    const int row0   = blockIdx.x * 32 + hw * 2;
    const int row1   = row0 + 1;

    const char * g_row0 = exp_base + int64_t(row0) * nb01;
    const char * u_row0 = exp_base + int64_t(row0 + n_ff) * nb01;
    const char * g_row1 = exp_base + int64_t(row1) * nb01;
    const char * u_row1 = exp_base + int64_t(row1 + n_ff) * nb01;

    float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;

    // 144 subblocks of 16 elements = 9 exact iterations x 16 lanes (zero remainder, 100% active!)
#pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        const int sub = iter * 16 + lane16; // 0..143
        const int bq8_idx = sub >> 1;       // 0..71
        const int i8      = (sub & 1) << 2; // 0 or 4
        const block_q8_1 & bx = s_x[bq8_idx];
        const int * q8p = reinterpret_cast<const int *>(bx.qs) + i8;
        const int q8_0 = q8p[0], q8_1 = q8p[1], q8_2 = q8p[2], q8_3 = q8p[3];
        const float d8 = __low2float(bx.ds);

        if constexpr (WTYPE == GGML_TYPE_NVFP4) {
            const int kbx = sub >> 2; // 0..35
            const int is  = sub & 3;  // 0..3
            g0 += nvfp4_subblock_dot(reinterpret_cast<const block_nvfp4 *>(g_row0) + kbx, is, q8_0, q8_1, q8_2, q8_3, d8);
            u0 += nvfp4_subblock_dot(reinterpret_cast<const block_nvfp4 *>(u_row0) + kbx, is, q8_0, q8_1, q8_2, q8_3, d8);
            g1 += nvfp4_subblock_dot(reinterpret_cast<const block_nvfp4 *>(g_row1) + kbx, is, q8_0, q8_1, q8_2, q8_3, d8);
            u1 += nvfp4_subblock_dot(reinterpret_cast<const block_nvfp4 *>(u_row1) + kbx, is, q8_0, q8_1, q8_2, q8_3, d8);
        } else {
            const int ih = sub & 1;
            g0 += q8_0_halfblock_dot(reinterpret_cast<const block_q8_0 *>(g_row0) + bq8_idx, ih, q8_0, q8_1, q8_2, q8_3, d8);
            u0 += q8_0_halfblock_dot(reinterpret_cast<const block_q8_0 *>(u_row0) + bq8_idx, ih, q8_0, q8_1, q8_2, q8_3, d8);
            g1 += q8_0_halfblock_dot(reinterpret_cast<const block_q8_0 *>(g_row1) + bq8_idx, ih, q8_0, q8_1, q8_2, q8_3, d8);
            u1 += q8_0_halfblock_dot(reinterpret_cast<const block_q8_0 *>(u_row1) + bq8_idx, ih, q8_0, q8_1, q8_2, q8_3, d8);
        }
    }

    // 16-lane half-warp reduction
#pragma unroll
    for (int off = 8; off >= 1; off >>= 1) {
        g0 += __shfl_xor_sync(0xffffffff, g0, off);
        u0 += __shfl_xor_sync(0xffffffff, u0, off);
        g1 += __shfl_xor_sync(0xffffffff, g1, off);
        u1 += __shfl_xor_sync(0xffffffff, u1, off);
    }

    __shared__ float s_out[32];
    if (lane16 == 0) {
        const float sc = gate_up_s ? __ldg(gate_up_s + expert_id) : 1.0f;
        s_out[hw * 2 + 0] = inkling2_cuda::situ_scalar(g0 * sc, u0 * sc, beta, inv_beta, lb, inv_lb);
        s_out[hw * 2 + 1] = inkling2_cuda::situ_scalar(g1 * sc, u1 * sc, beta, inv_beta, lb, inv_lb);
    }
    __syncthreads();

    // Warp 0 quantizes all 32 SiTU outputs directly from shared memory into 1 block_q8_1!
    if (tid < 32) {
        const float v = s_out[tid];
        float amax = fabsf(v);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        }
        const float d = amax / 127.0f;
        const int q = amax > 0.0f ? __float2int_rn(v * (127.0f / amax)) : 0;
        const int n_ff_blocks = n_ff >> 5; // 36
        block_q8_1 * out = h_q8_1 + (int64_t(tok) * n_exp_used + exp_slot) * n_ff_blocks + blockIdx.x;
        out->qs[tid] = (int8_t) q;
        if (tid == 0) {
            out->ds = make_half2(__float2half(d), __float2half(0.0f));
        }
    }
}

// FUSED MoE Stage 2: down_exps (NVFP4 or Q8_0) + expert_scale + Top-8 Weighted Reduction!
// Exact dimensions: K = 1152 (18 NVFP4 / 36 Q8_0 blocks -> 72 subblocks = 8 lanes x 9 iters!), D = 2304 (144 blocks of 16 rows!)
// Grid: (D / 16 = 144, n_tokens), Block: 128 threads (16 quarter-warps of 8 lanes -> 16 output rows/block!)
template <ggml_type WTYPE>
__launch_bounds__(128, 4)
static __global__ void inkling2_moe_down_reduce_kernel(
        const char * __restrict__ down_w,
        const block_q8_1 * __restrict__ h_q8_1,
        const int32_t * __restrict__ ids,
        const float * __restrict__ weights,
        const float * __restrict__ down_s,
        float * __restrict__ dst,
        const int64_t nb01,         // bytes per row of down_w
        const int64_t nb02,         // bytes per expert of down_w
        const int D,                // 2304
        const int n_exp_used,       // 8
        const int ids_stride,
        const int w_stride_exp,     // weights->nb[1] / 4
        const int w_stride_tok) {   // weights->nb[2] / 4
    constexpr int NFF_Q8 = 1152 / 32; // 36
    constexpr int MAX_EXP = 8;

    // Load all 8 experts' h_q8_1 (8 * 36 * 36 = 10,368 bytes) + expert IDs & folded scales ONCE into shared memory!
    __shared__ block_q8_1 s_h[MAX_EXP * NFF_Q8];
    __shared__ int32_t    s_eids[MAX_EXP];
    __shared__ float      s_w[MAX_EXP];

    const int tok = blockIdx.y;
    const int tid = threadIdx.x;

    const int total_u32 = n_exp_used * NFF_Q8 * 9;
    const uint32_t * src_u32 = reinterpret_cast<const uint32_t *>(h_q8_1 + int64_t(tok) * n_exp_used * NFF_Q8);
    uint32_t * dst_u32       = reinterpret_cast<uint32_t *>(s_h);
    for (int i = tid; i < total_u32; i += 128) {
        dst_u32[i] = src_u32[i];
    }
    if (tid < n_exp_used) {
        const int eid = ids[int64_t(tok) * ids_stride + tid];
        s_eids[tid] = eid;
        const float rw = weights[int64_t(tok) * w_stride_tok + int64_t(tid) * w_stride_exp];
        const float ds = down_s ? __ldg(down_s + eid) : 1.0f;
        s_w[tid] = rw * ds;
    }
    __syncthreads();

    const int qw    = tid >> 3; // 0..15 output row within block
    const int lane8 = tid & 7;  // 0..7 lane within quarter-warp
    const int row   = blockIdx.x * 16 + qw;
    if (row >= D) return;

    float total_acc = 0.0f;

    // Accumulate across all 8 experts directly in registers!
    for (int e = 0; e < n_exp_used; ++e) {
        const char * w_row = down_w + int64_t(s_eids[e]) * nb02 + int64_t(row) * nb01;
        const block_q8_1 * h_exp = s_h + e * NFF_Q8;
        float exp_acc = 0.0f;

        // 72 subblocks of 16 elements = 9 exact iterations x 8 lanes (zero remainder, 100% active!)
#pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            const int sub     = iter * 8 + lane8; // 0..71
            const int bq8_idx = sub >> 1;         // 0..35
            const int i8      = (sub & 1) << 2;   // 0 or 4
            const block_q8_1 & bh = h_exp[bq8_idx];
            const int * q8p = reinterpret_cast<const int *>(bh.qs) + i8;
            const int q8_0 = q8p[0], q8_1 = q8p[1], q8_2 = q8p[2], q8_3 = q8p[3];
            const float d8 = __low2float(bh.ds);

            if constexpr (WTYPE == GGML_TYPE_NVFP4) {
                const int kbx = sub >> 2; // 0..17
                const int is  = sub & 3;  // 0..3
                exp_acc += nvfp4_subblock_dot(reinterpret_cast<const block_nvfp4 *>(w_row) + kbx, is, q8_0, q8_1, q8_2, q8_3, d8);
            } else {
                const int ih = sub & 1;
                exp_acc += q8_0_halfblock_dot(reinterpret_cast<const block_q8_0 *>(w_row) + bq8_idx, ih, q8_0, q8_1, q8_2, q8_3, d8);
            }
        }
        total_acc = fmaf(exp_acc, s_w[e], total_acc);
    }

    // Single 8-lane quarter-warp reduction across all 8 experts combined!
    total_acc = inkling2_cuda::subwarp_sum8(total_acc);
    if (lane8 == 0) {
        dst[int64_t(tok) * D + row] = total_acc;
    }
}

// ============================================================================
// EXPERT-GROUPED BATCHED MoE KERNELS FOR T = 2..8 (DSpark Speculative Verify!)
// Deduplicates active experts across tokens so each expert's weights are read ONCE!
// Uses fixed static grids with early exit -> 100% CUDA-graph capturable!
// ============================================================================
struct InklingExpGroup {
    int32_t n_active;
    int32_t active_eid[64];
    int32_t group_count[64];
    int32_t group_tok[64 * 8];
    int32_t group_slot[64 * 8];
    float   group_w[64 * 8];
};

static __global__ void inkling2_group_experts_kernel(
        const int32_t * __restrict__ ids,
        const float   * __restrict__ weights,
        const float   * __restrict__ down_s,
        InklingExpGroup * __restrict__ grp,
        const int n_tokens,
        const int n_exp_used,
        const int ids_stride,
        const int w_stride_exp,
        const int w_stride_tok) {
    int8_t e2g[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) e2g[i] = -1;
    int n_act = 0;
    for (int tok = 0; tok < n_tokens; ++tok) {
        for (int slot = 0; slot < n_exp_used; ++slot) {
            const int eid = ids[tok * ids_stride + slot];
            int g = (eid >= 0 && eid < 128) ? int(e2g[eid]) : -1;
            if (g < 0) {
                g = n_act++;
                if (eid >= 0 && eid < 128) e2g[eid] = (int8_t) g;
                grp->active_eid[g]  = eid;
                grp->group_count[g] = 0;
            }
            const int idx = grp->group_count[g]++;
            if (idx < 8) {
                grp->group_tok[g * 8 + idx]  = tok;
                grp->group_slot[g * 8 + idx] = slot;
                const float rw = weights[tok * w_stride_tok + slot * w_stride_exp];
                const float ds = down_s ? __ldg(down_s + eid) : 1.0f;
                grp->group_w[g * 8 + idx] = rw * ds;
            }
        }
    }
    grp->n_active = n_act;
}

template <ggml_type WTYPE, int MAX_T>
__launch_bounds__(256, 2)
static __global__ void inkling2_moe_gate_up_situ_q8_1_grouped_kernel(
        const char * __restrict__ gate_up_w,
        const block_q8_1 * __restrict__ x_q8_1,
        const InklingExpGroup * __restrict__ grp,
        const float * __restrict__ gate_up_s,
        block_q8_1 * __restrict__ h_q8_1,
        const int64_t nb01,
        const int64_t nb02,
        const int n_ff,
        const int n_exp_used,
        const float beta,
        const float inv_beta,
        const float lb,
        const float inv_lb) {
    const int g = blockIdx.y;
    if (g >= grp->n_active) return;

    constexpr int D = 2304;
    constexpr int NQ8 = D / 32; // 72

    const int cnt = grp->group_count[g];
    const int expert_id = grp->active_eid[g];
    const int tid = threadIdx.x;

    __shared__ block_q8_1 s_x[MAX_T][NQ8];
    __shared__ int32_t s_tok[MAX_T];
    __shared__ int32_t s_slot[MAX_T];

    if (tid < cnt) {
        s_tok[tid]  = grp->group_tok[g * 8 + tid];
        s_slot[tid] = grp->group_slot[g * 8 + tid];
    }
    __syncthreads();

    for (int m = 0; m < cnt; ++m) {
        const uint32_t * src_u32 = reinterpret_cast<const uint32_t *>(x_q8_1 + int64_t(s_tok[m]) * NQ8);
        uint32_t * dst_u32       = reinterpret_cast<uint32_t *>(s_x[m]);
#pragma unroll
        for (int i = tid; i < NQ8 * 9; i += 256) {
            dst_u32[i] = src_u32[i];
        }
    }
    __syncthreads();

    const char * exp_base = gate_up_w + int64_t(expert_id) * nb02;
    const int hw     = tid >> 4;
    const int lane16 = tid & 15;
    const int row0   = blockIdx.x * 32 + hw * 2;
    const int row1   = row0 + 1;

    const char * g_row0 = exp_base + int64_t(row0) * nb01;
    const char * u_row0 = exp_base + int64_t(row0 + n_ff) * nb01;
    const char * g_row1 = exp_base + int64_t(row1) * nb01;
    const char * u_row1 = exp_base + int64_t(row1 + n_ff) * nb01;

    float g0[MAX_T], u0[MAX_T], g1[MAX_T], u1[MAX_T];
#pragma unroll
    for (int m = 0; m < MAX_T; ++m) {
        g0[m] = 0.0f; u0[m] = 0.0f; g1[m] = 0.0f; u1[m] = 0.0f;
    }

#pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        const int sub = iter * 16 + lane16;
        const int bq8_idx = sub >> 1;
        const int i8      = (sub & 1) << 2;

        DecodedSub16 wg0, wu0, wg1, wu1;
        if constexpr (WTYPE == GGML_TYPE_NVFP4) {
            const int kbx = sub >> 2;
            const int is  = sub & 3;
            wg0 = decode_nvfp4_sub16(reinterpret_cast<const block_nvfp4 *>(g_row0) + kbx, is);
            wu0 = decode_nvfp4_sub16(reinterpret_cast<const block_nvfp4 *>(u_row0) + kbx, is);
            wg1 = decode_nvfp4_sub16(reinterpret_cast<const block_nvfp4 *>(g_row1) + kbx, is);
            wu1 = decode_nvfp4_sub16(reinterpret_cast<const block_nvfp4 *>(u_row1) + kbx, is);
        } else {
            const int ih = sub & 1;
            wg0 = decode_q8_0_sub16(reinterpret_cast<const block_q8_0 *>(g_row0) + bq8_idx, ih);
            wu0 = decode_q8_0_sub16(reinterpret_cast<const block_q8_0 *>(u_row0) + bq8_idx, ih);
            wg1 = decode_q8_0_sub16(reinterpret_cast<const block_q8_0 *>(g_row1) + bq8_idx, ih);
            wu1 = decode_q8_0_sub16(reinterpret_cast<const block_q8_0 *>(u_row1) + bq8_idx, ih);
        }

        for (int m = 0; m < cnt; ++m) {
            const block_q8_1 & bx = s_x[m][bq8_idx];
            const int * q8p = reinterpret_cast<const int *>(bx.qs) + i8;
            const int q8_0 = q8p[0], q8_1 = q8p[1], q8_2 = q8p[2], q8_3 = q8p[3];
            const float d8 = __low2float(bx.ds);
            g0[m] += dot_decoded_sub16(wg0, q8_0, q8_1, q8_2, q8_3, d8);
            u0[m] += dot_decoded_sub16(wu0, q8_0, q8_1, q8_2, q8_3, d8);
            g1[m] += dot_decoded_sub16(wg1, q8_0, q8_1, q8_2, q8_3, d8);
            u1[m] += dot_decoded_sub16(wu1, q8_0, q8_1, q8_2, q8_3, d8);
        }
    }

    __shared__ float s_out[32];
    const float sc = gate_up_s ? __ldg(gate_up_s + expert_id) : 1.0f;
    const int n_ff_blocks = n_ff >> 5;

    for (int m = 0; m < cnt; ++m) {
        float vg0 = g0[m], vu0 = u0[m], vg1 = g1[m], vu1 = u1[m];
#pragma unroll
        for (int off = 8; off >= 1; off >>= 1) {
            vg0 += __shfl_xor_sync(0xffffffff, vg0, off);
            vu0 += __shfl_xor_sync(0xffffffff, vu0, off);
            vg1 += __shfl_xor_sync(0xffffffff, vg1, off);
            vu1 += __shfl_xor_sync(0xffffffff, vu1, off);
        }
        if (lane16 == 0) {
            s_out[hw * 2 + 0] = inkling2_cuda::situ_scalar(vg0 * sc, vu0 * sc, beta, inv_beta, lb, inv_lb);
            s_out[hw * 2 + 1] = inkling2_cuda::situ_scalar(vg1 * sc, vu1 * sc, beta, inv_beta, lb, inv_lb);
        }
        __syncthreads();

        if (tid < 32) {
            const float v = s_out[tid];
            float amax = fabsf(v);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
            }
            const float d = amax / 127.0f;
            const int q = amax > 0.0f ? __float2int_rn(v * (127.0f / amax)) : 0;
            block_q8_1 * out = h_q8_1 + (int64_t(s_tok[m]) * n_exp_used + s_slot[m]) * n_ff_blocks + blockIdx.x;
            out->qs[tid] = (int8_t) q;
            if (tid == 0) {
                out->ds = make_half2(__float2half(d), __float2half(0.0f));
            }
        }
        __syncthreads();
    }
}

template <ggml_type WTYPE, int MAX_T>
__launch_bounds__(128, 3)
static __global__ void inkling2_moe_down_grouped_kernel(
        const char * __restrict__ down_w,
        const block_q8_1 * __restrict__ h_q8_1,
        const InklingExpGroup * __restrict__ grp,
        float * __restrict__ dst,
        const int64_t nb01,
        const int64_t nb02,
        const int D,
        const int n_exp_used) {
    const int g = blockIdx.y;
    if (g >= grp->n_active) return;

    constexpr int NFF_Q8 = 1152 / 32; // 36

    const int cnt = grp->group_count[g];
    const int expert_id = grp->active_eid[g];
    const int tid = threadIdx.x;

    __shared__ block_q8_1 s_h[MAX_T][NFF_Q8];
    __shared__ int32_t s_tok[MAX_T];
    __shared__ int32_t s_slot[MAX_T];
    __shared__ float   s_w[MAX_T];

    if (tid < cnt) {
        s_tok[tid]  = grp->group_tok[g * 8 + tid];
        s_slot[tid] = grp->group_slot[g * 8 + tid];
        s_w[tid]    = grp->group_w[g * 8 + tid];
    }
    __syncthreads();

    for (int m = 0; m < cnt; ++m) {
        const uint32_t * src_u32 = reinterpret_cast<const uint32_t *>(
            h_q8_1 + (int64_t(s_tok[m]) * n_exp_used + s_slot[m]) * NFF_Q8);
        uint32_t * dst_u32 = reinterpret_cast<uint32_t *>(s_h[m]);
        for (int i = tid; i < NFF_Q8 * 9; i += 128) {
            dst_u32[i] = src_u32[i];
        }
    }
    __syncthreads();

    const int qw    = tid >> 3;
    const int lane8 = tid & 7;
    const int row   = blockIdx.x * 16 + qw;
    if (row >= D) return;

    const char * w_row = down_w + int64_t(expert_id) * nb02 + int64_t(row) * nb01;
    float acc[MAX_T];
#pragma unroll
    for (int m = 0; m < MAX_T; ++m) acc[m] = 0.0f;

#pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        const int sub     = iter * 8 + lane8;
        const int bq8_idx = sub >> 1;
        const int i8      = (sub & 1) << 2;

        DecodedSub16 wd;
        if constexpr (WTYPE == GGML_TYPE_NVFP4) {
            const int kbx = sub >> 2;
            const int is  = sub & 3;
            wd = decode_nvfp4_sub16(reinterpret_cast<const block_nvfp4 *>(w_row) + kbx, is);
        } else {
            const int ih = sub & 1;
            wd = decode_q8_0_sub16(reinterpret_cast<const block_q8_0 *>(w_row) + bq8_idx, ih);
        }

        for (int m = 0; m < cnt; ++m) {
            const block_q8_1 & bh = s_h[m][bq8_idx];
            const int * q8p = reinterpret_cast<const int *>(bh.qs) + i8;
            const int q8_0 = q8p[0], q8_1 = q8p[1], q8_2 = q8p[2], q8_3 = q8p[3];
            const float d8 = __low2float(bh.ds);
            acc[m] += dot_decoded_sub16(wd, q8_0, q8_1, q8_2, q8_3, d8);
        }
    }

    for (int m = 0; m < cnt; ++m) {
        const float v = inkling2_cuda::subwarp_sum8(acc[m]);
        if (lane8 == 0) {
            atomicAdd(dst + int64_t(s_tok[m]) * D + row, v * s_w[m]);
        }
    }
}

void ggml_cuda_op_inkling_moe_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int flags    = ggml_get_op_params_i32(dst, 0);
    const float beta   = ggml_get_op_params_f32(dst, 1);
    const float lb     = ggml_get_op_params_f32(dst, 2);
    const bool has_gus = (flags & 1) != 0;
    const bool has_ds  = (flags & 2) != 0;

    const ggml_tensor * x         = dst->src[0];
    const ggml_tensor * gate_up_w = dst->src[1];
    const ggml_tensor * down_w    = dst->src[2];
    const ggml_tensor * ids       = dst->src[3];
    const ggml_tensor * weights   = dst->src[4];
    int s = 5;
    const float * gate_up_s = has_gus ? (const float *) dst->src[s++]->data : nullptr;
    const float * down_s    = has_ds  ? (const float *) dst->src[s++]->data : nullptr;

    const int D          = (int) x->ne[0];         // 2304
    const int n_tokens   = (int) x->ne[1];         // 1..8
    const int n_ff       = (int) down_w->ne[0];    // 1152
    const int n_exp_used = (int) ids->ne[0];       // 8
    const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
    const int w_stride_exp = (int) (weights->nb[1] / sizeof(float));
    const int w_stride_tok = (int) (weights->nb[2] / sizeof(float));

    GGML_ASSERT(D == 2304 && n_ff == 1152 && n_exp_used <= 8 && n_tokens <= 8);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const int x_blocks_per_tok = D / 32;    // 72
    const int h_blocks_per_exp = n_ff / 32; // 36

    ggml_cuda_pool_alloc<block_q8_1> x_q8_1(pool, size_t(n_tokens) * x_blocks_per_tok);
    ggml_cuda_pool_alloc<block_q8_1> h_q8_1(pool, size_t(n_tokens) * n_exp_used * h_blocks_per_exp);

    // 1. Quantize x -> Q8_1
    inkling2_quantize_x_q8_1_kernel<<<dim3((x_blocks_per_tok + 7) / 8, n_tokens), 256, 0, stream>>>(
        (const float *) x->data, x_q8_1.ptr, x_blocks_per_tok, n_tokens);

    const float inv_beta = 1.0f / beta;
    const float inv_lb   = lb > 0.0f ? 1.0f / lb : 0.0f;

    if (n_tokens == 1) {
        // Ultra-fast zero-atomic single-token decode path
        const dim3 gu_grid(h_blocks_per_exp, n_exp_used, 1);
        if (gate_up_w->type == GGML_TYPE_NVFP4) {
            inkling2_moe_gate_up_situ_q8_1_kernel<GGML_TYPE_NVFP4><<<gu_grid, 256, 0, stream>>>(
                (const char *) gate_up_w->data, x_q8_1.ptr, (const int32_t *) ids->data, gate_up_s, h_q8_1.ptr,
                gate_up_w->nb[1], gate_up_w->nb[2], n_ff, n_exp_used, ids_stride, beta, inv_beta, lb, inv_lb);
        } else if (gate_up_w->type == GGML_TYPE_Q8_0) {
            inkling2_moe_gate_up_situ_q8_1_kernel<GGML_TYPE_Q8_0><<<gu_grid, 256, 0, stream>>>(
                (const char *) gate_up_w->data, x_q8_1.ptr, (const int32_t *) ids->data, gate_up_s, h_q8_1.ptr,
                gate_up_w->nb[1], gate_up_w->nb[2], n_ff, n_exp_used, ids_stride, beta, inv_beta, lb, inv_lb);
        } else {
            GGML_ABORT("inkling_moe_ffn: unsupported gate_up_w type");
        }

        const dim3 dn_grid((D + 15) / 16, 1);
        if (down_w->type == GGML_TYPE_NVFP4) {
            inkling2_moe_down_reduce_kernel<GGML_TYPE_NVFP4><<<dn_grid, 128, 0, stream>>>(
                (const char *) down_w->data, h_q8_1.ptr, (const int32_t *) ids->data,
                (const float *) weights->data, down_s, (float *) dst->data,
                down_w->nb[1], down_w->nb[2], D, n_exp_used, ids_stride, w_stride_exp, w_stride_tok);
        } else if (down_w->type == GGML_TYPE_Q8_0) {
            inkling2_moe_down_reduce_kernel<GGML_TYPE_Q8_0><<<dn_grid, 128, 0, stream>>>(
                (const char *) down_w->data, h_q8_1.ptr, (const int32_t *) ids->data,
                (const float *) weights->data, down_s, (float *) dst->data,
                down_w->nb[1], down_w->nb[2], D, n_exp_used, ids_stride, w_stride_exp, w_stride_tok);
        } else {
            GGML_ABORT("inkling_moe_ffn: unsupported down_w type");
        }
        return;
    }

    // Expert-grouped batched path for T = 2..8 (DSpark verify!)
    ggml_cuda_pool_alloc<InklingExpGroup> grp(pool, 1);
    inkling2_group_experts_kernel<<<1, 1, 0, stream>>>(
        (const int32_t *) ids->data, (const float *) weights->data, down_s, grp.ptr,
        n_tokens, n_exp_used, ids_stride, w_stride_exp, w_stride_tok);

    const int max_groups = n_tokens * n_exp_used;
    const dim3 gu_grid(h_blocks_per_exp, max_groups);
    const dim3 dn_grid((D + 15) / 16, max_groups);

    if (n_tokens <= 4) {
        if (gate_up_w->type == GGML_TYPE_NVFP4) {
            inkling2_moe_gate_up_situ_q8_1_grouped_kernel<GGML_TYPE_NVFP4, 4><<<gu_grid, 256, 0, stream>>>(
                (const char *) gate_up_w->data, x_q8_1.ptr, grp.ptr, gate_up_s, h_q8_1.ptr,
                gate_up_w->nb[1], gate_up_w->nb[2], n_ff, n_exp_used, beta, inv_beta, lb, inv_lb);
        } else {
            inkling2_moe_gate_up_situ_q8_1_grouped_kernel<GGML_TYPE_Q8_0, 4><<<gu_grid, 256, 0, stream>>>(
                (const char *) gate_up_w->data, x_q8_1.ptr, grp.ptr, gate_up_s, h_q8_1.ptr,
                gate_up_w->nb[1], gate_up_w->nb[2], n_ff, n_exp_used, beta, inv_beta, lb, inv_lb);
        }
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, size_t(n_tokens) * D * sizeof(float), stream));
        if (down_w->type == GGML_TYPE_NVFP4) {
            inkling2_moe_down_grouped_kernel<GGML_TYPE_NVFP4, 4><<<dn_grid, 128, 0, stream>>>(
                (const char *) down_w->data, h_q8_1.ptr, grp.ptr, (float *) dst->data,
                down_w->nb[1], down_w->nb[2], D, n_exp_used);
        } else {
            inkling2_moe_down_grouped_kernel<GGML_TYPE_Q8_0, 4><<<dn_grid, 128, 0, stream>>>(
                (const char *) down_w->data, h_q8_1.ptr, grp.ptr, (float *) dst->data,
                down_w->nb[1], down_w->nb[2], D, n_exp_used);
        }
    } else {
        if (gate_up_w->type == GGML_TYPE_NVFP4) {
            inkling2_moe_gate_up_situ_q8_1_grouped_kernel<GGML_TYPE_NVFP4, 8><<<gu_grid, 256, 0, stream>>>(
                (const char *) gate_up_w->data, x_q8_1.ptr, grp.ptr, gate_up_s, h_q8_1.ptr,
                gate_up_w->nb[1], gate_up_w->nb[2], n_ff, n_exp_used, beta, inv_beta, lb, inv_lb);
        } else {
            inkling2_moe_gate_up_situ_q8_1_grouped_kernel<GGML_TYPE_Q8_0, 8><<<gu_grid, 256, 0, stream>>>(
                (const char *) gate_up_w->data, x_q8_1.ptr, grp.ptr, gate_up_s, h_q8_1.ptr,
                gate_up_w->nb[1], gate_up_w->nb[2], n_ff, n_exp_used, beta, inv_beta, lb, inv_lb);
        }
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, size_t(n_tokens) * D * sizeof(float), stream));
        if (down_w->type == GGML_TYPE_NVFP4) {
            inkling2_moe_down_grouped_kernel<GGML_TYPE_NVFP4, 8><<<dn_grid, 128, 0, stream>>>(
                (const char *) down_w->data, h_q8_1.ptr, grp.ptr, (float *) dst->data,
                down_w->nb[1], down_w->nb[2], D, n_exp_used);
        } else {
            inkling2_moe_down_grouped_kernel<GGML_TYPE_Q8_0, 8><<<dn_grid, 128, 0, stream>>>(
                (const char *) down_w->data, h_q8_1.ptr, grp.ptr, (float *) dst->data,
                down_w->nb[1], down_w->nb[2], D, n_exp_used);
        }
    }
}
