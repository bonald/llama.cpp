#pragma once

#include "common.cuh"

void ggml_cuda_op_inkling_attn_res      (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_l2_clip       (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_short_conv    (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_loop_ctrl_pre (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_loop_ctrl_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_loop_gate_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_mul_sigmoid   (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_ffn       (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
