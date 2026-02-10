"""
Simplified toy implmentation
non-causal attention
Fixed block sizes (BLOCK_M=64, BLOCK_N=64)
Simple pointer-based memory access
Test correctness vs PyTorch's attention


TODO:
Add causal masking
Experiment with block sizes
Add timing benchmarks

TODO:
Try different BLOCK_M/BLOCK_N combinations
Profile memory access patterns
Compare performance
"""

import torch
import triton
import triton.language as tl

#simplified inner loop
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, 
                    K_block_ptr, V_block_ptr, 
                    qk_scale,
                    BLOCK_M: tl.constexpr, 
                    BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr,
                    N_CTX: tl.constexpr):


    offsetk_y = 0
    offsetv_y = 0
    # simplified version: no stages, no causal masking initially
    for start_n in range(0, N_CTX, BLOCK_N):
        # load K, V blocks
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr).T
        qk = tl.dot(q, k)
        # compute QK
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # update m_i, l_i with online softmax
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # update acc
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, O, 
              sm_scale,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vn, stride_vk,
              stride_oz, stride_oh, stride_om, stride_ok,
              Z, H, N_CTX,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              HEAD_DIM: tl.constexpr):
    
    # get which block we're computing
    start_m = tl.program_id(0)  # Which Q block
    off_hz = tl.program_id(1)   # Which batch/head
    
    # decompose batch and head indices
    off_z = off_hz // H
    off_h = off_hz % H
    
    # compute Q block offset
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    
    # load Q block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs)
    
    # initialize accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    # call inner loop
    K_base = K + qvk_offset
    V_base = V + qvk_offset
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, K_base, V_base,
        stride_kn, stride_kk, stride_vn, stride_vk,
        sm_scale, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX
    )
    
    # final normalization
    acc = acc / l_i[:, None]
    
    # store output
    o_ptrs = O + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))

# Python wrapper
def flash_attention_forward(q, k, v, sm_scale=None):
    """
    q, k, v: [BATCH, N_HEADS, N_CTX, HEAD_DIM]
    """
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    
    # default scale
    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # allocate output
    o = torch.empty_like(q)
    
    # block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    
    # launch grid: (num_Q_blocks, batch * heads)
    grid = (triton.cdiv(N_CTX, BLOCK_M), BATCH * N_HEADS)
    
    _attn_fwd[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BATCH, N_HEADS, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
    )
    
    return o
