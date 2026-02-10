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

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  
                    K_block_ptr, V_block_ptr,
                    qk_scale,
                    BLOCK_M: tl.constexpr, 
                    BLOCK_N: tl.constexpr,
                    HEAD_DIM: tl.constexpr,
                    N_CTX: tl.constexpr):
    
    #loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr)
        k = k.T
        
        qk = tl.dot(q, k)
        
        #online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        
        #correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        
        #load v block
        v = tl.load(V_block_ptr)
        v = v.to(tl.float32)
        p = p.to(tl.float32)
        
        #accumulate
        acc = tl.dot(p, v, acc)
        
        #running statistics
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        #next blocks
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
    

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1) 
    

    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    
    #load q
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs)
    
    #initialize acc
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    
    #block pointers
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    
    #inner loop
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr,
        sm_scale, BLOCK_M, BLOCK_N, HEAD_DIM, N_CTX
    )
    
    #normalization
    acc = acc / l_i[:, None]
    
    #output
    o_ptrs = O + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))

#wrapper
def flash_attention_forward(q, k, v, sm_scale=None):
    #q, k, v
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    

    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    o = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
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
