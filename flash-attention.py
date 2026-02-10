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

#TODO: wrapper
def flash_attention_forward(q, k, v, causal, sm_scale):
    #setup: allocate output, compute grid dimensions
    #launch: _attn_fwd kernel
    #return: output tensor

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
    # Simplified version: no stages, no causal masking initially
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V blocks
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr).T
        qk = tl.dot(q, k)
        # Compute QK
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # Update m_i, l_i with online softmax
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # Update acc
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        
    return acc, l_i, m_i

# TODO: Outer kernel
@triton.jit  
def _attn_fwd(Q, K, V, O, sm_scale,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vn, stride_vk,
              stride_oz, stride_oh, stride_om, stride_ok,
              Z, H, N_CTX,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, 
              HEAD_DIM: tl.constexpr):
    
    #get program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    #compute offsets
    #load Q block
    #initialize acc, l_i, m_i
    #call _attn_fwd_inner
    #normalize: acc / l_i
    #store output
    pass

# TODO: Python wrapper
def flash_attention(q, k, v, causal=False, sm_scale=None):
    BLOCK_M, BLOCK_N = 64, 64
    #compute grid
    #launch kernel
    #return output
    pass

# TODO: benchmark
def benchmark():
    #compare to torch.nn.functional.scaled_dot_product_attention
    #or naive attention implementation
    pass