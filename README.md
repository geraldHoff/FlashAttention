# FlashAttention

CUDA/C++: https://github.com/Dao-AILab/flash-attention/
Triton: https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py 

This is a Triton implementation of Flash Attention v2.

# Flash Attention Algorithm

## Core Attention Algorithm Mapping

    Standard Attention: Attention(Q,K,V) = softmax(QK^T / √d) V
    This implementation breaks this into memory-efficient blocks using online softmax with running statistics.

## Key Structural Components
1. Inner Computation Loop (_attn_fwd_inner)
    This is the core part of the algorithm, processes attention in blocks:

    QK Computation: Loads K blocks, computes dot products with Q
    Online Softmax with Numerical Stability:

    Tracks running maximum (m_i, m_ij) to prevent overflow
    Computes correction factors (alpha) when the max changes
    Updates running sum of exponentials (l_i, l_ij)


    Three Processing Stages:

    Stage 1: Non-causal tokens (before current position)
    Stage 2: Causal diagonal (requires masking)
    Stage 3: All tokens (non-causal mode)


    Output Accumulation: Computes weighted sum with V, correcting for changing statistics

2. Main Forward Kernel (_attn_fwd, lines 179-271)
Orchestrates the computation:

    Initializes accumulators (acc, l_i, m_i)
    Handles tensor descriptors for efficient memory access
    Calls _attn_fwd_inner for each stage
    Performs final normalization: output = acc / l_i
    Stores the running max M for backward pass
