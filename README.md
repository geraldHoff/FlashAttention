# FlashAttention

This is a Triton implementation of Flash Attention v2.

Examples drawn from:
CUDA/C++: https://github.com/Dao-AILab/flash-attention/
Triton: https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Testing and benchmarking: https://colab.research.google.com/drive/1q8LwFCDiQNomSsqLe3T3mCkY4pq9TZVh?usp=sharing

# Flash Attention Algorithm

Standard Attention: Attention(Q,K,V) = softmax(QK^T / √d) V
This implementation breaks this into memory-efficient blocks using online softmax with running statistics.
Simplified algorithm, no tuning to hardware or causal masking

1. Inner Computation Loop (_attn_fwd_inner)
    Processes attention in blocks:

    QK Computation: Loads K blocks, computes dot products with Q
    Online Softmax with Numerical Stability:

    Tracks running maximum (m_i, m_ij)
    Computes correction factors (alpha)
    Updates running sum of exponentials (l_i, l_ij)

    Output Accumulation: Computes weighted sum with V, correcting for changing statistics

2. Main Forward Kernel (_attn_fwd, lines 179-271)
Orchestrates the computation:

    Initializes accumulators (acc, l_i, m_i)
    Handles tensor descriptors for efficient memory access
    Calls _attn_fwd_inner for each stage
    Performs final normalization: output = acc / l_i
    Stores the running max M for backward pass
