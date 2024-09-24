from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.


    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    inv_freq = inv_freq.to(device)  # Ensure frequencies are on the same device

    # 3. Compute the angles for each position
    # Positions range from 0 to seqlen-1
    positions = torch.arange(seqlen, device=device).unsqueeze(1)  # Shape: (seqlen, 1)
    angles = positions * inv_freq  # Shape: (seqlen, head_dim/2)

    # 4. Compute sine and cosine of the angles
    sin_angles = torch.sin(angles)  # Shape: (seqlen, head_dim/2)
    cos_angles = torch.cos(angles)  # Shape: (seqlen, head_dim/2)

    # 5. Reshape sin and cos for broadcasting
    # New shape: (1, seqlen, 1, head_dim/2)
    sin_angles = sin_angles[None, :, None, :]  # Add batch and head dimensions
    cos_angles = cos_angles[None, :, None, :]  # Add batch and head dimensions

    # 6. Apply the rotary transformation to the query
    # q_rotated = q_real * cos - q_imag * sin
    # q_imag_rotated = q_real * sin + q_imag * cos
    query_rotated_real = query_real * cos_angles - query_imag * sin_angles
    query_rotated_imag = query_real * sin_angles + query_imag * cos_angles

    # 7. Apply the rotary transformation to the key
    # k_rotated = k_real * cos - k_imag * sin
    # k_imag_rotated = k_real * sin + k_imag * cos
    key_rotated_real = key_real * cos_angles - key_imag * sin_angles
    key_rotated_imag = key_real * sin_angles + key_imag * cos_angles

    # 8. Recombine the real and imaginary parts
    # Stack along the last dimension and reshape back to original head_dim
    # Shape after stack: (batch_size, seqlen, n_heads, head_dim/2, 2)
    # Then reshape to (batch_size, seqlen, n_heads, head_dim)
    query_out = torch.stack([query_rotated_real, query_rotated_imag], dim=-1).reshape(*query.shape)
    key_out = torch.stack([key_rotated_real, key_rotated_imag], dim=-1).reshape(*key.shape)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out