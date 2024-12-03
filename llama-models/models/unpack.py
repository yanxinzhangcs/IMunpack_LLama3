import torch

# Unpacks the matrix A column-wise with respect to sparsity.
# If the elements of A are greater than or equal to a threshold based on bit width, it splits them into lower and higher bits.
# A and B are unpacked simultaneously, and the scaling factors are adjusted accordingly.
def unpack_column(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)  # Threshold value based on the bit width
    unpacked_A = [A.clone().detach()]  # List to store the unpacked A matrices
    unpacked_B = [B.clone().detach()]  # List to store the unpacked B matrices
    scales = [scales]  # List to store scaling factors

    while True:
        # Split matrix A into low and high bit values
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale

        # Calculate sparsity by determining how many elements exceed the scale threshold
        sparsity = torch.mean((unpacked_A[-1].abs() >= scale).float(), dim=0)
        sparsity_mask = sparsity > 0  # Create a mask for columns with sparse values
        count_sparsity = sparsity_mask.int().sum().item()

        if count_sparsity == 0:  # If no sparsity, stop unpacking
            break

        # Replace the sparse columns with their low bit values and store the high bit values
        unpacked_A[-1][:, sparsity_mask] = low_bit_vals[:, sparsity_mask]
        unpacked_A.append(high_bit_vals[:, sparsity_mask])
        unpacked_B.append(unpacked_B[-1][:, sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)  # Update the scaling factors

    # Concatenate the unpacked results
    unpacked_A = torch.cat(unpacked_A, dim=1)
    unpacked_B = torch.cat(unpacked_B, dim=1)
    scales = torch.cat(scales, dim=0)
    return unpacked_A, unpacked_B, scales

# Unpacks the matrix A row-wise in a similar fashion to column-wise unpacking.
def unpack_row(A, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = [A.clone().detach()]
    indices = [torch.arange(A.shape[0], device=A.device)]  # List to store row indices
    scales = [torch.ones(A.shape[0], device=A.device, dtype=torch.int)]  # List to store scaling factors

    while True:
        # Split the matrix into low and high bit values
        low_bit_vals = unpacked_A[-1] % scale
        high_bit_vals = unpacked_A[-1] // scale

        # Calculate sparsity row-wise
        sparsity = torch.mean((unpacked_A[-1].abs() >= scale).float(), dim=1)
        sparsity_mask = sparsity > 0  # Mask for rows with sparse values
        count_sparsity = sparsity_mask.int().sum().item()

        if count_sparsity == 0:  # If no sparsity, stop unpacking
            break

        # Replace sparse rows with their low bit values and store the high bit values
        unpacked_A[-1][sparsity_mask, :] = low_bit_vals[sparsity_mask, :]
        unpacked_A.append(high_bit_vals[sparsity_mask, :])
        indices.append(indices[-1][sparsity_mask])
        scales.append(scales[-1][sparsity_mask] * scale)

    # Concatenate the unpacked results
    unpacked_A = torch.cat(unpacked_A, dim=0)
    indices = torch.cat(indices, dim=0)
    scales = torch.cat(scales, dim=0)
    return unpacked_A, indices, scales

# Expands matrix M by adding extra rows or columns depending on the dimension specified
def expend_mat(M, size, dim):
    if dim == 0:  # Expand by adding extra rows
        extra = torch.zeros(size, M.shape[1], dtype=M.dtype, device=M.device)
        return torch.cat([M, extra], dim=0)
    elif dim == 1:  # Expand by adding extra columns
        extra = torch.zeros(M.shape[0], size, dtype=M.dtype, device=M.device)
        return torch.cat([M, extra], dim=1)
    else:
        raise Exception()  # Raise an error for invalid dimensions

# Expands a vector v by adding extra elements
def expend_vec(v, size):
    extra = torch.zeros(size, dtype=v.dtype, device=v.device)
    return torch.cat([v, extra], dim=0)

# Unpacks both rows and columns of matrix A and matrix B using a more general approach
def unpack_both(A, B, scales, bit_width):
    scale = 2 ** (bit_width - 1)
    unpacked_A = A.clone().detach()
    unpacked_B = B.clone().detach()
    Pi_indices = torch.arange(A.shape[0], device=A.device)  # Row indices for A
    Pi_scales = torch.ones(A.shape[0], device=A.device, dtype=torch.int)  # Scale factors for rows

    insert_pointer_i = A.shape[0]
    insert_pointer_j = A.shape[1]

    while True:
        # Mask to identify values that exceed the scale threshold
        sparsity_mask = unpacked_A.abs() >= scale
        col_sparsity = torch.sum(sparsity_mask.int(), dim=1)  # Column-wise sparsity
        row_sparsity = torch.sum(sparsity_mask.int(), dim=0)  # Row-wise sparsity

        col_val, col_idx = torch.max(col_sparsity, dim=0)
        row_val, row_idx = torch.max(row_sparsity, dim=0)

        if col_val == 0 and row_val == 0:  # Stop when no sparse values remain
            break

        if col_val >= row_val:  # Unpack by column
            if insert_pointer_i >= unpacked_A.shape[0]:  # Expand rows if needed
                unpacked_A = expend_mat(unpacked_A, A.shape[0], 0)
                Pi_indices = expend_vec(Pi_indices, A.shape[0])
                Pi_scales = expend_vec(Pi_scales, A.shape[0])

            vals = unpacked_A[col_idx, :]  # Unpack the selected column

            # Split values into high and low bit representations
            unpacked_A[insert_pointer_i, :] = vals // scale
            unpacked_A[col_idx, :] = vals % scale

            Pi_indices[insert_pointer_i] = Pi_indices[col_idx]
            Pi_scales[insert_pointer_i] = Pi_scales[col_idx] * scale
            insert_pointer_i += 1
        else:  # Unpack by row
            if insert_pointer_j >= unpacked_A.shape[1]:  # Expand columns if needed
                unpacked_A = expend_mat(unpacked_A, A.shape[1], 1)
                unpacked_B = expend_mat(unpacked_B, B.shape[1], 1)
                scales = expend_vec(scales, A.shape[1])

            vals = unpacked_A[:, row_idx]  # Unpack the selected row

            # Split values into high and low bit representations
            unpacked_A[:, insert_pointer_j] = vals // scale
            unpacked_A[:, row_idx] = vals % scale

            unpacked_B[:, insert_pointer_j] = unpacked_B[:, row_idx]
            scales[insert_pointer_j] = scales[row_idx] * scale
            insert_pointer_j += 1

    # Crop the unpacked matrices to their final sizes
    unpacked_A = unpacked_A[:insert_pointer_i, :insert_pointer_j]
    unpacked_B = unpacked_B[:, :insert_pointer_j]
    Pi_indices = Pi_indices[:insert_pointer_i]
    Pi_scales = Pi_scales[:insert_pointer_i]
    scales = scales[:insert_pointer_j]
    return unpacked_A, unpacked_B, Pi_indices, Pi_scales, scales

# Function that calls the appropriate unpacking strategy based on the input
def unpack(A, B, scales, bit_width, strategy):
    if strategy == unpack_row:
        A, Pi_indices, Pi_scales = unpack_row(A, bit_width)
    elif strategy == unpack_column:
        A, B, scales = unpack_column(A, B, scales, bit_width)
        Pi_indices = torch.arange(A.shape[0], device=A.device)
        Pi_scales = torch.ones(A.shape[0], device=A.device, dtype=A.dtype)
    elif strategy == unpack_both:
        A, B, Pi_indices, Pi_scales, scales = unpack_both(A, B, scales, bit_width)
    return A, B, Pi_indices, Pi_scales, scales

# Performs matrix multiplication between two unpacked matrices A and B, applying the appropriate scaling.
def scaled_matmul(unpacked_A, unpacked_B, scales):
    return torch.matmul(unpacked_A.float() * scales.float(), unpacked_B.T.float())

# Re-packs the rows of matrix A using the original indices and scaling factors.
def pack_row(A, indices, scales):
    A, scales = A.float(), scales.float()
    packed_A = torch.zeros(indices.max() + 1, A.shape[1], device=A.device, dtype=A.dtype)
    packed_A.index_add_(0, indices, A * scales[:, None])
    return packed_A

# Re-packs the transposed rows of matrix A using the original indices and scaling factors.
def pack_transposed_row(A, indices, scales):
    A, scales = A.float(), scales.float()
    packed_A = torch.zeros(A.shape[0], indices.max() + 1, device=A.device, dtype=A.dtype)
    packed_A.index_add_(1, indices, A * scales)
    return packed_A
