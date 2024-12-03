import torch
import math
import json
from unpack import unpack_row, unpack_column, unpack_both, unpack, scaled_matmul, pack_row, pack_transposed_row

A = (torch.randn(1, 2) * 8).int()
B = (torch.randn(4, 2) * 8).int()
C = torch.matmul(A, B.T)
bit_width = 4
print(A)
print(B)
print(C)
scales = torch.ones(A.shape[1], device = A.device, dtype = A.dtype)
Au, Be, APi_indices, APi_scales, scales_u = unpack(A, B, scales, bit_width, unpack_row)
#Beu, Aue, BPi_indices, BPi_scales, scales_uu = unpack(Be, Au, scales_u, bit_width, unpack_row)
print(Au)
print(Be)
print(APi_indices)
print(APi_scales)
print(scales_u)
AueSuuBeu = scaled_matmul(Au, Be, scales_u)
APiAueSuuBeu = pack_row(AueSuuBeu, APi_indices, APi_scales)
#APiAueSuuBeuBPi = pack_transposed_row(APiAueSuuBeu, BPi_indices, BPi_scales)

print((C - APiAueSuuBeu).abs().max())