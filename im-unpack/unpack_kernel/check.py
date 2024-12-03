from kernel import construct_matrix, profile, row_unpack, col_unpack
import torch
from unpack import *
import cupy
import time
def reverse_with_einsum(S, c1, c2, exp_factor):
    target_rows = len(c1)
    target_cols = len(c2)
    exp_factor = exp_factor - 1
    total_rows = c1[-1] 
    total_cols = c2[-1]

    # 构建行权重矩阵
    W_row = torch.zeros((target_rows, total_rows), dtype=S.dtype, device=S.device)
    row_start = 0
    for i in range(target_rows):
        row_end = c1[i]
        n_rows = row_end - row_start
        idxs = torch.arange(n_rows, device=S.device)
        weights = (2 ** exp_factor) ** idxs
        W_row[i, row_start:row_end] = weights
        row_start = row_end
    #print(W_row)
    # 构建列权重矩阵
    W_col = torch.zeros((target_cols, total_cols), dtype=S.dtype, device=S.device)
    col_start = 0
    for k in range(target_cols):
        col_end = c2[k]
        n_cols = col_end - col_start
        idxs = torch.arange(n_cols, device=S.device)
        weights = (2 ** exp_factor) ** idxs
        W_col[k, col_start:col_end] = weights
        col_start = col_end
    #print(W_col)
    # 使用 einsum 进行矩阵乘法
    final_matrix = torch.einsum('ij,jk,kl->il', W_row, S, W_col.T)

    return final_matrix
if __name__ == "__main__":
    # define the input matrix
    x = torch.empty(2, 1024).cuda()  # 创建一个空的张量，大小为 (2, 1024) 在 GPU 上
    x[0, :] = 0.91  # 第一行全为 8
    x[1, :] = 0.23  # 第二行全为 9
    sc = 127 / 0.0177 * 0.5
    # transform Torch 
    #print(x.to(torch.float32))
    # print the result
    cum_cnt, x_cuda = row_unpack(x.to(torch.float32), 127 / 0.0177 * 0.5, bits = 8)
    x_other = x_cuda.to(torch.int32)
    #print(x_other)
    #print(cum_cnt)
    x_cuda = cupy.asarray(x_cuda)
    x_other = cupy.asarray(x_other)
    #print(x_cuda)
    xx_cuda = cupy.matmul(x_cuda,x_other.T)
    #print(xx_cuda)
    xx_cuda = torch.as_tensor(xx_cuda, device='cuda')
    xx_cuda = xx_cuda.to(dtype=torch.float32)
    start_time = time.time()

    xx = reverse_with_einsum(xx_cuda,cum_cnt,cum_cnt,8)
    end_time = time.time()

    print(f"函数运行时间: {end_time - start_time:.2f} 秒")
    #print(xx)
    print(torch.matmul(x, x.T))
    print(xx/sc/sc)
    

    
    
   
    