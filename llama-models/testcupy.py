import torch

def reverse(S, c1, c2, exp_factor):
    target_rows = len(c1)
    target_cols = len(c2)
    exp_factor = exp_factor - 1
    restored_matrix = torch.zeros((target_rows, target_cols), dtype=S.dtype, device=S.device)
    row_start = 0
    for i in range(target_rows):
        row_end = c1[i]
        weight = 1  
        for j in range(row_start, row_end):
            restored_matrix[i, :] += S[j, :] * weight
            weight *= 2 ** exp_factor  
        row_start = row_end
    print(restored_matrix)
    col_start = 0
    final_matrix = torch.zeros_like(restored_matrix)  
    for k in range(target_cols):
        col_end = c2[k]
        weight = 1  
        for l in range(col_start, col_end):
            final_matrix[:, k] += restored_matrix[:, l] * weight
            weight *= 2 ** exp_factor
        col_start = col_end
    
    return final_matrix


# 测试代码
if __name__ == "__main__":
    # 示例输入
    S = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]], dtype=torch.float32)
    c1 = [2, 4, 7]  # 行累加和
    c2 = [1, 2]     # 列累加和
    exp_factor = 4  # 指数因子

    # 调用函数
    restored = revers(S, c1, c2, exp_factor)

    # 打印结果
    print("原始矩阵 S:")
    print(S)
    print("\n还原后的矩阵:")
    print(restored)