import torch
import torch.nn as nn

def matrix_ops(size=1024):
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    for _ in range(10):
        c = torch.mm(a, b)
        d = torch.sigmoid(c)
        e = d * a
    return e

def inefficient_loop(tensor):
    result = []
    for i in range(tensor.size(0)):
        row_sum = 0
        for j in range(tensor.size(1)):
            row_sum += tensor[i][j].item()
        result.append(row_sum)
    return result

def run():
    print("Running matrix operations...")
    output = matrix_ops()
    print(f"Matrix ops done. Output shape: {output.shape}")
    print("Running inefficient loop...")
    data = torch.randn(512, 512)
    sums = inefficient_loop(data)
    print(f"Done. Sum Sample: {sums[0]:.4f}")



run()
