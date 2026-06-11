import torch
import numpy as np

print("✅ 导入成功！开始测试 Torch 和 NumPy 的友谊小船...")

# 1. 用 NumPy 创建一个数组
numpy_array = np.array([1, 2, 3, 4, 5])
print(f"这是 NumPy 的数组: {numpy_array}")

# 2. 把 NumPy 数组直接变成 PyTorch 的张量 (Tensor)
torch_tensor = torch.from_numpy(numpy_array)
print(f"这是转换成 PyTorch 的张量: {torch_tensor}")

# 3. 检验一下它们的内存是不是共享的（深层联系）
numpy_array[0] = 999
print(f"\n🔄 修改 NumPy 数组后，PyTorch 张量变了吗？")
print(f"PyTorch 张量变成了: {torch_tensor}")

if torch_tensor[0] == 999:
    print("\n🎉 完美！NumPy 和 PyTorch 已经手拉手成功会师！你可以开始你的 AI 大业了！")
else:
    print("❌ 好像哪里不对劲...")