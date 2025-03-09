import torch

print(torch.cuda.is_available())  # 返回 True 表示 GPU 可用
print(torch.cuda.get_device_name(0))  # 打印 GPU 名称
