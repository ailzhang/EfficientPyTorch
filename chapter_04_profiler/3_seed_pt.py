import torch


def generate_random_seq(device):
    return torch.rand((3, 3), device=device)


print(
    f"""不设置随机种子时，每次运行生成的序列都是不同的
CPU: {generate_random_seq('cpu')}
CUDA: {generate_random_seq('cuda')}"""
)

# 为所有PyTorch后端设置生成随机数的种子
seed = 32
torch.manual_seed(seed)

print(
    f"""设置随机种子后，每次运行都会生成相同的序列
CPU: {generate_random_seq('cpu')}
CUDA: {generate_random_seq('cuda')}"""
)

# 第一次运行代码结果
# 不设置随机种子时，每次运行生成的序列都是不同的
# CPU: tensor([[0.8485, 0.6379, 0.6855],
#         [0.0954, 0.7357, 0.3545],
#         [0.9822, 0.1272, 0.9752]])
# CUDA: tensor([[0.5688, 0.7038, 0.6558],
#         [0.1524, 0.8050, 0.7368],
#         [0.5904, 0.2899, 0.4835]], device='cuda:0')
# 设置随机种子后，每次运行都会生成相同的序列
# CPU: tensor([[0.8757, 0.2721, 0.4141],
#         [0.7857, 0.1130, 0.5793],
#         [0.6481, 0.0229, 0.5874]])
# CUDA: tensor([[0.6619, 0.2778, 0.7292],
#         [0.8970, 0.0063, 0.7033],
#         [0.9305, 0.2407, 0.3767]], device='cuda:0')

# 相同代码，第二次运行结果
# 不设置随机种子时，每次运行生成的序列都是不同的
# CPU: tensor([[0.3968, 0.4038, 0.7816],
#         [0.1577, 0.8753, 0.8638],
#         [0.3971, 0.2644, 0.1432]])
# CUDA: tensor([[0.4933, 0.2223, 0.5825],
#         [0.6528, 0.9796, 0.3861],
#         [0.7478, 0.2834, 0.7953]], device='cuda:0')
# 设置随机种子后，每次运行都会生成相同的序列
# CPU: tensor([[0.8757, 0.2721, 0.4141],
#         [0.7857, 0.1130, 0.5793],
#         [0.6481, 0.0229, 0.5874]])
# CUDA: tensor([[0.6619, 0.2778, 0.7292],
#         [0.8970, 0.0063, 0.7033],
#         [0.9305, 0.2407, 0.3767]], device='cuda:0')
