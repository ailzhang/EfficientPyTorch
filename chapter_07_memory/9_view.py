import torch

shape = [256, 1024, 1024]
t = torch.ones(shape, device="cuda:0")

print(f"Current memory used: {torch.cuda.memory_allocated()/1024/1024/1024}GB")
# Current memory used: 1.0GB

v1 = t.view(-1)
v1[0] = -1  # t[0][0][0]也被更新了
assert v1[0] == t[0][0][0] == -1
print(f"Current memory used: {torch.cuda.memory_allocated()/1024/1024/1024}GB")
# Current memory used: 1.0GB


v2 = t[0]
v2[0][1] = 2  # t[0][0][1]也被更新了
assert v2[0][1] == t[0][0][1] == 2
print(f"Current memory used: {torch.cuda.memory_allocated()/1024/1024/1024}GB")
# Current memory used: 1.0GB
