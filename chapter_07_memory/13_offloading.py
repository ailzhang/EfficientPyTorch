import torch
import torch.nn as nn


class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.layer1 = nn.Linear(50000, 50000)
        self.layer2 = nn.Linear(50000, 50000)

    # OOM on a GPU with 24GB
    # def forward(self, x):
    #     x = self.layer1(x)
    #     x = torch.relu(x)
    #     x = self.layer2(x)
    #     x = torch.relu(x)
    #     return x

    def forward(self, x):
        self.layer1.to("cuda")
        x = self.layer1(x)
        x = torch.relu(x)
        self.layer1.to("cpu")

        self.layer2.to("cuda")
        x = self.layer2(x)
        x = torch.relu(x)
        self.layer2.to("cpu")
        return x


model = LargeModel().to("cuda")
input_data = torch.randn(10, 50000).to("cuda")
output = model(input_data)

print(f"前向过程中GPU显存占用峰值: {torch.cuda.max_memory_allocated()/1024/1024/1024}GB")
# 前向过程中GPU显存占用峰值: 9.328798770904541GB

loss = output.sum()
loss.backward()
