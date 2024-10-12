import time

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

# 设置batchsize
batch_size = 4

transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=10)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


def train_num_batches(trainloader, model, device, num_batches):
    for i, data in enumerate(trainloader, 0):
        if i >= num_batches:
            break

        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


# 热身
train_num_batches(trainloader, model, device, num_batches=5)
num_batches = len(trainloader) / batch_size

start = time.perf_counter()
train_num_batches(trainloader, model, device, num_batches=num_batches)
torch.cuda.synchronize()
end = time.perf_counter() - start
print(f"batch_size={batch_size} 运行时间: {end * 1000} ms")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_num_batches(trainloader, model, device, num_batches=10)
prof.export_chrome_trace(f"traces/PROF_resnet18_batchsize={batch_size}.json")
