import torch
import time
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.optim import SGD
from torch.utils.data import TensorDataset


class SimpleCNN(nn.Module):
    def __init__(self, input_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        return out


def train(dataset, model, use_amp):
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for batch_data in dataset:
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            result = model(batch_data[0])
            loss = result.sum()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
