import torch
import torch.nn as nn
from torch.utils.data import Dataset


def set_seed(seed: int = 37) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)


class MyTrainDataset(Dataset):
    def __init__(self, num, size):
        self.num = num
        self.data = [
            (
                torch.rand(size, dtype=torch.float),
                torch.tensor([i / num], dtype=torch.float),
            )
            for i in range(num)
        ]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index]


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(512, 10240, bias=True)
        self.fc2 = nn.Linear(10240, 10240, bias=True)
        self.fc3 = nn.Linear(10240, 1, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
