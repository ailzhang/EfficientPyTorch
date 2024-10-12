import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import SimpleNet, MyTrainDataset


def train(model, optimizer, train_data, device_id):
    model = model.to(device_id)
    for i, (src, target) in enumerate(train_data):
        src = src.to(device_id)
        target = target.to(device_id)
        optimizer.zero_grad()
        output = model(src)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f"[GPU{device_id}]: batch {i}/{len(train_data)}, loss: {loss}")


def main(device_id):
    model = SimpleNet()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    batchsize_per_gpu = 32
    dataset = MyTrainDataset(num=2048, size=512)
    train_data = DataLoader(dataset, batch_size=batchsize_per_gpu)

    train(model, optimizer, train_data, device_id)


if __name__ == "__main__":
    device_id = 0
    main(device_id)
