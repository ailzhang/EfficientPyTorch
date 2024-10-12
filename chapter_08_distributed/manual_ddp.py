import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import SimpleNet, MyTrainDataset

import os
import torch.distributed as dist
import torch.multiprocessing as mp


# (3) 初始化分布式通信组
def setup(rank, device_id, world_size, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    torch.cuda.set_device(device_id)


def train(model, optimizer, train_data, rank, device_id, world_size):
    for i, (src, target) in enumerate(train_data):
        src = src.to(device_id)
        target = target.to(device_id)
        optimizer.zero_grad()
        output = model(src)
        loss = F.mse_loss(output, target)
        loss.backward()

        # (5) 每个批次训练结束后进行梯度同步
        grads = [t.grad.data for t in model.parameters()]
        for grad in grads:
            grad.div_(world_size)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        optimizer.step()
        print(f"[GPU{rank}]: batch {i}/{len(train_data)}, loss: {loss}")


def main(rank, world_size, backend):
    device_id = rank
    setup(rank, device_id, world_size, backend)

    model = SimpleNet().to(device_id)

    # (4) 初始化模型并参数同步
    params = [t.detach() for t in model.parameters()]
    for param in params:
        dist.broadcast(param, 0)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    batchsize_per_gpu = 4096
    dataset = MyTrainDataset(num=40960, size=512)

    # (1) 数据分割
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    train_data = DataLoader(dataset, batch_size=batchsize_per_gpu, sampler=sampler)

    train(model, optimizer, train_data, rank, device_id, world_size)


if __name__ == "__main__":
    # (2) 多进程启动和管理
    world_size = 2
    mp.spawn(main, args=(world_size, "nccl"), nprocs=world_size, join=True)
