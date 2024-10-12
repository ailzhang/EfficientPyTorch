import time
import torch


def my_work():
    # 需要计时的操作
    sz = 64
    x = torch.randn((sz, sz))


if __name__ == "__main__":
    # 热身
    num_warmup = 5
    for i in range(num_warmup):
        start = time.perf_counter()
        my_work()
        end = time.perf_counter()
        t = end - start
        print(f"热身#{i}: {t * 1000 :.6f}ms")

    # 多次运行取平均
    repeat = 30
    start = time.perf_counter()
    for _ in range(repeat):
        my_work()
    end = time.perf_counter()

    t = (end - start) / repeat
    print(f"{repeat}次取平均: {t * 1000:.6f}ms")

# 热身#0: 0.317707ms
# 热身#1: 0.023586ms
# 热身#2: 0.016913ms
# 热身#3: 0.016409ms
# 热身#4: 0.015868ms
# 30次取平均: 0.014164ms
