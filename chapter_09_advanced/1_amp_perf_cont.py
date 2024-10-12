N, C, H, W = 32, 3, 256, 256  # Example dimensions

data = torch.randn(10, N, C, H, W, device="cuda")
dataset = TensorDataset(data)

model = SimpleCNN(C).to("cuda")

# warm up
train(dataset, model, use_amp=False)
torch.cuda.synchronize()
# 测量未使用AMP时的时间和性能图谱
start_time = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train(dataset, model, use_amp=False)
    torch.cuda.synchronize()
prof.export_chrome_trace("traces/PROF_wo_amp.json")
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Float32 Time: {elapsed} seconds")

# warm up
train(dataset, model, use_amp=True)
torch.cuda.synchronize()
# 测量使用AMP后的时间和性能图谱
start_time = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train(dataset, model, use_amp=True)
    torch.cuda.synchronize()
prof.export_chrome_trace("traces/PROF_amp.json")
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"Float16 Time: {elapsed} seconds")
