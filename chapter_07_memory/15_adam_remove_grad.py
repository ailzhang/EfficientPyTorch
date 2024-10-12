torch.cuda.memory._record_memory_history()

device = "cuda:0"
shape = [4]
params = generate_params(device, shape)
out = run(params)

# 设置一个优化器字典，方便后续引用
optimizer_dict = {
    p: torch.optim.Adam([p], foreach=False) for p in [w0, w1, w2, w3, w4, w5]
}


# 定义一个优化器钩子，这个钩子会调用step()和zero_grad()函数
def optimizer_hook(parameter) -> None:
    optimizer_dict[parameter].step()
    optimizer_dict[parameter].zero_grad()


# 设置钩子在梯度更新后被调用
for p in model.parameters():
    p.register_post_accumulate_grad_hook(optimizer_hook)

out.backward()

torch.cuda.memory._dump_snapshot("traces/adam_remove_grad.pickle")
