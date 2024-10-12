def train(model, optimizer, trainloader, num_iters):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, batch in enumerate(trainloader, 0):
            if i >= num_iters:
                break
            data = batch[0].cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()

            loss.backward()
            optimizer.step()

    prof.export_chrome_trace(f"traces/PROF_non_blocking.json")


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize([512, 512])]
)
trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, pin_memory=True, num_workers=4)


# non_blocking
train(model, optimizer, trainloader, num_iters=20)
