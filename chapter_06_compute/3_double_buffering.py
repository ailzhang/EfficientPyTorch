def train(model, optimizer, trainloader, num_iters):
    # Create two CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    submit_stream = stream1
    running_stream = stream2
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, batch in enumerate(trainloader, 0):
            if i >= num_iters:
                break

            with torch.cuda.stream(submit_stream):
                data = batch[0].cuda(non_blocking=True)
                submit_stream.wait_stream(running_stream)

                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = output.sum()

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Alternate between the two streams
            submit_stream = stream2 if submit_stream == stream1 else stream1
            running_stream = stream2 if running_stream == stream1 else stream1

    prof.export_chrome_trace(f"PROF_double_buffering_wait_after_data.json")
