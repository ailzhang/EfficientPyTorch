from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f"gpt_{config.model.model_type}")


...


model = GPT(config.model)
batch = [t.to(trainer.device) for t in next(iter(trainer.train_loader))]
writer.add_graph(model, batch)
writer.close()
