from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook

model.register_comm_hook(None, noop_hook)
