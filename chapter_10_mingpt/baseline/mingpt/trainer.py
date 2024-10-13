"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from torch.profiler import profile, record_function, ProfilerActivity

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 0
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 32
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def prepare(self):
        # No.4 Enable Torch Compile
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.optimizer = self.model.configure_optimizers(self.config)

        # No.5 Enable AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # setup the dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        self.model.train()

    def run_num_samples(self, num_samples):
        batch_size = self.config.batch_size
        assert num_samples % batch_size == 0

        num_iters = num_samples // batch_size
        self.run(num_iters)

    def run(self, max_num_iters):
        iter_num = 0
        while True:
            for batch in self.train_loader:
                with record_function(f"train_{iter_num}"):
                    # No.3 non_blocking
                    batch = [t.to(self.device, non_blocking=True) for t in batch]
                    x, y = batch

                    # forward the model
                    # No.4 Enable AMP
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits, self.loss = self.model(x, y)

                    # backprop and update the parameters
                    self.model.zero_grad(set_to_none=True)

                    # No.4 Enable AMP
                    self.scaler.scale(self.loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.trigger_callbacks('on_batch_end')
                    iter_num += 1

                    if iter_num >= max_num_iters:
                        return
