import os
import sys

import torch
import time
import numpy as np

from mingpt.model import GPT
from mingpt.trainer import Trainer, accelerator
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from mingpt.char_dataset import CharDataset

from torch.profiler import profile, ProfilerActivity

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt2-large'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    # configs
    C.trainer.batch_size = 32
    C.trainer.num_workers = 4

    return C

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    print(config)

    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)
    trainer.prepare()

    # warm up
    trainer.run(10)
    torch.cuda.synchronize()

    # Memory
    MB = 1024 * 1024
    print("Peak allocated VRAM: ", torch.cuda.max_memory_allocated() / MB, " MB")

    # evaluate time
    measured_runtimes = []

    num_samples = 1024 // accelerator.state.num_processes
    num_repeats = 5
    for i in range(num_repeats):
        start = time.perf_counter()

        trainer.run_num_samples(num_samples)

        torch.cuda.synchronize()
        end = time.perf_counter()
        measured_runtimes.append(end - start)

    average_runtime = sum(measured_runtimes) / len(measured_runtimes)
    print("Ave Runtime: ", average_runtime, " seconds")
    print("Std: ", np.std(measured_runtimes), " seconds")
