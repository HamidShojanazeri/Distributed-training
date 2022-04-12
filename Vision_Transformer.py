import torch
import os
import torch.distributed as dist
from typing import Dict, Union, Any, Tuple
from torch.utils.data import Dataset
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
import argparse
from vit_pytorch.deepvit import DeepViT

def is_sm_run():
    return "TRAINING_JOB_NAME" in os.environ


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', choices=['tiny', 'small', 'medium', 'large'], default='large', type=str)
    parser.add_argument('--strategy', choices=['fsdp', 'cpu'], default='fsdp')
    parser.add_argument('--model_dir', default='/tmp', type=str)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('-mp', '--mixed_precision', default=False, action='store_true')
    args, _ = parser.parse_known_args()
    return args


def initialize_process_group(setup_args: Dict[str, Union[int, str]], backend: str = 'nccl') -> None:
    """
    Initialize process group.
    """
    master_addr, master_port = setup_args['master_addr'], setup_args['master_port']
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist_url = f'tcp://{master_addr}:{master_port}'
    args = {'backend': backend,
            'rank': setup_args['global_rank'],
            'world_size': setup_args['world_size'],
            # 'init_method': dist_url
            }
    dist.init_process_group(**args)


def get_setup_defaults(local_rank: int) -> Dict[str, Union[str, int]]:
    gpus_per_node = torch.cuda.device_count()
    world_size = get_num_nodes() * gpus_per_node
    node_rank = get_node_rank()
    global_rank = (node_rank * gpus_per_node) + local_rank
    # print(f'global rank is {global_rank}')
    # default_port = 12355#2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    ddp_setup_args = {'global_rank': global_rank,
                      'node_rank': node_rank,
                      'local_rank': local_rank,
                      'world_size': world_size,
                      'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
                      'master_port': '12355'}#os.environ.get('MASTER_PORT', str(default_port))}
    return ddp_setup_args


def run_fsdp(local_rank: int, *args: Any) -> None:
    args = args[0]
    setup_args = get_setup_defaults(local_rank=local_rank)
    initialize_process_group(setup_args)
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    dataset = FakeDataset()
    log_every = args.log_every
    model = build_model(args.model_size)

    model = model.to(torch.cuda.current_device())

    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')

    from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
    from torch.distributed.fsdp.wrap import default_auto_wrap_policy
    my_auto_wrap_policy = partial(default_auto_wrap_policy, min_num_params=20000)
    model = FullyShardedDataParallel(model,
                                     fsdp_auto_wrap_policy=my_auto_wrap_policy,
                                     cpu_offload=CPUOffload(offload_params=False))

    if torch.distributed.get_rank() == 0:
        print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, amsgrad=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4,
                                              pin_memory=False)
    loss_function = torch.nn.CrossEntropyLoss()
    mixed_precision = args.mixed_precision

    t0 = time.perf_counter()
    for batch_index, (inputs, target) in enumerate(data_loader, start=1):
        inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
            target.to(torch.cuda.current_device()), -1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(mixed_precision):
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_index % log_every == 0 and torch.distributed.get_rank() == 0:
            print(f'step: {batch_index}: time taken for the last {log_every} steps is {time.perf_counter() - t0}')
            t0 = time.perf_counter()

    dist.destroy_process_group()


def get_num_nodes() -> int:
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
        return len(cluster_inf['hosts'])
    return 1


def get_node_rank() -> int:
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
        return cluster_inf['hosts'].index(cluster_inf['current_host'])
    return 0


class FakeDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        self._input_shape = kwargs.get('input_shape', [3, 256, 256])
        self._input_type = kwargs.get('input_type', torch.float32)
        self._len = kwargs.get('len', 1000000)
        self._num_classes = kwargs.get('num_classes', 1000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_image = torch.randn(self._input_shape, dtype=self._input_type)
        label = torch.tensor(data=[index % self._num_classes], dtype=torch.int64)
        return rand_image, label


def build_model(model_size: str):
    model_args = dict()
    if model_size == 'tiny':
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 1,
            "heads": 1,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
    if model_size == 'small':
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 59,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
    if model_size == 'medium':
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 357,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
    if model_size == 'large':
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 952,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
    model = DeepViT(**model_args)

    return model




if __name__ == '__main__':
    args = parse_args()
    gpus_per_machine = torch.cuda.device_count()
    mp.spawn(fn=run_fsdp,
             args=(args,),
             nprocs=gpus_per_machine,
             join=True)
