import os
import torch
import torch.distributed as dist
from typing import Dict, Union, Any, Tuple
from torch.utils.data import Dataset
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
import argparse
from vit_pytorch.deepvit import DeepViT, Residual
import gc
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
# from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import subprocess
# bfloat16 support verification imports (network and gpu native support)
import torch.cuda.nccl as nccl
from distutils.version import LooseVersion

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)


os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
# os.environ['NCCL_ALGO'] = 'RING'
# os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['RDMAV_FORK_SAFE'] = '1'
# os.environ['NCCL_MIN_NRINGS'] = '8'
# os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'


# torch.backends.cuda.matmul.allow_tf32 = True

bf16_ready = (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
)

# save memory as gb
gb_unit_size = 1024 ** 3


def is_sm_run():
    return "TRAINING_JOB_NAME" in os.environ


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', choices=['tiny', 'small', 'medium', 'large'], default='large' if is_sm_run() else 'tiny', type=str)
    parser.add_argument('--strategy', choices=['fsdp', 'cpu'], default='fsdp' if is_sm_run() else 'cpu')
    parser.add_argument('--model_dir', default='/tmp', type=str)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--activation_checkpointing',  action='store_true', default=False)
    # parser.add_argument('-mp', '--mixed_precision', default=False, action='store_true')
    args, _ = parser.parse_known_args()
    return args


def initialize_process_group(setup_args: Dict[str, Union[int, str]], backend: str = 'nccl') -> None:
    """
    Initialize process group.
    """
    master_addr, master_port = setup_args['master_addr'], setup_args['master_port']
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    args = {'backend': backend if torch.cuda.is_available() else 'gloo',
            'rank': setup_args['global_rank'],
            'world_size': setup_args['world_size']}
    dist.init_process_group(**args)


def get_setup_defaults(local_rank: int) -> Dict[str, Union[str, int]]:
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    world_size = get_num_nodes() * gpus_per_node
    node_rank = get_node_rank()
    global_rank = (node_rank * gpus_per_node) + local_rank
    print(f'local rank {local_rank} global rank {global_rank} world size {world_size}')
    ddp_setup_args = {'global_rank': global_rank,
                      'node_rank': node_rank,
                      'local_rank': local_rank,
                      'world_size': world_size,
                      'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
                      'master_port': '12355'}  # os.environ.get('MASTER_PORT', str(default_port))}
    return ddp_setup_args


def run_fsdp(local_rank: int, *args: Any) -> None:

    print('/opt/amazon/efa/bin/fi_info')
    subprocess.run(["/opt/amazon/efa/bin/fi_info", "-p", "efa"])
    print('ls -l /dev/infiniband/uverbs0')
    subprocess.run(["ls", "-l", "/dev/infiniband/uverbs0"])
    
    # print_efa_info()
    # fsdp params count (min_num_params)
    fsdp_params_count_min = 20000

    # mixed precision policies

    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    # verify b16 native support:

    mp_policy = None
    # bf16_ready = False
    if bf16_ready:
        mp_policy = bfSixteen  # set to None to run with fp32
        if local_rank == 0:
            print(f"--> Running with bfloat16 mixed precision")
    else:
        if local_rank == 0:
            print(f"--> Warning - bf16 support not available.  Reverting to fp32")

    args = args[0]
    setup_args = get_setup_defaults(local_rank=local_rank)
    initialize_process_group(setup_args)
    if torch.cuda.is_available() and torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    print(f"finished init for local rank {local_rank}")

    dataset = FakeDataset()
    log_every = args.log_every
    model = build_model(args.model_size)

    # model = model.to(torch.cuda.current_device())
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')

    from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    # gc.collect()
    # torch.cuda.synchronize()
    my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Residual,})
    fetching_policy = BackwardPrefetch(BackwardPrefetch.BACKWARD_PRE)
    # model = FullyShardedDataParallel(model,
    #                                  auto_wrap_policy=my_auto_wrap_policy,
    #  
    #                                cpu_offload=CPUOffload(offload_params=True))
    check_fn = lambda submodule: isinstance(submodule, Residual)
    non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    model = FullyShardedDataParallel(model,
                                     auto_wrap_policy=my_auto_wrap_policy,
                                    #  mixed_precision=mp_policy,
                                     sharding_strategy=ShardingStrategy.FULL_SHARD, # SHARD_GRAD_OP
                                    #  device_id=torch.cuda.current_device(),
                                    #  backward_prefetch = fetching_policy,
                                    #  cpu_offload=CPUOffload(offload_params=True)
                                    #  sharding_strategy=ShardingStrategy.SHARD_GRAD_OP# SHARD_GRAD_OP
                                
                                     )

    # move to gpu
    if torch.cuda.is_available():
        model.to(torch.cuda.current_device()) 
    
    # scaler = ShardedGradScaler()

   

    apply_activation_checkpointing_wrapper(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    # if torch.distributed.get_rank() == 0:
    #     print(model)

    batch_size_training = 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, amsgrad=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_training, num_workers=16,
                                              pin_memory=False)
    loss_function = torch.nn.CrossEntropyLoss()
    # mixed_precision = args.mixed_precision
    # memory and timing tracking
    if torch.distributed.get_rank() == 0:
        tracking_mem_allocs = []
        tracking_mem_reserved = []
        tracking_duration = []
    print( "data loader size *****************", len(data_loader))
    t0 = time.perf_counter()
    for batch_index, (inputs, target) in enumerate(data_loader, start=1):
        if torch.cuda.is_available():
            inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
                target.to(torch.cuda.current_device()), -1)
        else:
            targets = torch.squeeze(target,-1)
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast(mixed_precision):
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()

        # update durations and memory tracking
        if torch.cuda.is_available() and torch.distributed.get_rank() == 0:
            mini_batch_time = time.perf_counter() - t0
            tracking_duration.append(mini_batch_time)
            tracking_mem_allocs.append(torch.cuda.memory_allocated() / gb_unit_size)
            tracking_mem_reserved.append(torch.cuda.memory_reserved() / gb_unit_size)

        if batch_index % log_every == 0 and torch.distributed.get_rank() == 0:
            print(f'step: {batch_index}: time taken for the last {log_every} steps is {mini_batch_time}')

        # reset timer
        t0 = time.perf_counter()
        if batch_index > 20:
            break

    # memory summary
    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:

        stable_sum = sum(tracking_duration[1:])
        stable_avg = stable_sum / 20
        stable_avg = round(stable_avg, 4)
        print(f"minibatch durations Average: {stable_avg}")
        print(f"******************************************")
        print(f"minibatch durations: {tracking_duration}")
        print(f"running mem allocs: {tracking_mem_allocs}")
        print(f"running mem reserved: {tracking_mem_reserved}")
        print(
            f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}"
        )

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

def print_efa_info():
    import subprocess
    print('/opt/amazon/efa/bin/fi_info')
    subprocess.run(["/opt/amazon/efa/bin/fi_info", "-p", "efa"])
    print('ls -l /dev/infiniband/uverbs0')
    subprocess.run(["ls", "-l", "/dev/infiniband/uverbs0"])

if __name__ == '__main__':
    if get_node_rank() == 0:
        import sys, shlex, subprocess

        cmd = shlex.split(
            f'{sys.executable} -m torch.utils.collect_env'
        )
        proc = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        print(proc.stdout)
    args = parse_args()
    gpus_per_machine = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(fn=run_fsdp,
             args=(args,),
             nprocs=gpus_per_machine,
             join=True)
