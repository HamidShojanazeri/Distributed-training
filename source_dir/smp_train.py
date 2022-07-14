import torch
import os
import torch.distributed as dist
from typing import Tuple
from torch.utils.data import Dataset
import time
import argparse
from vit_pytorch.deepvit import DeepViT
from apex import amp
gb_unit_size = 1024 ** 3

def is_sm_run():
    return "TRAINING_JOB_NAME" in os.environ


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', choices=['tiny', 'small', 'medium', 'large'], default='large', type=str)
    parser.add_argument('--model_dir', default='/tmp', type=str)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('-mp', '--mixed_precision', default=True, action='store_false')
    parser.add_argument('-smp', '--smp', default=False, action='store_true')
    parser.add_argument(
        "--allreduce_post_accumulation",
        default=1,
        type=int,
        help="Whether to do allreduces during gradient accumulation steps.",
    )
    parser.add_argument(
        "--allreduce_post_accumulation_fp16",
        default=1,
        type=int,
        help="Whether to do fp16 allreduce post accumulation.",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0.0,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    args, _ = parser.parse_known_args()
    return args


def get_num_nodes() -> int:
    """x
        return number of nodes in job cluster.
    Returns:
        integer
    """
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
        return len(cluster_inf['hosts'])
    return 1


def get_node_rank() -> int:
    """
        returns node rank
    Returns:
        int
    """
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

    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')

    return model


def register_tp(device):
    def init_hook(dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        kwargs = {
            "hidden_width": dim,
            "num_layers": depth,
            "num_attention_heads": heads,
            "attention_head_size": dim_head,
            "intermediate_size": mlp_dim,
            "attention_dropout_prob": dropout,
            "embedding_dropout_prob": dropout,
            "hidden_dropout_prob": dropout,
        }
        return (), kwargs

    def forward_hook(x):
        return ((x, torch.zeros(x.shape[0], 1, 1, x.shape[1], dtype=torch.float32, device=device)),), {}

    def return_hook(x):
        return x[0]

    from vit_pytorch.deepvit import Transformer 
    smp.tp_register_with_module(Transformer, smp.nn.DistributedTransformer, init_hook, forward_hook, return_hook)

if __name__ == '__main__':
    args = parse_args()
    gpus_per_machine = torch.cuda.device_count()
    import smdistributed.modelparallel.torch as smp
    smp_config = {
        "ddp": True,
        "tensor_parallel_degree": 8,
        "shard_optimizer_state": True,

    }

    smp.init(smp_config)

    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda", smp.local_rank())
    register_tp(device)

    model = build_model(args.model_size)
    smp.set_tensor_parallelism(model.transformer)

    dataset = FakeDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,
                                              pin_memory=False)
    model = smp.DistributedModel(model, trace_device="gpu")
    
    if smp.tp_size() > 1:
        transformer_layers = model.module.module.transformer.seq_layers
        smp.set_activation_checkpointing(transformer_layers, pack_args_as_tuple=True, strategy="each")
    else:
        transformer_layers = model.module.module.transformer.layers
        for layer in transformer_layers.children():
            for m in layer.children():
                smp.set_activation_checkpointing(m)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, amsgrad=True)
    optimizer = smp.DistributedOptimizer(optimizer)
    loss_function = torch.nn.CrossEntropyLoss()
    if args.mixed_precision:
        if args.loss_scale == 0:
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level="O2",
                loss_scale="dynamic",
                cast_model_outputs=torch.float16,
            )
        else:
            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level="O2",
                loss_scale=args.loss_scale,
                cast_model_outputs=torch.float16,
            )
    if torch.distributed.get_rank() == 0:
        print(model)

    if torch.distributed.get_rank() == 0:
        tracking_mem_allocs = []
        tracking_mem_reserved = []
        tracking_duration = []
    model.train()
    t0 = time.perf_counter()


    @smp.step
    def train_step(model, inputs, targets):
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        if args.mixed_precision:
            print("################# mixed percision ##########")
            with amp.scale_loss(
                loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation
            ) as scaled_loss:
                model.backward(scaled_loss)
        else:
            if args.smp > 0:
                model.backward(loss)
            else:
                loss.backward()
        model.backward(loss)
        return loss


    for batch_index, (inputs, target) in enumerate(data_loader, start=1):
        if batch_index % 4 == 0:
            print(f'batch index is {batch_index} rank is {torch.distributed.get_rank()}')
        inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
            target.to(torch.cuda.current_device()), -1)
        optimizer.zero_grad()

        loss_mb = train_step(model, inputs, targets)
        loss = loss_mb.reduce_mean()
        
        optimizer.step()
        if torch.cuda.is_available() and torch.distributed.get_rank() == 0:

            mini_batch_time = time.perf_counter() - t0
            tracking_duration.append(mini_batch_time)
            tracking_mem_allocs.append(torch.cuda.memory_allocated() / gb_unit_size)
            tracking_mem_reserved.append(torch.cuda.memory_reserved() / gb_unit_size)

        if batch_index % args.log_every == 0 and torch.distributed.get_rank() == 0:
            print(f'step: {batch_index}:time taken for the last {args.log_every} steps is {time.perf_counter() - t0}')
            t0 = time.perf_counter()
        if batch_index % 10 == 0:
            break
    if torch.cuda.is_available() and torch.distributed.get_rank() == 0:

        stable_sum = sum(tracking_duration[1:])
        stable_avg = stable_sum / 10
        stable_avg = round(stable_avg, 4)
        print("Len of dataset***************************", len(data_loader) )
        print(f"minibatch durations Average: {stable_avg}")
        print(f"******************************************")
        print(f"minibatch durations: {tracking_duration}")
        print(f"running mem allocs: {tracking_mem_allocs}")
        print(f"running mem reserved: {tracking_mem_reserved}")
        print(
            f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}"
        )
