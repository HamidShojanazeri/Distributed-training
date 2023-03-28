import torch
import torch.nn as nn
import functools
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import Dataset
from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block

from contextlib import contextmanager

	
# available models
# t5-base
# google/t5-v1_1-small #52M
# google/t5-v1_1-base #185M
# google/t5-v1_1-large #611M
# google/t5-v1_1-xl  #2b
# google/t5-v1_1-xxl #8b
# t5-11b


def configure(model_name):
    configure = T5Config()
    if model_name == "t5-small":
        configure = T5Config(
            d_ff=1024,
            d_kv=64,
            d_model=512,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=8,
            num_heads=6,
            num_layers=8,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "t5-base":
        configure = T5Config(
            d_ff=2048,
            d_kv=64,
            d_model=768,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=12,
            num_heads=12,
            num_layers=12,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "t5-large":
        configure = T5Config(
            d_ff=2816,
            d_kv=64,
            d_model=1024,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=16,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "t5-xl":
        configure = T5Config(
            d_ff=5120,
            d_kv=64,
            d_model=2048,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=32,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "t5-xxl":
        configure = T5Config(
            d_ff=10240,
            d_kv=64,
            d_model=4096,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=64,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "t5-11b":
        configure = T5Config(
            d_ff=65536,
            d_kv=128,
            d_model=1024,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            n_positions=512,
            num_heads=128,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    return configure




@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.
    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.
    Example:
    ```pyton
    import torch.nn as nn
    from accelerate import init_empty_weights
    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```
    <Tip warning={true}>
    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    </Tip>
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")), **kwargs)
            
    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_reg


def get_inputs(batch_size, device):
    seq_len = 256
    return {
        "source_ids": torch.randint(
            1, 32000, (batch_size, seq_len), dtype=torch.long, device=device
        ),
        "source_mask": torch.randint(
            1, 32000, (batch_size, seq_len), dtype=torch.long, device=device
        ),
        "target_ids": torch.randint(
            1, 32000, (batch_size, seq_len), dtype=torch.long, device=device
        ),
        "target_mask": torch.randint(
            1, 32000, (batch_size, seq_len), dtype=torch.long, device=device
        ),
    }


def get_loss(dist_model, inputs):
    out = dist_model(
        input_ids=inputs["source_ids"],
        attention_mask=inputs["source_mask"],
        labels=inputs["target_ids"],
    )
    return out["loss"]

def get_fsdp_wrapping_policy():
    return ModuleWrapPolicy({T5Block})


def apply_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, T5Block)

    apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

def get_flops(model_name, batch_size):
    cfg = configure(model_name)
    # So total FLOP estimate for T5 fwd + backward with AC is:
    #B = batch size
    #s = sequence length (1024 in our code)
    #d_model, d_kv, num_heads, d_ff are input during T5 construction
    #(6B*s*d_model*d_kv*num_heads + 2*(d_kv*num_heads)^2*batch*seq + 2*(d_kv*num_heads)^2*d_model + 2d_model^2*d_kv*num_heads + 4B*s*d_ff*d_model) * (num_layers + num_decoders) * 3
    d_kv = cfg.d_kv
    S = 1024
    num_heads = cfg.num_heads
    d_model = cfg.d_model
    d_ff = cfg.d_ff
    flops= 3 * (cfg.num_layers + cfg.num_decoder_layers) * ( (6 * batch_size * S * cfg.d_model*cfg.d_kv*cfg.num_heads + 2 * batch_size * S *(d_kv*num_heads)**2) + 2*d_kv*num_heads*d_model**2 + 4 * S * d_ff * d_model)
    return flops

def build(model_name: str, device="cpu"):    
    with init_empty_weights():
        return T5ForConditionalGeneration(configure(model_name))



def sync_all_device():
    # setup() has already configured CUDA_VISIBLE_DEVICES such that each
    # process exclusively works on its own set of devices. So it's safe to
    # do device sync here
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def print_memory_summary(prefix, device):
    # rank = int(os.getenv("RANK"))
    # if rank == 0:
    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0)
    print(
        f"{prefix}, GPU peak memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB, "
        f"GPU peak memory reserved: {torch.cuda.max_memory_reserved(device) // 1e9}GB, "
        f"GPU peak memory active: {peak_memory_active // 1e9}GB"
    )
    torch.cuda.reset_peak_memory_stats(device)


def print_latency(prefix, start_event, end_event):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(f"{prefix}: {start_event.elapsed_time(end_event) / 1000}sec")




if __name__=="__main__":
    model = build("t5-small", device="cuda")
    print_memory_summary("t5", "cuda")

