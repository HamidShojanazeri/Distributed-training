from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # seed
    seed: int = 2022

    # model
    model_name = "t5-small"  # "google/t5-v1_1-small"
    tokenizer = "t5-large"
    # available models
    ## t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b

    # save models
    save_model: bool = True
    save_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies
    fsdp_unit_size = 1000000
    use_mixed_precision: bool = True

    # activation checkpointing
    hf_activation_checkpointing: bool = False
    fsdp_activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 5
    num_epochs: int = 1

    # validation
    run_validation: bool = True
    val_batch_size = 4
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    use_child_tuning: bool = True
    use_mirror_optimizer = False
    lr: float = 4e-8

    use_task_free: bool = True
    use_fisher_matrix: bool = False
    percent_F: float = 0.35
