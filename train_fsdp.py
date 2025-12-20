import os
import fire
import json
import time
import numpy as np
from dataclasses import asdict
from contextlib import nullcontext
from llama import LLaMAConfig, LLaMA, LLaMABlock, Fp8LLaMA, Fp8LLaMABlock
from mistral import MistralConfig, Mistral, MistralBlock, Fp8Mistral, Fp8MistralBlock
import itertools
from collections.abc import Iterable

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import destroy_process_group

from torch.profiler import profile, ProfilerActivity, record_function

from torch.utils.data import DataLoader, IterableDataset
from torch.distributed.tensor import DTensor

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp
# from straggler_detection import get_straggler_gpus, info, warn

# FSDP2 imports
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
# FSDP imports
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)

class RandData(IterableDataset):
    def __init__(self, vocab_size, max_seq_len, total_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.total_size = total_size

    def __len__(self):
        return self.total_size

    def __iter__(self):
        datas = []
        for i in range(self.total_size):
            input = torch.randint(
                self.vocab_size, [self.max_seq_len], dtype=torch.int64)
            label = torch.cat(
                [input[:-1], torch.randint(self.vocab_size, [1])])
            datas.append((input, label))
        return iter(datas)


def create_dummy_data_loader(world_size, batch_size, num_iteration, model_config):
    dataset = RandData(model_config.vocab_size,
                       model_config.max_seq_len, batch_size*num_iteration*world_size)
    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=world_size, pin_memory=True, shuffle=False
    )
    return data_loader


# https://github.com/pytorch/torchtitan/blob/55c63c14594107363b8e286c1742efe3efbeda7c/torchtitan/distributed/utils.py#L341
@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    torch.nn.utils.clip_grads_with_norm_(
        parameters, max_norm, total_norm, foreach)
    return total_norm


def train(
    config_file: str,
    model_name: str,
    num_iteration: int = 128,
    grad_accumlate_pre_steps: int = 8,  # steps to accumlate gradient
    enable_compile: bool = False,
    seed: int = 1024,  # to ensure reproducible
    use_pytorch_profiler: bool = False,
    output_dir: str = '.',
    wait: int = 9,
    active: int = 1,
    use_fsdp2: bool = True,
):

    if use_pytorch_profiler:
        schedule = torch.profiler.schedule(
            wait=wait,
            warmup=0,
            active=active,
            repeat=0,
        )

        def export_trace(prof):
            rank = int(os.environ["RANK"])
            prof.export_chrome_trace(
                f"{output_dir}/pytorch_trace_gpu{rank}_step{prof.step_num}.json")

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                     schedule=schedule,
                     on_trace_ready=export_trace,
                     ) as prof:
            _train(
                config_file,
                model_name,
                num_iteration,
                grad_accumlate_pre_steps,
                enable_compile,
                seed,
                prof,
                use_fsdp2=use_fsdp2,
            )
    else:
        _train(
            config_file,
            model_name,
            num_iteration,
            grad_accumlate_pre_steps,
            enable_compile,
            seed,
            use_fsdp2=use_fsdp2,
        )


def _train(
    config_file: str,
    model_name: str,
    num_iteration: int = 128,
    grad_accumlate_pre_steps: int = 8,  # steps to accumlate gradient
    enable_compile: bool = False,
    seed: int = 1024,  # to ensure reproducible
    prof=None,
    use_fsdp2: bool = True,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    assert rank == local_rank, "This script is intended to run on single node for testing"
    world_size = int(os.environ["WORLD_SIZE"])
    # Construct process group
    if local_rank == 0:
        print("Initing communication")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    assert dist.get_rank() == local_rank
    # Configure training setup
    if local_rank == 0:
        print("Using config", config_file)
    with open(config_file) as f:
        config = json.load(f)

    if model_name == "llama":
        model_config = LLaMAConfig(**config)
    elif model_name == "mistral":
        model_config = MistralConfig(**config)
    else:
        print("model not supported. please pass either llama or mistral as param")
    if local_rank == 0:
        print("Creating model with config: ", model_config)

    enable_fp8 = model_config.enable_fp8

    if use_fsdp2:
        with torch.device('meta'):
            if enable_fp8:  # add more model
                enable_compile = False
                if rank == 0:
                    print(
                        'PyTorch compile currently doesn\'t work with Transformer Engine.')
                if model_name == "llama":
                    layer_class = Fp8LLaMABlock
                    model = Fp8LLaMA(**asdict(model_config))
                elif model_name == "mistral":
                    layer_class = Fp8MistralBlock
                    model = Fp8Mistral(**asdict(model_config))
            else:
                if model_name == "llama":
                    layer_class = LLaMABlock
                    model = LLaMA(**asdict(model_config))
                elif model_name == "mistral":
                    layer_class = MistralBlock
                    model = Mistral(**asdict(model_config))
    else:
        if enable_fp8:  # add more model
            enable_compile = False
            if rank == 0:
                print('PyTorch compile currently doesn\'t work with Transformer Engine.')
            if model_name == "llama":
                layer_class = Fp8LLaMABlock
                model = Fp8LLaMA(**asdict(model_config))
            elif model_name == "mistral":
                layer_class = Fp8MistralBlock
                model = Fp8Mistral(**asdict(model_config))
        else:
            if model_name == "llama":
                layer_class = LLaMABlock
                model = LLaMA(**asdict(model_config))
            elif model_name == "mistral":
                layer_class = MistralBlock
                model = Mistral(**asdict(model_config))

    # Need to calculate before wrapping in FSDP
    model_config.estimate_flops_per_token(model, model_config.batch_size)

    if local_rank == 0:
        print(
            f"Loaded model on CPU with number of parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        print(f"Original Model:\n{model}")

    if use_fsdp2:
        fsdp_kwargs = {
            'mp_policy': MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
            )
        }
        for module in model.modules():
            if isinstance(module, layer_class):
                fully_shard(module, **fsdp_kwargs)

        fully_shard(model, **fsdp_kwargs)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta")
        model.to_empty(device='cuda')
    else:
        model = FSDP(
            model,
            device_id=local_rank,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
            ),
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={layer_class}),
            use_orig_params=True,
            limit_all_gathers=True,
        )
        if enable_fp8:
            assert not use_fsdp2, "Doesn't work for FSDPv2"
            prepare_te_modules_for_fsdp(model)
            fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
            fp8_recipe = DelayedScaling(
                fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
            all_worker = dist.new_group(backend='nccl')

    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # Print out allocated device memory
    batch_size = model_config.batch_size
    pre_mem_use = torch.cuda.memory_allocated(
        device=f"cuda:{local_rank}") * 1e-6
    flops_per_iter = model_config.flops_per_token * \
        (batch_size * model_config.max_seq_len)
    if local_rank == 0:
        print(f"GPU memory use = {pre_mem_use}MB")
        print("TFLOP per iteration:", flops_per_iter/1e12)
    # PyTorch compile
    if enable_compile:
        if rank == 0:
            print(f'Compiling model....')
        model = torch.compile(model)

    model.train()
    iter_times = []

    data_loader = create_dummy_data_loader(
        world_size, batch_size, num_iteration, model_config)
    last_time = time.time()

    for step_idx, data_batch in enumerate(data_loader):
        with record_function(f"Iteration{step_idx}"):

            with record_function("GetInputLabels"):
                input, labels = data_batch
                input = input.to(local_rank)
                labels = labels.to(local_rank)
            fp8_context = nullcontext() if not enable_fp8 else te.fp8_autocast(
                enabled=enable_fp8, fp8_recipe=fp8_recipe, fp8_group=all_worker)
            with torch.amp.autocast('cuda', torch.bfloat16), fp8_context:
                weight_cache = enable_fp8 and (
                    step_idx % grad_accumlate_pre_steps == 0)
                logits = model(input, is_first_microbatch=weight_cache)
                with record_function("cel"):
                    loss = F.cross_entropy(
                        logits.flatten(0, 1), labels.flatten())
                    loss /= grad_accumlate_pre_steps

            with record_function("b_l"):
                loss.backward()

            if (step_idx + 1) % grad_accumlate_pre_steps == 0:
                with record_function("b_ga"):
                    # https://github.com/foundation-model-stack/fms-fsdp/blob/0fdb43dcfd31ab093f8d873b58b0b531dd0818b1/fms_fsdp/utils/train_utils.py#L94
                    # https://github.com/foundation-model-stack/foundation-model-stack/blob/d55a9f2ade65ef4157cdfd928300874e2348e5d0/fms/training/trainer.py#L36
                    if use_fsdp2:
                        clip_grad_norm_(
                            [p for p in model.parameters()], 1.0, foreach=True)
                    else:
                        model.clip_grad_norm_(1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if rank == 0:
                current_time = time.time()
                iter_time = current_time-last_time
                token_per_sec = (
                    batch_size * model_config.max_seq_len)/iter_time
                print(
                    f"Step: {step_idx}; TFLOP/s: {flops_per_iter/iter_time/1e12}; iteration time: {iter_time}; token per second: {token_per_sec}")
                iter_times.append(iter_time)
                last_time = current_time

            if prof:
                prof.step()
            if (step_idx+1) == num_iteration:
                break

    if rank == 0:
        iter_times = np.array(iter_times)
        avg_iter_time = np.mean(iter_times)

        print("Avg token per second:", (batch_size *
              model_config.max_seq_len)/avg_iter_time)
        print("Avg iter time:", avg_iter_time)
        print("TFLOP per iteration:", flops_per_iter/1e12)
        print("Avg TFLOP/s,", flops_per_iter/avg_iter_time/1e12)
        peak_memory = torch.cuda.max_memory_allocated(
            device=f"cuda:{local_rank}") * 1e-6
        print(f"Peak memory use = {peak_memory}MB")

    torch.cuda.empty_cache()
    dist.barrier()
    destroy_process_group()


if __name__ == '__main__':
    fire.Fire(train)
