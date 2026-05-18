import os
import fire
import json
import time
import numpy as np
from dataclasses import asdict
from contextlib import nullcontext
from llama import LLaMAConfig, LLaMA, LLaMABlock, Fp8LLaMA, Fp8LLaMABlock
from mistral import MistralConfig, Mistral, MistralBlock, Fp8Mistral, Fp8MistralBlock
import math
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


def info(msg: str):
    print(f"STRAGGLER INFO: {msg}")


def warn(msg: str):
    print(f"STRAGGLER WARN: {msg}")

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


pending_counter = 0
wait_counter = 0
max_lead = 0


def set_power_cap(gpu_num: int, watts: int, grpc_socket: str) -> bool:
    import grpc
    import grpc.experimental

    protos, services = grpc.protos_and_services("freq.proto")
    response = services.FreqServer.SetPower(
        protos.PowerReq(gpu=gpu_num, watts=watts),
        f"unix://{grpc_socket}",
        insecure=True,
    )
    if response.ack == 0:
        return True
    if response.ack == 1:
        return False
    raise ValueError(f"Unexpected ack: {response.ack}")


class PowerTuner:
    def __init__(
        self,
        *,
        output_dir: str,
        world_size: int,
        adjust_steps: int,
        wait_steps: int,
        initial_power_cap: int,
        power_budget: int,
        max_adj: int,
        use_sum: bool,
        use_last: bool,
        use_max: bool,
        use_global: bool,
        grpc_socket: str,
        max_power: int,
    ):
        if not (use_sum or use_last or use_max):
            raise ValueError("Power tuning requires one of use_sum, use_last, or use_max")

        self.output_dir = output_dir
        self.world_size = world_size
        self.adjust_steps = adjust_steps
        self.wait_steps = wait_steps
        self.max_adj = max_adj
        self.use_sum = use_sum
        self.use_last = use_last
        self.use_max = use_max
        self.use_global = use_global
        self.grpc_socket = grpc_socket
        self.max_power = max_power
        self.fake_max_power = initial_power_cap + power_budget
        self.gpu_power = [initial_power_cap for _ in range(world_size)]
        self.gpu_pending = [[] for _ in range(world_size)]

    def initialize(self):
        for gpu_num, power_cap in enumerate(self.gpu_power):
            info(f"Initializing GPU{gpu_num} power cap to {power_cap} W")
            if set_power_cap(gpu_num, power_cap, self.grpc_socket):
                info("  Success")
            else:
                warn("  Failed setting power")

    def update_from_trace(self, step_num: int):
        from straggler_detection import get_straggler_gpus

        global pending_counter
        global wait_counter
        global max_lead

        gpu_traces = tuple(
            f"{self.output_dir}/pytorch_trace_gpu{gpu_num}_step{step_num}.json"
            for gpu_num in range(self.world_size)
        )
        for gpu_trace in gpu_traces:
            assert os.path.exists(gpu_trace), f"{gpu_trace} is missing"

        straggler_gpus, max_lead = get_straggler_gpus(
            gpu_traces,
            self.max_adj,
            invert=True,
            max_lead=max_lead if self.use_global else 0,
            use_sum=self.use_sum,
            use_max=self.use_max,
            use_last=self.use_last,
        )

        info("Pending power deltas:")
        for gpu_num, delta in straggler_gpus.items():
            info(f"  GPU{gpu_num}: {delta:.3f} W")

        if wait_counter < self.wait_steps:
            info(f"Waiting steps {self.wait_steps - wait_counter} left")
            wait_counter += 1
            return

        if pending_counter != self.adjust_steps - 1:
            pending_counter += 1
            for gpu_num, delta in straggler_gpus.items():
                self.gpu_pending[gpu_num].append(delta)
            return

        pending_counter = 0
        avg_deltas = {}
        for gpu_num, delta in straggler_gpus.items():
            self.gpu_pending[gpu_num].append(delta)
            avg_delta = np.median(self.gpu_pending[gpu_num]).astype(int)
            self.gpu_pending[gpu_num] = []
            avg_deltas[gpu_num] = avg_delta

        for gpu_num, avg_delta in avg_deltas.items():
            self.gpu_power[gpu_num] += avg_delta

        total_power = sum(self.gpu_power)
        power_delta = math.ceil((total_power - self.fake_max_power * self.world_size) / self.world_size)
        info(f"Total power after normalization: {total_power - power_delta * self.world_size} W")
        assert total_power - power_delta * self.world_size <= self.fake_max_power * self.world_size

        gpu_delta = 0
        for gpu_num in avg_deltas.keys():
            self.gpu_power[gpu_num] -= power_delta
            gpu_delta = max(gpu_delta, self.gpu_power[gpu_num] - self.max_power)
        for gpu_num in avg_deltas.keys():
            self.gpu_power[gpu_num] -= gpu_delta

        underutil = self.fake_max_power * self.world_size - sum(self.gpu_power)
        assert underutil >= 0, f"{-1 * underutil} W over the limit"
        if underutil > 0:
            warn(f"Operating {underutil} W lower than node cap")

        info("Final power deltas:")
        for gpu_num, avg_delta in avg_deltas.items():
            new_cap = self.gpu_power[gpu_num]
            assert new_cap <= self.max_power
            info(f"  GPU{gpu_num}: delta={avg_delta:.3f} W cap={new_cap} W")
            if not set_power_cap(gpu_num, new_cap, self.grpc_socket):
                warn("  Failed setting power")
                self.gpu_power[gpu_num] -= avg_delta


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
    power_man: bool = False,
    adjust_steps: int = 3,
    wait_steps: int = 50,
    initial_power_cap: int = 700,
    power_budget: int = 0,
    max_adj: int = 15,
    max_power: int = 700,
    use_sum: bool = True,
    use_last: bool = False,
    use_max: bool = False,
    use_global: bool = True,
    grpc_socket: str = "/tmp/freq.sock",
    use_fsdp2: bool = True,
):

    if use_pytorch_profiler:
        schedule = torch.profiler.schedule(
            wait=wait,
            warmup=0,
            active=active,
            repeat=0,
        )

        power_tuner = None
        if power_man:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            power_tuner = PowerTuner(
                output_dir=output_dir,
                world_size=world_size,
                adjust_steps=adjust_steps,
                wait_steps=wait_steps,
                initial_power_cap=initial_power_cap,
                power_budget=power_budget,
                max_adj=max_adj,
                max_power=max_power,
                use_sum=use_sum,
                use_last=use_last,
                use_max=use_max,
                use_global=use_global,
                grpc_socket=grpc_socket,
            )
            if rank == 0:
                power_tuner.initialize()

        def export_trace(prof):
            rank = int(os.environ["RANK"])
            prof.export_chrome_trace(
                f"{output_dir}/pytorch_trace_gpu{rank}_step{prof.step_num}.json")
            if power_tuner is not None:
                dist.barrier()
                if dist.get_rank() == 0:
                    power_tuner.update_from_trace(prof.step_num)
                dist.barrier()

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
        if power_man:
            warn("Power tuning requires --use_pytorch_profiler True; continuing without tuning")
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
        model.to(device='cuda')
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
