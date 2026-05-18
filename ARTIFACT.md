# Lit Silicon Benchmark & Evaluation (AMD)

The AMD artifact also supports tuning power caps, and uses a more optimized AMD training framework, Primus:

[AMD Lit Silicon artifact + tuning](https://github.com/UnaryLab/lit_silicon_tuning_amd)

# Lit Silicon Benchmark & Evaluation (NVIDIA)

This repo provides an FSDP training benchmark that can be used to detect the "Lit Silicon" effect.
We are assuming the visualization will be done on a local computer while the benchmark will be run on a remote node.
However, if a GUI is available on the remote node, all steps can be completed there.
Each step will be annotated with **(local)** or **(remote)** to designate where it should be performed.

## Setup

### Python virtual environment **(local & remote)**

Create a python virtual environment if you don't have one already.
This will be used for installing Chopper.

```
python -m venv .ls_venv
. .ls_venv/bin/activate
```

1) Clone this benchmark and install Chopper **(local & remote)**

```
git clone --recursive https://github.com/UnaryLab/lit_silicon.git
cd lit_silicon/chopper
pip install .
cd ..
```

2) Install host-side tuning dependencies **(remote, only needed for `POWER_MAN=1`)**

Power-cap tuning starts `freq_server.py` on the host before launching training inside the container.
Because that server runs outside `pytorch.sif`, the remote host Python environment needs gRPC installed.
Install this in the same environment that will run `sbatch run_pytorch.sh`:

```
pip install grpcio grpcio-tools
```

The container also needs gRPC and trace parsing dependencies for the training process.
Those are installed by `pytorch.def` when the image is built.

3) Build the container **(remote)**

> [!NOTE]
> While we use apptainer and slurm, docker can also be used. Adjust the scripts as needed.

Build `pytorch.sif` before running the benchmark. `run_pytorch.sh` expects this image to exist.

```
sbatch build_pytorch.sh
```

## Running **(remote)**

1) Run the benchmark

This benchmark will run pytorch FSDPv2 training with batch size one sequence length 4k (b1s4), b2s4, and b1s8.
Raw traces will be inside a folder named the hostname, with the batch size and sequence length number as subfolders (e.g., if node `foobar` ran the benchmark, traces are in `foobar/b1s4`, `foobar/b2s4`, and `foobar/b1s8`).
Make sure the container build step above has completed successfully first.

```
sbatch run_pytorch.sh
```

Power-cap tuning can be enabled with `POWER_MAN=1`.
This starts `freq_server.py`, exports profiler traces, computes per-GPU cap adjustments from straggler lead, and sends the requested caps to the local gRPC server.
The server sets power caps with `sudo nvidia-smi -i <gpu> -pl <power cap>`.
The default initial and maximum power cap is 700 W, which is suitable for H100 SXM-class systems but should be overridden for lower-TDP GPUs such as H100 PCIe.

```
ITERS=1200 POWER_MAN=1 sbatch run_pytorch.sh
```

The tuning experiments use three power-management scenarios.
For H100 SXM systems, use these environment variable sets:
The total node power budget is always `(INITIAL_POWER_CAP + POWER_BUDGET) * number_of_gpus`, while `MAX_POWER` is the hard per-GPU cap.

**GPU-Red** starts every GPU at the maximum 700 W cap.
The power manager can then reduce power on leader GPUs while leaving the straggler near the maximum.

```
ITERS=1200 POWER_MAN=1 INITIAL_POWER_CAP=700 MAX_POWER=700 POWER_BUDGET=0 sbatch run_pytorch.sh
```

**GPU-Realloc** starts every GPU 100 W below the maximum, at 600 W.
The power manager can reallocate within the 700 W per-GPU maximum, so the straggler can receive more power while leaders are reduced.

```
ITERS=1200 POWER_MAN=1 INITIAL_POWER_CAP=600 MAX_POWER=700 POWER_BUDGET=0 sbatch run_pytorch.sh
```

**CPU-Slosh** starts every GPU at 600 W and adds a 20 W per-GPU budget.
This lets each GPU rise as high as 620 W in aggregate budget terms, so the straggler can receive extra power while leaders give power back.

```
ITERS=1200 POWER_MAN=1 INITIAL_POWER_CAP=600 MAX_POWER=700 POWER_BUDGET=20 sbatch run_pytorch.sh
```

2) Merge traces using Chopper

This convenience script calls Chopper to aggregate all raw traces into a single pickle file.
Pass the directory you would like to merge as an argument (e.g., `hostname/b1s4` to merge batch size one sequence length 4k traces).

> [!WARNING]
> Do not pass **just** the hostname as the directory. If you did, results from `b1s4`, `b2s4`, and `b1s8` would all be merged together.

```
./chopper.sh hostname/bXsX
```

Now, the pickle file will be inside the directory passed and named `ts.pkl` (e.g., `hostname/bXsX/ts.pkl`).

## Visualization **(local)**

1) Copy the pickle to your local computer for visualization

Copy it any way you'd like, `rsync` is not required.

```
rsync -avzh <login_node>:/data/lit_silicon/hostname/bXsX/ts.pkl .
```

2) Open the Chopper GUI

```
python -m chopper.window
```

3) Select `straggler_per_gpu` under `available plots`

If `ts.pkl` isn't located inside the current directory, select the check box for `data args` and change the `dirs` entry to the directory `ts.pkl` is located.
You can also zoom in on a few iterations by changing `idx_start` and `idx_end` in the `draw args` (e.g., `idx_start=5` and `idx_end=10` to view samples 5-9).

4) Click `load data`, then click `redraw plot` once it becomes available.

In a system suffering from "Lit Silicon", you will observe one or more GPU consistently has a lower "lead" value than the others:

![Lit Silicon Example](misc/lit_silicon_example.png)

> In the above example, GPU2 is clearly the straggler, GPU6 is close, and other GPUs are leaders that all reach an equilibrium where the lead stops increasing due to increase communication overlap.
For more details check out [our paper on arxiv](https://arxiv.org/abs/2511.09861)!
