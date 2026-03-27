# flocora_eusipco24
flocora - Eusipco 2024

The code for this paper was built on top of the [flower framework](https://github.com/adap/flower/tree/main/src/py/flwr)

Different from the original version, I have lastly tested it with python 3.10, torch 2.5 and flwr 1.12.0 + ray 2.10 (flwr[simulation]), however I have not performed extensive tests with it. The original version of the code was tested with python 3.8, torch 2.0 and flwr 1.8.0.
## Running on a SLURM cluster (sbatch)

If your environment requires `sbatch`, use the provided template:

```bash
sbatch run_slurm_ray.sbatch
```

The sbatch script is intentionally minimal and executes:

```bash
python main_ray.py
```

Ray resource alignment is handled in Python: when running under SLURM, the project automatically derives `ray_cpu`/`ray_gpu` from `SLURM_CPUS_PER_TASK` and `SLURM_GPUS_ON_NODE` (unless explicitly overridden via CLI), reducing Ray scheduling mismatches.

The sbatch template also exports common CPU-thread env variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`).
By default these are pinned to `1` to avoid CPU oversubscription when Ray runs many client workers concurrently on the same node.

You can override experiment defaults at submit time, for example:

```bash
python main_ray.py --num_clients 20 --samp_rate 0.2 --model resnet8 --strategy fedavg
```

This keeps Ray's per-client resource requests aligned with the resources actually allocated by SLURM.

`#SBATCH --nodes=1` and `#SBATCH --ntasks=1` mean SLURM launches one main Python process on one node. Ray then schedules multiple client workers inside that allocation; it does **not** force every FL client to be single-threaded by itself.
