#!/usr/bin/env python3
"""
Aquila-GS Multi-Seed Directed Evolution (multi-GPU)

Architecture (mirrors aquila_train_multi.py):
  - Fixed pool of worker processes (mp.Process)
  - Shared GPU queue: each worker grabs a GPU from the queue, runs one seed, returns GPU
  - Seeds dispatched via task queue — no CUDA tensors in multiprocessing objects
  - spawn start method for CUDA compatibility

Usage:
    python aquila_evolve_multi.py \\
        --model-dir path/to/model \\
        --vcf input.vcf.gz \\
        --direction-file trait_direction.tsv \\
        --sites-to-evolve snp_list.txt \\
        --output-dir multi_evolve_1000 \\
        --strategy combinatorial --iterations 300 --n-seeds 1000
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Worker process
# ─────────────────────────────────────────────────────────────────────────────

def worker_process(
    worker_id: int,
    gpu_queue: mp.Queue,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_dir: str,
    vcf: str,
    output_base: str,
    strategy: str,
    iterations: int,
    top_k: int,
    min_improve: float,
    mode: str,
    direction_file: str,
    sites_to_evolve: str,
    ref_vcf: str,
    pheno: str,
    phased: bool,
    mc_samples: int,
    python_exec: str,
    homozygous: bool,
):
    """
    Worker: pulls a GPU from gpu_queue, runs one seed, returns GPU, repeats.
    """
    import os as _os
    import subprocess as _subprocess
    from pathlib import Path as _Path

    print(f"Worker {worker_id} started (PID: {os.getpid()})")
    script = str(_Path(__file__).resolve().parent / 'aquila_evolve.py')

    def build_cmd(seed: int, out_dir: _Path) -> List[str]:
        cmd = [python_exec, script,
               '--model-dir',   str(_Path(model_dir).resolve()),
               '--vcf',        str(_Path(vcf).resolve()),
               '--output-dir', str(out_dir),
               '--strategy',   strategy,
               '--iterations', str(iterations),
               '--top-k',      str(top_k),
               '--min-improve', str(min_improve),
               '--mode',       mode,
               '--mc-samples', str(mc_samples),
               '--seed',       str(seed)]
        if direction_file:
            cmd += ['--direction-file', str(_Path(direction_file).resolve())]
        if pheno:
            cmd += ['--pheno', pheno]
        if sites_to_evolve:
            cmd += ['--sites-to-evolve', str(_Path(sites_to_evolve).resolve())]
        if ref_vcf:
            cmd += ['--ref-vcf', str(_Path(ref_vcf).resolve())]
        if not phased:
            cmd.append('--unphased')
        if homozygous:
            cmd.append('--homozygous')
        return cmd

    while True:
        gpu_id = None
        try:
            # Get task from queue
            task = task_queue.get()
            if task is None:
                print(f"Worker {worker_id} shutting down")
                break

            # Dynamically acquire a GPU
            gpu_id = gpu_queue.get()

            # Set CUDA_VISIBLE_DEVICES BEFORE any torch import
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            seed = task['seed']
            total = task['total']
            out_dir = _Path(output_base) / f"seed_{seed:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_file = out_dir / "run.log"

            cmd = build_cmd(seed, out_dir)
            with open(log_file, 'w') as fh:
                proc = _subprocess.run(cmd, stdout=fh, stderr=_subprocess.STDOUT)
            rc = proc.returncode

            # Find evolved VCF
            stem = _Path(vcf).stem.replace('.vcf', '')
            suffix = '.vcf.gz' if _Path(vcf).suffix == '.gz' else '.vcf'
            evolved_vcf = next(out_dir.glob(f"{stem}_evolve{suffix}"), None)

            ok = rc == 0 and evolved_vcf is not None
            result_queue.put({
                'seed': seed, 'success': ok, 'exit_code': rc,
                'evolved_vcf': str(evolved_vcf) if evolved_vcf else '',
                'output_dir': str(out_dir),
            })

            print(f"[Worker{worker_id} GPU{gpu_id}] seed={seed:04d} ({total}) → "
                  f"{'OK' if ok else f'FAIL rc={rc}'}", flush=True)

            task_queue.task_done()

            # Return GPU to queue
            gpu_queue.put(gpu_id)

        except Exception as e:
            import traceback
            print(f"Worker {worker_id} (GPU{gpu_id}) ERROR: {e}")
            print(f"  {traceback.format_exc()}", flush=True)
            result_queue.put({
                'seed': task.get('seed', -1) if 'task' in dir() else -1,
                'success': False, 'exit_code': -1,
                'evolved_vcf': '', 'output_dir': str(output_base),
            })
            task_queue.task_done()
            if gpu_id is not None:
                try:
                    gpu_queue.put(gpu_id)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Aquila-GS Multi-Seed Directed Evolution (multi-GPU parallel)')
    parser.add_argument('--model-dir',       type=str, required=True)
    parser.add_argument('--vcf',            type=str, required=True)
    parser.add_argument('--direction-file',  type=str, default=None)
    parser.add_argument('--pheno',          type=str, default=None)
    parser.add_argument('--output-dir',      type=str, required=True)
    parser.add_argument('--strategy',         type=str, default='combinatorial',
                        choices=['screening', 'combinatorial'])
    parser.add_argument('--iterations',       type=int, default=300)
    parser.add_argument('--top-k',            type=int, default=4)
    parser.add_argument('--min-improve',      type=float, default=0.0001)
    parser.add_argument('--n-seeds',          type=int, default=1000)
    parser.add_argument('--mode',             type=str, default='maximize',
                        choices=['maximize', 'minimize'])
    parser.add_argument('--sites-to-evolve',  type=str, default=None)
    parser.add_argument('--ref-vcf',          type=str, default=None)
    parser.add_argument('--mc-samples',        type=int, default=1)
    parser.add_argument('--phased',           action='store_true', default=True)
    parser.add_argument('--unphased',        dest='phased', action='store_false')
    parser.add_argument('--qtn-list',         type=str, default=None)
    parser.add_argument('--gpus',              type=str, default=None,
                        help='Comma-separated GPU IDs, e.g. 0,1,2. Default: all available.')
    parser.add_argument('--n-workers-per-gpu', type=int, default=1,
                        help='Worker processes per GPU (default: 1, one seed per GPU at a time)')
    parser.add_argument('--homozygous', action='store_true', default=False,
                        help='Only generate homozygous (REF/REF or ALT/ALT) mutation candidates.')
    return parser.parse_args()


def main():
    args = parse_args()

    # CUDA spawn required for multiprocessing with CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # GPU detection
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        import torch
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]

    n_gpus = len(gpu_ids)
    n_seeds = args.n_seeds
    n_workers_per_gpu = args.n_workers_per_gpu
    total_workers = n_gpus * n_workers_per_gpu

    print(f"{'='*60}")
    print(f"Aquila-GS Multi-GPU Evolution")
    print(f"{'='*60}")
    print(f"  Seeds:       {n_seeds}  (0 .. {n_seeds-1})")
    print(f"  GPUs:       {gpu_ids}  ({n_gpus} devices)")
    print(f"  Workers/GPU: {n_workers_per_gpu}  → {total_workers} total workers")
    print(f"  Iter:       {args.iterations}  |  Strategy: {args.strategy}")
    print(f"  Output:     {output_base}")

    # GPU queue — each slot = one available GPU
    gpu_queue = mp.Queue()
    for gpu_id in gpu_ids:
        for _ in range(n_workers_per_gpu):
            gpu_queue.put(gpu_id)

    # Task queue
    task_queue = mp.JoinableQueue()
    for seed in range(n_seeds):
        task_queue.put({'seed': seed, 'total': n_seeds})

    result_queue = mp.Queue()

    # Launch workers
    workers = []
    for wid in range(total_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                wid, gpu_queue, task_queue, result_queue,
                args.model_dir, args.vcf, str(output_base),
                args.strategy, args.iterations, args.top_k, args.min_improve,
                args.mode, args.direction_file, args.sites_to_evolve,
                args.ref_vcf, args.pheno, args.phased, args.mc_samples,
                sys.executable, args.homozygous,
            )
        )
        p.start()
        workers.append(p)

    # Add termination sentinels for workers
    for _ in range(total_workers):
        task_queue.put(None)

    # Wait for workers
    for p in workers:
        p.join()

    # Collect results
    results = {}
    while not result_queue.empty():
        r = result_queue.get()
        results[r['seed']] = r

    succeeded = sum(1 for r in results.values() if r['success'])
    failed = n_seeds - succeeded
    print(f"\n{'='*60}")
    print(f"Done. {succeeded}/{n_seeds} OK, {failed} failed.")

    for seed, r in sorted(results.items()):
        if not r['success']:
            print(f"  FAIL seed={seed} rc={r['exit_code']}")

    # Merge VCFs
    evolved_paths = [r['evolved_vcf'] for r in results.values() if r['success'] and r['evolved_vcf']]
    print(f"\n[Merge] {len(evolved_paths)} evolved VCFs")

    if evolved_paths:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from aquila.evolve import merge_evolved_vcfs, compute_qtn_allele_freq_changes

        merged_vcf = output_base / "merged_evolved.vcf.gz"
        merge_evolved_vcfs(args.vcf, evolved_paths, str(merged_vcf), sample_prefix="seed")
        print(f"  Saved: {merged_vcf}")

        if args.qtn_list:
            with open(args.qtn_list) as f:
                qtn_ids = [l.strip() for l in f if l.strip()]
            af_df = compute_qtn_allele_freq_changes(args.vcf, evolved_paths, qtn_ids)
            af_path = output_base / "qtn_af_change_summary.tsv"
            af_df.to_csv(af_path, sep='\t', index=False)
            print(f"  QTN summary: {af_path}")

    pd.DataFrame([{'seed': s, **r} for s, r in sorted(results.items())]
                 ).to_csv(output_base / "run_summary.tsv", sep='\t', index=False)
    print(f"  Summary: {output_base / 'run_summary.tsv'}")


if __name__ == '__main__':
    main()
