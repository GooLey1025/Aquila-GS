# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Spawn-safe scheduling helpers for independent outer-fold jobs."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable, Iterable, Sequence


FoldExecutor = Callable[["FoldJob", str, int], Any]


@dataclass(frozen=True)
class FoldJob:
    """Serializable outer-fold work item."""

    fold_id: int
    payload: Any = None


@dataclass(frozen=True)
class FoldJobResult:
    """Serializable success or failure from a fold worker."""

    fold_id: int
    value: Any = None
    device: str = "cpu"
    worker_seed: int = 0
    error: str | None = None
    traceback: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def derive_seed(
    global_seed: int,
    fold_id: int = 0,
    trial_id: int = 0,
    inner_fold_id: int = 0,
    *extra: int,
) -> int:
    """Derive a stable 31-bit seed from nested-CV coordinates."""
    coordinates = (
        int(global_seed),
        int(fold_id),
        int(trial_id),
        int(inner_fold_id),
        *map(int, extra),
    )
    digest = hashlib.blake2b(
        ":".join(map(str, coordinates)).encode("ascii"),
        digest_size=8,
        person=b"aquila-cv",
    ).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


derive_worker_seed = derive_seed


def detect_gpu_ids(requested: int | Sequence[int] | None = None) -> list[int]:
    """Detect usable CUDA device indices, optionally applying a request."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    available = list(range(torch.cuda.device_count()))
    if requested is None:
        return available
    if isinstance(requested, int):
        if requested < 0:
            raise ValueError("Requested GPU count must be nonnegative")
        return available[:requested]
    selected = [int(device) for device in requested]
    invalid = [device for device in selected if device not in available]
    if invalid:
        raise ValueError(f"Unavailable GPU indices: {invalid}")
    if len(set(selected)) != len(selected):
        raise ValueError("GPU indices must be unique")
    return selected


def share_memory_tensors(value: Any) -> Any:
    """Move every CPU tensor in a nested structure into shared memory."""
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Only CPU tensors can be moved into shared memory")
        return value.share_memory_()
    if isinstance(value, dict):
        return {key: share_memory_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [share_memory_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(share_memory_tensors(item) for item in value)
    return value


share_nested_tensors = share_memory_tensors


class GPUFoldQueue:
    """Execute one independent fold at a time on each available GPU."""

    def __init__(
        self,
        executor: FoldExecutor,
        *,
        gpu_ids: Sequence[int] | None = None,
        global_seed: int = 42,
        start_method: str = "spawn",
        raise_on_error: bool = False,
    ) -> None:
        self.executor = executor
        self.gpu_ids = (
            list(gpu_ids) if gpu_ids is not None else detect_gpu_ids()
        )
        self.global_seed = int(global_seed)
        self.start_method = start_method
        self.raise_on_error = raise_on_error

    def run(self, jobs: Iterable[FoldJob]) -> list[FoldJobResult]:
        """Run jobs dynamically and return results ordered by fold ID."""
        work = [job if isinstance(job, FoldJob) else FoldJob(int(job)) for job in jobs]
        if len({job.fold_id for job in work}) != len(work):
            raise ValueError("Fold jobs must have unique fold IDs")
        if not work:
            return []
        if not self.gpu_ids:
            results = [
                _execute_job(
                    job,
                    self.executor,
                    "cpu",
                    self.global_seed,
                )
                for job in work
            ]
        else:
            results = self._run_spawned(work)
        results.sort(key=lambda result: result.fold_id)
        failures = [result for result in results if not result.succeeded]
        if failures and self.raise_on_error:
            details = "; ".join(
                f"fold {result.fold_id}: {result.error}" for result in failures
            )
            raise RuntimeError(f"Fold execution failed: {details}")
        return results

    def _run_spawned(self, jobs: Sequence[FoldJob]) -> list[FoldJobResult]:
        context = mp.get_context(self.start_method)
        task_queue = context.Queue()
        result_queue = context.Queue()
        for job in jobs:
            task_queue.put(job)
        worker_count = min(len(self.gpu_ids), len(jobs))
        for _ in range(worker_count):
            task_queue.put(None)
        workers = []
        for worker_index, gpu_id in enumerate(self.gpu_ids[:worker_count]):
            process = context.Process(
                target=_fold_worker,
                args=(
                    worker_index,
                    int(gpu_id),
                    task_queue,
                    result_queue,
                    self.executor,
                    self.global_seed,
                ),
            )
            process.start()
            workers.append(process)

        results = []
        while len(results) < len(jobs):
            try:
                results.append(result_queue.get(timeout=1.0))
            except Empty:
                if not any(process.is_alive() for process in workers):
                    break
        for process in workers:
            process.join()
        received = {result.fold_id for result in results}
        for job in jobs:
            if job.fold_id not in received:
                results.append(
                    FoldJobResult(
                        fold_id=job.fold_id,
                        error="Worker exited without returning a result",
                    )
                )
        return results


def execute_fold_jobs(
    jobs: Iterable[FoldJob],
    executor: FoldExecutor,
    *,
    gpu_ids: Sequence[int] | None = None,
    global_seed: int = 42,
    raise_on_error: bool = False,
) -> list[FoldJobResult]:
    """Convenience wrapper around :class:`GPUFoldQueue`."""
    return GPUFoldQueue(
        executor,
        gpu_ids=gpu_ids,
        global_seed=global_seed,
        raise_on_error=raise_on_error,
    ).run(jobs)


run_fold_jobs = execute_fold_jobs


@dataclass(frozen=True)
class GPUWorkResult:
    """Serializable success or failure from a generic GPU worker."""

    job_id: int
    value: Any = None
    device: str = "cpu"
    error: str | None = None
    traceback: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def execute_gpu_jobs(
    jobs: Sequence[Any],
    worker: Callable[..., Any],
    gpu_ids: Sequence[int],
    *,
    job_id_attr: str = "job_id",
    worker_args: tuple[Any, ...] = (),
    start_method: str = "spawn",
    raise_on_error: bool = True,
) -> list[GPUWorkResult]:
    """Run independent jobs across GPUs with dynamic scheduling.

    ``worker`` must be a top-level callable with signature
    ``worker(job, device, *worker_args) -> value``.
    Each job must expose an integer ``job_id`` attribute (or ``job_id_attr``).
    """
    work = list(jobs)
    if not work:
        return []
    ids = [int(getattr(job, job_id_attr)) for job in work]
    if len(set(ids)) != len(ids):
        raise ValueError("GPU jobs must have unique job IDs")
    devices = [f"cuda:{int(gpu_id)}" for gpu_id in gpu_ids]
    if not devices:
        results = [
            _execute_gpu_job(job, worker, "cpu", worker_args, job_id_attr)
            for job in work
        ]
    else:
        results = _run_gpu_jobs_spawned(
            work,
            worker,
            devices,
            worker_args=worker_args,
            job_id_attr=job_id_attr,
            start_method=start_method,
        )
    results.sort(key=lambda result: result.job_id)
    failures = [result for result in results if not result.succeeded]
    if failures and raise_on_error:
        details = "; ".join(
            f"job {result.job_id}: {result.error}" for result in failures
        )
        raise RuntimeError(f"GPU job execution failed: {details}")
    return results


def _run_gpu_jobs_spawned(
    jobs: Sequence[Any],
    worker: Callable[..., Any],
    devices: Sequence[str],
    *,
    worker_args: tuple[Any, ...],
    job_id_attr: str,
    start_method: str,
) -> list[GPUWorkResult]:
    context = mp.get_context(start_method)
    task_queue = context.Queue()
    result_queue = context.Queue()
    for job in jobs:
        task_queue.put(job)
    worker_count = min(len(devices), len(jobs))
    for _ in range(worker_count):
        task_queue.put(None)
    workers = []
    for device in devices[:worker_count]:
        process = context.Process(
            target=_gpu_job_worker,
            args=(
                device,
                task_queue,
                result_queue,
                worker,
                worker_args,
                job_id_attr,
            ),
        )
        process.start()
        workers.append(process)

    results: list[GPUWorkResult] = []
    while len(results) < len(jobs):
        try:
            results.append(result_queue.get(timeout=1.0))
        except Empty:
            if not any(process.is_alive() for process in workers):
                break
    for process in workers:
        process.join()
    received = {result.job_id for result in results}
    for job in jobs:
        job_id = int(getattr(job, job_id_attr))
        if job_id not in received:
            results.append(
                GPUWorkResult(
                    job_id=job_id,
                    error="Worker exited without returning a result",
                )
            )
    return results


def _gpu_job_worker(
    device: str,
    task_queue: Any,
    result_queue: Any,
    worker: Callable[..., Any],
    worker_args: tuple[Any, ...],
    job_id_attr: str,
) -> None:
    if device.startswith("cuda:"):
        import torch

        torch.cuda.set_device(int(device.split(":", 1)[1]))
        torch.backends.cudnn.benchmark = True
    while True:
        job = task_queue.get()
        if job is None:
            return
        result_queue.put(
            _execute_gpu_job(job, worker, device, worker_args, job_id_attr)
        )


def _execute_gpu_job(
    job: Any,
    worker: Callable[..., Any],
    device: str,
    worker_args: tuple[Any, ...],
    job_id_attr: str,
) -> GPUWorkResult:
    job_id = int(getattr(job, job_id_attr))
    try:
        value = worker(job, device, *worker_args)
        return GPUWorkResult(job_id=job_id, value=value, device=device)
    except Exception as error:
        return GPUWorkResult(
            job_id=job_id,
            device=device,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )


class PersistentGPUPool:
    """Long-lived one-process-per-GPU pool with a dynamic shared task queue.

    Unlike ``execute_gpu_jobs``, sentinels are only sent on ``shutdown()``, so
    workers stay alive and pull newly submitted work (e.g. later outer folds or
    final refits) instead of exiting when an early batch of jobs completes.
    """

    def __init__(
        self,
        gpu_ids: Sequence[int],
        worker: Callable[..., Any],
        *,
        worker_args: tuple[Any, ...] = (),
        job_id_attr: str = "job_id",
        start_method: str = "spawn",
    ) -> None:
        if not gpu_ids:
            raise ValueError("PersistentGPUPool requires at least one GPU id")
        self._worker = worker
        self._worker_args = worker_args
        self._job_id_attr = job_id_attr
        self._context = mp.get_context(start_method)
        self._task_queue = self._context.Queue()
        self._result_queue = self._context.Queue()
        self._devices = [f"cuda:{int(gpu_id)}" for gpu_id in gpu_ids]
        self._workers: list[Any] = []
        self._closed = False
        for device in self._devices:
            process = self._context.Process(
                target=_gpu_job_worker,
                args=(
                    device,
                    self._task_queue,
                    self._result_queue,
                    worker,
                    worker_args,
                    job_id_attr,
                ),
            )
            process.start()
            self._workers.append(process)

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(self._devices)

    def submit(self, job: Any) -> None:
        if self._closed:
            raise RuntimeError("PersistentGPUPool is already shut down")
        self._task_queue.put(job)

    def submit_many(self, jobs: Sequence[Any]) -> None:
        for job in jobs:
            self.submit(job)

    def get(self, timeout: float | None = None) -> GPUWorkResult:
        if timeout is None:
            return self._result_queue.get()
        return self._result_queue.get(timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        if not self._closed:
            self._closed = True
            for _ in self._workers:
                self._task_queue.put(None)
        if wait:
            for process in self._workers:
                process.join()

    def __enter__(self) -> "PersistentGPUPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


def _fold_worker(
    worker_index: int,
    gpu_id: int,
    task_queue: Any,
    result_queue: Any,
    executor: FoldExecutor,
    global_seed: int,
) -> None:
    while True:
        job = task_queue.get()
        if job is None:
            return
        result_queue.put(
            _execute_job(
                job,
                executor,
                f"cuda:{gpu_id}",
                global_seed,
                worker_index=worker_index,
            )
        )


def _execute_job(
    job: FoldJob,
    executor: FoldExecutor,
    device: str,
    global_seed: int,
    worker_index: int = 0,
) -> FoldJobResult:
    seed = derive_seed(global_seed, job.fold_id)
    try:
        value = executor(job, device, seed)
        return FoldJobResult(
            fold_id=job.fold_id,
            value=value,
            device=device,
            worker_seed=seed,
        )
    except Exception as error:
        return FoldJobResult(
            fold_id=job.fold_id,
            device=device,
            worker_seed=seed,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
