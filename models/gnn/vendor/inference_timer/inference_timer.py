"""Time one warmup and five measured runs of a Python or PyTorch workload."""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Protocol

__all__ = ["InferenceTimer", "benchmark", "sweep_cpu_cache"]


class _TorchDevice(Protocol):
    """Shape of a torch.device for type checking without importing PyTorch."""

    @property
    def type(self) -> str: ...

    @property
    def index(self) -> int | None: ...


def _timing_device(device: str) -> tuple[str, Callable[[], None] | None]:
    """Resolve the device name and its completion barrier once per experiment."""
    match = re.fullmatch(r"(cpu|cuda|mps|xpu)(?::([0-9]+))?", device)
    if match is None:
        raise ValueError(
            f"Unsupported device {device!r}. Use cpu, cuda, mps, or xpu (e.g. cuda:1)."
        )
    backend_name, index_text = match.groups()
    if backend_name == "cpu":
        return device, None

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ModuleNotFoundError(f"Timing {device!r} requires PyTorch.") from exc

    backend = getattr(torch, backend_name, None)
    synchronize = getattr(backend, "synchronize", None)
    if synchronize is None:
        raise RuntimeError(f"This PyTorch installation cannot synchronize the {backend_name} backend.")
    if backend_name == "mps":
        if index_text is not None and int(index_text) != 0:
            raise ValueError("PyTorch MPS timing supports device mps:0.")
        return "mps:0", synchronize

    index = backend.current_device() if index_text is None else int(index_text)
    return f"{backend_name}:{index}", partial(synchronize, index)


class InferenceTimer:
    """Collect six timed blocks, then save a CSV and return the trimmed mean in ms."""

    runs = 6

    def __init__(
        self,
        *,
        csv_path: str | Path = "inference.csv",
        device: str | _TorchDevice = "cpu",
        before_each: Callable[[], object] | None = None,
    ) -> None:
        self._path = Path(csv_path)
        self._device, self._sync = _timing_device(str(device))
        self._before_each = before_each
        self._elapsed_ns: list[int] = []
        self._active = False
        self._mean_ms: float | None = None

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Time one successful block. Run 0 is the warmup."""
        if self._active:
            raise RuntimeError("A timing block is already active on this timer.")
        if len(self._elapsed_ns) >= self.runs:
            raise RuntimeError(
                "All six runs are complete. Use a new timer for another experiment."
            )

        self._active = True
        try:
            if self._before_each is not None:
                self._before_each()
            if self._sync is not None:
                self._sync()
            start_ns = perf_counter_ns()
            yield
            if self._sync is not None:
                self._sync()
            elapsed_ns = perf_counter_ns() - start_ns
            self._elapsed_ns.append(elapsed_ns)
        finally:
            self._active = False

    def finish(self) -> float:
        """Write all samples and return the arithmetic mean of the middle three in ms."""
        if self._mean_ms is not None:
            return self._mean_ms
        if self._active or len(self._elapsed_ns) != self.runs:
            raise RuntimeError(
                f"Finish requires six completed runs. Recorded {len(self._elapsed_ns)}."
            )

        # Sort run numbers so tied times still identify two distinct samples.
        ranked = sorted(range(1, self.runs), key=self._elapsed_ns.__getitem__)
        kept = ranked[1:-1]
        mean_ms = sum(self._elapsed_ns[i] for i in kept) / (len(kept) * 1_000_000)
        status = ["warmup"] + ["kept"] * len(ranked)
        status[ranked[0]] = "excluded_fastest"
        status[ranked[-1]] = "excluded_slowest"

        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete report even if another experiment uses the same path.
        with TemporaryDirectory(dir=self._path.parent) as directory:
            report = Path(directory) / "report.csv"
            with report.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("run", "status", "elapsed_ms", "device"))
                for run, ns in enumerate(self._elapsed_ns):
                    ms, remainder_ns = divmod(ns, 1_000_000)
                    writer.writerow((run, status[run], f"{ms}.{remainder_ns:06d}", self._device))
                writer.writerow(("", "trimmed_mean", mean_ms, self._device))
            report.replace(self._path)

        self._mean_ms = mean_ms
        return mean_ms


def benchmark(
    infer: Callable[[], object],
    *,
    csv_path: str | Path = "inference.csv",
    device: str | _TorchDevice = "cpu",
    before_each: Callable[[], object] | None = None,
) -> float:
    """Run infer() six times, write the CSV, and return the trimmed mean in ms.

    Pass a device string or the input tensor's torch.device for accelerator
    timing. before_each, if supplied, runs outside each timed interval.
    """
    timer = InferenceTimer(csv_path=csv_path, device=device, before_each=before_each)
    for _ in range(timer.runs):
        with timer.measure():
            output = infer()
        # Keep the output alive through synchronization, then release it between runs.
        del output
    return timer.finish()


def sweep_cpu_cache(size_mib: int = 64) -> None:
    """Touch a host buffer at 64-byte intervals to encourage CPU cache eviction."""
    if type(size_mib) is not int or size_mib < 1:
        raise ValueError("size_mib must be a positive integer.")
    buffer = bytearray(size_mib * 1024 * 1024)
    for offset in range(0, len(buffer), 64):
        buffer[offset] = 1
