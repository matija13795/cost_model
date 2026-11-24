# Inference Timer

Measure the wall-clock latency of **one model call on one input**. The package follows PolyBench/C’s five-measurement trimming method and adds one warmup for inference. A single call to `inference_timer.benchmark()` runs the experiment, writes a CSV, and returns milliseconds per call.

## Install

Unzip the classroom package and run the following command in the Python environment that runs your model. The wheel requires Python 3.10 or newer.

```sh
python -m pip install ./inference_timer-1.0.1-py3-none-any.whl
```

For a one-file assignment, you can instead copy `inference_timer.py` next to your script. Ordinary CPU Python timing uses only the standard library. The PyTorch examples use your existing PyTorch installation.

## Time a single input

This example is runnable. In your own code, substitute your trained model and one representative input. Create both before calling the timer.

```python
import torch
import inference_timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Linear(128, 10).to(device).eval()
sample = torch.randn(128, device=device)  # One input with 128 features.
# If your model requires a leading batch dimension, use sample = sample.unsqueeze(0).
# The resulting shape is (1, 128) and the batch size is one.

with torch.inference_mode():
    mean_ms = inference_timer.benchmark(
        lambda: model(sample), device=sample.device, csv_path="single_input.csv"
    )
print(f"{mean_ms:.3f} ms per input")
```

Pass a callable such as `lambda: model(sample)`. The timer invokes it six times on the same input. For plain Python on CPU, `inference_timer.benchmark(my_function)` is sufficient. If you deliberately use a larger batch, the returned unit is milliseconds per **whole batch**. [PyTorch input shapes for `Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

## How the timing works

PolyBench/C’s `time_benchmark.sh` runs the instrumented benchmark five times, removes the two extremes, and checks how much the remaining times vary. The inference timer uses the following protocol in a single Python process. [PolyBench/C documentation, “Accurate performance timing”](https://www.cs.colostate.edu/~pouchet/software/polybench/)

1. **Warm up once.** Call the same inference function used for measurement. Save this duration as run `0` and exclude it before ranking the measurements. This extra call allows first-use work, including compilation, to occur.
2. **Time five more calls.** Keep the model, input, shape, dtype, and device fixed. Read `time.perf_counter_ns()` before and after each call. On an accelerator, wait for the selected device before the first timestamp to drain earlier work, and again before the end timestamp to include execution of the current call.
3. **Remove two measurements.** Rank runs `1` through `5` by duration. Remove one fastest and one slowest. For ties, remove the earliest minimum and the latest maximum, so exactly three runs remain even if every duration is identical.
4. **Report the arithmetic mean.** Add the three retained durations, divide by three, and convert nanoseconds to milliseconds. The CSV preserves the original run order. Preparation and file writing occur outside the timed interval.

For example, with a 2 ms warmup and measured times of **14, 7, 10, 30, 9 ms**, the warmup is set aside first. The measured extremes are **7** and **30**. The reported mean is **(14 + 10 + 9) / 3 = 11 ms**.

## Which PolyBench features are supported?

| Feature | Package behavior | How to request it |
| --- | --- | --- |
| Five measurements, trim two extremes, average three | Automatic for every completed experiment | `inference_timer.benchmark(...)` or the loop API below |
| Separate inference warmup | Automatic classroom addition, recorded as run `0` | Included in both APIs |
| Wall-clock latency | Host elapsed time, including completion of work on one selected accelerator | Default CPU, or pass `device=sample.device` |
| Cache preparation | Optional best-effort CPU cache sweep. PolyBench/C normally flushes caches before timing. This package leaves cache state alone by default. | `before_each=inference_timer.sweep_cpu_cache` |
| PolyBench’s 5% variability check | Unimplemented. Every complete experiment reports its mean, including widely spaced retained times. | Inspect the three `kept` CSV rows when assessing variability |
| Cycle counts, PAPI hardware counters, real-time scheduling | Outside the package | No API for these controls |
| GPU cache flushing, isolated kernel timing, distributed or multi-device synchronization | Outside the package | Each timer measures one device and reports host wall time |

The supported device values are `cpu`, `cuda`, `mps` (Apple), and `xpu` (Intel), as strings or `torch.device` objects. An indexed device such as `cuda:1` selects that device. For an accelerator, use a PyTorch build with the matching backend. Pass the device doing the work, including when a function moves a CPU input internally. The wall time includes Python dispatch and the final device wait. [CUDA](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html) · [MPS](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html) · [XPU](https://docs.pytorch.org/docs/stable/generated/torch.xpu.synchronize.html)

## Keep an existing loop

Using the `model` and `sample` from the single-input example, wrap just the forward pass in `measure()`. This form also gives you the model output.

```python
timer = inference_timer.InferenceTimer(device=sample.device, csv_path="loop.csv")
with torch.inference_mode():
    for _ in range(timer.runs):        # One warmup plus five measurements
        with timer.measure():
            prediction = model(sample)
        # Read or use prediction here, outside the measured block.
        del prediction
mean_ms = timer.finish()
```

`finish()` requires six successful blocks. A block that raises an exception contributes no sample. Use a fresh timer for another experiment.

## Compiled models and optional preparation

Call `torch.compile` on the model once, outside the timer. The first invocation of that compiled model takes place inside the excluded warmup. If PyTorch compiles again on a later call, that cost belongs to the later measurement. Keep model loading, input construction, transfers and output inspection outside the callback when measuring just the forward pass. `model.eval()` sets evaluation behavior for layers such as dropout, and `torch.inference_mode()` disables gradient tracking. [PyTorch compilation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) · [Inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)

```python
compiled_model = torch.compile(model)
with torch.inference_mode():
    mean_ms = inference_timer.benchmark(
        lambda: compiled_model(sample), device=sample.device, csv_path="compiled.csv"
    )
```

Both APIs accept `before_each`, a zero-argument preparation function that runs outside the clock before **every** call, including warmup. For a CPU cache experiment, pass `before_each=inference_timer.sweep_cpu_cache`. It touches one byte at 64-byte intervals in a 64 MiB host buffer. Set `before_each=lambda: inference_timer.sweep_cpu_cache(128)` to use 128 MiB. Actual eviction depends on the CPU and cache size. The same hook can reset a stateful workload between calls.

## Read the CSV

| Column | Meaning |
| --- | --- |
| `run` | `0` for warmup, `1`–`5` for measurements, blank for the summary |
| `status` | `warmup`, `kept`, `excluded_fastest`, `excluded_slowest`, or `trimmed_mean` |
| `elapsed_ms` | A raw duration expressed in milliseconds to six decimals, or the same floating-point mean the API returns in the summary row |
| `device` | The measured device, including the resolved accelerator index |

The default file is `inference.csv`. A successful experiment replaces the previous CSV at the same path. Give each configuration its own filename to keep its results. [Sample CSV from the plain Python example](examples/sample_report.csv)

## Imports and API names

The distribution is called `inference-timer`. The wheel installs one top-level Python module, `inference_timer`. Its three public names are `benchmark`, `InferenceTimer`, and `sweep_cpu_cache`. The examples use `import inference_timer` so that `inference_timer.benchmark(...)`, Python’s `timeit.Timer`, and PyTorch’s `torch.utils.benchmark.Timer` can be used in the same script. Existing explicit imports such as `from inference_timer import benchmark` also work. [Python `timeit`](https://docs.python.org/3/library/timeit.html) · [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

## Run the tests

From the extracted folder, after installing the wheel, run the following commands. Isolated mode (`-I`) ensures the tests import the installed timer. The tests also exercise the one-file installation separately.

```sh
python -m compileall -q -f inference_timer.py examples tests
python -I -m unittest discover -s tests -v
python -I examples/plain_python.py
```

In an environment with PyTorch installed, the same suite runs eager single-input inference, a batch of size one, graph capture with `torch.compile`, and import checks with `torch.utils.benchmark`. CUDA, MPS, and XPU integration tests run on available devices. For native CPU code generation as well, set `INFERENCE_TIMER_TEST_INDUCTOR=1` in your environment before running the tests. You can also run `python -I examples/pytorch_inference.py` to produce `pytorch.csv`.

**Local verification for 1.0.1.** On x86-64 Linux, the wheel was installed in clean CPython 3.12.14 and 3.13.5 environments. Both versions byte-compiled the module, examples, and tests and passed all 18 standard-library tests. The wheel also timed a single-input dense layer with NumPy 2.3.5 on Python 3.12 and NumPy 2.2.4 on Python 3.13. Eight real PyTorch tests were skipped because these interpreters had no PyTorch installation. The standard-library suite checks accelerator wait ordering with simulated device work.

The included GitHub Actions workflow configures additional runs on Linux, macOS, and Windows when the extracted project is placed at the root of a GitHub repository. Its Linux PyTorch job enables the native compiler test.
