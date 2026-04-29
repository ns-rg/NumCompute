import time
import numpy as np
import tracemalloc
from numcompute.stats import mean, mean_loop, std, std_loop
from numcompute.utils import softmax, softmax_loop
from numcompute.sort_search import topk, topk_loop

def benchmark(func, *args, repeats=5, warmup=2, track_memory=False):

    # Warm-up
    for _ in range(warmup):
        func(*args)

    times = []
    memory_usage = []

    for _ in range(repeats):

        if track_memory:
            tracemalloc.start()

        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()

        times.append(end - start)

        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_usage.append(peak)

    times = np.array(times)

    result = {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times, ddof=1)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "runs": times.tolist(),
    }

    if track_memory:
        result["memory_bytes"] = {
            "mean": int(np.mean(memory_usage)),
            "max": int(np.max(memory_usage)),
        }

    return result

def validate(loop_fn, vector_fn, *args, tol=1e-6):

    out1 = loop_fn(*args)
    out2 = vector_fn(*args)

    if isinstance(out1, tuple) and isinstance(out2, tuple):
        if len(out1) != len(out2):
            raise ValueError("Tuple outputs have different lengths.")

        for a, b in zip(out1, out2):
            if not np.allclose(a, b, atol=tol, equal_nan=True):
                raise ValueError("Tuple outputs are not equal within tolerance.")
        return True
    
    if isinstance(out1, np.ndarray) or isinstance(out2, np.ndarray):
        if not np.allclose(out1, out2, atol=tol, equal_nan=True):
            raise ValueError("Outputs are not equal within tolerance.")
        return True
    
    if not np.isclose(out1, out2, atol=tol, equal_nan=True):
        raise ValueError("Outputs are not equal within tolerance.")
    else:
        if abs(out1 - out2) > tol:
            raise ValueError("Outputs are not equal within tolerance.")

    return True

def compare(loop_fn, vector_fn, *args, repeats=5, validate_first=True):

    if validate_first:
        validate(loop_fn, vector_fn, *args)

    loop_res = benchmark(loop_fn, *args, repeats=repeats)
    vec_res = benchmark(vector_fn, *args, repeats=repeats)

    speedup = loop_res["mean_time"] / vec_res["mean_time"]

    return {
        "loop": loop_res,
        "vectorized": vec_res,
        "speedup": float(speedup),
    }

def benchmark_scaling(func, sizes, repeats=5, seed=42):

    np.random.seed(seed)
    results = []

    for n in sizes:
        data = np.random.rand(n).astype(np.float64)

        res = benchmark(func, data, repeats=repeats)

        results.append({
            "size": n,
            "mean_time": res["mean_time"],
            "std_time": res["std_time"],
        })

    return results

def run_performance_table(data):
    tasks = [
        ("Mean", mean_loop, mean),
        ("Std Dev", std_loop, std),
        ("Softmax", softmax_loop, softmax),
        ("Top-K", lambda x: topk_loop(x,3), lambda x: topk(x,3)),
    ]

    print("\nPerformance Comparison (Vectorised vs Python Loops)\n")
    print(f"{'Task':<12} {'Loop (s)':<12} {'Vectorised (s)':<16} {'Speedup':<10}")
    print("-" * 55)

    for name, loop_fn, vec_fn in tasks:
        result = compare(loop_fn, vec_fn, data)

        loop_t = result["loop"]["mean_time"]
        vec_t = result["vectorized"]["mean_time"]
        speed = result["speedup"]

        print(f"{name:<12} {loop_t:<12.6f} {vec_t:<16.6f} {speed:<10.2f}")
