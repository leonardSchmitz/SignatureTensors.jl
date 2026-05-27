import numpy as np
import time
import gc
import iisignature

def benchmark_iisignature(ds, ms, k, num_samples=100):
    wall_time = {}
    for d in ds:
        for m in ms:
            times = []
            for _ in range(num_samples):
                path = np.random.randint(-20, 20, size=(m, d))
                # warmup
                iisignature.sig(path, k)
                gc.collect()
                start = time.perf_counter()
                iisignature.sig(path, k)
                end = time.perf_counter()
                times.append((end - start) * 1000.0)  # ms
            wall_time[(d, m)] = np.median(times)

    ds_list = list(ds)
    ms_list = list(ms)
    wall_matrix = np.zeros((len(ds_list), len(ms_list)))
    for (d, m), t in wall_time.items():
        i = ds_list.index(d)
        j = ms_list.index(m)
        wall_matrix[i, j] = t

    return wall_matrix
if __name__ == "__main__":
    ds = [10, 20, 30, 40, 50, 60]
    ms = [10, 20, 30, 40, 50, 60]
    k = 3
    result = benchmark_iisignature(ds, ms, k, num_samples=10)
    print(result)
