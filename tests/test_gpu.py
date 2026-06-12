"""Quick CuPy / RawKernel smoke test."""
import cupy as cp

a = cp.arange(10, dtype=cp.float32) ** 2
print("compute test:", float(a.sum()))

src = r'extern "C" __global__ void f(float* x){ x[threadIdx.x] *= 2.0f; }'
k = cp.RawKernel(src, "f")
k((1,), (10,), (a,))
print("rawkernel test:", float(a.sum()))
print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
