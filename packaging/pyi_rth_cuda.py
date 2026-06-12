"""PyInstaller runtime hook: point CuPy at the bundled NVRTC + CUDA headers."""
import os
import sys

base = getattr(sys, "_MEIPASS", None)
if base:
    cuda_rt = os.path.join(base, "nvidia", "cuda_runtime")
    if os.path.isdir(cuda_rt):
        os.environ.setdefault("CUDA_PATH", cuda_rt)
    for sub in (("nvidia", "cuda_nvrtc", "bin"),
                ("nvidia", "cuda_runtime", "bin")):
        d = os.path.join(base, *sub)
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
