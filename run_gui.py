"""Launcher for the RocketCFD GUI.

`RocketCFD.exe --selftest` runs a small headless GPU simulation to verify
the packaged CUDA pipeline works on this machine.
"""
import sys


def selftest():
    """Headless GPU smoke test. Writes %TEMP%/rocketcfd_selftest.log because
    windowed builds have no console for tracebacks."""
    import os
    import tempfile
    import traceback

    log_path = os.path.join(tempfile.gettempdir(), "rocketcfd_selftest.log")
    log = open(log_path, "w", encoding="utf-8")

    def out(msg):
        log.write(msg + "\n")
        log.flush()
        try:
            print(msg)
        except Exception:
            pass

    try:
        out(f"frozen: {getattr(sys, 'frozen', False)}")
        out(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '<unset>')}")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np
        out(f"numpy {np.__version__}")
        import cupy as cp
        out(f"cupy {cp.__version__}")
        out(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        from rocketcfd.config import SimConfig
        from rocketcfd.mask import load_mask
        from rocketcfd.sample import make_nozzle_png
        from rocketcfd.solver import GPUSolver

        path = os.path.join(tempfile.gettempdir(), "rocketcfd_selftest.png")
        make_nozzle_png(path, 200, 200)
        cfg = SimConfig()
        cfg.inlet_ramp_steps = 10
        mask = load_mask(path, cfg.meters_per_pixel)
        solver = GPUSolver(mask, cfg)
        solver.step(50)
        snap = solver.snapshot()
        mach = float(np.nanmax(snap["fields"]["Mach"]))
        assert np.isfinite(mach) and mach > 0.0
        out(f"SELFTEST OK — 50 GPU steps, Mach max {mach:.3f}")
    except Exception:
        out("SELFTEST FAILED:")
        out(traceback.format_exc())
        log.close()
        sys.exit(1)
    log.close()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        from rocketcfd.gui.main import main
        main()
