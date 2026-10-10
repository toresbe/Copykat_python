"""NVML per-process VRAM sampler, without a new package dependency.

API/structure: https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
https://docs.nvidia.com/deploy/archive/R535/nvml-api/structnvmlProcessInfo__t.html
"""

import contextlib
import ctypes as c
from pathlib import Path
from typing import ClassVar


class ProcessInfo(c.Structure):
    _fields_: ClassVar[list[tuple[str, type]]] = [
        ("pid", c.c_uint),
        ("usedGpuMemory", c.c_ulonglong),
        ("gpuInstanceId", c.c_uint),
        ("computeInstanceId", c.c_uint),
    ]


class Monitor:
    def __init__(self):
        self.lib = c.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handle = c.c_void_p()
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [c.c_uint, c.POINTER(c.c_void_p)]
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, c.byref(self.handle)))
        self.query = self.lib.nvmlDeviceGetComputeRunningProcesses_v3
        self.query.argtypes = [c.c_void_p, c.POINTER(c.c_uint), c.POINTER(ProcessInfo)]

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError("NVML returned " + str(code))

    def processes(self):
        capacity = 64
        while True:
            count = c.c_uint(capacity)
            rows = (ProcessInfo * capacity)()
            code = self.query(self.handle, c.byref(count), rows)
            if code == 7:
                capacity = max(capacity * 2, count.value + 16)
                continue
            self.check(code)
            return {int(r.pid): int(r.usedGpuMemory) for r in rows[: count.value] if r.usedGpuMemory != 2**64 - 1}

    def sample(self, pid):
        ids = set()
        stack = [pid]
        while stack:
            p = stack.pop()
            if p in ids:
                continue
            ids.add(p)
            with contextlib.suppress(OSError):
                stack.extend(map(int, Path(f"/proc/{p}/task/{p}/children").read_text().split()))
        return sum(value for p, value in self.processes().items() if p in ids)
