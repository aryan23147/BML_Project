import ctypes
import os
import tracemalloc
import numpy as np
import torch


def compute_cl_metrics(R):
    """
    R: numpy array of shape (T, T) where R[t, i] is test accuracy on task i after training task t.
    Matches Section 5 of the proposal:
      - A_T: average accuracy across all T tasks after training task T-1
      - BWT: average change in accuracy on task i from right after learning task i to after task T-1
      - Forgetting: average drop from best accuracy on task i (up to T-2) to final accuracy after task T-1
    """
    R = np.asarray(R, dtype=np.float64)
    T = R.shape[0]
    final_acc = float(np.mean(R[T - 1, :]))
    if T <= 1:
        return {'A_T': final_acc, 'BWT': 0.0, 'Forgetting': 0.0}

    bwt = float(np.mean([R[T - 1, i] - R[i, i] for i in range(T - 1)]))
    forgetting = float(np.mean([np.max(R[i:T, i]) - R[T - 1, i] for i in range(T - 1)]))
    return {
        'A_T': final_acc,
        'BWT': bwt,
        'Forgetting': forgetting,
    }


def learning_curve_auc(epoch_accs):
    """Mean accuracy over training epochs on a new task (to assess forward transfer / adaptation speed)."""
    if len(epoch_accs) == 0:
        return 0.0
    return float(np.mean(epoch_accs))


def get_peak_gpu_memory_mb():
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_peak_cpu_memory_mb():
    """Return process peak CPU RAM (PeakWorkingSetSize on Windows, ru_maxrss on POSIX, or tracemalloc fallback) in MB."""
    if os.name == 'nt':
        try:
            import ctypes.wintypes as wt
            k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            k32.GetCurrentProcess.restype = wt.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wt.DWORD]
            psapi.GetProcessMemoryInfo.restype = wt.BOOL
            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            handle = k32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return float(pmc.PeakWorkingSetSize / (1024 ** 2))
        except Exception:
            pass
    else:
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            # Linux ru_maxrss is in KB; macOS is in bytes
            return float(ru.ru_maxrss / 1024.0)
        except Exception:
            pass

    if tracemalloc.is_tracing():
        _, peak = tracemalloc.get_traced_memory()
        return float(peak / (1024 ** 2))
    return 0.0
