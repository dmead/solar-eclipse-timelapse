"""Pin worker processes to distinct performance cores.

This is a hybrid CPU: an i9-14900K has 8 P-cores with SMT (logical 0-15) and 16
E-cores without (16-31). Windows 10 has no Thread Director, so its placement is a
heuristic - a good one here, measured mid-render at 57-93% on the even P-core
logical CPUs and under 15% on every E-core, but not an exact one. Two things leak:
a worker sometimes lands on an SMT sibling of a core another worker already has,
and nothing stops one migrating onto an E-core, which for this workload is a large
step down in single-thread throughput.

Giving each worker its own PHYSICAL P-core removes both. It only helps while the
worker count is at or below the number of P-cores; past that the pool has to share
and the OS is better placed to decide how, so `plan` returns nothing and the
callers leave placement alone.

The efficiency class comes from GetLogicalProcessorInformationEx rather than from
the numbering. The numbering happens to put P-cores first on this part, and that
is a convention, not a guarantee.
"""

import ctypes as C
import os
from ctypes import wintypes

__all__ = ["core_groups", "performance_cpus", "plan", "pin"]

_RELATION_PROCESSOR_CORE = 0

# PROCESSOR_RELATIONSHIP, past the 8-byte SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
# header: Flags at +8, EfficiencyClass at +9, 20 reserved, GroupCount at +30, then
# GROUP_AFFINITY with Mask at +32.
_OFF_EFFICIENCY = 9
_OFF_MASK = 32


def core_groups():
    """[(efficiency_class, [logical cpu, ...]), ...], one entry per physical core.

    Empty on anything that is not Windows, or if the query fails - callers treat
    that as "no plan" and leave scheduling to the OS.
    """
    if os.name != "nt":
        return []
    try:
        k32 = C.WinDLL("kernel32", use_last_error=True)
        n = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, None,
                                             C.byref(n))
        buf = (C.c_ubyte * n.value)()
        if not k32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, buf,
                                                    C.byref(n)):
            return []
        out, off = [], 0
        while off < n.value:
            size = C.cast(C.byref(buf, off + 4), C.POINTER(wintypes.DWORD))[0]
            if not size:
                break
            eff = buf[off + _OFF_EFFICIENCY]
            mask = C.cast(C.byref(buf, off + _OFF_MASK), C.POINTER(C.c_size_t))[0]
            out.append((eff, [i for i in range(64) if mask >> i & 1]))
            off += size
        return out
    except Exception:
        return []


def performance_cpus():
    """One logical CPU per physical core of the fastest class, in order.

    One per core, not all of them: two workers on the two SMT threads of one core
    share its execution units, which is most of the point of avoiding this.
    """
    groups = core_groups()
    if not groups:
        return []
    top = max(e for e, _ in groups)
    if sum(1 for e, _ in groups if e == top) == len(groups):
        return []                      # uniform CPU - nothing to prefer
    return [cpus[0] for e, cpus in groups if e == top and cpus]


def plan(n_workers):
    """One CPU per worker, or [] if the OS should decide."""
    cpus = performance_cpus()
    if not cpus or n_workers > len(cpus):
        return []
    return cpus[:n_workers]


def pin(cpu):
    """Confine this process to one logical CPU. False if it could not be done."""
    try:
        import psutil
        psutil.Process().cpu_affinity([cpu])
        return True
    except Exception:
        return False
