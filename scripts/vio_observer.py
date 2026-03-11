#!/usr/bin/env python3
# Copyright (c) 2026, Long Vuong
# SPDX-License-Identifier: BSD-3-Clause
"""
EKF-VIO process observer — live CPU and memory monitor for a running VIO process.

Usage:
  ./scripts/vio_observer.py                    # auto-detect any VIO process
  ./scripts/vio_observer.py --name euroc_runner
  ./scripts/vio_observer.py --pid 12345
  ./scripts/vio_observer.py --interval 0.5     # 500 ms refresh
"""

import argparse
import os
import signal
import sys
import time

# Process names to search for when no --pid/--name given
VIO_PROC_NAMES = [
    "vio_main",
    "vio_node",
]

# ── ANSI helpers ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
CYAN   = "\033[36m"
MUTED  = "\033[90m"
CLEAR  = "\033[2J\033[H"


def color_pct(pct: float) -> str:
    if pct < 50:
        return GREEN
    if pct < 80:
        return YELLOW
    return RED


def bar(pct: float, width: int = 18) -> str:
    filled = min(int(pct / 100 * width), width)
    c = color_pct(pct)
    return c + "█" * filled + MUTED + "░" * (width - filled) + RESET


def fmt_kb(kb: int) -> str:
    if kb >= 1024 * 1024:
        return f"{kb / 1024 / 1024:.1f} GB"
    if kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    return f"{kb} kB"


def parse_kb(s: str) -> int:
    try:
        return int(s.split()[0])
    except (IndexError, ValueError):
        return 0


# ── /proc readers ─────────────────────────────────────────────────────────────

def find_pid(name: str) -> int | None:
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/comm") as f:
                if f.read().strip() == name:
                    return int(entry)
        except OSError:
            continue
    return None


def proc_name(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


def read_proc_ticks(pid: int) -> tuple[int | None, int]:
    """Return (total_ticks, num_threads) from /proc/<pid>/stat."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        return int(fields[13]) + int(fields[14]), int(fields[19])
    except (OSError, IndexError, ValueError):
        return None, 0


def read_total_cpu_ticks() -> int:
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    return sum(int(x) for x in parts)


def read_status(pid: int) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    result[k.strip()] = v.strip()
    except OSError:
        pass
    return result


def list_threads(pid: int) -> list[int]:
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task") if t.isdigit()]
    except OSError:
        return []


def read_thread_ticks(pid: int, tid: int) -> tuple[int | None, str]:
    try:
        with open(f"/proc/{pid}/task/{tid}/stat") as f:
            fields = f.read().split()
        ticks = int(fields[13]) + int(fields[14])
        with open(f"/proc/{pid}/task/{tid}/comm") as f:
            name = f.read().strip()
        return ticks, name
    except (OSError, IndexError, ValueError):
        return None, "?"


# ── Stats accumulator ────────────────────────────────────────────────────────

class Stats:
    """Accumulate min/max/mean for a scalar series."""

    def __init__(self) -> None:
        self._n    = 0
        self._sum  = 0.0
        self._min  = float("inf")
        self._max  = float("-inf")

    def record(self, v: float) -> None:
        self._n   += 1
        self._sum += v
        self._min  = min(self._min, v)
        self._max  = max(self._max, v)

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    @property
    def min(self) -> float:
        return self._min if self._n else 0.0

    @property
    def max(self) -> float:
        return self._max if self._n else 0.0


def render_summary(
    pid: int,
    name: str,
    elapsed: float,
    samples: int,
    cpu: Stats,
    rss: Stats,
    thread_cpu: dict[str, Stats],
) -> None:
    out = ["\n", "═" * 64, "\n"]
    out.append(f"{BOLD}{CYAN}Session Summary{RESET}  "
               f"{MUTED}pid {pid} · {name} · {samples} samples · {elapsed:.0f}s{RESET}\n")
    out.append("─" * 64 + "\n\n")

    # CPU
    out.append(f"  {BOLD}CPU %{RESET}\n")
    out.append(f"    avg  {color_pct(cpu.mean)}{BOLD}{cpu.mean:5.1f}%{RESET}   "
               f"min {GREEN}{cpu.min:5.1f}%{RESET}   "
               f"max {color_pct(cpu.max)}{cpu.max:5.1f}%{RESET}\n\n")

    # Memory
    out.append(f"  {BOLD}RSS{RESET}\n")
    out.append(f"    avg  {GREEN}{BOLD}{fmt_kb(int(rss.mean)):>8}{RESET}   "
               f"min {fmt_kb(int(rss.min)):>8}   "
               f"max {YELLOW}{fmt_kb(int(rss.max)):>8}{RESET}\n\n")

    # Per-thread (sorted by avg CPU desc)
    if thread_cpu:
        out.append(f"  {BOLD}{'THREAD NAME':<24} {'AVG%':>6}  {'MIN%':>6}  {'MAX%':>6}{RESET}\n")
        out.append("  " + "─" * 48 + "\n")
        ranked = sorted(thread_cpu.items(), key=lambda kv: -kv[1].mean)
        for tname, st in ranked:
            if st.n < 2:
                continue
            tc = color_pct(st.mean)
            out.append(
                f"  {tname:<24} "
                f"{tc}{st.mean:5.1f}%{RESET}  "
                f"{GREEN}{st.min:5.1f}%{RESET}  "
                f"{color_pct(st.max)}{st.max:5.1f}%{RESET}\n"
            )

    out.append("\n" + "═" * 64 + "\n")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


# ── Render ────────────────────────────────────────────────────────────────────

def render(
    pid: int,
    name: str,
    cpu_pct: float,
    num_threads: int,
    num_cpus: int,
    vm_rss: int,
    vm_peak: int,
    vm_virt: int,
    thread_stats: list[tuple[int, str, float]],
    interval: float,
) -> None:
    out = [CLEAR]
    out.append(
        f"{BOLD}{CYAN}EKF-VIO Observer{RESET}  "
        f"{MUTED}pid {pid} · {name} · {num_cpus} CPU · "
        f"Δt {interval:.1f}s{RESET}\n"
    )
    out.append("─" * 64 + "\n\n")

    # CPU
    cpu_color = color_pct(cpu_pct)
    out.append(
        f"  {BOLD}CPU {RESET} {bar(cpu_pct)}  "
        f"{cpu_color}{BOLD}{cpu_pct:5.1f}%{RESET}  "
        f"{MUTED}of {num_cpus * 100:.0f}% total  ·  {num_threads} threads{RESET}\n\n"
    )

    # Memory
    out.append(
        f"  {BOLD}RSS {RESET} {GREEN}{BOLD}{fmt_kb(vm_rss):>8}{RESET}  "
        f"{MUTED}peak {fmt_kb(vm_peak)}  ·  virt {fmt_kb(vm_virt)}{RESET}\n\n"
    )

    # Per-thread table
    out.append(
        f"  {BOLD}{'TID':<7} {'NAME':<22} {'CPU%':>6}  {'USAGE'}{RESET}\n"
    )
    out.append("  " + "─" * 58 + "\n")
    for tid, tname, tpct in thread_stats[:14]:
        tc = color_pct(tpct)
        out.append(
            f"  {MUTED}{tid:<7}{RESET} {tname:<22} "
            f"{tc}{tpct:5.1f}%{RESET}  {bar(tpct, 14)}\n"
        )

    out.append(f"\n  {MUTED}Ctrl+C to exit{RESET}\n")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EKF-VIO process observer")
    parser.add_argument("--pid",      type=int,   default=None, help="Target PID")
    parser.add_argument("--name",     type=str,   default=None, help="Process name")
    parser.add_argument("--interval", type=float, default=1.0,  help="Refresh seconds (default 1.0)")
    args = parser.parse_args()

    num_cpus = os.cpu_count() or 1

    # Resolve PID
    pid = args.pid
    if pid is None:
        candidates = [args.name] if args.name else VIO_PROC_NAMES
        for n in candidates:
            pid = find_pid(n)
            if pid:
                break
        if pid is None:
            print(
                f"{RED}No VIO process found. "
                f"Start one of: {VIO_PROC_NAMES}{RESET}\n"
                f"Or pass --pid / --name explicitly."
            )
            sys.exit(1)

    name = proc_name(pid)
    print(f"Attaching to  pid={pid}  ({name}) ...\n")

    # Graceful exit
    running = [True]
    def _stop(sig, frame):  # noqa: ANN001
        running[0] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Accumulators
    cpu_stats: Stats = Stats()
    rss_stats: Stats = Stats()
    thread_cpu_stats: dict[str, Stats] = {}
    session_start = time.monotonic()

    # Initial snapshot
    prev_proc, _ = read_proc_ticks(pid)
    prev_total   = read_total_cpu_ticks()
    prev_threads: dict[int, tuple[int, str]] = {}
    for tid in list_threads(pid):
        t, n = read_thread_ticks(pid, tid)
        if t is not None:
            prev_threads[tid] = (t, n)

    time.sleep(args.interval)

    while running[0]:
        # ── sample ──
        cur_proc, num_threads = read_proc_ticks(pid)
        cur_total = read_total_cpu_ticks()

        if cur_proc is None:
            sys.stdout.write(f"\n{RED}Process {pid} ({name}) has exited.{RESET}\n")
            break

        delta_proc  = cur_proc  - (prev_proc or 0)
        delta_total = cur_total - prev_total
        cpu_pct = (delta_proc / delta_total * num_cpus * 100) if delta_total > 0 else 0.0
        cpu_pct = max(0.0, min(cpu_pct, num_cpus * 100))

        status  = read_status(pid)
        vm_rss  = parse_kb(status.get("VmRSS",  "0 kB"))
        vm_peak = parse_kb(status.get("VmPeak", "0 kB"))
        vm_virt = parse_kb(status.get("VmSize", "0 kB"))

        # per-thread
        cur_threads: dict[int, tuple[int, str]] = {}
        thread_stats: list[tuple[int, str, float]] = []
        for tid in list_threads(pid):
            t, n = read_thread_ticks(pid, tid)
            if t is None:
                continue
            cur_threads[tid] = (t, n)
            if tid in prev_threads and delta_total > 0:
                dt   = t - prev_threads[tid][0]
                tpct = dt / delta_total * num_cpus * 100
                thread_stats.append((tid, n, max(0.0, tpct)))
        thread_stats.sort(key=lambda x: -x[2])

        # Accumulate
        cpu_stats.record(cpu_pct)
        rss_stats.record(vm_rss)
        for _, tname, tpct in thread_stats:
            if tname not in thread_cpu_stats:
                thread_cpu_stats[tname] = Stats()
            thread_cpu_stats[tname].record(tpct)

        render(
            pid, name, cpu_pct, num_threads, num_cpus,
            vm_rss, vm_peak, vm_virt, thread_stats, args.interval,
        )

        prev_proc    = cur_proc
        prev_total   = cur_total
        prev_threads = cur_threads

        time.sleep(args.interval)

    elapsed = time.monotonic() - session_start
    if cpu_stats.n > 0:
        render_summary(pid, name, elapsed, cpu_stats.n,
                       cpu_stats, rss_stats, thread_cpu_stats)
    else:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
