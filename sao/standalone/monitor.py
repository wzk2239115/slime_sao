#!/usr/bin/env python3
"""SAO 集群监控面板 — 从共享 NFS 读取所有机器状态。

在任意机器上运行，不需要 SSH：

  BASH_ENV= python3 sao/standalone/monitor.py           # 实时刷新
  BASH_ENV= python3 sao/standalone/monitor.py --once     # 只看一次
"""
from __future__ import annotations

import os, sys, re, glob, json, time
from datetime import datetime
from collections import defaultdict

WORKDIR = "/home/jovyan/h800fast/wangzekai/slime_sao"
QUEUE_DIR = f"{WORKDIR}/queue"
LOG_DIR = f"{WORKDIR}/logs"
CKPT_DIR = f"{WORKDIR}/checkpoints/sao"

# ANSI colors
C_RESET = "\033[0m"
C_GREEN = "\033[32m"
C_RED   = "\033[31m"
C_YEL   = "\033[33m"
C_CYAN  = "\033[36m"
C_BOLD  = "\033[1m"
C_DIM   = "\033[2m"


def parse_rollout_log(logfile):
    """Parse a rollout worker log, return stats."""
    hostname = os.path.basename(logfile).replace("rollout_", "").replace(".log", "")

    stats = {
        "hostname": hostname,
        "trajectories": 0,
        "last_reward": None,
        "avg_reward": None,
        "last_len": None,
        "rate": None,
        "elapsed_min": None,
        "last_line": "",
        "log_size": 0,
        "log_mtime": 0,
    }

    try:
        stats["log_size"] = os.path.getsize(logfile)
        stats["log_mtime"] = os.path.getmtime(logfile)
    except OSError:
        return stats

    try:
        with open(logfile) as f:
            lines = f.readlines()
    except Exception:
        return stats

    if not lines:
        return stats

    stats["last_line"] = lines[-1].strip()

    # Parse progress lines: [hostname:N] total=N avg100=Y rate=W/min elapsed=Vmin
    for line in reversed(lines):
        m = re.search(r'\[.*?:(\d+)\]\s+total=(\d+)\s+avg100=([\d.]+).*?rate=([\d.]+)/min\s+elapsed=([\d.]+)min', line)
        if m:
            stats["trajectories"] = int(m.group(2))
            stats["avg_reward"] = float(m.group(3))
            stats["rate"] = float(m.group(4))
            stats["elapsed_min"] = float(m.group(5))
            break
        # Old format: [N] r=X avg100=Y len=Z rate=W/min elapsed=Vmin
        m = re.search(r'\[(\d+)\]\s+r=([\d.]+)\s+avg100=([\d.]+)\s+len=(\d+)\s+rate=([\d.]+)/min\s+elapsed=([\d.]+)min', line)
        if m:
            stats["trajectories"] = int(m.group(1)) + 1
            stats["last_reward"] = float(m.group(2))
            stats["avg_reward"] = float(m.group(3))
            stats["last_len"] = int(m.group(4))
            stats["rate"] = float(m.group(5))
            stats["elapsed_min"] = float(m.group(6))
            break

    # Check for errors/stuck
    last_5 = "".join(lines[-5:]).lower()
    if "traceback" in last_5 or "error" in last_5:
        stats["status"] = "ERROR"
    elif "waiting for sglang to reload" in last_5:
        stats["status"] = "RELOADING"
    elif time.time() - stats["log_mtime"] > 600:  # 10 min no update
        stats["status"] = "STALE"
    else:
        stats["status"] = "ACTIVE"

    return stats


def count_queue():
    """Count trajectories in queue."""
    pending = len(glob.glob(f"{QUEUE_DIR}/pending/traj_*.json"))
    done = len(glob.glob(f"{QUEUE_DIR}/done/traj_*.json"))
    return pending, done


def get_recent_rewards(n=20):
    """Get rewards from recent done trajectories."""
    files = sorted(glob.glob(f"{QUEUE_DIR}/done/traj_*.json"), key=os.path.getmtime)[-n:]
    rewards = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            rewards.append(d.get("reward", 0))
        except Exception:
            pass
    return rewards


def get_trainer_status():
    """Parse latest trainer log (by modification time)."""
    logs = sorted(glob.glob(f"{LOG_DIR}/trainer_*.log"), key=os.path.getmtime)
    if not logs:
        return None

    stats = {"log_file": os.path.basename(logs[-1])}
    try:
        with open(logs[-1]) as f:
            lines = f.readlines()
    except Exception:
        return stats

    for line in reversed(lines):
        m = re.search(r'(\d+)/(\d+)\s*\((\d+)%\).*?r=([\d.]+)\(avg20=([\d.]+)\)\s+al=([-\d.]+)\s+cl=([\d.]+)\s+clip=(\d+)%', line)
        if m:
            stats["step"] = int(m.group(1))
            stats["total"] = int(m.group(2))
            stats["pct"] = int(m.group(3))
            stats["reward"] = float(m.group(4))
            stats["avg20"] = float(m.group(5))
            stats["actor_loss"] = float(m.group(6))
            stats["critic_loss"] = float(m.group(7))
            stats["clip"] = int(m.group(8))
            break

    # Check if training is stuck
    try:
        mtime = os.path.getmtime(logs[-1])
        stats["active"] = (time.time() - mtime) < 300  # updated in last 5 min
    except OSError:
        stats["active"] = False

    return stats


def get_checkpoints():
    """List checkpoints."""
    if not os.path.exists(CKPT_DIR):
        return []
    ckpts = sorted([d for d in os.listdir(CKPT_DIR) if d.startswith("step_")])
    return ckpts


def format_duration(seconds):
    """Format duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        return f"{seconds/3600:.1f}h"


def render(once=False):
    """Render the dashboard."""
    while True:
        os.system("clear" if once else "")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{C_BOLD}╔{'═'*78}╗{C_RESET}")
        print(f"{C_BOLD}║{'SAO Cluster Monitor':^78}║{C_RESET}")
        print(f"{C_BOLD}╚{'═'*78}╝{C_RESET}")
        print(f"  {C_DIM}{now}{C_RESET}\n")

        # === Queue ===
        pending, done = count_queue()
        print(f"{C_BOLD}QUEUE{C_RESET}")
        print(f"  Pending: {C_CYAN}{pending}{C_RESET}  Done: {C_GREEN}{done}{C_RESET}  Total generated: {pending+done}")

        # Recent rewards
        recent_rewards = get_recent_rewards(50)
        if recent_rewards:
            avg_r = sum(recent_rewards) / len(recent_rewards)
            n_correct = sum(1 for r in recent_rewards if r == 1.0)
            bar_len = 30
            filled = int(bar_len * avg_r)
            bar = C_GREEN + "█" * filled + C_DIM + "░" * (bar_len - filled) + C_RESET
            print(f"  Recent reward (last {len(recent_rewards)}): {bar} {avg_r:.1%} ({n_correct}/{len(recent_rewards)} correct)")

        # === Trainer ===
        print()
        trainer = get_trainer_status()
        if trainer and "step" in trainer:
            step = trainer["step"]
            total = trainer["total"]
            pct = trainer["pct"]
            bar_len = 30
            filled = int(bar_len * step / total)
            bar = C_GREEN + "█" * filled + C_DIM + "░" * (bar_len - filled) + C_RESET

            active_str = C_GREEN + "●" + C_RESET if trainer.get("active") else C_RED + "○" + C_RESET
            print(f"{C_BOLD}TRAINER{C_RESET} {active_str}")
            print(f"  [{bar}] {step}/{total} ({pct}%)")
            print(f"  reward={C_CYAN}{trainer['reward']:.2f}{C_RESET}(avg20={trainer['avg20']:.2f})  "
                  f"al={trainer['actor_loss']:.4f}  cl={trainer['critic_loss']:.4f}  "
                  f"clip={trainer['clip']}%")
        elif trainer:
            print(f"{C_BOLD}TRAINER{C_RESET} {C_DIM}(waiting for data...){C_RESET}")
        else:
            print(f"{C_BOLD}TRAINER{C_RESET} {C_RED}not running{C_RESET}")

        # === Checkpoints ===
        ckpts = get_checkpoints()
        if ckpts:
            print(f"  Checkpoints: {', '.join(ckpts[-5:])}")

        # === Inference Machines ===
        print(f"\n{C_BOLD}INFERENCE MACHINES{C_RESET}")
        print(f"  {'Machine':<30} {'Status':<10} {'#Trajs':>7} {'Rate':>8} {'AvgRwd':>7} {'LastLen':>8} {'Elapsed':>8}")
        print(f"  {'─'*30} {'─'*10} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

        rollout_logs = sorted(glob.glob(f"{LOG_DIR}/rollout_*.log"))
        total_trajs = 0
        total_rate = 0
        n_active = 0

        for logfile in rollout_logs:
            s = parse_rollout_log(logfile)
            total_trajs += s["trajectories"]
            total_rate += s["rate"] or 0

            # Status indicator
            status = s.get("status", "?")
            if status == "ACTIVE":
                status_str = C_GREEN + "● ACTIVE" + C_RESET
                n_active += 1
            elif status == "RELOADING":
                status_str = C_YEL + "↻ RELOAD" + C_RESET
            elif status == "STALE":
                status_str = C_RED + "✗ STALE" + C_RESET
            elif status == "ERROR":
                status_str = C_RED + "✗ ERROR" + C_RESET
            else:
                status_str = C_DIM + "? " + status + C_RESET

            elapsed = f"{s['elapsed_min']:.0f}m" if s["elapsed_min"] else "-"
            avg_r = f"{s['avg_reward']:.2f}" if s["avg_reward"] is not None else "-"
            last_len = f"{s['last_len']}" if s["last_len"] else "-"
            rate = f"{s['rate']:.1f}/m" if s["rate"] else "-"

            # Truncate hostname
            host = s["hostname"][:28]

            print(f"  {host:<30} {status_str:<25} {s['trajectories']:7d} {rate:>8} {avg_r:>7} {last_len:>8} {elapsed:>8}")

        # Totals
        print(f"  {'─'*30} {'─'*10} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")
        print(f"  {C_BOLD}{'TOTAL':<30}{C_RESET} {n_active} active   {total_trajs:7d} {total_rate:.1f}/m")

        # === Warnings ===
        warnings = []
        if pending == 0 and done < 8:
            warnings.append("Queue empty! Inference machines may not be generating.")
        if pending > 200:
            warnings.append(f"Queue backlog: {pending} pending. Trainer may be too slow.")
        if trainer and not trainer.get("active"):
            warnings.append("Trainer hasn't updated in 5+ minutes. May be stuck or crashed.")
        for logfile in rollout_logs:
            s = parse_rollout_log(logfile)
            if s.get("status") == "ERROR":
                warnings.append(f"{s['hostname']}: rollout worker has errors. Check log.")
            elif s.get("status") == "STALE":
                warnings.append(f"{s['hostname']}: no activity for 10+ minutes. May be stuck.")
        if recent_rewards and all(r == 0 for r in recent_rewards):
            warnings.append("ALL recent rewards are 0! Check reward function before training continues.")

        if warnings:
            print(f"\n{C_BOLD}⚠️  WARNINGS{C_RESET}")
            for w in warnings:
                print(f"  {C_YEL}⚠{C_RESET}  {w}")

        # === Footer ===
        print(f"\n{C_DIM}  Ctrl+C to exit | Logs: ls {LOG_DIR}/ | Queue: {QUEUE_DIR}/{C_RESET}")

        if once:
            break
        time.sleep(15)  # Refresh every 15 seconds


if __name__ == "__main__":
    setup_path = f"/home/jovyan/h800fast/wangzekai/slime_sao"
    if os.path.exists(setup_path):
        sys.path.insert(0, setup_path)

    once = "--once" in sys.argv
    try:
        render(once=once)
    except KeyboardInterrupt:
        print("\n\nExited.")
