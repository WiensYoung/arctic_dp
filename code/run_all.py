#!/usr/bin/env python3
"""按阶段依次运行实验, 实验内部按任务并行。

用法:
    # 默认全局 28 个 worker, 输出到 results
    python run_all.py

    # 自定义全局 worker 预算和输出目录
    python run_all.py --jobs 32 --out /scratch/arctic_dp

    # 只跑冒烟验证
    python run_all.py --phase smoke

    # 跳过冒烟, 直接跑主实验
    python run_all.py --phase main

    # 只跑统计汇总 (主实验跑完后)
    python run_all.py --phase stats

    # 断点续跑 (已完成的 task 自动跳过)
    python run_all.py --resume
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------- 实验定义 ----------

@dataclass
class Experiment:
    """一个实验任务。"""
    name: str
    config: str
    out_subdir: str
    extra_args: List[str] = field(default_factory=list)
    phase: str = "main"           # smoke / main / stats / traces
    optional: bool = False        # True = 失败不阻塞后续
    needs_data: bool = False      # True = 需要真实数据


EXPERIMENTS: List[Experiment] = [
    # ---- Phase: smoke ----
    Experiment(
        name="smoke",
        config="configs/sci1/sci1_smoke.yaml",
        out_subdir="smoke",
        extra_args=["--no-traces"],
        phase="smoke",
    ),
    Experiment(
        name="artifact_check",
        config="configs/sci1/sci1_artifact_check.yaml",
        out_subdir="artifact_check",
        extra_args=["--no-traces"],
        phase="smoke",
    ),
    Experiment(
        name="method_smoke",
        config="configs/sci1/sci1_method_smoke.yaml",
        out_subdir="method_smoke",
        extra_args=["--no-traces"],
        phase="smoke",
    ),
    Experiment(
        name="method_fast",
        config="configs/sci1/sci1_method_fast.yaml",
        out_subdir="method_fast",
        extra_args=["--no-traces"],
        phase="smoke",
    ),

    # ---- Phase: main ----
    Experiment(
        name="paper_full",
        config="configs/sci1/sci1_paper_full.yaml",
        out_subdir="paper_full",
        extra_args=["--skip-statistics"],
        phase="main",
    ),
    Experiment(
        name="submission",
        config="configs/sci1/sci1_submission.yaml",
        out_subdir="submission",
        extra_args=["--skip-statistics"],
        phase="main",
    ),
    Experiment(
        name="ablation",
        config="configs/sci1/sci1_ablation.yaml",
        out_subdir="ablation",
        phase="main",
    ),
    Experiment(
        name="method_paper_parallel",
        config="configs/sci1/sci1_method_paper_parallel.yaml",
        out_subdir="method_paper_parallel",
        extra_args=[],  # YAML 已默认 skip_statistics: true
        phase="main",
    ),
    Experiment(
        name="method_paper_small",
        config="configs/sci1/sci1_method_paper_small.yaml",
        out_subdir="method_paper_small",
        phase="main",
    ),
    Experiment(
        name="paper_small",
        config="configs/sci1/sci1_paper_small.yaml",
        out_subdir="paper_small",
        phase="main",
    ),
    Experiment(
        name="runtime",
        config="configs/sci1/sci1_runtime.yaml",
        out_subdir="runtime",
        phase="main",
    ),
    Experiment(
        name="data_sensitivity",
        config="configs/sci1/sci1_data_sensitivity.yaml",
        out_subdir="data_sensitivity",
        phase="main",
    ),
    Experiment(
        name="scale_comparison",
        config="configs/sci1/sci1_scale_comparison.yaml",
        out_subdir="scale_comparison",
        phase="main",
    ),
    Experiment(
        name="fullscale_experimental",
        config="configs/sci1/sci1_fullscale_experimental.yaml",
        out_subdir="fullscale_experimental",
        extra_args=["--allow-experimental-full-scale"],
        phase="main",
    ),

    # ---- Phase: stats ----
    Experiment(
        name="paper_full_stats",
        config="configs/sci1/sci1_paper_full.yaml",
        out_subdir="paper_full",
        extra_args=["--statistics-only"],
        phase="stats",
    ),
    Experiment(
        name="submission_stats",
        config="configs/sci1/sci1_submission.yaml",
        out_subdir="submission",
        extra_args=["--statistics-only"],
        phase="stats",
    ),
    Experiment(
        name="method_parallel_stats",
        config="configs/sci1/sci1_method_paper_parallel.yaml",
        out_subdir="method_paper_parallel",
        extra_args=["--statistics-only"],
        phase="stats",
    ),

    # ---- Phase: traces ----
    Experiment(
        name="representative_traces",
        config="configs/sci1/sci1_representative_traces.yaml",
        out_subdir="representative_traces",
        phase="traces",
    ),
    Experiment(
        name="real_replay_h1",
        config="configs/sci1/sci1_real_replay_h1.yaml",
        out_subdir="real_replay_h1",
        phase="traces",
        optional=True,
        needs_data=True,
    ),
]


# ---------- 实验队列 ----------

def set_thread_env():
    """限制 BLAS/OpenMP 线程, 防止多进程过度竞争。"""
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _stop_experiment(proc: subprocess.Popen):
    """终止当前实验并回收进程; POSIX 上同时终止其 worker 进程组。"""
    def send_signal(sig):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            elif sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except ProcessLookupError:
            pass

    send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            send_signal(signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()
    finally:
        # 即使 runner 已退出, 仍清除没有响应 SIGTERM 的 worker。
        if os.name == "posix":
            send_signal(signal.SIGKILL)


def run_phase(
    experiments: List[Experiment],
    out_root: Path,
    jobs: int,
    resume: bool,
    log_dir: Path,
    code_dir: Path,
) -> dict:
    """依次运行实验, 将全局 jobs 预算交给当前实验的任务进程池。

    一个实验退出后才启动下一个, 返回 {name: returncode}。
    启动失败按非零退出处理; 等待异常或用户中断时清理当前子进程。
    """
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")
    results: dict[str, int] = {}

    for exp in experiments:
        out_dir = out_root / exp.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "arctic_quasi_dp.sci1.runner",
            "--config", exp.config,
            "--out", str(out_dir),
            "--jobs", str(jobs),
            "--parallel-backend", "process",
        ]
        if resume:
            cmd.append("--resume")
        cmd.extend(exp.extra_args)

        log_path = log_dir / f"{exp.name}.log"
        print(f"  [启动] {exp.name} (worker 预算: {jobs})")
        print(f"         cmd: {' '.join(cmd)}")
        print(f"         log: {log_path}")

        with open(log_path, "w", encoding="utf-8") as log_fh:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(code_dir),
                    env={**os.environ},
                    # Linux 上中断时连同任务 worker 一起清理。
                    start_new_session=(os.name == "posix"),
                )
            except OSError as exc:
                print(f"  [启动失败] {exp.name}: {exc}")
                log_fh.write(f"Failed to start experiment: {exc}\n")
                results[exp.name] = 1
                continue

            try:
                results[exp.name] = proc.wait()
                if results[exp.name] != 0 and os.name == "posix":
                    # runner 异常退出时, 避免遗留 worker 与下一个实验重叠。
                    _stop_experiment(proc)
            except BaseException:
                _stop_experiment(proc)
                raise

        rc = results[exp.name]
        status = "✓ 完成" if rc == 0 else f"✗ 失败 (code={rc})"
        print(f"  [{status}] {exp.name}")

    return results


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser(description="按阶段依次运行 Arctic-DP 实验, 实验内部按任务并行")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="输出根目录 (默认: results, 相对于项目根目录)")
    parser.add_argument("--jobs", type=int, default=28,
                        help="全局 CPU worker 预算, 实验依次运行 (正整数, 默认: 28)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续跑, 跳过已完成的 task (默认开启)")
    parser.add_argument("--no-resume", action="store_true",
                        help="禁用断点续跑, 全部重算")
    parser.add_argument("--phase", choices=["smoke", "main", "stats", "traces", "all"],
                        default="all", help="只跑指定阶段 (默认: all)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="跳过冒烟验证, 直接跑主实验")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")

    resume = not args.no_resume
    code_dir = Path(__file__).resolve().parent
    out_root = args.out
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    set_thread_env()

    phases = {
        "smoke":  [e for e in EXPERIMENTS if e.phase == "smoke"],
        "main":   [e for e in EXPERIMENTS if e.phase == "main"],
        "stats":  [e for e in EXPERIMENTS if e.phase == "stats"],
        "traces": [e for e in EXPERIMENTS if e.phase == "traces"],
    }

    run_phases: List[str]
    if args.phase == "all":
        run_phases = ["smoke", "main", "stats", "traces"]
    else:
        run_phases = [args.phase]

    if args.skip_smoke and "smoke" in run_phases:
        run_phases.remove("smoke")

    print("=" * 60)
    print(" Arctic-DP 全量实验队列")
    print(f" 输出目录: {out_root}")
    print(f" 全局 worker 预算: {args.jobs} (实验依次运行)")
    print(f" 断点续跑: {'是' if resume else '否'}")
    print(f" 执行阶段: {' → '.join(run_phases)}")
    print(f" 日志目录: {log_dir}")
    print("=" * 60)

    all_results: dict[str, int] = {}
    t_start = time.time()

    for phase_name in run_phases:
        phase_exps = phases[phase_name]
        if not phase_exps:
            continue

        print(f"\n{'=' * 60}")
        print(f" Phase: {phase_name} ({len(phase_exps)} 个实验)")
        print(f"{'=' * 60}")

        results = run_phase(
            phase_exps, out_root, args.jobs, resume, log_dir, code_dir,
        )
        all_results.update(results)

        # smoke 阶段失败则中止
        if phase_name == "smoke":
            failed = [n for n, rc in results.items() if rc != 0]
            if failed:
                print(f"\n[错误] 冒烟测试失败: {failed}, 中止运行")
                print(f"查看日志: {log_dir}/")
                sys.exit(1)
            print("\n冒烟验证全部通过 ✓")

    # 汇总
    elapsed = time.time() - t_start
    elapsed_min = elapsed / 60

    print(f"\n{'=' * 60}")
    print(f" 全部完成  耗时: {elapsed_min:.1f} 分钟")
    print(f"{'=' * 60}")
    print()

    # 打印结果表
    print(f"{'实验':<35} {'状态':<10} {'日志'}")
    print("-" * 80)
    for exp in EXPERIMENTS:
        if exp.name not in all_results:
            continue
        rc = all_results[exp.name]
        status = "✓ 通过" if rc == 0 else ("✗ 失败" if not exp.optional else "⊘ 跳过")
        log_path = log_dir / f"{exp.name}.log"
        print(f"  {exp.name:<33} {status:<10} {log_path}")

    # 打印结果目录
    print(f"\n结果目录:")
    for d in sorted(out_root.iterdir()):
        if d.is_dir() and d.name != "logs":
            # 统计 task 完成数
            tasks_dir = d / "raw" / "tasks"
            if tasks_dir.exists():
                done = len(list(tasks_dir.glob("*.done")))
                total = len(list(tasks_dir.glob("*.json")))
                print(f"  {d}/  ({done}/{total} tasks)")
            else:
                print(f"  {d}/")

    # 退出码
    failed_required = [
        n for n, rc in all_results.items()
        if rc != 0 and not any(e.name == n and e.optional for e in EXPERIMENTS)
    ]
    if failed_required:
        print(f"\n[警告] {len(failed_required)} 个必要实验失败: {failed_required}")
        sys.exit(1)

    print("\n全部成功 ✓")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
