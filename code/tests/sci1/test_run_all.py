"""Batch scheduling regressions; never import or run the simulation runner."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "arctic_run_all", Path(__file__).resolve().parents[2] / "run_all.py"
)
run_all = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = run_all
_spec.loader.exec_module(run_all)


@pytest.fixture
def paths(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    return tmp_path, logs, Path(run_all.__file__).parent


def experiment(name, **kwargs):
    return run_all.Experiment(name, f"{name}.yaml", name, **kwargs)


@pytest.mark.parametrize("jobs", [1, 3, 28])
@pytest.mark.parametrize("count", [0, 4, 10])
def test_global_worker_budget(monkeypatch, paths, jobs, count):
    root, logs, code = paths
    active = 0
    handles = []

    class Child:
        def wait(self):
            nonlocal active
            active -= jobs
            return 0

    def popen(cmd, **kwargs):
        nonlocal active
        active += int(cmd[cmd.index("--jobs") + 1])
        assert active <= jobs  # earlier experiments must have exited
        assert cmd[cmd.index("--parallel-backend") + 1] == "process"
        assert "--resume" in cmd
        assert "--no-traces" in cmd
        assert kwargs["cwd"] == str(code)
        assert kwargs["start_new_session"] == (os.name == "posix")
        handles.append(kwargs["stdout"])
        return Child()

    monkeypatch.setattr(run_all.subprocess, "Popen", popen)
    exps = [experiment(str(i), extra_args=["--no-traces"]) for i in range(count)]
    assert run_all.run_phase(exps, root, jobs, True, logs, code) == {
        exp.name: 0 for exp in exps
    }
    assert len(handles) == count
    assert active == 0
    assert all(handle.closed for handle in handles)


def test_real_lightweight_children_exit_status_and_logs(monkeypatch, paths):
    root, logs, code = paths
    real_popen = subprocess.Popen
    names = iter(["bad", "good"])

    def popen(cmd, **kwargs):
        name = next(names)
        assert "--resume" not in cmd
        # Substitute a tiny Python process, never the experiment runner.
        return real_popen(
            [sys.executable, "-c", f"print('{name}'); raise SystemExit({7 if name == 'bad' else 0})"],
            **kwargs,
        )

    monkeypatch.setattr(run_all.subprocess, "Popen", popen)
    assert run_all.run_phase(
        [experiment("bad"), experiment("good")], root, 2, False, logs, code
    ) == {"bad": 7, "good": 0}
    assert (logs / "bad.log").read_text().strip() == "bad"
    assert (logs / "good.log").read_text().strip() == "good"


def test_launch_failure_is_recorded_and_queue_continues(monkeypatch, paths):
    root, logs, code = paths
    handles = []

    def popen(cmd, **kwargs):
        handles.append(kwargs["stdout"])
        if len(handles) == 1:
            raise OSError("cannot launch")
        return type("Child", (), {"wait": lambda self: 0})()

    monkeypatch.setattr(run_all.subprocess, "Popen", popen)
    assert run_all.run_phase(
        [experiment("bad"), experiment("good")], root, 2, True, logs, code
    ) == {"bad": 1, "good": 0}
    assert "cannot launch" in (logs / "bad.log").read_text()
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_wait_exception_cleans_child_before_propagating(monkeypatch, paths, error):
    root, logs, code = paths
    handles = []
    stopped = []

    class Child:
        def wait(self):
            raise error()

    child = Child()

    def popen(cmd, **kwargs):
        handles.append(kwargs["stdout"])
        return child

    monkeypatch.setattr(run_all.subprocess, "Popen", popen)
    monkeypatch.setattr(run_all, "_stop_experiment", stopped.append)
    with pytest.raises(error):
        run_all.run_phase([experiment("one"), experiment("two")], root, 2, True, logs, code)
    assert stopped == [child]
    assert len(handles) == 1
    assert handles[0].closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker process groups")
@pytest.mark.parametrize("timeout", [False, True])
def test_stop_signals_worker_group_and_reaps_runner(monkeypatch, timeout):
    signals = []
    waits = []

    class Child:
        pid = 12345

        def wait(self, **kwargs):
            waits.append(kwargs)
            if timeout and kwargs:
                raise subprocess.TimeoutExpired("runner", 5)
            return -signal.SIGTERM

    monkeypatch.setattr(run_all.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    run_all._stop_experiment(Child())
    assert signals[0] == (12345, signal.SIGTERM)
    assert signals[-1] == (12345, signal.SIGKILL)
    assert waits[0] == {"timeout": 5}
    if timeout:
        assert waits[1] == {}


@pytest.mark.parametrize("jobs", [0, -1])
def test_invalid_budget_rejected_before_side_effects(monkeypatch, tmp_path, jobs):
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--jobs", str(jobs), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exc:
        run_all.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError):
        run_all.run_phase([], tmp_path, jobs, True, tmp_path, tmp_path)


@pytest.mark.parametrize("args,expected", [
    ([], ["smoke", "main", "stats", "traces"]),
    (["--phase", "smoke"], ["smoke"]),
    (["--phase", "main"], ["main"]),
    (["--phase", "stats"], ["stats"]),
    (["--phase", "traces"], ["traces"]),
    (["--skip-smoke"], ["main", "stats", "traces"]),
    (["--phase", "smoke", "--skip-smoke"], []),
])
def test_phase_selection_order_and_resume(monkeypatch, tmp_path, args, expected):
    calls = []

    def phase(exps, root, jobs, resume, logs, code):
        calls.append(exps[0].phase)
        assert all(exp.phase == calls[-1] for exp in exps)
        assert jobs == 3
        assert resume is True
        return {exp.name: 0 for exp in exps}

    monkeypatch.setattr(run_all, "run_phase", phase)
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--out", str(tmp_path), "--jobs", "3", *args])
    run_all.main()
    assert calls == expected


@pytest.mark.parametrize("failed_phase,optional,exit_code,expected", [
    ("smoke", False, 1, ["smoke"]),
    ("main", False, 1, ["smoke", "main", "stats", "traces"]),
    ("traces", True, 0, ["smoke", "main", "stats", "traces"]),
])
def test_failure_propagation(monkeypatch, tmp_path, failed_phase, optional, exit_code, expected):
    calls = []
    exps = [experiment(name, phase=name, optional=optional if name == failed_phase else False)
            for name in ["smoke", "main", "stats", "traces"]]
    monkeypatch.setattr(run_all, "EXPERIMENTS", exps)

    def phase(exps, root, jobs, resume, logs, code):
        name = exps[0].phase
        calls.append(name)
        assert resume is False
        return {name: -15 if name == failed_phase else 0}

    monkeypatch.setattr(run_all, "run_phase", phase)
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--out", str(tmp_path), "--no-resume"])
    if exit_code:
        with pytest.raises(SystemExit) as exc:
            run_all.main()
        assert exc.value.code == exit_code
    else:
        run_all.main()
    assert calls == expected


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker process groups")
def test_abnormal_exit_cleans_workers_before_next_launch(monkeypatch, paths):
    root, logs, code = paths
    events = []

    class Child:
        def wait(self):
            return -9

    def popen(cmd, **kwargs):
        events.append("launch")
        return Child()

    monkeypatch.setattr(run_all.subprocess, "Popen", popen)
    monkeypatch.setattr(run_all, "_stop_experiment", lambda proc: events.append("cleanup"))
    assert run_all.run_phase(
        [experiment("one"), experiment("two")], root, 2, True, logs, code
    ) == {"one": -9, "two": -9}
    assert events == ["launch", "cleanup", "launch", "cleanup"]
