"""
Generate dynamic flexible job shop scheduling (DFJSP) training data.

Scale: 4 machines, 6 initial jobs, rush orders (1-2 jobs) at ~50% progress.
Labels: approximate-optimal reschedule from OR-Tools CP-SAT.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

NUM_MACHINES = 4
NUM_INITIAL_JOBS = 6
MIN_OPS_PER_JOB = 3
MAX_OPS_PER_JOB = 5
MIN_PROC_TIME = 1
MAX_PROC_TIME = 15
MIN_RUSH_JOBS = 1
MAX_RUSH_JOBS = 2

SOLVER_TIME_LIMIT_S = 3.0
HORIZON_PADDING = 10


@dataclass
class OperationAlt:
    machine: int
    duration: int


@dataclass
class ScheduledOp:
    job_id: int
    op_idx: int
    machine: int
    start: int
    duration: int

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass
class Instance:
    initial_jobs: list[list[list[OperationAlt]]]
    rush_jobs: list[list[list[OperationAlt]]]
    insertion_time: int = 0
    initial_makespan: int = 0
    final_makespan: int = 0
    completed_at_insertion: list[ScheduledOp] = field(default_factory=list)
    in_progress_at_insertion: list[ScheduledOp] = field(default_factory=list)
    waiting_at_insertion: list[tuple[int, int]] = field(default_factory=list)
    final_schedule: list[ScheduledOp] = field(default_factory=list)


def _eligible_machines(rng: random.Random) -> list[int]:
    count = rng.randint(1, NUM_MACHINES)
    return sorted(rng.sample(range(NUM_MACHINES), count))


def _random_job(rng: random.Random) -> list[list[OperationAlt]]:
    num_ops = rng.randint(MIN_OPS_PER_JOB, MAX_OPS_PER_JOB)
    job: list[list[OperationAlt]] = []
    for _ in range(num_ops):
        machines = _eligible_machines(rng)
        alts = [
            OperationAlt(m, rng.randint(MIN_PROC_TIME, MAX_PROC_TIME))
            for m in machines
        ]
        job.append(alts)
    return job


def _estimate_horizon(jobs: list[list[list[OperationAlt]]]) -> int:
    total = sum(min(a.duration for a in op) for job in jobs for op in job)
    return total + HORIZON_PADDING


def _solve_fjsp(
    jobs: list[list[list[OperationAlt]]],
    *,
    release_times: dict[int, int] | None = None,
    min_start_times: dict[tuple[int, int], int] | None = None,
    fixed_ops: dict[tuple[int, int], ScheduledOp] | None = None,
    time_limit: float = SOLVER_TIME_LIMIT_S,
) -> tuple[list[ScheduledOp], int, str]:
    """Flexible job shop solver using CP-SAT optional intervals."""
    release_times = release_times or {}
    min_start_times = min_start_times or {}
    fixed_ops = fixed_ops or {}

    model = cp_model.CpModel()
    horizon = _estimate_horizon(jobs)

    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    intervals_by_machine: dict[int, list[cp_model.IntervalVar]] = {
        m: [] for m in range(NUM_MACHINES)
    }
    presences: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    all_keys: list[tuple[int, int]] = []

    for job_id, job in enumerate(jobs):
        for op_idx, alts in enumerate(job):
            key = (job_id, op_idx)
            all_keys.append(key)
            start = model.new_int_var(0, horizon, f"s_{job_id}_{op_idx}")
            end = model.new_int_var(0, horizon, f"e_{job_id}_{op_idx}")
            starts[key] = start
            ends[key] = end

            if key in fixed_ops:
                fixed = fixed_ops[key]
                model.add(start == fixed.start)
                model.add(end == fixed.end)
                interval = model.new_interval_var(
                    fixed.start,
                    fixed.duration,
                    fixed.end,
                    f"iv_fix_{job_id}_{op_idx}",
                )
                intervals_by_machine[fixed.machine].append(interval)
                continue

            alt_vars: list[cp_model.BoolVar] = []
            for alt in alts:
                presence = model.new_bool_var(f"p_{job_id}_{op_idx}_m{alt.machine}")
                presences[(job_id, op_idx, alt.machine)] = presence
                alt_vars.append(presence)
                interval = model.new_optional_interval_var(
                    start,
                    alt.duration,
                    end,
                    presence,
                    f"iv_{job_id}_{op_idx}_m{alt.machine}",
                )
                intervals_by_machine[alt.machine].append(interval)
            model.add_exactly_one(alt_vars)

            if op_idx == 0 and job_id in release_times:
                model.add(start >= release_times[job_id])
            if key in min_start_times:
                model.add(start >= min_start_times[key])

    for job_id, job in enumerate(jobs):
        for op_idx in range(len(job) - 1):
            model.add(ends[(job_id, op_idx)] <= starts[(job_id, op_idx + 1)])

    for machine_intervals in intervals_by_machine.values():
        if len(machine_intervals) > 1:
            model.add_no_overlap(machine_intervals)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [ends[k] for k in all_keys])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 4
    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], -1, status_name

    schedule: list[ScheduledOp] = []
    for job_id, job in enumerate(jobs):
        for op_idx, alts in enumerate(job):
            key = (job_id, op_idx)
            if key in fixed_ops:
                fixed = fixed_ops[key]
                schedule.append(fixed)
                continue
            chosen_machine, duration = -1, 0
            for alt in alts:
                presence = presences.get((job_id, op_idx, alt.machine))
                if presence is not None and solver.value(presence):
                    chosen_machine = alt.machine
                    duration = alt.duration
                    break
            schedule.append(
                ScheduledOp(
                    job_id=job_id,
                    op_idx=op_idx,
                    machine=chosen_machine,
                    start=int(solver.value(starts[key])),
                    duration=duration,
                )
            )

    schedule.sort(key=lambda op: (op.start, op.job_id, op.op_idx))
    return schedule, int(solver.value(makespan)), status_name


def _build_instance(rng: random.Random) -> Instance | None:
    initial_jobs = [_random_job(rng) for _ in range(NUM_INITIAL_JOBS)]
    rush_jobs = [
        _random_job(rng) for _ in range(rng.randint(MIN_RUSH_JOBS, MAX_RUSH_JOBS))
    ]

    init_schedule, init_mk, init_status = _solve_fjsp(initial_jobs)
    if init_status not in ("OPTIMAL", "FEASIBLE") or init_mk <= 0:
        return None

    insertion_time = max(1, init_mk // 2)

    completed = [op for op in init_schedule if op.end <= insertion_time]
    in_progress = [op for op in init_schedule if op.start < insertion_time < op.end]
    waiting = {
        (op.job_id, op.op_idx)
        for op in init_schedule
        if op.start >= insertion_time
    }

    fixed: dict[tuple[int, int], ScheduledOp] = {
        (op.job_id, op.op_idx): op for op in completed + in_progress
    }

    all_jobs = initial_jobs + rush_jobs
    release_times = {
        job_id: insertion_time
        for job_id in range(NUM_INITIAL_JOBS, len(all_jobs))
    }
    min_start_times = {(job_id, op_idx): insertion_time for job_id, op_idx in waiting}

    final_schedule, final_mk, final_status = _solve_fjsp(
        all_jobs,
        release_times=release_times,
        min_start_times=min_start_times,
        fixed_ops=fixed,
    )
    if final_status not in ("OPTIMAL", "FEASIBLE") or final_mk <= 0:
        return None

    return Instance(
        initial_jobs=initial_jobs,
        rush_jobs=rush_jobs,
        insertion_time=insertion_time,
        initial_makespan=init_mk,
        final_makespan=final_mk,
        completed_at_insertion=completed,
        in_progress_at_insertion=in_progress,
        waiting_at_insertion=sorted(waiting),
        final_schedule=final_schedule,
    )


def _machine_status_text(inst: Instance) -> str:
    lines = [f"当前时刻 t={inst.insertion_time}，各机器状态："]
    for m in range(NUM_MACHINES):
        running = [op for op in inst.in_progress_at_insertion if op.machine == m]
        done_count = sum(1 for op in inst.completed_at_insertion if op.machine == m)
        if running:
            op = running[0]
            remaining = op.end - inst.insertion_time
            lines.append(
                f"  机器{m + 1}：正在加工 J{op.job_id + 1}-O{op.op_idx + 1}，"
                f"剩余 {remaining} 单位时间"
            )
        else:
            lines.append(f"  机器{m + 1}：空闲（已完成 {done_count} 道工序）")
    return "\n".join(lines)


def _format_job_text(job_id: int, job: list[list[OperationAlt]], prefix: str) -> str:
    lines = [f"{prefix} J{job_id + 1}："]
    for op_idx, alts in enumerate(job):
        alt_text = " / ".join(f"M{a.machine + 1}({a.duration})" for a in alts)
        lines.append(f"  工序{op_idx + 1}：可选 {alt_text}")
    return "\n".join(lines)


def _build_description(inst: Instance) -> str:
    parts = [
        "【动态柔性作业车间调度问题】",
        f"车间有 {NUM_MACHINES} 台机器（M1-M{NUM_MACHINES}），"
        f"初始 {NUM_INITIAL_JOBS} 个作业，每道工序可在多台机器上加工，加工时间因机器而异。",
        f"初始计划完工时间（仅含初始作业）约为 {inst.initial_makespan}。",
        "",
        "── 初始作业 ──",
    ]
    for j, job in enumerate(inst.initial_jobs):
        parts.append(_format_job_text(j, job, "作业"))

    parts.extend(
        [
            "",
            f"── 急单插队（t={inst.insertion_time}，约为初始进度的 50%）──",
            _machine_status_text(inst),
            f"已完成 {len(inst.completed_at_insertion)} 道工序，"
            f"{len(inst.in_progress_at_insertion)} 道工序进行中，"
            f"{len(inst.waiting_at_insertion)} 道工序尚未开始。",
            f"此时插入 {len(inst.rush_jobs)} 个急单作业：",
        ]
    )
    for i, rush_job in enumerate(inst.rush_jobs):
        parts.append(_format_job_text(NUM_INITIAL_JOBS + i, rush_job, "急单"))

    parts.extend(
        [
            "",
            "请在满足工序先后顺序和机器互斥的前提下，重新调度所有未完成工序及急单，"
            f"目标是最小化总完工时间（makespan）。",
        ]
    )
    return "\n".join(parts)


def _schedule_to_actions(schedule: list[ScheduledOp]) -> list[dict[str, int]]:
    return [
        {
            "job": op.job_id + 1,
            "operation": op.op_idx + 1,
            "machine": op.machine + 1,
            "start": op.start,
            "duration": op.duration,
        }
        for op in schedule
    ]


def _generate_sample(rng: random.Random, instance_id: str) -> dict[str, Any] | None:
    inst = _build_instance(rng)
    if inst is None:
        return None

    return {
        "instance_id": instance_id,
        "description": _build_description(inst),
        "optimal_action": {
            "type": "reschedule",
            "insertion_time": inst.insertion_time,
            "makespan": inst.final_makespan,
            "schedule": _schedule_to_actions(inst.final_schedule),
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate_dataset(count: int, prefix: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 30

    while len(records) < count and attempts < max_attempts:
        attempts += 1
        sample = _generate_sample(rng, f"{prefix}_{len(records):04d}")
        if sample is not None:
            records.append(sample)

    if len(records) < count:
        raise RuntimeError(
            f"Only generated {len(records)}/{count} samples after {attempts} attempts."
        )
    return records


def main() -> None:
    out_dir = Path(__file__).parent
    train_path = out_dir / "train_data.jsonl"
    test_path = out_dir / "test_data.jsonl"

    print("Generating 200 training samples ...")
    train_data = generate_dataset(200, "train", seed=42)
    _write_jsonl(train_path, train_data)
    print(f"  -> {train_path} ({len(train_data)} lines)")

    print("Generating 50 test samples ...")
    test_data = generate_dataset(50, "test", seed=2024)
    _write_jsonl(test_path, test_data)
    print(f"  -> {test_path} ({len(test_data)} lines)")

    sample = train_data[0]
    print("\nSample record preview:")
    print(f"  instance_id   : {sample['instance_id']}")
    print(f"  description   : {sample['description'][:120]}...")
    print(
        f"  optimal_action: makespan={sample['optimal_action']['makespan']}, "
        f"{len(sample['optimal_action']['schedule'])} scheduled ops"
    )


if __name__ == "__main__":
    main()
