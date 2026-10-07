"""MOOSE batch adapter using AutoPF's MatEnsemble launcher."""

from __future__ import annotations

import json
import os
import importlib
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

from ..schema import SimulationBatchHandle, SimulationRequest, SimulationResult


ArgumentBuilder = Callable[[SimulationRequest], Sequence[str]]
ResultLoader = Callable[[SimulationRequest, Path], Union[Mapping[str, Any], SimulationResult]]


def _default_arguments(request: SimulationRequest) -> list[str]:
    values = {**request.condition, **request.parameters}
    return [f"{key}={value}" for key, value in sorted(values.items())]


def _default_result(request: SimulationRequest, directory: Path) -> Mapping[str, Any]:
    return {
        "status": "completed",
        "data_references": {"run_directory": str(directory)},
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class MatEnsembleMOOSEBackend:
    """Run each theory batch with ``autopf.utils.automoose``.

    ``submit`` only materializes a durable batch specification.  The runner
    checkpoints that handle before ``collect`` launches MatEnsemble.  A
    completed batch receipt makes subsequent collection idempotent; normal
    MatEnsemble restart files remain available if execution is interrupted.
    """

    backend_name = "autopf-matensemble-moose-v1"

    def __init__(
        self,
        *,
        moose_app: str | os.PathLike[str],
        base_input: str | os.PathLike[str],
        output_root: str | os.PathLike[str],
        num_cores: int,
        argument_builder: ArgumentBuilder | None = None,
        result_loader: ResultLoader | None = None,
        write_restart_freq: int = 10,
        buffer_time: float = 0.5,
        adaptive_load_balance: bool = True,
    ) -> None:
        self.moose_app = str(Path(moose_app).resolve())
        self.base_input = str(Path(base_input).resolve())
        self.output_root = Path(output_root).resolve()
        self.num_cores = int(num_cores)
        self.argument_builder = argument_builder or _default_arguments
        self.result_loader = result_loader or _default_result
        self.write_restart_freq = int(write_restart_freq)
        self.buffer_time = float(buffer_time)
        self.adaptive_load_balance = bool(adaptive_load_balance)

    def submit(self, requests: Sequence[SimulationRequest]) -> SimulationBatchHandle:
        rows = tuple(requests)
        iteration = rows[0].iteration if rows else 0
        handle_id = f"moose-batch-{iteration:04d}"
        batch_dir = self.output_root / handle_id
        run_dirs = [batch_dir / "runs" / request.simulation_id for request in rows]
        payload = {
            "schema": "autopf.closed_loop.moose_batch.v1",
            "handle_id": handle_id,
            "moose_app": self.moose_app,
            "base_input": self.base_input,
            "num_cores": self.num_cores,
            "requests": [request.to_dict() for request in rows],
            "arguments": [list(self.argument_builder(request)) for request in rows],
            "run_directories": [str(path) for path in run_dirs],
        }
        _atomic_json(batch_dir / "batch_request.json", payload)
        return SimulationBatchHandle(
            backend=self.backend_name,
            handle_id=handle_id,
            requests=rows,
            metadata={"batch_directory": str(batch_dir)},
        )

    def collect(self, handle: SimulationBatchHandle) -> Sequence[SimulationResult]:
        if handle.backend != self.backend_name:
            raise ValueError(f"Handle backend {handle.backend!r} does not match {self.backend_name!r}.")
        batch_dir = Path(handle.metadata["batch_directory"]).resolve()
        receipt = batch_dir / "batch_results.json"
        if receipt.exists():
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            return [SimulationResult.from_dict(row) for row in payload["results"]]

        request_payload = json.loads((batch_dir / "batch_request.json").read_text(encoding="utf-8"))
        run_dirs = [Path(path) for path in request_payload["run_directories"]]
        for directory in run_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        # Imported only on the compute side, where MatEnsemble is installed.
        from autopf.utils import automoose

        params = {
            "total_jobs": len(handle.requests),
            "base_input": self.base_input,
            "arg_list": request_payload["arguments"],
            "num_cores": self.num_cores,
            "directory_list": [str(path) for path in run_dirs],
        }
        previous = Path.cwd()
        try:
            os.chdir(batch_dir)
            automoose(
                self.moose_app,
                params,
                write_restart_freq=self.write_restart_freq,
                buffer_time=self.buffer_time,
                adaptive_load_balance=self.adaptive_load_balance,
            )
        finally:
            os.chdir(previous)

        results = []
        for request, run_dir in zip(handle.requests, run_dirs):
            loaded = self.result_loader(request, run_dir)
            if isinstance(loaded, SimulationResult):
                result = loaded
            else:
                payload = dict(loaded)
                references = dict(payload.pop("data_references", {}))
                metadata = dict(payload.pop("metadata", {}))
                status = str(payload.pop("status", "completed"))
                result = SimulationResult(
                    simulation_id=request.simulation_id,
                    iteration=request.iteration,
                    status=status,
                    outputs=payload,
                    data_references=references,
                    metadata={"backend": self.backend_name, **metadata},
                )
            results.append(result)
        _atomic_json(receipt, {"schema": "autopf.closed_loop.moose_results.v1", "results": [r.to_dict() for r in results]})
        return results


def _load_symbol(specification: str):
    module_name, separator, symbol_name = specification.partition(":")
    if not separator:
        raise ValueError(f"Import specification {specification!r} must use module:function syntax.")
    return getattr(importlib.import_module(module_name), symbol_name)


class MatEnsembleAsyncTSBackend:
    """Dynamically replenish MOOSE chores after every completed field.

    The command-builder hook must return a command whose final structured
    MatEnsemble result contains ``simulation_id``, ``status``, and either an
    inline field or a field reference.  AutoPF serializes POD--GP updates with a
    filesystem lock, so each replacement query sees every earlier completion
    even when many MOOSE chores finish together.
    """

    backend_name = "autopf-matensemble-moose-async-ts-v1"

    def __init__(
        self,
        *,
        output_root: str | os.PathLike[str],
        command_builder: str,
        command_config: Mapping[str, Any] | None = None,
        num_cores: int = 64,
        task_name: str = "moose-autopf-async-ts",
        task_environment: Mapping[str, str] | None = None,
        buffer_time: float = 0.0,
        log_delay: float = 5.0,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.command_builder = str(command_builder)
        self.command_config = dict(command_config or {})
        self.num_cores = int(num_cores)
        self.task_name = str(task_name)
        self.task_environment = dict(task_environment or {})
        self.buffer_time = float(buffer_time)
        self.log_delay = float(log_delay)

    def submit(self, requests: Sequence[SimulationRequest]) -> SimulationBatchHandle:
        rows = tuple(requests)
        if not rows:
            raise ValueError("Asynchronous TS needs at least one initial MOOSE request.")
        contexts = {request.metadata.get("async_context_path") for request in rows}
        if len(contexts) != 1 or None in contexts:
            raise ValueError("Every initial request must reference the same async_context_path.")
        iteration = rows[0].iteration
        handle_id = f"moose-async-ts-{iteration:05d}"
        batch_dir = self.output_root / handle_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "autopf.closed_loop.async_moose_batch.v1",
            "handle_id": handle_id,
            "requests": [request.to_dict() for request in rows],
            "async_context_path": next(iter(contexts)),
            "command_builder": self.command_builder,
            "command_config": self.command_config,
            "num_cores": self.num_cores,
            "task_name": self.task_name,
        }
        _atomic_json(batch_dir / "batch_request.json", payload)
        return SimulationBatchHandle(
            backend=self.backend_name,
            handle_id=handle_id,
            requests=rows,
            metadata={
                "batch_directory": str(batch_dir),
                "adaptive_requests": True,
                "async_context_path": next(iter(contexts)),
            },
        )

    def collect(self, handle: SimulationBatchHandle) -> Sequence[SimulationResult]:
        if handle.backend != self.backend_name:
            raise ValueError(f"Handle backend {handle.backend!r} does not match {self.backend_name!r}.")
        batch_dir = Path(handle.metadata["batch_directory"])
        receipt_path = batch_dir / "batch_results.json"
        if receipt_path.exists():
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            return [SimulationResult.from_dict(row) for row in payload["results"]]

        from matensemble.chore import ExecutableChoreSpec
        from matensemble.model import Resources
        from matensemble.pipeline import Pipeline

        payload = json.loads((batch_dir / "batch_request.json").read_text(encoding="utf-8"))
        builder = _load_symbol(payload["command_builder"])
        context_path = str(payload["async_context_path"])
        pipe = Pipeline(basedir=str(batch_dir / "workflow"))

        @pipe.chore(name=self.task_name)
        def registry_placeholder() -> None:
            raise RuntimeError("registry-only placeholder must never execute")

        @pipe.strategy(
            bolo_list=[self.task_name],
            name="autopf-online-pod-gp-ts-next",
            num_tasks=1,
            cores_per_task=1,
            gpus_per_task=0,
            mpi=False,
            env=self.task_environment,
            inherit_env=True,
        )
        def replenish(result: dict[str, Any]):
            from autopf.closed_loop.async_controller import incorporate_and_propose
            from autopf.closed_loop.backends.matensemble import _load_symbol
            from autopf.closed_loop.schema import SimulationRequest
            from matensemble.chore import ExecutableChoreSpec
            from matensemble.model import Resources

            proposed = incorporate_and_propose(result, context_path)
            if proposed is None:
                return None
            request = SimulationRequest.from_dict(proposed)
            command = _load_symbol(payload["command_builder"])(
                request, "{workdir}", payload["command_config"]
            )
            spec = ExecutableChoreSpec(
                command=command,
                name=payload["task_name"],
                nnodes=1,
                resources=Resources(
                    num_tasks=int(payload["num_cores"]),
                    cores_per_task=1,
                    gpus_per_task=0,
                    mpi=True,
                    env=self.task_environment,
                    inherit_env=True,
                ),
            )
            return spec

        for row in payload["requests"]:
            request = SimulationRequest.from_dict(row)
            chore = pipe.exec(
                command=["placeholder"],
                name=self.task_name,
                num_tasks=self.num_cores,
                cores_per_task=1,
                gpus_per_task=0,
                mpi=True,
                env=self.task_environment,
            )
            chore.nnodes = 1
            chore.command = builder(request, chore.workdir, payload["command_config"])
        future = pipe.submit(
            buffer_time=self.buffer_time,
            log_delay=self.log_delay,
            set_cpu_affinity=True,
            set_gpu_affinity=False,
            adaptive=True,
        )
        future.result()
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
        state = json.loads(Path(context["state_path"]).read_text(encoding="utf-8"))
        if state["pending"]:
            raise RuntimeError(f"Asynchronous TS ended with pending requests: {list(state['pending'])}")
        trained = set(state["trained_simulation_ids"])
        results = []
        for row in state["completed_results"]:
            metadata = dict(row.get("metadata", {}))
            metadata["online_conditioned"] = (
                row.get("status") == "completed" and row["simulation_id"] in trained
            )
            results.append(SimulationResult.from_dict({**row, "metadata": metadata}))
        receipt = {
            "schema": "autopf.closed_loop.async_moose_results.v1",
            "results": [result.to_dict() for result in results],
            "launched": state["launched"],
            "completed": state["completed"],
            "failed": state["failed"],
            "online_update_seconds": state["online_update_seconds"],
            "acquisition_seconds": state["acquisition_seconds"],
            "workflow_directory": str(pipe._base_dir),
        }
        _atomic_json(receipt_path, receipt)
        return results
