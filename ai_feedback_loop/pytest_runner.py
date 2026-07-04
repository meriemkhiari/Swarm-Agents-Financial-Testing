from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

from pydantic import BaseModel


class FailureRecord(BaseModel):
    node_id: str
    test_file: str
    test_name: str
    message: str
    traceback: str


class PytestRunResult(BaseModel):
    passed: int
    failed: int
    total: int
    failures: list[FailureRecord]


def run_pytest(target: str, project_root: str) -> PytestRunResult:
    report_path = Path(project_root) / f".feedback_loop/report_{uuid.uuid4().hex}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            target,
            "--json-report",
            f"--json-report-file={report_path}",
            "-q",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    report_data = json.loads(report_path.read_text())
    summary = report_data.get("summary", {})
    failures: list[FailureRecord] = []

    for test in report_data.get("tests", []):
        if test.get("outcome") != "failed":
            continue
        call_phase = test.get("call", {})
        node_id = test["nodeid"]
        failures.append(
            FailureRecord(
                node_id=node_id,
                test_file=node_id.split("::")[0],
                test_name=node_id.split("::")[-1],
                message=str(call_phase.get("longrepr", "unknown failure")),
                traceback=str(call_phase.get("longrepr", "")),
            )
        )

    return PytestRunResult(
        passed=summary.get("passed", 0),
        failed=summary.get("failed", 0),
        total=summary.get("total", 0),
        failures=failures,
    )
