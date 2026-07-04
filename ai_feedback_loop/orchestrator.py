from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ai_feedback_loop.agents import build_agents
from ai_feedback_loop.config import FeedbackLoopConfig
from ai_feedback_loop.defect_filer import file_defect
from ai_feedback_loop.pytest_runner import FailureRecord, run_pytest
from ai_feedback_loop.vector_store import DefectMemoryStore


def _extract_fenced_block(text: str, language: str) -> str:
    pattern = rf"```{language}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _read_source_under_test(config: FeedbackLoopConfig, node_id: str) -> str:
    test_file = Path(config.project_root) / node_id.split("::")[0]
    return test_file.read_text() if test_file.exists() else ""


def _apply_patch(config: FeedbackLoopConfig, patch_text: str, target_file: str) -> bool:
    target_path = Path(config.project_root) / target_file
    if not target_path.exists():
        return False

    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    shutil.copy(target_path, backup_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as diff_file:
        diff_file.write(patch_text)
        diff_path = diff_file.name

    result = subprocess.run(
        ["patch", "-p1", "--fuzz=2", "-i", diff_path],
        cwd=config.project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        shutil.copy(backup_path, target_path)
        return False
    return True


def process_failure(
    config: FeedbackLoopConfig,
    store: DefectMemoryStore,
    failure: FailureRecord,
) -> bool:
    agents = build_agents(config)
    similar = store.query_similar(failure.traceback, n_results=3)
    similar_context = "\n\n".join(f"{entry['node_id']}: {entry['document']}" for entry in similar)
    source_under_test = _read_source_under_test(config, failure.node_id)

    analysis_prompt = (
        f"Failing test: {failure.node_id}\n"
        f"Traceback:\n{failure.traceback}\n\n"
        f"Test source file content:\n{source_under_test}\n\n"
        f"Similar historical defects:\n{similar_context or 'none found'}"
    )

    agents["user_proxy"].initiate_chat(
        agents["analyzer"],
        message=analysis_prompt,
        max_turns=1,
        clear_history=True,
    )
    analysis_text = agents["analyzer"].last_message()["content"]

    agents["user_proxy"].initiate_chat(
        agents["fix_proposer"],
        message=f"Root cause analysis:\n{analysis_text}\n\nSource file:\n{source_under_test}",
        max_turns=1,
        clear_history=True,
    )
    proposed_patch = _extract_fenced_block(agents["fix_proposer"].last_message()["content"], "diff")

    agents["user_proxy"].initiate_chat(
        agents["critic"],
        message=f"Analysis:\n{analysis_text}\n\nProposed patch:\n{proposed_patch}",
        max_turns=1,
        clear_history=True,
    )
    verdict = agents["critic"].last_message()["content"].strip()

    status = "rejected_by_critic"
    applied = False

    if verdict.upper().startswith("APPROVE") and config.auto_apply_patches:
        applied = _apply_patch(config, proposed_patch, failure.test_file)
        status = "patch_applied" if applied else "patch_apply_failed"
    elif verdict.upper().startswith("APPROVE"):
        status = "approved_awaiting_manual_apply"

    agents["user_proxy"].initiate_chat(
        agents["defect_filer"],
        message=f"Analysis:\n{analysis_text}\n\nPatch status: {status}\n\nPatch:\n{proposed_patch}",
        max_turns=1,
        clear_history=True,
    )

    file_defect(
        config=config,
        store=store,
        node_id=failure.node_id,
        analysis=analysis_text,
        patch=proposed_patch,
        status=status,
    )

    return applied


def run_feedback_loop(config: FeedbackLoopConfig) -> None:
    store = DefectMemoryStore(config)

    for attempt in range(1, config.max_repair_attempts + 1):
        result = run_pytest(config.tests_path, config.project_root)
        print(f"attempt {attempt}: {result.passed} passed, {result.failed} failed, {result.total} total")

        if result.failed == 0:
            print("all tests passing, feedback loop complete")
            return

        any_patch_applied = False
        for failure in result.failures:
            applied = process_failure(config, store, failure)
            any_patch_applied = any_patch_applied or applied

        if not any_patch_applied:
            print(
                "no patches were auto-applied this round "
                "(auto_apply_patches is False or every patch was rejected); "
                "review filed defect records under defects_path and stopping to avoid an unproductive loop"
            )
            return

    print(f"reached max_repair_attempts={config.max_repair_attempts} with failures remaining")
