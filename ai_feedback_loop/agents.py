from __future__ import annotations

import autogen

from ai_feedback_loop.config import FeedbackLoopConfig


def build_llm_config(config: FeedbackLoopConfig) -> dict[str, object]:
    return {
        "config_list": [
            {
                "model": config.ollama_model,
                "base_url": f"{config.ollama_base_url}/v1",
                "api_key": "ollama",
            }
        ],
        "temperature": 0.0,
        "cache_seed": None,
    }


def build_agents(config: FeedbackLoopConfig) -> dict[str, autogen.ConversableAgent]:
    llm_config = build_llm_config(config)

    analyzer = autogen.AssistantAgent(
        name="failure_analyzer",
        system_message=(
            "You are a senior test-failure analyst for a fraud-detection swarm system. "
            "Given a pytest failure traceback, the failing test source, the source under test, "
            "and any similar historical defects retrieved from memory, produce a concise root-cause "
            "analysis. State the defect class, the exact line implicated, and why it fails. "
            "Do not propose a fix in this message, analysis only."
        ),
        llm_config=llm_config,
    )

    fix_proposer = autogen.AssistantAgent(
        name="fix_proposer",
        system_message=(
            "You are a fix-proposal agent. Given a root-cause analysis, produce a minimal unified "
            "diff patch against the implicated source file. Output only a fenced diff code block, "
            "no prose outside it. The patch must be the smallest change that resolves the root cause "
            "without altering unrelated behavior."
        ),
        llm_config=llm_config,
    )

    critic = autogen.AssistantAgent(
        name="patch_critic",
        system_message=(
            "You are a patch critic. Review the proposed diff against the root-cause analysis and the "
            "original test intent. Respond with exactly one line: APPROVE or REJECT: <reason>. "
            "Reject any patch that changes test assertions to force a pass rather than fixing the "
            "underlying defect."
        ),
        llm_config=llm_config,
    )

    defect_filer = autogen.AssistantAgent(
        name="defect_filer",
        system_message=(
            "You write structured defect records. Given the root-cause analysis and final patch status, "
            "output a JSON object with keys: title, severity, root_cause, fix_summary, status. "
            "Output only the JSON object, nothing else."
        ),
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="orchestrator_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    return {
        "analyzer": analyzer,
        "fix_proposer": fix_proposer,
        "critic": critic,
        "defect_filer": defect_filer,
        "user_proxy": user_proxy,
    }
