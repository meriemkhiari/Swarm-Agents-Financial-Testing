from __future__ import annotations

from ai_feedback_loop.config import FeedbackLoopConfig
from ai_feedback_loop.orchestrator import run_feedback_loop


def main() -> None:
    config = FeedbackLoopConfig()
    run_feedback_loop(config)


if __name__ == "__main__":
    main()
