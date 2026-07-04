from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ai_feedback_loop.config import FeedbackLoopConfig
from ai_feedback_loop.vector_store import DefectMemoryStore


def file_defect(
    config: FeedbackLoopConfig,
    store: DefectMemoryStore,
    node_id: str,
    analysis: str,
    patch: str,
    status: str,
) -> str:
    defect_id = uuid.uuid4().hex
    defects_dir = Path(config.project_root) / config.defects_path
    defects_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "defect_id": defect_id,
        "node_id": node_id,
        "analysis": analysis,
        "patch": patch,
        "status": status,
        "filed_at": datetime.now(timezone.utc).isoformat(),
    }
    defect_path = defects_dir / f"{defect_id}.json"
    defect_path.write_text(json.dumps(record, indent=2))

    store.add_defect(
        defect_id=defect_id,
        document=f"{node_id}\n{analysis}\n{patch}",
        metadata={"node_id": node_id, "status": status},
    )
    return defect_id
