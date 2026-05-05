from __future__ import annotations

import os
import sys
from pathlib import Path

# Add service-local src directory so chatbot imports are self-contained.
SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from docs_chatbot_service.api import app as app_module
from docs_chatbot_service.core.service import RetrievalService


def _rebind_service_index_root() -> None:
    index_root = Path(os.getenv("CHATBOT_INDEX_ROOT", "data/index"))
    # Resolve relative path from repo root for predictable local/dev behavior.
    if not index_root.is_absolute():
        index_root = (REPO_ROOT / index_root).resolve()
    app_module.service = RetrievalService(index_root=index_root)


_rebind_service_index_root()
app = app_module.app
