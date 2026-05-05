from __future__ import annotations

import re
from typing import List

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]
