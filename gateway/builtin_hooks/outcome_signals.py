"""Implicit outcome signal detection.

Fires after each agent response. Detects success/failure patterns
from the user's NEXT message without any LLM call.
"""

import re
from typing import List, Optional, Tuple

NEGATIVE_PATTERNS = [
    (r"(?:no|not)\s+(?:what|that|this|right)", "correction"),
    (r"(?:i\s+(?:said|asked|meant|wanted))", "repeat_request"),
    (r"(?:try\s+again|redo|undo|revert|wrong)", "explicit_rejection"),
    (r"(?:that'?s?\s+(?:not|wrong|incorrect|broken))", "explicit_rejection"),
    (r"(?:you\s+(?:missed|forgot|ignored|skipped))", "correction"),
    (r"(?:that\s+doesn'?t?\s+(?:work|help|answer))", "explicit_rejection"),
]

POSITIVE_PATTERNS = [
    (r"(?:thanks|thank\s+you|perfect|great|awesome|nice|good\s+job)", "explicit_approval"),
    (r"(?:that'?s?\s+(?:it|right|correct|perfect|exactly))", "confirmation"),
    (r"(?:ship\s+it|lgtm|looks\s+good|merge\s+it|approved)", "approval"),
    (r"(?:well\s+done|nailed\s+it|excellent|brilliant)", "explicit_approval"),
]


def detect_implicit_outcome(
    user_messages: List,
    assistant_messages: List,
) -> Optional[Tuple[str, str]]:
    """Return (outcome, signal_source) or None.

    Only fires when confidence is HIGH — missing a signal
    is better than recording a wrong one.
    """
    if not user_messages:
        return None

    last = user_messages[-1]
    last_user = last.lower().strip() if isinstance(last, str) else ""
    if not last_user:
        return None

    for pattern, _ in NEGATIVE_PATTERNS:
        if re.search(pattern, last_user, re.IGNORECASE):
            return ("negative", "auto_detect")

    for pattern, _ in POSITIVE_PATTERNS:
        if re.search(pattern, last_user, re.IGNORECASE):
            return ("positive", "auto_detect")

    return None
