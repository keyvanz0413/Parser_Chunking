import re
from typing import List

class TagDetector:
    """Rule-based tag detection using regex and keywords."""
    
    TAG_RULES = {
        "definition": [r"(?i)\bdefined\s+as\b", r"(?i)\brefers\s+to\b", r"(?i)^[A-Z][a-z\s]+:\s"],
        "procedure": [r"(?i)^step\s+\d+", r"(?i)^firstly,", r"(?i)\bhow\s+to\b"],
        "example": [r"(?i)\bfor\s+example\b", r"(?i)\bfor\s+instance\b", r"(?i)\be\.g\.,"],
        "theorem": [r"(?i)^theorem\b", r"(?i)^lemma\b", r"(?i)^proof\b"],
    }
    
    IMPERATIVE_VERBS = {"calculate", "compute", "determine", "find", "solve", "evaluate", "identify", "analyze"}

    def detect_tags(self, text: str) -> List[str]:
        tags = []
        for tag, patterns in self.TAG_RULES.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    tags.append(tag)
                    break
        return tags

    def detect_imperative(self, text: str) -> bool:
        first_word = text.split()[0].lower() if text.split() else ""
        return first_word in self.IMPERATIVE_VERBS
