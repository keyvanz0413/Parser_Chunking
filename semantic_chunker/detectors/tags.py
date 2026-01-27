import re
from typing import List, Dict

class TagDetector:
    """
    Rule-based tag detection using regex and keywords.
    Enhanced with academic/mathematical patterns per Reddit best practices.
    """
    
    TAG_RULES = {
        # Definition patterns (enhanced)
        "definition": [
            r"\bis\s+defined\s+as\b",
            r"\brefers\s+to\b",
            r"\bis\s+a\s+(?:type|form|kind)\s+of\b",
            r"\bcan\s+be\s+defined\b",
            r"\bmeans\s+that\b",
            r"\bwe\s+define\b",
            r"\blet\s+\w+\s+be\b",  # "Let X be..."
            r":\s*(?:a|an|the)\s+\w+\s+(?:that|which)",  # Colon definitions
        ],
        # Theorem/Lemma/Proof patterns (NEW - Academic)
        "theorem": [
            r"^(?:Theorem|Lemma|Proposition|Corollary)\s*\d*",
            r"\btheorem\s+\d+",
            r"\blemma\s+\d+",
            r"\bproposition\s+\d+",
            r"\bcorollary\s+\d+",
        ],
        "proof": [
            r"^Proof[:\.]",
            r"\bproof\s+of\b",
            r"\bQ\.E\.D\.",
            r"∎",  # QED symbol
            r"\bwe\s+(?:prove|show)\s+that\b",
        ],
        # Rule/Principle patterns
        "rule": [
            r"\bmust\s+(?:be|have|always)\b",
            r"\brequired\s+to\b",
            r"\bprinciple\s+(?:of|that)\b",
            r"\balways\s+(?:results|leads)\b",
            r"\bnever\s+(?:should|can)\b",
            r"\blaw\s+of\b",
            r"\bproperty\s+(?:of|that)\b",
        ],
        # Application patterns
        "application": [
            r"\bin\s+practice\b",
            r"\bapplied\s+to\b",
            r"\bused\s+(?:to|for|in)\b",
            r"\bpractical\s+(?:use|application)\b",
            r"\breal-world\b",
        ],
        # Example patterns
        "example": [
            r"\bfor\s+example\b",
            r"\bfor\s+instance\b",
            r"\bsuch\s+as\b",
            r"\bconsider\s+(?:the|a)\b",
            r"\bsuppose\s+(?:that|we)\b",
            r"\billustrat(?:e|ed|ion)\b",
            r"^Example\s*\d*",
            r"\bcase\s+study\b",
        ],
        # Reference patterns (Figure/Table/Equation) - Enhanced for Linkage
        "reference": [
            r"\b(?:figure|fig\.?|table|tbl\.?|exhibit|chart|appendix|box|equation|eq\.?)\s+\d+",
            r"\bsee\s+(?:figure|fig\.?|table|tbl\.?|eq\.?)\b",
            r"\bas\s+shown\s+in\b",
            r"\billustrated\s+in\b",
            r"\b(?:referenced|discussed|presented)\s+in\b",
            r"\[(?:Table|Figure|Eq\.?)\s+\d+\]",      # Bracketed references
            r"\((?:see|refer\s+to)?\s*(?:Figure|Fig\.?|Table|Tbl\.?|Equation|Eq\.?)\s+\d+\)", # Parenthetical
            r"^(?:Table|Figure|Fig\.?|Equation|Eq\.?)\s+\d+", # Block Captions at start of segment
        ],
        # Procedure/Steps patterns (enhanced with imperative detection)
        "procedure": [
            r"\bstep\s+\d+\b",
            r"^\s*\d+\.\s+[A-Z]",  # Numbered list starting with capital
            r"\bprocedure\b",
            r"\bprocess\s+(?:of|for)\b",
            r"\balgorithm\b",
            r"\bmethod\s+(?:for|to)\b",
            r"^(?:First|Second|Third|Finally|Next|Then),?\s",
        ],
        # Key term patterns
        "key_term": [
            r"\bkey\s+(?:term|concept|point|idea)\b",
            r"\bimportant(?:ly)?\b",
            r"\bessential(?:ly)?\b",
            r"\bfundamental\b",
            r"\bcritical(?:ly)?\b",
            r"\bsignificant(?:ly)?\b",
        ],
        # Summary patterns
        "summary": [
            r"\bin\s+summary\b",
            r"\bto\s+summarize\b",
            r"\bin\s+conclusion\b",
            r"\boverall\b",
            r"\bto\s+recap\b",
            r"^Summary\b",
        ],
        # Introduction patterns (NEW)
        "introduction": [
            r"^Introduction\b",
            r"\bthis\s+(?:chapter|section)\s+(?:introduces|presents|discusses)\b",
            r"\boverview\s+of\b",
            r"\bin\s+this\s+(?:chapter|section)\b",
        ],
        # Comparison patterns
        "comparison": [
            r"\bcompare(?:d)?\s+(?:to|with)\b",
            r"\bversus\b",
            r"\bunlike\b",
            r"\bsimilar\s+to\b",
            r"\bdifferent\s+from\b",
            r"\bin\s+contrast\b",
            r"\bon\s+the\s+other\s+hand\b",
        ],
        # Caution/Warning patterns
        "caution": [
            r"\bcaution\b",
            r"\bwarning\b",
            r"\bbe\s+careful\b",
            r"\bavoid\b",
            r"\bdo\s+not\b",
            r"\bnote\s+that\b",
            r"\bimportant:\b",
        ],
        # Assumption patterns
        "assumption": [
            r"^\s*(?:Suppose|Assume|Imagine)\b", # Sentence starters (Hypothesis)
            r"\bassume\s+(?:that|we)\b",
            r"\bassuming\b",
            r"\bgiven\s+that\b",
            r"\bsuppose\s+(?!to\b)(?:that|we|\w+)\b", # Exclude "supposed to"
            r"\bunder\s+the\s+assumption\b",
        ],
        # Evidence patterns (Data-driven)
        "evidence": [
            r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b',
            r'\b(?:p\s*[<>=]\s*[\d.]+|significant(?:ly)?)\b',
            r'\b(?:correlation|r\s*=|R²\s*=)\b',
            r'\b(?:increase[ds]?|decrease[ds]?|grew|rose|fell|declined)\s+(?:by|to)\s+[\d.]+',
            r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?',
            r'\b\d+\s*(?:bps|basis\s+points?|bp)\b',
            r'\b(?:mean|median|average|std|standard\s+deviation)\s*[=:\s]*[\d.]+',
        ],
        # Interpretation patterns
        "interpretation": [
            r"\b(this\s+(?:means|implies|suggests)|interpret|in\s+other\s+words|effectively)\b",
        ],
        # Conclusion/Summary patterns
        "conclusion": [
            r"\b(therefore|thus|hence|in\s+conclusion|as\s+a\s+result|consequently|overall|ultimately|in\s+short|the\s+lesson|takeaway|the\s+result\s+is)\b",
            r"^Summary\b",
            r"\bin\s+summary\b",
        ],
        # Mechanism patterns (Causal chains / System operation)
        "mechanism": [
            r"\b(?:transmission|arbitrage|leverage|fluctuat(?:e|es|ing|ion))\b", # High weight
            r"\b(?:rate|price|premium|spread|return|portfolio|asset|liquidity|yield)\b", # Financial domain
            r"\b(?:leads?\s+to|caus(?:e|es|ing)|result(?:s|ing)\s+in|dri(?:ve|ves|ving)|trigge(?:r|rs|ring)|due\s+to|because\s+of)\b",
            r"\b(?:if|when).*(?:then|lead|result|cause|driven|triggered)\b",
            r"\b(?:impact(?:s|ing)?|influenc(?:e|es|ing)|affect(?:s|ing)?)\b",
            r"\b(?:mechanism|operation|functionality|process|result)\b", # Standard/Low weight
        ],
        # Contrast patterns (Intra-sentence opposing logic)
        "contrast": [
            r"\b(?:but|however|although|even\s+though|whereas|despite|nevertheless|nonetheless|conversely|on\s+the\s+contrary)\b",
            r"\b(?:instead\s+of|rather\s+than|as\s+opposed\s+to)\b",
            r"\b(?:while|yet)\b.*,", # "While X, Y" or "X, yet Y"
        ],
    }
    
    # Imperative verbs for procedure detection
    IMPERATIVE_VERBS = {
        "calculate", "compute", "determine", "find", "solve", "evaluate",
        "identify", "analyze", "compare", "explain", "describe", "define",
        "list", "state", "prove", "show", "demonstrate", "derive", "verify",
        "apply", "use", "consider", "note", "observe", "recall", "remember"
    }
    
    def __init__(self):
        # Compile regex patterns for performance
        self.compiled_rules = {}
        for tag, patterns in self.TAG_RULES.items():
            self.compiled_rules[tag] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns
            ]
    
    def detect_tags(self, text: str) -> List[str]:
        """Detect all matching tags for a given text."""
        detected = []
        for tag, patterns in self.compiled_rules.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected.append(tag)
                    break  # One match per tag is enough
        return detected
    
    def detect_imperative(self, text: str) -> bool:
        """Check if text starts with an imperative verb."""
        words = text.strip().split()
        if words:
            first_word = words[0].lower().rstrip('.,;:')
            return first_word in self.IMPERATIVE_VERBS
        return False
