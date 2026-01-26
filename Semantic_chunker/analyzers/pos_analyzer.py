import re
import logging
from typing import List, Dict, Any, Optional
from ..detectors.tags import TagDetector

# spaCy will be loaded lazily to avoid import errors if not installed
nlp = None

class POSAnalyzer:
    """
    Uses spaCy for POS tagging and sentence role identification.
    
    Enhanced with:
    - Imperative verb detection
    - Evidence detection (statistics, percentages, trends)
    - Heading context awareness
    - Sentence position features
    """
    
    # Evidence patterns for detecting data-driven sentences
    EVIDENCE_PATTERNS = [
        r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b',                    # Percentages
        r'\b(?:p\s*[<>=]\s*[\d.]+|significant(?:ly)?)\b',        # Statistical significance
        r'\b(?:correlation|r\s*=|R²\s*=)\b',                     # Correlation measures
        r'\b(?:increase[ds]?|decrease[ds]?|grew|rose|fell|declined)\s+(?:by|to)\s+[\d.]+',  # Trends
        r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', # Financial values
        r'\b\d+\s*(?:bps|basis\s+points?|bp)\b',                 # Basis points
        r'\b(?:CI|confidence\s+interval)\s*[=:\s]*',             # Confidence intervals
        r'\b(?:mean|median|average|std|standard\s+deviation)\s*[=:\s]*[\d.]+', # Statistics
        r'\b(?:ratio|rate|yield)\s+(?:of|is|was|=)\s*[\d.]+',    # Ratios/rates
    ]
    
    # Heading keywords that suggest specific roles
    HEADING_ROLE_HINTS = {
        'example': 'example',
        'case study': 'example',
        'illustration': 'example',
        'limitation': 'limitation',
        'constraint': 'limitation',
        'caveat': 'limitation',
        'method': 'procedure',
        'approach': 'procedure',
        'procedure': 'procedure',
        'how to': 'procedure',
        'result': 'evidence',
        'finding': 'evidence',
        'data': 'evidence',
        'empirical': 'evidence',
        'conclusion': 'conclusion',
        'summary': 'conclusion',
        'takeaway': 'conclusion',
        'definition': 'definition',
        'what is': 'definition',
        'assumption': 'assumption',
        'hypothesis': 'assumption',
    }
    
    # Short sentence exceptions - don't mark as irrelevant
    SHORT_SENTENCE_EXCEPTIONS = [
        r'\d+(?:\.\d+)?%',              # Percentages
        r'\$[\d,]+',                     # Currency
        r'(?:figure|table|exhibit)\s*\d', # References
        r'[=<>±]\s*\d',                  # Math expressions
        r'\b\d+\s*(?:bps|bp)\b',         # Basis points
    ]
    
    # Bloom's Taxonomy Verbs for Learning Objectives
    BLOOM_LO_VERBS = {
        # Remember/Understand
        'list', 'describe', 'state', 'identify', 'name', 'label', 'recall', 'select', 'match',
        'explain', 'clarify', 'discuss', 'report', 'review', 'summarize', 'illustrate',
        # Apply/Analyze
        'apply', 'implement', 'solve', 'use', 'compute', 'calculate', 'estimate', 'demonstrate',
        'analyze', 'differentiate', 'distinguish', 'compare', 'contrast', 'examine',
        # Evaluate/Create
        'evaluate', 'assess', 'judge', 'appraise', 'defend', 'justify', 'critique',
        'create', 'design', 'formulate', 'compose', 'construct', 'develop', 'derive',
        # Additional instructional verbs
        'specify', 'understand', 'outline', 'point'
    }
    
    # Priority-ordered roles for sequential detection
    ROLE_PRIORITY = [
        "reference", "theorem", "definition", "assumption", 
        "conclusion", "contrast", "comparison", "evidence", 
        "procedure", "mechanism", "interpretation", "limitation", 
        "application", "example"
    ]

    def __init__(self, model_name: str = "en_core_web_md"):
        global nlp
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load(model_name)
            except Exception as e:
                # Fallback to sm if md not found, or raise
                try:
                    import spacy
                    nlp = spacy.load("en_core_web_sm")
                except:
                    raise RuntimeError(f"Could not load spaCy: {e}") from e
        self.nlp = nlp
        self.tag_detector = TagDetector()
        
        # Compile patterns for efficiency
        self._short_exceptions = [re.compile(p, re.IGNORECASE) for p in self.SHORT_SENTENCE_EXCEPTIONS]
        self._boilerplate = [
            re.compile(r"^\s*(see\s+(?:also|below|above)|continued|ibid)\s*$", re.I),
            re.compile(r"\b(?:https?://|www\.|mhhe\.com)\b", re.I),
            re.compile(r"\b(?:all\s+rights\s+reserved|copyright)\b", re.I),
            re.compile(r"\bmcgraw\s*hill\b", re.I)
        ]
    
    
    def analyze_sentences(self, text: str, heading_path: str = "", 
                         forced_role: Optional[str] = None,
                         chunk_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze text and return sentences with roles.
        
        Enhanced with heading context and position awareness.
        """
        if self.nlp is None:
            raise RuntimeError("POSAnalyzer.nlp is None. This should not happen if initialization passed.")
        
        doc = self.nlp(text)
        sentences = []
        sent_list = list(doc.sents)
        total_sentences = len(sent_list)
        
        # Get heading context hint
        heading_hint = self._get_heading_hint(heading_path)
        
        for i, sent in enumerate(sent_list):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            
            # Extract POS tags
            pos_tags = [token.pos_ for token in sent]
            
            # Check for imperative
            is_imperative = self._is_imperative_sentence(sent)
            
            # Calculate relative position
            relative_pos = i / max(total_sentences, 1)
            
            # Detect sentence form (NEW)
            form = self._detect_form(sent_text, pos_tags, is_imperative)
            
            # Determine role/function with enhanced context
            prev_role = sentences[-1]["role"] if sentences else None
            
            role = self._determine_role(
                sent_text, i, pos_tags, is_imperative,
                heading_hint=heading_hint,
                relative_pos=relative_pos,
                is_first=(i == 0),
                is_last=(i == total_sentences - 1),
                form=form,
                prev_role=prev_role,
                chunk_type=chunk_type,
                heading_path=heading_path
            )
            
            # Forced Role Linkage: Override role if forced or if vague in structural blocks
            if forced_role:
                role = forced_role
            elif heading_hint == "header" and role == "explanation":
                role = "topic"
            
            # Detect all applicable tags for this sentence
            sentence_tags = self.tag_detector.detect_tags(sent_text)
            
            # --- Composite Semantic Implementation: Primary Role + Attributes ---
            secondary_attributes = {}
            
            # Specialized Case: Question + Reference
            if role == "question" and "reference" in sentence_tags:
                secondary_attributes["has_reference"] = True
                # Extract ref_target (e.g., "Table 1.1")
                ref_match = re.search(r'\b(?:Figure|Table|Exhibit|Equation|Formula|Eq\.?|Fig\.?)\s+\d+(?:\.\d+)?', sent_text, re.I)
                if ref_match:
                    secondary_attributes["ref_target"] = ref_match.group(0)

            sentences.append({
                "text": sent_text,
                "role": role,
                "secondary_attributes": secondary_attributes, # NEW: Sub-tag chaining
                "tags": sentence_tags,
                "form": form,
                "pos_tags": pos_tags[:10],
                "is_imperative": is_imperative
            })
        
        return sentences
    
    def pre_analyze_role(self, text: str) -> Optional[str]:
        """
        Quickly pre-analyze a segment's role using keyword and POS heuristics.
        Used for early role detection to influence chunking/adsorption.
        """
        if not text or len(text.strip()) < 5:
            return None
            
        text_lower = text.lower().strip()
        
        # 1. LO Pattern: "learning objectives", "by the end of this...", etc.
        lo_markers = [
            r'\blearning\s+objectives?\b',
            r'\bat\s+the\s+end\s+of\b',
            r'\byou\s+will\s+be\s+able\s+to\b',
            r'\bafter\s+studying\b'
        ]
        for pattern in lo_markers:
            if re.search(pattern, text_lower):
                return 'learning_objective'
                
        # 2. Verb-Initial LO: "Specify...", "Calculate...", etc.
        # Use simple split for speed, if it starts with a Bloom verb
        words = text_lower.split()
        if words and words[0] in self.BLOOM_LO_VERBS:
            # Check length to avoid short noise like "Explain" as a header
            if len(words) > 3:
                return 'learning_objective'
                
        # 3. Structural Header types
        if re.match(r'^(?:Table|Figure|Exhibit|Equation|Formula)\s*\d+', text, re.IGNORECASE):
            # These are handled by CaptionBondingHelper usually, but good to have
            return None
            
        return None

    def _get_heading_hint(self, heading_path: str) -> Optional[str]:
        """Extract role hint from heading path."""
        if not heading_path:
            return None
        
        heading_lower = heading_path.lower()
        for keyword, role in self.HEADING_ROLE_HINTS.items():
            if keyword in heading_lower:
                return role
        return None
    
    def _detect_form(self, text: str, pos_tags: List[str], is_imperative: bool) -> str:
        """
        Detect sentence form (Layer 1 in discourse taxonomy).
        
        Forms:
        - declarative: Statement (default)
        - interrogative: Question (ends with ?)
        - imperative: Command (verb-initial without subject)
        - formula: Mathematical expression
        - fragment: Incomplete/short text
        
        Returns:
            One of: declarative, interrogative, imperative, formula, fragment
        """
        text_stripped = text.strip()
        
        # Interrogative: ends with question mark
        if text_stripped.endswith('?'):
            return "interrogative"
        
        # Imperative: already detected
        if is_imperative:
            return "imperative"
        
        # Formula: multiple math operators or equation patterns
        if re.search(r'[=+\-*/^]{2,}', text) or re.search(r'\b\w+\s*=\s*\w+', text):
            return "formula"
        
        # Fragment: very short or no verb
        if len(text_stripped) < 10:
            return "fragment"
        if pos_tags and "VERB" not in pos_tags and "AUX" not in pos_tags:
            if len(text_stripped) < 30:
                return "fragment"
        
        # Default: declarative statement
        return "declarative"
    
    def _is_imperative_sentence(self, sent) -> bool:
        """
        Check if sentence is imperative (command form).
        Imperatives typically start with a base form verb (VB) without a subject.
        """
        tokens = list(sent)
        if not tokens:
            return False
        
        first_token = tokens[0]
        
        # Check if first token is a verb in base form
        if first_token.pos_ == "VERB" and first_token.tag_ in ["VB", "VBP"]:
            # Additional check: no subject before the verb
            if first_token.dep_ == "ROOT":
                return True
        
        # Also check using our keyword list
        first_word = first_token.text.lower()
        return first_word in self.tag_detector.IMPERATIVE_VERBS
    
    def _determine_role(self, text: str, position: int, pos_tags: List[str], 
                        is_imperative: bool, heading_hint: Optional[str] = None,
                        relative_pos: float = 0.5, is_first: bool = False,
                        is_last: bool = False, form: str = "declarative",
                        prev_role: Optional[str] = None,
                        chunk_type: Optional[str] = None,
                        heading_path: Optional[str] = None) -> str:
        """Determines role using priority-ordered rules and TagDetector."""
        text_stripped = text.strip()
        text_lower = text.lower()
        
        # 1. Structural/Format roles (highest priority)
        if form == "interrogative":
            return "question"

        # 2. Rule-based roles from TagDetector (the core logic)
        detected_tags = self.tag_detector.detect_tags(text)
        
        # PRIORITY CHECK: If it's a reference, return immediately regardless of length/noise
        if "reference" in detected_tags:
            return "reference"
            
        # 3. Noise/Irrelevant detection (moved after reference check)
        if len(text_stripped) < 15 and not any(p.search(text) for p in self._short_exceptions):
            return "irrelevant"
        if any(p.search(text_lower) for p in self._boilerplate):
            return "irrelevant"

        # 4. Other Priority Roles
        for role in self.ROLE_PRIORITY:
            if role == "reference": continue # Already handled
            if role in detected_tags:
                # Domain Constraint: mechanism only in main body explanation/example chunks
                if role == "mechanism":
                    # 1. Ignore if in front matter
                    if heading_path and re.search(r"front\s*matter", heading_path, re.I):
                        continue
                    # 2. Ignore if contains "Note:" or "McGraw Hill" or administrative words
                    if re.search(r"\bNote:|\bMcGraw\s*Hill\b|\b(?:chapter|author|text|preface|acknowledgment)\b", text_lower):
                        continue
                    # 3. Only in的主体块 (explanation, example, or generic paragraph)
                    if chunk_type and chunk_type not in ["explanation", "example", "paragraph"]:
                        continue
                
                # Domain Constraint: assumption refinement
                if role == "assumption":
                    # High priority starters: Suppose/Assume/Imagine (but not 'supposed to')
                    starts_with_hypo = re.match(r"^\s*(?:Suppose|Assume|Imagine)\b", text)
                    if starts_with_hypo:
                        # Highest priority, bypass some other checks
                        return "assumption"
                    
                    # In Exercises/Cases, assumptions are very common
                    is_case_context = chunk_type in ["exercise", "question", "example"] or \
                                      (heading_path and re.search(r"exercise|problem|case", heading_path, re.I))
                    
                    if not is_case_context and not starts_with_hypo:
                        # In normal text, without a strong starter, it might just be a declarative "assume"
                        # We might want to be more conservative
                        if len(text_stripped.split()) > 40: # Too long for a typical hypothesis
                            continue
                        
                # Specific logic for procedure: needs verb or specific sequence word
                if role == "procedure" and not is_imperative and not re.match(r"^(?:first|second|next|then|finally)\b", text_lower):
                    continue
                return role

        # 4. Heading Context Boost
        if heading_hint and heading_hint != "explanation":
            return heading_hint

        # 5. Specific Logic for Topic (First sentence of a block)
        if is_first and len(text_stripped.split()) < 15 and "VERB" in pos_tags:
            # Topic should not start with complex conjunctions
            if not re.search(r"^(?:Because|Since|Although|While|If|Whereas)\b", text):
                return "topic"
            
        return "explanation"
    
    def get_last_n_sentences(self, text: str, n: int = 2) -> str:
        """Extract the last N sentences from text for overlap."""
        if self.nlp is None:
            # Fallback: simple split
            sentences = text.split('. ')
            return '. '.join(sentences[-n:]) if len(sentences) >= n else text
        
        doc = self.nlp(text)
        sents = list(doc.sents)
        if len(sents) <= n:
            return text
        return ' '.join([s.text for s in sents[-n:]])
