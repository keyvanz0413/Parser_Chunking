"""
Logic Segmenter Module v2.0 (Upgraded)

Processes flat_segments from parser_docling.py and applies:
1. POS Tagging (via spaCy) - Sentence role identification with Imperative detection
2. Rule-Based Grouping - List aggregation, definition detection, etc.
3. Tag Enrichment - 20+ semantic tags from taxonomy (enhanced with Theorem/Lemma/Proof)
4. Context Overlap - Each chunk carries context from previous chunk for RAG continuity
5. Token-based Length Control - Prevents extreme chunk sizes

Input: JSON output from parser_docling.py
Output: Enriched chunks with roles, tags, context overlap, and metadata
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# spaCy will be loaded lazily to avoid import errors if not installed
nlp = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ChunkingConfig:
    """Configuration for chunking behavior."""
    # Token limits (approximate, using word count as proxy)
    MIN_CHUNK_WORDS = 30          # Merge if below this
    MAX_CHUNK_WORDS = 500         # Split if above this
    TARGET_CHUNK_WORDS = 200      # Ideal chunk size
    
    # Overlap settings
    OVERLAP_SENTENCES = 2         # Number of sentences to carry over
    ENABLE_OVERLAP = True         # Toggle overlap feature
    
    # Buffer settings
    MAX_BUFFER_SEGMENTS = 5       # Max segments before forced flush
    
    # Short paragraph merge threshold
    SHORT_PARAGRAPH_WORDS = 50    # Paragraphs shorter than this may be merged


# =============================================================================
# Data Models (Schema for Sunday Deliverable)
# =============================================================================

@dataclass
class SentenceRole:
    """Represents a sentence with its identified role."""
    text: str
    role: str  # topic, definition, example, conclusion, evidence, procedural, imperative
    pos_tags: List[str] = field(default_factory=list)
    is_imperative: bool = False


@dataclass
class EnrichedChunk:
    """
    Core Schema v2.0 - The standardized output format with overlap support.
    
    Fields:
    - chunk_id: Unique identifier
    - heading_path: Breadcrumb context (e.g., "Chapter 1 > Section 1.1")
    - chunk_type: Primary type (definition, procedure, explanation, list, theorem, etc.)
    - content: Full text content
    - context_prefix: Overlap text from previous chunk (for RAG continuity)
    - sentences: List of sentences with roles
    - tags: Semantic tags from 20+ taxonomy
    - source_segments: Original segment IDs for traceability
    - page_range: Page numbers covered by this chunk
    - word_count: Approximate word count for length analysis
    """
    chunk_id: str
    heading_path: str
    chunk_type: str
    content: str
    context_prefix: str = ""  # NEW: Overlap from previous chunk
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source_segments: List[str] = field(default_factory=list)
    page_range: List[int] = field(default_factory=list)
    depth: int = 0
    word_count: int = 0  # NEW: For length analysis
    

# =============================================================================
# Tag Rules (20+ Tags - Enhanced with Theorem/Lemma/Proof)
# =============================================================================

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
        # Formula/Equation patterns (refined to reduce false positives)
        "formula": [
            r"\bequation\s+\d+",
            r"\bformula\s+(?:for|to)\b",
            r"\bwhere\s+[A-Z]\s*=",
            r"\b[A-Z]\s*=\s*[A-Z\d]",  # Variable assignments
            r"∑|∫|∂|√",  # Math symbols
            r"\bderivative\s+of\b",
            r"\bintegral\s+of\b",
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
        # Visual reference patterns
        "visual_ref": [
            r"\bfigure\s+\d+",
            r"\btable\s+\d+",
            r"\bexhibit\s+\d+",
            r"\bsee\s+(?:figure|table)\b",
            r"\bas\s+shown\s+in\b",
            r"\billustrated\s+in\b",
        ],
        # Exercise/Problem patterns (NEW)
        "exercise": [
            r"^(?:Exercise|Problem|Question)\s*\d*",
            r"\bsolve\s+(?:the|for)\b",
            r"\bfind\s+the\s+value\b",
            r"\bcalculate\s+the\b",
            r"\bdetermine\s+(?:the|whether)\b",
        ],
        # Assumption patterns (NEW)
        "assumption": [
            r"\bassume\s+(?:that|we)\b",
            r"\bassuming\b",
            r"\bgiven\s+that\b",
            r"\bsuppose\s+(?:that|we)\b",
            r"\bunder\s+the\s+assumption\b",
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


# =============================================================================
# POS Analyzer (Enhanced with Imperative Detection)
# =============================================================================

class POSAnalyzer:
    """
    Uses spaCy for POS tagging and sentence role identification.
    Enhanced with imperative verb detection per Reddit best practices.
    """
    
    def __init__(self, model_name: str = "en_core_web_md"):
        global nlp
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}. POS analysis disabled.")
                nlp = None
        self.nlp = nlp
        self.tag_detector = TagDetector()
    
    def analyze_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyze text and return sentences with roles.
        
        Roles:
        - topic: Introduces main idea (usually first sentence)
        - definition: Contains definition pattern
        - example: Contains example indicator
        - conclusion: Summarizes or concludes
        - procedural: Step-by-step instruction
        - imperative: Command/instruction starting with verb
        - evidence: Supporting evidence or data
        - explanation: Default explanatory content
        """
        if self.nlp is None:
            # Fallback: simple sentence split without POS
            return [{"text": text.strip(), "role": "explanation", "pos_tags": [], "is_imperative": False}]
        
        doc = self.nlp(text)
        sentences = []
        
        for i, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Extract POS tags
            pos_tags = [token.pos_ for token in sent]
            
            # Check for imperative (verb at start)
            is_imperative = self._is_imperative_sentence(sent)
            
            # Determine role based on position, content, and POS
            role = self._determine_role(sent_text, i, pos_tags, is_imperative)
            
            sentences.append({
                "text": sent_text,
                "role": role,
                "pos_tags": pos_tags[:10],  # Limit for storage
                "is_imperative": is_imperative
            })
        
        return sentences
    
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
                        is_imperative: bool) -> str:
        """
        Determine sentence role based on content, position, and POS.
        
        15 Roles (aligned with Senior's requirements):
        
        === Core Content Roles ===
        1. definition - Concept definitions
        2. explanation - General explanatory content (default)
        3. example - Examples and illustrations
        4. formula - Mathematical expressions
        5. procedure - Step-by-step instructions (includes imperative)
        
        === Semantic Relationship Roles ===
        6. mechanism - How something works
        7. assumption - Prerequisites/starting conditions
        8. interpretation - Explaining meaning/implications
        9. limitation - Constraints/boundaries
        10. comparison - Comparing/contrasting concepts
        11. application - Practical use cases
        
        === Structural Roles ===
        12. reference - Figure/Table/Citation references
        13. conclusion - Summary/concluding statements
        14. topic - Opening/main idea sentence
        
        === Special ===
        15. irrelevant - Low semantic value (boilerplate, filler)
        """
        text_lower = text.lower()
        text_stripped = text.strip()
        
        # ============ IRRELEVANT (check first) ============
        # Short sentences with no semantic content
        if len(text_stripped) < 15:
            return "irrelevant"
        # Boilerplate patterns
        if re.search(r"^\s*(see\s+(?:also|below|above)|page\s+\d+|continued|ibid)\s*$", text_lower):
            return "irrelevant"
        
        # ============ PROCEDURE (imperative + step patterns) ============
        if is_imperative:
            return "procedure"
        if re.search(r"\b(step\s+\d|first|second|third|finally)\b", text_lower):
            return "procedure"
        
        # ============ DEFINITION ============
        if re.search(r"\bis\s+(defined\s+as|a\s+\w+\s+that)\b", text_lower):
            return "definition"
        if re.search(r"\brefers?\s+to\b|\bmeans?\s+that\b", text_lower):
            return "definition"
        
        # ============ FORMULA ============
        if re.search(r"[=+\-*/^].*[=+\-*/^]", text):
            return "formula"
        if re.search(r"\bequation\b|\bformula\b|\bwhere\s+\w+\s*=", text_lower):
            return "formula"
        
        # ============ REFERENCE (per Senior: "Figure Y shows...") ============
        if re.search(r"\b(figure|table|exhibit|chart)\s+\d", text_lower):
            return "reference"
        if re.search(r"(?:as\s+)?(?:shown|illustrated|presented)\s+in", text_lower):
            return "reference"
        
        # ============ MECHANISM (per Senior) ============
        if re.search(r"\b(mechanism|process|works?\s+by|functions?\s+by|how\s+\w+\s+works?)\b", text_lower):
            return "mechanism"
        
        # ============ INTERPRETATION (per Senior) ============
        if re.search(r"\b(this\s+(?:means|implies|suggests)|interpret|in\s+other\s+words)\b", text_lower):
            return "interpretation"
        
        # ============ LIMITATION (per Senior) ============
        if re.search(r"\b(limitation|constraint|caveat|does\s+not\s+(?:apply|work)|only\s+works?\s+(?:when|if))\b", text_lower):
            return "limitation"
        
        # ============ COMPARISON (includes contrast per Senior) ============
        if re.search(r"\b(compar|contrast|unlike|whereas|similar|differ)\b", text_lower):
            return "comparison"
        if re.search(r"\b(more|less|greater|smaller)\s+than\b", text_lower):
            return "comparison"
        
        # ============ ASSUMPTION (per Senior) ============
        if re.search(r"\b(assume|assuming|given\s+that|suppose|provided\s+that)\b", text_lower):
            return "assumption"
        if re.search(r"^(?:If|When|Suppose|Assume|Given)\b", text):
            return "assumption"
        
        # ============ APPLICATION ============
        if re.search(r"\b(in\s+practice|applies?\s+to|used\s+(?:in|for)|practical|real-world)\b", text_lower):
            return "application"
        
        # ============ EXAMPLE ============
        if re.search(r"\b(for\s+example|for\s+instance|such\s+as|consider\s+the|e\.g\.)\b", text_lower):
            return "example"
        
        # ============ CONCLUSION ============
        if re.search(r"\b(therefore|thus|hence|in\s+conclusion|as\s+a\s+result|consequently)\b", text_lower):
            return "conclusion"
        
        # ============ TOPIC (first sentence with verb) ============
        if position == 0 and "VERB" in pos_tags:
            return "topic"
        
        # ============ DEFAULT ============
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


# =============================================================================
# Logic Segmenter (Main Class - Enhanced)
# =============================================================================

class LogicSegmenter:
    """
    Main segmenter that processes flat_segments from parser_docling.py.
    
    Pipeline v2.0:
    1. Load segments from JSON
    2. Group segments by logical rules (lists, procedures, definitions)
    3. Analyze with POS tagging (enhanced with imperative detection)
    4. Apply context overlap for RAG continuity
    5. Enrich with tags (20+ including Theorem/Lemma/Proof)
    6. Apply length constraints (min/max word counts)
    7. Output standardized chunks
    """
    
    def __init__(self, use_pos: bool = True, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.tag_detector = TagDetector()
        self.pos_analyzer = POSAnalyzer() if use_pos else None
        self.chunk_counter = 0
        self.previous_chunk_text = ""  # For overlap
    
    def process_file(self, input_json_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a parser_docling.py output file.
        
        Args:
            input_json_path: Path to the JSON from parser_docling
            output_path: Optional path to save enriched output
            
        Returns:
            Dictionary with enriched chunks and statistics
        """
        logger.info(f"Processing: {input_json_path}")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flat_segments = data.get('flat_segments', [])
        metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(flat_segments)} segments")
        
        # Process segments into chunks
        chunks = self._process_segments(flat_segments)
        
        # Post-process: merge short chunks and apply overlap
        chunks = self._post_process_chunks(chunks)
        
        # Build output
        result = {
            "metadata": {
                **metadata,
                "total_chunks": len(chunks),
                "total_segments": len(flat_segments),
                "processing_version": "2.0",
                "features": ["context_overlap", "imperative_detection", "theorem_tagging", "length_control"],
                "processing_stats": self._calculate_stats(chunks)
            },
            "chunks": [asdict(c) for c in chunks]
        }
        
        if output_path:
            self._save_json(result, output_path)
        
        return result
    
    def _process_segments(self, segments: List[Dict]) -> List[EnrichedChunk]:
        """
        Core processing logic:
        1. Group related segments (lists, procedures)
        2. Create enriched chunks
        """
        chunks = []
        buffer = []
        current_heading_path = ""
        
        for seg in segments:
            seg_type = seg.get('type', 'Paragraph')
            heading_path = seg.get('heading_path', '')
            
            # Rule 1: Headers start new chunks
            if seg_type == 'Header':
                # Flush buffer
                if buffer:
                    chunks.append(self._create_chunk(buffer, current_heading_path))
                    buffer = []
                current_heading_path = heading_path
                # Headers themselves become chunks
                chunks.append(self._create_chunk([seg], heading_path, chunk_type="header"))
                continue
            
            # Rule 2: ListItems should be grouped together
            if seg_type == 'ListItem':
                # If buffer has non-list items, flush first
                if buffer and buffer[-1].get('type') != 'ListItem':
                    chunks.append(self._create_chunk(buffer, current_heading_path))
                    buffer = []
                buffer.append(seg)
                continue
            
            # Rule 3: If we have ListItems in buffer and current is not ListItem, flush
            if buffer and buffer[-1].get('type') == 'ListItem' and seg_type != 'ListItem':
                chunks.append(self._create_chunk(buffer, current_heading_path, chunk_type="list"))
                buffer = []
            
            # Rule 4: Tables and Pictures are standalone chunks
            if seg_type in ['Table', 'Picture', 'Formula']:
                if buffer:
                    chunks.append(self._create_chunk(buffer, current_heading_path))
                    buffer = []
                chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                continue
            
            # Rule 4.5: Learning Objectives are standalone chunks
            if seg_type == 'LearningObjective':
                if buffer:
                    chunks.append(self._create_chunk(buffer, current_heading_path))
                    buffer = []
                chunks.append(self._create_chunk([seg], heading_path, chunk_type="learning_objective"))
                continue
            
            # Rule 5: Check for theorem/proof block starters
            text = seg.get('text', '')
            if re.match(r'^(?:Theorem|Lemma|Proposition|Corollary|Proof)\s*\d*', text, re.IGNORECASE):
                if buffer:
                    chunks.append(self._create_chunk(buffer, current_heading_path))
                    buffer = []
                buffer.append(seg)
                continue
            
            # Default: Add to buffer
            buffer.append(seg)
            
            # Rule 6: Flush if buffer exceeds threshold (prevents very long chunks)
            if len(buffer) >= self.config.MAX_BUFFER_SEGMENTS:
                chunks.append(self._create_chunk(buffer, current_heading_path))
                buffer = []
        
        # Flush remaining buffer
        if buffer:
            chunks.append(self._create_chunk(buffer, current_heading_path))
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """
        Post-processing:
        1. Merge short adjacent chunks under same heading
        2. Add context overlap from previous chunk
        """
        if not chunks:
            return chunks
        
        processed = []
        
        for i, chunk in enumerate(chunks):
            # Add context overlap (if enabled and not first chunk)
            if self.config.ENABLE_OVERLAP and i > 0 and chunk.chunk_type not in ['header', 'picture', 'table']:
                prev_chunk = chunks[i - 1]
                if prev_chunk.content and prev_chunk.chunk_type not in ['header', 'picture', 'table']:
                    # Get last N sentences from previous chunk
                    if self.pos_analyzer:
                        overlap_text = self.pos_analyzer.get_last_n_sentences(
                            prev_chunk.content, 
                            self.config.OVERLAP_SENTENCES
                        )
                    else:
                        # Fallback: take last 100 chars
                        overlap_text = prev_chunk.content[-100:] if len(prev_chunk.content) > 100 else ""
                    
                    # Only add if meaningful
                    if len(overlap_text.split()) > 5:
                        chunk.context_prefix = f"[Previous context: {overlap_text}]"
            
            processed.append(chunk)
        
        # Merge very short chunks (optional second pass)
        merged = self._merge_short_chunks(processed)
        
        return merged
    
    def _merge_short_chunks(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """Merge consecutive short paragraphs under the same heading."""
        if len(chunks) < 2:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Skip non-mergeable types
            if current.chunk_type in ['header', 'picture', 'table', 'list', 'formula']:
                merged.append(current)
                i += 1
                continue
            
            # Check if current is short and can be merged with next
            if (current.word_count < self.config.SHORT_PARAGRAPH_WORDS and 
                i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                
                # Merge if same heading and next is also short enough
                if (next_chunk.heading_path == current.heading_path and
                    next_chunk.chunk_type in ['explanation', 'example'] and
                    current.word_count + next_chunk.word_count < self.config.MAX_CHUNK_WORDS):
                    
                    # Create merged chunk
                    merged_content = current.content + " " + next_chunk.content
                    merged_tags = list(set(current.tags + next_chunk.tags))
                    merged_sources = current.source_segments + next_chunk.source_segments
                    
                    merged_chunk = EnrichedChunk(
                        chunk_id=current.chunk_id,
                        heading_path=current.heading_path,
                        chunk_type=self._infer_chunk_type_from_tags(merged_tags),
                        content=merged_content,
                        context_prefix=current.context_prefix,
                        sentences=current.sentences + next_chunk.sentences,
                        tags=merged_tags,
                        source_segments=merged_sources,
                        page_range=[
                            min(current.page_range[0] if current.page_range else 0, 
                                next_chunk.page_range[0] if next_chunk.page_range else 0),
                            max(current.page_range[1] if len(current.page_range) > 1 else 0,
                                next_chunk.page_range[1] if len(next_chunk.page_range) > 1 else 0)
                        ],
                        depth=current.depth,
                        word_count=len(merged_content.split())
                    )
                    merged.append(merged_chunk)
                    i += 2  # Skip both chunks
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _create_chunk(self, segments: List[Dict], heading_path: str, 
                      chunk_type: Optional[str] = None) -> EnrichedChunk:
        """Create an enriched chunk from a list of segments."""
        self.chunk_counter += 1
        chunk_id = f"chunk_{self.chunk_counter:04d}"
        
        # Combine text
        full_text = " ".join([s.get('text', '') for s in segments]).strip()
        
        # Word count
        word_count = len(full_text.split())
        
        # Get source segment IDs
        source_ids = [s.get('segment_id', '') for s in segments if s.get('segment_id')]
        
        # Get page range
        pages = [s.get('page', 0) for s in segments if s.get('page')]
        page_range = [min(pages), max(pages)] if pages else []
        
        # Get depth
        depth = segments[0].get('depth', 0) if segments else 0
        
        # Detect tags
        tags = self.tag_detector.detect_tags(full_text)
        
        # Check for imperative start
        if self.tag_detector.detect_imperative(full_text):
            if "procedure" not in tags:
                tags.append("procedure")
        
        # Analyze sentences with POS
        sentences = []
        if self.pos_analyzer and full_text:
            sentences = self.pos_analyzer.analyze_sentences(full_text)
        
        # Determine chunk type if not provided
        if chunk_type is None:
            chunk_type = self._infer_chunk_type(segments, tags, sentences)
        
        return EnrichedChunk(
            chunk_id=chunk_id,
            heading_path=heading_path,
            chunk_type=chunk_type,
            content=full_text,
            context_prefix="",  # Will be filled in post-processing
            sentences=sentences,
            tags=tags,
            source_segments=source_ids,
            page_range=page_range,
            depth=depth,
            word_count=word_count
        )
    
    def _infer_chunk_type(self, segments: List[Dict], tags: List[str], 
                          sentences: List[Dict]) -> str:
        """Infer chunk type from segments, tags, and sentence analysis."""
        seg_types = [s.get('type', '') for s in segments]
        
        # Check segment types first
        if all(t == 'ListItem' for t in seg_types):
            return "list"
        
        # Check for theorem/proof (highest priority for academic content)
        if "theorem" in tags:
            return "theorem"
        if "proof" in tags:
            return "proof"
        
        # Check for imperative sentences (indicates procedure)
        if sentences:
            imperative_count = sum(1 for s in sentences if s.get('is_imperative', False))
            if imperative_count > 0 and imperative_count >= len(sentences) / 2:
                return "procedure"
        
        # Check other tags
        if "definition" in tags:
            return "definition"
        if "procedure" in tags:
            return "procedure"
        if "example" in tags:
            return "example"
        if "exercise" in tags:
            return "exercise"
        if "formula" in tags:
            return "formula"
        if "summary" in tags:
            return "summary"
        
        return "explanation"
    
    def _infer_chunk_type_from_tags(self, tags: List[str]) -> str:
        """Infer chunk type from tags only (for merged chunks)."""
        priority = ["theorem", "proof", "definition", "procedure", "example", 
                    "exercise", "formula", "summary"]
        for tag in priority:
            if tag in tags:
                return tag
        return "explanation"
    
    def _calculate_stats(self, chunks: List[EnrichedChunk]) -> Dict[str, Any]:
        """Calculate processing statistics."""
        type_counts = {}
        tag_counts = {}
        word_counts = []
        overlap_count = 0
        imperative_count = 0
        
        for chunk in chunks:
            # Count types
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
            # Count tags
            for tag in chunk.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            # Word counts
            word_counts.append(chunk.word_count)
            # Overlap
            if chunk.context_prefix:
                overlap_count += 1
            # Imperative sentences
            for sent in chunk.sentences:
                if sent.get('is_imperative', False):
                    imperative_count += 1
        
        return {
            "chunk_types": type_counts,
            "tag_distribution": tag_counts,
            "avg_sentences_per_chunk": sum(len(c.sentences) for c in chunks) / len(chunks) if chunks else 0,
            "avg_words_per_chunk": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "chunks_with_overlap": overlap_count,
            "imperative_sentences_detected": imperative_count
        }
    
    def _save_json(self, data: Dict, path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved enriched chunks to: {path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Default paths
    input_path = "testFloder/outputs/docling_json/An Introduction to Derivatives and Risk Management.json"
    output_path = "testFloder/outputs/chunks/An Introduction to Derivatives and Risk Management.json"
    
    # Allow CLI override
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Check if input exists
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Process with custom config
    config = ChunkingConfig()
    config.ENABLE_OVERLAP = True
    config.OVERLAP_SENTENCES = 2
    
    segmenter = LogicSegmenter(use_pos=True, config=config)
    result = segmenter.process_file(input_path, output_path)
    
    # Print summary
    stats = result['metadata']['processing_stats']
    print(f"\n{'='*70}")
    print(f"LogicSegmenter v2.0 - Processing Complete!")
    print(f"{'='*70}")
    print(f"Total Chunks: {result['metadata']['total_chunks']}")
    print(f"Total Segments: {result['metadata']['total_segments']}")
    print(f"Features: {', '.join(result['metadata']['features'])}")
    print(f"\n--- Word Count Stats ---")
    print(f"  Avg words/chunk: {stats['avg_words_per_chunk']:.1f}")
    print(f"  Min: {stats['min_words']}, Max: {stats['max_words']}")
    print(f"\n--- New Features ---")
    print(f"  Chunks with context overlap: {stats['chunks_with_overlap']}")
    print(f"  Imperative sentences detected: {stats['imperative_sentences_detected']}")
    print(f"\n--- Chunk Types ---")
    for t, count in sorted(stats['chunk_types'].items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    print(f"\n--- Tag Distribution (Top 12) ---")
    for tag, count in sorted(stats['tag_distribution'].items(), key=lambda x: -x[1])[:12]:
        print(f"  {tag}: {count}")
