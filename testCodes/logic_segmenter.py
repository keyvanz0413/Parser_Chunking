"""
Logic Segmenter Module v2.3 (Three-Phase Pipeline)

Processes flat_segments from parser_docling.py and applies:
1. Reading Order Correction (NEW v2.3) - Column-aware segment reordering for multi-column layouts
2. Heading Path Reconstruction (NEW v2.3) - Stack-based hierarchy rebuilding with backfilling
3. POS Tagging (via spaCy) - Sentence role identification with Imperative detection
4. Rule-Based Grouping - List aggregation, definition detection, etc.
5. Tag Enrichment - 20+ semantic tags from taxonomy (enhanced with Theorem/Lemma/Proof)
6. Context Overlap - Each chunk carries context from previous chunk for RAG continuity
7. Token-based Length Control - Prevents extreme chunk sizes
8. Cross-Page Continuation Detection - Hyphenation, open clause, fragment detection

Three-Phase Pipeline Architecture:
- Phase 1: Column-based reading order correction (left-col then right-col)
- Phase 2: Heading stack reconstruction (ancestor stack algorithm)
- Phase 3: Same-page backfilling for late-discovered L1 headers

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
    
    # Cross-page continuation settings
    ENABLE_CONTINUATION_DETECTION = True  # Enable cross-page continuation
    CONTINUATION_STYLE_THRESHOLD = 0.6    # Minimum style similarity score for continuation
    CONTINUATION_COLUMN_STRICT = True     # Require same column for continuation
    
    # Three-Phase Pipeline settings (NEW v2.3)
    ENABLE_READING_ORDER_CORRECTION = True   # Phase 1: Column-based reordering
    ENABLE_HEADING_RECONSTRUCTION = True     # Phase 2: Stack-based hierarchy rebuild
    ENABLE_BACKFILL_CORRECTION = True        # Phase 3: Same-page L1 backfilling
    
    # Column detection thresholds
    COLUMN_DETECTION_THRESHOLD = 0.45        # x < page_width * threshold => left column
    PAGE_WIDTH_DEFAULT = 612.0               # Default PDF page width (8.5" × 72dpi)



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
    Core Schema v2.1 - The standardized output format with overlap and continuation support.
    
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
    
    NEW (v2.1) - Cross-page continuation:
    - is_cross_page: True if chunk spans multiple pages
    - continuation_type: 'none', 'full' (confident), 'partial' (needs review)
    - needs_review: True if continuation detection was uncertain
    - merge_evidence: Dict with reasons for merge decision (explainability)
    """
    chunk_id: str
    heading_path: str
    chunk_type: str
    content: str
    context_prefix: str = ""  # Overlap from previous chunk
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source_segments: List[str] = field(default_factory=list)
    page_range: List[int] = field(default_factory=list)
    depth: int = 0
    word_count: int = 0
    # NEW v2.1: Cross-page continuation fields
    is_cross_page: bool = False
    continuation_type: str = "none"  # none, full, partial
    needs_review: bool = False
    merge_evidence: Dict[str, Any] = field(default_factory=dict)  # Explainability
    

# =============================================================================
# Three-Phase Pipeline Processor (NEW v2.3)
# =============================================================================

class ReadingOrderCorrector:
    """
    Three-Phase Pipeline for Reading Order and Heading Path Correction.
    
    Addresses the common PDF parsing issue where:
    - Multi-column layouts cause incorrect reading order
    - Headers appearing late in scan order get wrong parent assignments
    - Same-page segments inherit stale heading paths
    
    Pipeline:
    - Phase 1: Column-based segment reordering (left-col then right-col per page)
    - Phase 2: Heading stack reconstruction (ancestor stack algorithm)
    - Phase 3: Same-page backfilling for late-discovered L1 headers
    """
    
    PATH_SEPARATOR = " > "
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {
            "pages_reordered": 0,
            "segments_moved": 0,
            "paths_reconstructed": 0,
            "backfill_corrections": 0
        }
    
    def process(self, segments: List[Dict]) -> List[Dict]:
        """
        Apply three-phase correction pipeline to segments.
        
        Args:
            segments: List of segment dictionaries from parser_docling
            
        Returns:
            Corrected segments with proper reading order and heading paths
        """
        result = segments
        
        # Phase 1: Reading Order Correction
        if self.config.ENABLE_READING_ORDER_CORRECTION:
            result = self._phase1_reorder_by_columns(result)
            logger.info(f"Phase 1: Reordered {self._stats['segments_moved']} segments across {self._stats['pages_reordered']} pages")
        
        # Phase 2: Heading Stack Reconstruction
        if self.config.ENABLE_HEADING_RECONSTRUCTION:
            result = self._phase2_rebuild_heading_paths(result)
            logger.info(f"Phase 2: Reconstructed {self._stats['paths_reconstructed']} heading paths")
        
        # Phase 3: Same-page Backfilling
        if self.config.ENABLE_BACKFILL_CORRECTION:
            result = self._phase3_backfill_same_page(result)
            logger.info(f"Phase 3: Applied {self._stats['backfill_corrections']} backfill corrections")
        
        return result
    
    def get_stats(self) -> Dict[str, int]:
        """Return processing statistics."""
        return self._stats.copy()
    
    # =========================================================================
    # Phase 1: Column-Based Reading Order Correction
    # =========================================================================
    
    def _phase1_reorder_by_columns(self, segments: List[Dict]) -> List[Dict]:
        """
        Reorder segments within each page by column (left first, then right).
        
        This fixes the common issue where a two-column page gets scanned
        left-to-right across both columns, causing interleaved text.
        
        Algorithm:
        1. Group segments by page
        2. For each page, determine if it's multi-column (based on x-positions)
        3. Split into left/right columns based on x-coordinate
        4. Sort each column by y-coordinate (top to bottom)
        5. Merge columns: process by y-position, alternating between left and right as needed
        """
        from collections import defaultdict
        
        # Group by page
        pages = defaultdict(list)
        for seg in segments:
            pages[seg.get('page', 1)].append(seg)
        
        result = []
        threshold = self.config.COLUMN_DETECTION_THRESHOLD
        page_width = self.config.PAGE_WIDTH_DEFAULT
        
        for page_num in sorted(pages.keys()):
            page_segs = pages[page_num]
            
            # Detect page width from bbox if available
            max_x = max((s.get('bbox', [0, 0, 0, 0])[2] for s in page_segs if s.get('bbox')), default=page_width)
            if max_x > 100:  # Reasonable page width
                page_width = max_x * 1.1  # Add margin
            
            mid_x = page_width * threshold
            
            # Separate into columns
            left_col = []
            right_col = []
            spanning = []
            
            for seg in page_segs:
                bbox = seg.get('bbox')
                if not bbox or len(bbox) < 4:
                    spanning.append(seg)
                    continue
                
                x_left, y_top, x_right, y_bottom = bbox[0], bbox[1], bbox[2], bbox[3]
                seg_width = x_right - x_left
                
                # Check if segment spans both columns (wide element)
                if x_left < mid_x and x_right > page_width * 0.55:
                    spanning.append(seg)
                elif x_left < mid_x:
                    left_col.append(seg)
                else:
                    right_col.append(seg)
            
            # Sort each group by y-coordinate (PDF y: higher = top of page)
            # Sort descending so top-of-page comes first
            left_col.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            right_col.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            spanning.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            
            # NEW v2.3: Improved merge strategy for textbook layouts
            # Instead of "spanning + left + right", we merge by y-position
            # This ensures continuation paragraphs stay in correct reading order
            
            # Combine all segments and sort by y (top to bottom)
            all_segs = []
            for seg in left_col + right_col + spanning:
                y_top = seg.get('bbox', [0, 0, 0, 0])[1]
                col_idx = seg.get('column_index', -1)
                # For same y-level, left column comes before right
                sort_key = (-y_top, 0 if col_idx <= 0 else 1)
                all_segs.append((sort_key, seg))
            
            all_segs.sort(key=lambda x: x[0])
            reordered = [seg for _, seg in all_segs]
            
            # Check if reordering actually changed anything
            original_ids = [s.get('segment_id') for s in page_segs]
            new_ids = [s.get('segment_id') for s in reordered]
            
            if original_ids != new_ids:
                self._stats['pages_reordered'] += 1
                self._stats['segments_moved'] += sum(1 for i, seg_id in enumerate(new_ids) if i < len(original_ids) and original_ids[i] != seg_id)
            
            result.extend(reordered)
        
        return result
    
    # =========================================================================
    # Phase 2: Heading Stack Reconstruction (Ancestor Stack Algorithm)
    # =========================================================================
    
    def _phase2_rebuild_heading_paths(self, segments: List[Dict]) -> List[Dict]:
        """
        Rebuild heading_path for all segments using the ancestor stack algorithm.
        
        Algorithm:
        1. Maintain a stack of (level, text) tuples representing current heading hierarchy
        2. For each Header segment:
           - Skip noise headers (concluded, continued, etc.) - they don't affect hierarchy
           - Pop all entries with level >= current level
           - Push current header onto stack
        3. For each non-Header segment:
           - Inherit the current stack's path
        
        This ensures consistent hierarchy regardless of original parsing order.
        """
        # Import noise patterns from ContinuationDetector
        noise_patterns = [
            r'^\s*\(?\s*concluded\s*\)?\s*$',
            r'^\s*\(?\s*continued\s*\)?\s*$',
            r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in noise_patterns]
        
        def is_noise_header(seg):
            if seg.get('type') != 'Header':
                return False
            text = seg.get('text', '').strip()
            return any(p.match(text) for p in compiled_patterns)
        
        stack = []  # [(level, text), ...]
        
        for seg in segments:
            if seg.get('type') == 'Header':
                # Skip noise headers - they don't affect hierarchy
                if is_noise_header(seg):
                    seg['is_noise_header'] = True
                    # Noise header inherits current path, doesn't change it
                    seg['heading_path'] = self.PATH_SEPARATOR.join([t for _, t in stack])
                    if seg.get('heading_path'):
                        seg['full_context_text'] = f"[Path: {seg['heading_path']}] {seg.get('text', '')}"
                    continue
                
                level = seg.get('level', 1)
                text = seg.get('text', '').strip()
                
                # Pop all headers at same or deeper level
                while stack and stack[-1][0] >= level:
                    stack.pop()
                
                # Push current header
                if text:
                    stack.append((level, text))
                
                # Update segment's heading_path
                old_path = seg.get('heading_path', '')
                new_path = self.PATH_SEPARATOR.join([t for _, t in stack])
                
                if old_path != new_path:
                    self._stats['paths_reconstructed'] += 1
                    seg['heading_path'] = new_path
                    seg['heading_path_original'] = old_path  # Keep original for debugging
            else:
                # Non-header: inherit current stack's path
                old_path = seg.get('heading_path', '')
                new_path = self.PATH_SEPARATOR.join([t for _, t in stack])
                
                if old_path != new_path:
                    self._stats['paths_reconstructed'] += 1
                    seg['heading_path'] = new_path
                    seg['heading_path_original'] = old_path
            
            # Update full_context_text
            if seg.get('heading_path'):
                seg['full_context_text'] = f"[Path: {seg['heading_path']}] {seg.get('text', '')}"
        
        return segments
    
    # =========================================================================
    # Phase 3: Same-Page Backfilling for Late-Discovered Headers
    # =========================================================================
    
    def _phase3_backfill_same_page(self, segments: List[Dict]) -> List[Dict]:
        """
        Correct heading paths for segments that appear before a major header on the same page.
        
        Scenario:
        - Page 30 has "The Investment Environment" (title) on the left at y=656
        - Page 30 has "CHAPTER 1" (L1 header) on the right at y=682
        - Due to column sorting, left content is processed first
        - The left content gets the wrong (stale) heading path
        
        Solution:
        - For each page, find the highest-level (lowest number) Header
        - If that header is not the first element on the page, backfill earlier segments
        """
        from collections import defaultdict
        
        # Group by page
        pages = defaultdict(list)
        for i, seg in enumerate(segments):
            pages[seg.get('page', 1)].append((i, seg))
        
        for page_num, page_items in pages.items():
            if len(page_items) < 2:
                continue
            
            # Find all headers on this page
            headers = [(i, seg) for i, seg in page_items if seg.get('type') == 'Header']
            if not headers:
                continue
            
            # Find the highest-priority header (lowest level number)
            top_header_idx, top_header = min(headers, key=lambda x: x[1].get('level', 99))
            top_level = top_header.get('level', 99)
            
            # Only backfill for L1 headers that aren't at the start
            if top_level > 1:
                continue
            
            # Find position of this header in page_items
            page_position = next((pos for pos, (i, s) in enumerate(page_items) if i == top_header_idx), None)
            if page_position is None or page_position == 0:
                continue
            
            # Backfill: update all segments before this header on the same page
            new_path_prefix = top_header.get('text', '').strip()
            if not new_path_prefix:
                continue
            
            for pos in range(page_position):
                global_idx, seg = page_items[pos]
                old_path = seg.get('heading_path', '')
                
                # Construct new path
                if old_path:
                    # Replace old root with new root
                    path_parts = old_path.split(self.PATH_SEPARATOR)
                    if path_parts[0] != new_path_prefix:
                        new_path = self.PATH_SEPARATOR.join([new_path_prefix] + path_parts[1:]) if len(path_parts) > 1 else new_path_prefix
                        seg['heading_path'] = new_path
                        seg['heading_path_backfilled'] = True
                        seg['heading_path_original'] = old_path
                        self._stats['backfill_corrections'] += 1
                else:
                    seg['heading_path'] = new_path_prefix
                    seg['heading_path_backfilled'] = True
                    self._stats['backfill_corrections'] += 1
                
                # Update full_context_text
                seg['full_context_text'] = f"[Path: {seg.get('heading_path', '')}] {seg.get('text', '')}"
        
        return segments


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
# Cross-Page Continuation Detector (Enhanced v2.2)
# =============================================================================

class ContinuationDetector:
    """
    Enhanced cross-page paragraph continuation detector v2.3.
    
    Implements the task requirements for accurate cross-page detection:
    - Incomplete sentences (missing terminal punctuation)
    - Hyphenation (word breaks across lines)
    - Abrupt clause endings (sentences ending with prepositions/conjunctions)
    - bbox position validation
    - Skip noise headers like "(concluded)", "(continued)" (NEW v2.3)
    
    Strategy:
    1. Check if previous segment is at page bottom and current is at page top
    2. Verify style consistency (column, text patterns)
    3. Calculate detailed continuation score with evidence
    4. Return continuation type with merge evidence for explainability
    5. Skip intervening noise headers when detecting cross-page continuations
    
    Returns:
    - 'full': High confidence continuation, auto-merge (score >= 0.7)
    - 'partial': Uncertain, needs human review (score 0.4-0.7)
    - 'none': Not a continuation (score < 0.4)
    """
    
    # Types that should NOT be merged across pages
    BREAK_TYPES = {'Header', 'Table', 'Picture', 'Formula', 'LearningObjective'}
    
    # Noise header patterns - these are page decorations, not real headers
    # They should be skipped when detecting paragraph continuations
    NOISE_HEADER_PATTERNS = [
        r'^\s*\(?\s*concluded\s*\)?\s*$',         # (concluded), concluded
        r'^\s*\(?\s*continued\s*\)?\s*$',         # (continued), continued
        r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',         # (cont'd), (contd)
        r'^\s*\(?\s*continuation\s*\)?\s*$',      # (continuation)
        r'^\s*\(?\s*continued on next page\s*\)?\s*$',
        r'^\s*\(?\s*continued from previous page\s*\)?\s*$',
        r'^\s*\(?\s*see next page\s*\)?\s*$',
        r'^\s*\(?\s*to be continued\s*\)?\s*$',
    ]
    
    # Score thresholds
    FULL_THRESHOLD = 0.7
    PARTIAL_THRESHOLD = 0.4
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._last_evidence = {}  # Store last detection's evidence
        # Compile noise patterns for efficiency
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_HEADER_PATTERNS]
    
    def _is_noise_header(self, seg: Dict[str, Any]) -> bool:
        """
        Check if a segment is a noise header (page decoration) that should be skipped.
        
        These are typically: (concluded), (continued), etc.
        """
        if seg.get('type') != 'Header':
            return False
        
        text = seg.get('text', '').strip()
        if not text:
            return False
        
        for pattern in self._noise_patterns:
            if pattern.match(text):
                return True
        
        return False
    
    def detect_continuation(self, prev_seg: Dict[str, Any], curr_seg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Detect if curr_seg is a continuation of prev_seg.
        
        Returns:
            Tuple of (continuation_type, evidence_dict)
            - continuation_type: 'full', 'partial', or 'none'
            - evidence_dict: Detailed reasons for the decision
        """
        evidence = {
            "prev_segment_id": prev_seg.get('segment_id', ''),
            "curr_segment_id": curr_seg.get('segment_id', ''),
            "rules_triggered": [],
            "rules_failed": [],
            "score_breakdown": {},
            "final_score": 0.0,
            "decision": "none",
        }
        
        if not self.config.ENABLE_CONTINUATION_DETECTION:
            evidence["rules_failed"].append("continuation_detection_disabled")
            self._last_evidence = evidence
            return 'none', evidence
        
        prev_page = prev_seg.get('page', 0)
        curr_page = curr_seg.get('page', 0)
        prev_type = prev_seg.get('type', 'Paragraph')
        curr_type = curr_seg.get('type', 'Paragraph')
        
        # =====================================================================
        # Rule 1: Break types never continue
        # =====================================================================
        if curr_type in self.BREAK_TYPES or prev_type in self.BREAK_TYPES:
            evidence["rules_failed"].append(f"break_type: prev={prev_type}, curr={curr_type}")
            self._last_evidence = evidence
            return 'none', evidence
        
        # =====================================================================
        # Rule 2: Must be on different pages (cross-page only)
        # =====================================================================
        if curr_page == prev_page:
            evidence["rules_failed"].append(f"same_page: {curr_page}")
            self._last_evidence = evidence
            return 'none', evidence
        
        # =====================================================================
        # Rule 3: Must be consecutive pages
        # =====================================================================
        if curr_page != prev_page + 1:
            evidence["rules_failed"].append(f"non_consecutive: prev={prev_page}, curr={curr_page}")
            self._last_evidence = evidence
            return 'none', evidence
        
        evidence["rules_triggered"].append("consecutive_pages")
        
        # =====================================================================
        # Rule 4: Check page position (bottom -> top)
        # =====================================================================
        prev_at_bottom = prev_seg.get('at_page_bottom', False)
        curr_at_top = curr_seg.get('at_page_top', False)
        
        # If position info is missing, estimate from bbox
        if 'at_page_bottom' not in prev_seg:
            prev_at_bottom = self._estimate_at_bottom(prev_seg)
        if 'at_page_top' not in curr_seg:
            curr_at_top = self._estimate_at_top(curr_seg)
        
        if not prev_at_bottom:
            evidence["rules_failed"].append("prev_not_at_bottom")
        else:
            evidence["rules_triggered"].append("prev_at_page_bottom")
            
        if not curr_at_top:
            evidence["rules_failed"].append("curr_not_at_top")
        else:
            evidence["rules_triggered"].append("curr_at_page_top")
        
        # Position check is a soft requirement - low score if fails
        position_valid = prev_at_bottom and curr_at_top
        
        # =====================================================================
        # Rule 5: Check column consistency
        # =====================================================================
        prev_col = prev_seg.get('column_index', 0)
        curr_col = curr_seg.get('column_index', 0)
        
        column_consistent = True
        if self.config.CONTINUATION_COLUMN_STRICT:
            # Spanning items (-1) always consistent
            if prev_col >= 0 and curr_col >= 0 and prev_col != curr_col:
                column_consistent = False
                evidence["rules_failed"].append(f"column_mismatch: prev={prev_col}, curr={curr_col}")
            else:
                evidence["rules_triggered"].append("column_consistent")
        
        # =====================================================================
        # Calculate comprehensive style score with breakdown
        # =====================================================================
        score, score_breakdown = self._calculate_enhanced_style_score(
            prev_seg, curr_seg, position_valid, column_consistent
        )
        
        evidence["score_breakdown"] = score_breakdown
        evidence["final_score"] = round(score, 3)
        
        # =====================================================================
        # Make decision based on score
        # =====================================================================
        if score >= self.FULL_THRESHOLD:
            decision = 'full'
            evidence["decision"] = "full (high confidence merge)"
        elif score >= self.PARTIAL_THRESHOLD:
            decision = 'partial'
            evidence["decision"] = "partial (needs review)"
        else:
            decision = 'none'
            evidence["decision"] = "none (score too low)"
        
        self._last_evidence = evidence
        
        if decision != 'none':
            logger.debug(f"Cross-page continuation: {prev_seg.get('segment_id')} -> "
                        f"{curr_seg.get('segment_id')} = {decision} (score: {score:.2f})")
        
        return decision, evidence
    
    def _calculate_enhanced_style_score(self, prev_seg: Dict[str, Any], curr_seg: Dict[str, Any],
                                        position_valid: bool, column_consistent: bool) -> Tuple[float, Dict]:
        """
        Calculate comprehensive continuation score with detailed breakdown.
        
        Enhanced scoring using task requirements:
        - Hyphenation detection (high weight)
        - Open clause endings (high weight)
        - Sentence completeness
        - Start pattern analysis
        """
        breakdown = {}
        total_score = 0.0
        
        prev_hints = prev_seg.get('style_hints', {})
        curr_hints = curr_seg.get('style_hints', {})
        
        # =====================================================================
        # Factor 1: Position validity (weight: 0.15)
        # =====================================================================
        weight = 0.15
        if position_valid:
            total_score += weight
            breakdown["position_valid"] = {"score": weight, "max": weight, "detail": "bottom->top verified"}
        else:
            breakdown["position_valid"] = {"score": 0, "max": weight, "detail": "position check failed"}
        
        # =====================================================================
        # Factor 2: Column consistency (weight: 0.10)
        # =====================================================================
        weight = 0.10
        if column_consistent:
            total_score += weight
            breakdown["column_consistent"] = {"score": weight, "max": weight, "detail": "same column"}
        else:
            breakdown["column_consistent"] = {"score": 0, "max": weight, "detail": "column mismatch"}
        
        # =====================================================================
        # Factor 3: Hyphenation - STRONG INDICATOR (weight: 0.20)
        # =====================================================================
        weight = 0.20
        if prev_hints.get('ends_with_hyphen', False):
            total_score += weight
            breakdown["hyphenation"] = {"score": weight, "max": weight, "detail": "word break at line end"}
        else:
            breakdown["hyphenation"] = {"score": 0, "max": weight, "detail": "no hyphenation"}
        
        # =====================================================================
        # Factor 4: Open clause ending - STRONG INDICATOR (weight: 0.15)
        # =====================================================================
        weight = 0.15
        if prev_hints.get('ends_with_open_clause', False):
            total_score += weight
            # Extract last word for evidence
            prev_text = prev_seg.get('text', '')
            last_word = prev_text.split()[-1] if prev_text.split() else ''
            breakdown["open_clause"] = {"score": weight, "max": weight, 
                                        "detail": f"ends with '{last_word}'"}
        else:
            breakdown["open_clause"] = {"score": 0, "max": weight, "detail": "complete clause"}
        
        # =====================================================================
        # Factor 5: Fragment pattern (weight: 0.10)
        # =====================================================================
        weight = 0.10
        if prev_hints.get('has_fragment_pattern', False):
            total_score += weight
            breakdown["fragment_pattern"] = {"score": weight, "max": weight, 
                                             "detail": "comma/ellipsis/equation ending"}
        else:
            breakdown["fragment_pattern"] = {"score": 0, "max": weight, "detail": "no fragment pattern"}
        
        # =====================================================================
        # Factor 6: Continuation start pattern (weight: 0.15)
        # =====================================================================
        weight = 0.15
        if curr_hints.get('starts_like_continuation', False):
            total_score += weight
            curr_text = curr_seg.get('text', '')
            first_word = curr_text.split()[0] if curr_text.split() else ''
            breakdown["continuation_start"] = {"score": weight, "max": weight,
                                               "detail": f"starts with '{first_word}'"}
        elif curr_hints.get('starts_lowercase', False):
            # Partial credit for lowercase start
            partial = weight * 0.7
            total_score += partial
            breakdown["continuation_start"] = {"score": partial, "max": weight,
                                               "detail": "starts lowercase"}
        else:
            breakdown["continuation_start"] = {"score": 0, "max": weight, "detail": "normal start"}
        
        # =====================================================================
        # Factor 7: Same heading path (weight: 0.05)
        # =====================================================================
        weight = 0.05
        if prev_seg.get('heading_path') == curr_seg.get('heading_path'):
            total_score += weight
            breakdown["same_heading"] = {"score": weight, "max": weight, "detail": "same section"}
        else:
            breakdown["same_heading"] = {"score": 0, "max": weight, "detail": "different sections"}
        
        # =====================================================================
        # Factor 8: Same segment type (weight: 0.05)
        # =====================================================================
        weight = 0.05
        if prev_seg.get('type') == curr_seg.get('type'):
            total_score += weight
            breakdown["same_type"] = {"score": weight, "max": weight, 
                                      "detail": prev_seg.get('type', 'unknown')}
        else:
            breakdown["same_type"] = {"score": 0, "max": weight, 
                                      "detail": f"{prev_seg.get('type')} -> {curr_seg.get('type')}"}
        
        # =====================================================================
        # Factor 9: Incomplete sentence (weight: 0.05)
        # =====================================================================
        weight = 0.05
        if prev_hints.get('ends_incomplete', False) and not prev_hints.get('has_terminal_punctuation', True):
            total_score += weight
            breakdown["incomplete_sentence"] = {"score": weight, "max": weight, 
                                                "detail": "no terminal punctuation"}
        else:
            breakdown["incomplete_sentence"] = {"score": 0, "max": weight, 
                                                "detail": "complete sentence"}
        
        # =====================================================================
        # Factor 10: STRONG COMBINATION - Incomplete + Lowercase start (NEW v2.3)
        # This is the strongest cross-page continuation signal
        # When prev ends without punctuation AND curr starts lowercase, it's almost
        # certainly a continuation, even if position/column checks fail
        # =====================================================================
        weight = 0.20  # High weight for this strong signal
        prev_incomplete = prev_hints.get('ends_incomplete', False) and not prev_hints.get('has_terminal_punctuation', True)
        curr_lowercase = curr_hints.get('starts_lowercase', False)
        
        if prev_incomplete and curr_lowercase:
            total_score += weight
            breakdown["strong_continuation_signal"] = {
                "score": weight, "max": weight, 
                "detail": "incomplete sentence + lowercase start (strong signal)"
            }
        else:
            breakdown["strong_continuation_signal"] = {
                "score": 0, "max": weight, 
                "detail": "no strong continuation pattern"
            }
        
        # Calculate summary
        breakdown["total"] = round(total_score, 3)
        breakdown["max_possible"] = 1.0
        
        return total_score, breakdown
    
    def _estimate_at_bottom(self, seg: Dict[str, Any]) -> bool:
        """Estimate if segment is at page bottom based on bbox."""
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4:
            return False
        # Assume page height ~842 (A4), bottom 15%
        return bbox[3] > 842 * 0.85
    
    def _estimate_at_top(self, seg: Dict[str, Any]) -> bool:
        """Estimate if segment is at page top based on bbox."""
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4:
            return False
        # Assume page height ~842 (A4), top 15%
        return bbox[1] < 842 * 0.15
    
    def get_last_evidence(self) -> Dict[str, Any]:
        """Return the evidence from the last detection call."""
        return self._last_evidence
    
    def _find_previous_content_segment(self, segments: List[Dict], current_index: int) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Find the previous content segment, skipping noise headers.
        
        Returns:
            Tuple of (previous_content_segment, list_of_skipped_noise_segments)
            Returns (None, []) if no valid previous segment found.
        """
        skipped = []
        for j in range(current_index - 1, -1, -1):
            seg = segments[j]
            if self._is_noise_header(seg):
                skipped.append(seg)
                continue
            # Found a non-noise segment
            return seg, skipped
        return None, skipped
    
    def annotate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Annotate all segments with continuation markers and evidence.
        
        Enhanced v2.3: Skips noise headers (like "(concluded)") when detecting
        cross-page continuations. This allows paragraphs to be correctly merged
        even when page decoration elements appear between them.
        
        Adds to each segment:
        - 'is_continuation': 'full', 'partial', or 'none'
        - 'continuation_evidence': Dict with detailed reasons
        - 'skipped_noise_headers': List of noise headers that were skipped (if any)
        """
        if len(segments) < 2:
            for seg in segments:
                seg['is_continuation'] = 'none'
                seg['continuation_evidence'] = {}
            return segments
        
        # First segment is never a continuation
        segments[0]['is_continuation'] = 'none'
        segments[0]['continuation_evidence'] = {}
        
        for i in range(1, len(segments)):
            curr_seg = segments[i]
            
            # Skip if current segment is a noise header
            if self._is_noise_header(curr_seg):
                curr_seg['is_continuation'] = 'none'
                curr_seg['continuation_evidence'] = {"skipped": "noise_header"}
                curr_seg['is_noise_header'] = True
                continue
            
            # Find previous content segment, skipping noise headers
            prev_seg, skipped_noise = self._find_previous_content_segment(segments, i)
            
            if prev_seg is None:
                curr_seg['is_continuation'] = 'none'
                curr_seg['continuation_evidence'] = {}
                continue
            
            # Detect continuation
            continuation, evidence = self.detect_continuation(prev_seg, curr_seg)
            
            # Record skipped noise headers in evidence
            if skipped_noise:
                evidence['skipped_noise_headers'] = [s.get('segment_id') for s in skipped_noise]
                evidence['skipped_noise_texts'] = [s.get('text', '')[:50] for s in skipped_noise]
            
            curr_seg['is_continuation'] = continuation
            curr_seg['continuation_evidence'] = evidence
            
            if continuation != 'none':
                skip_info = f" (skipped {len(skipped_noise)} noise headers)" if skipped_noise else ""
                logger.info(f"Cross-page continuation: {prev_seg.get('segment_id')} -> "
                           f"{curr_seg.get('segment_id')} ({continuation}, score: {evidence.get('final_score', 0):.2f}){skip_info}")
        
        return segments



# =============================================================================
# Logic Segmenter (Main Class - Enhanced)
# =============================================================================

class LogicSegmenter:
    """
    Main segmenter that processes flat_segments from parser_docling.py.
    
    Pipeline v2.3 (Three-Phase Architecture):
    1. Load segments from JSON
    2. Apply Three-Phase Correction Pipeline (NEW v2.3):
       - Phase 1: Column-based reading order correction
       - Phase 2: Heading stack reconstruction
       - Phase 3: Same-page backfilling for L1 headers
    3. Annotate segments with continuation markers
    4. Group segments by logical rules (lists, procedures, cross-page)
    5. Analyze with POS tagging (enhanced with imperative detection)
    6. Apply context overlap for RAG continuity
    7. Enrich with tags (20+ including Theorem/Lemma/Proof)
    8. Apply length constraints (min/max word counts)
    9. Output standardized chunks with continuation metadata
    """
    
    def __init__(self, use_pos: bool = True, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.tag_detector = TagDetector()
        self.pos_analyzer = POSAnalyzer() if use_pos else None
        self.continuation_detector = ContinuationDetector(self.config)
        self.reading_order_corrector = ReadingOrderCorrector(self.config)  # NEW v2.3
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
        
        # NEW v2.3: Apply Three-Phase Correction Pipeline
        # This must run BEFORE continuation detection to ensure correct reading order
        if (self.config.ENABLE_READING_ORDER_CORRECTION or 
            self.config.ENABLE_HEADING_RECONSTRUCTION or 
            self.config.ENABLE_BACKFILL_CORRECTION):
            flat_segments = self.reading_order_corrector.process(flat_segments)
            corrector_stats = self.reading_order_corrector.get_stats()
        else:
            corrector_stats = {}
        
        # Annotate segments with continuation markers
        if self.config.ENABLE_CONTINUATION_DETECTION:
            flat_segments = self.continuation_detector.annotate_segments(flat_segments)
            continuation_count = sum(1 for s in flat_segments if s.get('is_continuation') != 'none')
            logger.info(f"Detected {continuation_count} cross-page continuations")
        
        # Process segments into chunks
        chunks = self._process_segments(flat_segments)
        
        # Post-process: merge short chunks and apply overlap
        chunks = self._post_process_chunks(chunks)
        
        # Build feature list
        features = ["context_overlap", "imperative_detection", "theorem_tagging", "length_control"]
        if self.config.ENABLE_READING_ORDER_CORRECTION:
            features.append("reading_order_correction")
        if self.config.ENABLE_HEADING_RECONSTRUCTION:
            features.append("heading_reconstruction")
        if self.config.ENABLE_BACKFILL_CORRECTION:
            features.append("backfill_correction")
        if self.config.ENABLE_CONTINUATION_DETECTION:
            features.append("cross_page_continuation")
        
        # Calculate stats and include corrector stats
        processing_stats = self._calculate_stats(chunks)
        if corrector_stats:
            processing_stats['reading_order_correction'] = corrector_stats
        
        result = {
            "metadata": {
                **metadata,
                "total_chunks": len(chunks),
                "total_segments": len(flat_segments),
                "processing_version": "2.3",
                "features": features,
                "processing_stats": processing_stats
            },
            "chunks": [asdict(c) for c in chunks]
        }
        
        if output_path:
            self._save_json(result, output_path)
        
        return result
    
    def _process_segments(self, segments: List[Dict]) -> List[EnrichedChunk]:
        """
        Core processing logic v2.2:
        1. Handle cross-page continuations with evidence tracking
        2. Group related segments (lists, procedures)
        3. Create enriched chunks with continuation metadata and evidence
        """
        chunks = []
        buffer = []
        current_heading_path = ""
        has_cross_page = False  # Track if current buffer spans pages
        continuation_type = "none"  # Track continuation confidence
        merge_evidences = []  # Collect all evidence for this buffer
        
        for i, seg in enumerate(segments):
            seg_type = seg.get('type', 'Paragraph')
            heading_path = seg.get('heading_path', '')
            is_continuation = seg.get('is_continuation', 'none')
            continuation_evidence = seg.get('continuation_evidence', {})
            
            # =================================================================
            # Rule 0 (v2.2): Cross-page continuation handling with evidence
            # =================================================================
            # If this segment is marked as a continuation, do NOT flush buffer
            # Instead, continue accumulating to preserve paragraph integrity
            if is_continuation in ['full', 'partial']:
                # Track cross-page status
                has_cross_page = True
                if is_continuation == 'partial':
                    continuation_type = 'partial'
                elif continuation_type != 'partial':
                    continuation_type = 'full'
                
                # Collect evidence for explainability
                if continuation_evidence:
                    merge_evidences.append(continuation_evidence)
                
                # Add to buffer and continue (skip other rules)
                buffer.append(seg)
                
                # Log for debugging
                score = continuation_evidence.get('final_score', 0)
                logger.debug(f"Cross-page continuation: adding {seg.get('segment_id')} to buffer "
                           f"(type: {is_continuation}, score: {score:.2f})")
                
                # Still check buffer size limit
                if len(buffer) >= self.config.MAX_BUFFER_SEGMENTS * 2:  # Allow 2x for continuations
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                continue
            
            # =================================================================
            # Rule 1: Headers start new chunks (with noise header exception)
            # =================================================================
            if seg_type == 'Header':
                # Check if this is a noise header (concluded, continued, etc.)
                # Noise headers should NOT break the continuation chain
                is_noise = seg.get('is_noise_header', False) or self.continuation_detector._is_noise_header(seg)
                
                if is_noise:
                    # Noise header: add to buffer without flushing
                    # This preserves cross-page paragraph continuity
                    buffer.append(seg)
                    logger.debug(f"Skipping noise header {seg.get('segment_id')}: {seg.get('text', '')[:30]}")
                    continue
                
                # Real header: Flush buffer with cross-page metadata
                if buffer:
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                current_heading_path = heading_path
                # Headers themselves become chunks
                chunks.append(self._create_chunk([seg], heading_path, chunk_type="header"))
                continue
            
            # =================================================================
            # Rule 2: ListItems should be grouped together
            # =================================================================
            if seg_type == 'ListItem':
                # If buffer has non-list items, flush first
                if buffer and buffer[-1].get('type') != 'ListItem':
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                buffer.append(seg)
                continue
            
            # =================================================================
            # Rule 3: If we have ListItems in buffer and current is not ListItem, flush
            # =================================================================
            if buffer and buffer[-1].get('type') == 'ListItem' and seg_type != 'ListItem':
                chunk = self._create_chunk(buffer, current_heading_path, chunk_type="list")
                chunk.is_cross_page = has_cross_page
                chunk.continuation_type = continuation_type
                chunk.needs_review = (continuation_type == 'partial')
                chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                chunks.append(chunk)
                buffer = []
                has_cross_page = False
                continuation_type = "none"
                merge_evidences = []
            
            # =================================================================
            # Rule 4: Tables and Pictures are standalone chunks
            # =================================================================
            if seg_type in ['Table', 'Picture', 'Formula']:
                if buffer:
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                continue
            
            # =================================================================
            # Rule 4.5: Learning Objectives are standalone chunks
            # =================================================================
            if seg_type == 'LearningObjective':
                if buffer:
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                chunks.append(self._create_chunk([seg], heading_path, chunk_type="learning_objective"))
                continue
            
            # =================================================================
            # Rule 5: Check for theorem/proof block starters
            # =================================================================
            text = seg.get('text', '')
            if re.match(r'^(?:Theorem|Lemma|Proposition|Corollary|Proof)\s*\d*', text, re.IGNORECASE):
                if buffer:
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                buffer.append(seg)
                continue
            
            # =================================================================
            # Default: Add to buffer
            # =================================================================
            buffer.append(seg)
            
            # =================================================================
            # Rule 6: Flush if buffer exceeds threshold (with lookahead)
            # NEW v2.3: Check if next non-noise segment is a continuation
            #           If so, delay flush to preserve paragraph integrity
            # =================================================================
            if len(buffer) >= self.config.MAX_BUFFER_SEGMENTS:
                # Lookahead: check if upcoming segments form a continuation chain
                should_delay_flush = False
                for lookahead_idx in range(i + 1, min(i + 4, len(segments))):  # Look ahead up to 3 segments
                    future_seg = segments[lookahead_idx]
                    # Skip noise headers in lookahead
                    if future_seg.get('is_noise_header', False):
                        continue
                    # If we find a continuation, delay the flush
                    if future_seg.get('is_continuation') in ['full', 'partial']:
                        should_delay_flush = True
                        logger.debug(f"Delaying flush: upcoming {future_seg.get('segment_id')} is a continuation")
                    break  # Only check the first non-noise segment
                
                if not should_delay_flush:
                    chunk = self._create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
        
        # Flush remaining buffer
        if buffer:
            chunk = self._create_chunk(buffer, current_heading_path)
            chunk.is_cross_page = has_cross_page
            chunk.continuation_type = continuation_type
            chunk.needs_review = (continuation_type == 'partial')
            chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
            chunks.append(chunk)
        
        return chunks
    
    def _compile_merge_evidence(self, evidences: List[Dict]) -> Dict[str, Any]:
        """
        Compile multiple evidence dicts into a summary for explainability.
        
        Returns a dict with:
        - merge_count: Number of cross-page merges in this chunk
        - avg_confidence: Average continuation score
        - key_indicators: Most common triggering rules
        - details: List of individual evidence entries (summarized)
        """
        if not evidences:
            return {}
        
        # Calculate summary statistics
        scores = [e.get('final_score', 0) for e in evidences]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Collect all triggered rules
        all_triggers = []
        for e in evidences:
            all_triggers.extend(e.get('rules_triggered', []))
        
        # Count rule frequency
        rule_counts = {}
        for rule in all_triggers:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Get top indicators
        top_indicators = sorted(rule_counts.items(), key=lambda x: -x[1])[:5]
        
        # Summarize individual merges
        details = []
        for e in evidences:
            detail = {
                "from": e.get('prev_segment_id', ''),
                "to": e.get('curr_segment_id', ''),
                "score": e.get('final_score', 0),
                "decision": e.get('decision', 'none'),
            }
            # Add key score factors
            breakdown = e.get('score_breakdown', {})
            if breakdown:
                high_scorers = [(k, v['score']) for k, v in breakdown.items() 
                               if isinstance(v, dict) and v.get('score', 0) > 0]
                detail["contributing_factors"] = [f"{k}: {v:.2f}" for k, v in high_scorers]
            details.append(detail)
        
        return {
            "merge_count": len(evidences),
            "avg_confidence": round(avg_score, 3),
            "key_indicators": [f"{k}: {v}x" for k, v in top_indicators],
            "details": details
        }
    
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
        """Calculate processing statistics including cross-page metrics."""
        type_counts = {}
        tag_counts = {}
        word_counts = []
        overlap_count = 0
        imperative_count = 0
        
        # NEW v2.1: Cross-page statistics
        cross_page_count = 0
        full_continuation_count = 0
        partial_continuation_count = 0
        needs_review_count = 0
        
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
            
            # NEW v2.1: Cross-page continuation stats
            if chunk.is_cross_page:
                cross_page_count += 1
            if chunk.continuation_type == 'full':
                full_continuation_count += 1
            elif chunk.continuation_type == 'partial':
                partial_continuation_count += 1
            if chunk.needs_review:
                needs_review_count += 1
        
        return {
            "chunk_types": type_counts,
            "tag_distribution": tag_counts,
            "avg_sentences_per_chunk": sum(len(c.sentences) for c in chunks) / len(chunks) if chunks else 0,
            "avg_words_per_chunk": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "chunks_with_overlap": overlap_count,
            "imperative_sentences_detected": imperative_count,
            # NEW v2.1: Cross-page continuation stats
            "cross_page_chunks": cross_page_count,
            "full_continuations": full_continuation_count,
            "partial_continuations": partial_continuation_count,
            "chunks_needing_review": needs_review_count
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
    
    # Base directory for the project
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "outputs" / "docling_json"
    output_dir = base_dir / "outputs" / "Chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for JSON files from parser_docling
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        print("Please run parser_docling.py first.")
        sys.exit(0)
    
    print(f"Found {len(json_files)} files to process.")
    
    # Initialize segmenter
    config = ChunkingConfig()
    config.ENABLE_OVERLAP = True
    config.OVERLAP_SENTENCES = 2
    segmenter = LogicSegmenter(use_pos=True, config=config)
    
    for input_path in json_files:
        output_path = output_dir / input_path.name
        
        print(f"\n{'='*70}")
        print(f"Processing: {input_path.name}")
        print(f"Output: {output_path}")
        print(f"{'='*70}")
        
        try:
            result = segmenter.process_file(str(input_path), str(output_path))
            
            # Print summary for this file
            stats = result['metadata']['processing_stats']
            print(f"✓ Done - Generated {result['metadata']['total_chunks']} chunks")
            print(f"  Avg words/chunk: {stats['avg_words_per_chunk']:.1f}")
            print(f"  Cross-page continuations: {stats.get('cross_page_continuations', {}).get('total', 0)}")
        except Exception as e:
            print(f"✗ Error processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"All files processed successfully!")
    print(f"{'='*70}")

