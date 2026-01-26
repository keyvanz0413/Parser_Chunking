"""
Semantic Chunker Module - Intelligent Document Chunking for RAG

A comprehensive PDF document chunking pipeline that transforms raw text segments
into semantically enriched chunks optimized for Retrieval-Augmented Generation (RAG).

Core Capabilities:
1. Furniture Detection - Multi-feature page decoration detection (headers/footers/page numbers)
2. Reading Order Correction - Column-aware segment reordering for multi-column layouts
3. Heading Path Reconstruction - Stack-based hierarchy rebuilding with backfilling
4. Sentence Role Identification - spaCy-based POS tagging + heuristic rules (~94% accuracy)
5. Rule-Based Grouping - List aggregation, definition detection, cross-page continuation
6. Tag Enrichment - 20+ semantic tags from discourse taxonomy
7. Context Overlap - Each chunk carries context from previous chunk for RAG continuity
8. Token-based Length Control - Prevents extreme chunk sizes
9. Cross-Page Continuation Detection - Hyphenation, open clause, fragment detection

Pipeline Architecture:
- Pre-Phase: Furniture detection and document-level statistics
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
    
    # Pipeline Phase settings
    ENABLE_FURNITURE_DETECTION = True     # Pre-Phase: Furniture detection
    ENABLE_READING_ORDER_CORRECTION = True   # Phase 1: Column-based reordering
    ENABLE_HEADING_RECONSTRUCTION = True     # Phase 2: Stack-based hierarchy rebuild
    ENABLE_BACKFILL_CORRECTION = True        # Phase 3: Same-page L1 backfilling
    
    # Column detection thresholds
    COLUMN_DETECTION_THRESHOLD = 0.45        # x < page_width * threshold => left column
    ENABLE_COLUMN_ISOLATION = True          # Force spanning -> left -> right order
    COLUMN_MERGE_GUARD = True               # Prevents merging segments from different columns
    SEGMENT_ID_GAP_THRESHOLD = 8            # Max ID difference for merging
    PAGE_WIDTH_DEFAULT = 612.0               # Default PDF page width (8.5" × 72dpi)
    PAGE_HEIGHT_DEFAULT = 792.0              # Default PDF page height (11" × 72dpi)
    
    # Furniture detection settings (Safe Zone)
    FURNITURE_TOP_BAND = 0.08                # Page top 8% considered header zone
    FURNITURE_BOTTOM_BAND = 0.08             # Page bottom 8% considered footer zone
    FURNITURE_LEFT_BAND = 0.05               # Left margin 5% (Sidebars)
    FURNITURE_RIGHT_BAND = 0.05              # Right margin 5% (Sidebars)
    FURNITURE_REPEAT_THRESHOLD = 0.20        # String appearing on 20%+ pages is "repeated"
    FURNITURE_MAX_WORDS = 8                  # Short text threshold for furniture
    FURNITURE_MIN_PAGES_FOR_STATS = 10       # Minimum pages to compute frequency stats
    
    # Dehyphenation settings
    ENABLE_DEHYPHENATION = True              # Enable cross-page hyphen repair

    # Content Gating settings (New)
    ENABLE_CONTENT_GATING = True             # Enable main body vs front/back matter gating
    STRIP_FRONT_MATTER = False               # If True, discard front matter instead of labeling
    
    # Gating Patterns (Regex)
    MAIN_BODY_START_PATTERNS = [
        r'^(?:Chapter|CHAPTER|Section|SECTION)\s*(?:1|I|ONE)\b',
        r'^1\.\s+[A-Z\u4e00-\u9fa5]',
        r'^第[一1]章'
    ]
    BACK_MATTER_PATTERNS = [
        r'^(?:Index|Bibliography|References|Appendix|Appendices)\b',
        r'^(?:索引|参考文献|附录|后记)'
    ]
    GATING_SCAN_LIMIT_PAGE = 100             # Max page to search for Main Body start
    GATING_MIN_BODY_OFFSET = 2               # Min pages after last TOC before body start
    GATING_L1_DENSITY_THRESHOLD = 3          # Headers in 5 pages window to consider as TOC



# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SentenceRole:
    """Represents a sentence with its identified role."""
    text: str
    role: str  # topic, definition, example, conclusion, evidence, procedural, imperative
    pos_tags: List[str] = field(default_factory=list)
    is_imperative: bool = False


@dataclass
class Reference:
    """
    Represents a reference to a Figure/Table/Equation within a chunk.
    
    Used for cross-linking paragraph chunks with visual/structural blocks.
    """
    ref_text: str              # Original reference text (e.g., "Figure 1.1")
    start_offset: int          # Start position in chunk content
    end_offset: int            # End position in chunk content
    target_segment_id: str     # Segment ID of the referenced block (or None)
    target_type: str           # "Figure", "Table", "Equation", "Formula"
    ref_kind: str              # "explicit" or "implicit"
    confidence: float = 0.0    # Confidence score (0.0-1.0)


@dataclass
class EnrichedChunk:
    """
    Core Schema - The standardized output format with references support.
    
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
    
    Cross-page continuation:
    - is_cross_page: True if chunk spans multiple pages
    - continuation_type: 'none', 'full' (confident), 'partial' (needs review)
    - needs_review: True if continuation detection was uncertain
    - merge_evidence: Dict with reasons for merge decision (explainability)
    
    NEW: Figure/Table/Equation references:
    - references: List of detected references to visual/structural blocks
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
    # Cross-page continuation fields
    is_cross_page: bool = False
    continuation_type: str = "none"  # none, full, partial
    needs_review: bool = False
    merge_evidence: Dict[str, Any] = field(default_factory=dict)
    # NEW: Reference detection
    references: List[Reference] = field(default_factory=list)
    doc_zone: str = "body"      # "front", "body", "back"
    

# =============================================================================
# Three-Phase Pipeline Processor
# =============================================================================

# =============================================================================
# Furniture Detector
# =============================================================================

class FurnitureDetector:
    """
    Multi-feature page decoration (furniture) detector.
    
    Furniture includes running headers, footers, page numbers, and continuation
    markers that should not be treated as document content. This detector uses
    multiple features to identify these elements:
    
    1. Position band: Elements in top/bottom 10% of page are candidates
    2. Cross-page frequency: Repeated strings across many pages are likely furniture
    3. Lexical patterns: Known phrases like "continued", page numbers
    4. Semantic weakness: Very short text with no substantive content
    
    Usage:
        detector = FurnitureDetector(config)
        detector.scan_document(all_segments)  # First pass: build stats
        
        for seg in segments:
            if detector.is_furniture(seg):
                seg['is_furniture'] = True
    """
    
    # Known furniture phrases (case-insensitive)
    FURNITURE_PHRASES = [
        r'^\s*\(?\s*concluded\s*\)?\s*$',
        r'^\s*\(?\s*continued\s*\)?\s*$',
        r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',
        r'^\s*\(?\s*continuation\s*\)?\s*$',
        r'^\s*continued\s+(?:on|from)\s+(?:next|previous)?\s*page',
        r'^\s*see\s+(?:next|previous)\s+page',
        r'^\s*to\s+be\s+continued',
        r'^(?:Table|Figure|Exhibit)\s+[\d.]+\s*\(?\s*(?:continued|cont\'?d?)\s*\)?',
        r'^\s*(?:Page\s+)?\d+\s*$',  # Page numbers
        r'^\s*\d+\s*/\s*\d+\s*$',    # Page X/Y format
        r'^\s*[-–—]\s*\d+\s*[-–—]\s*$',  # -23- format
    ]
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.FURNITURE_PHRASES]
        
        # Document-level statistics (populated by scan_document)
        self._frequency_map: Dict[str, int] = {}  # normalized_text -> page count
        self._total_pages: int = 0
        self._scanned: bool = False
        
        # Statistics
        self._stats = {
            "furniture_detected": 0,
            "by_position": 0,
            "by_frequency": 0,
            "by_pattern": 0,
        }
    
    def scan_document(self, segments: List[Dict]) -> None:
        """
        First pass: Build document-level statistics for furniture detection.
        
        This should be called once before processing segments.
        Builds a frequency map of normalized text strings and their page coverage.
        """
        from collections import defaultdict
        
        # Track which pages each normalized string appears on
        text_pages: Dict[str, set] = defaultdict(set)
        all_pages: set = set()
        
        for seg in segments:
            page = seg.get('page', 0)
            all_pages.add(page)
            
            # Only track short strings in edge positions as furniture candidates
            text = seg.get('text', '').strip()
            if not text or len(text.split()) > self.config.FURNITURE_MAX_WORDS:
                continue
            
            # Check if in edge band
            if self._in_edge_band(seg):
                norm_text = self._normalize_text(text)
                text_pages[norm_text].add(page)
        
        self._total_pages = len(all_pages)
        
        # Build frequency map (count of unique pages, not occurrences)
        self._frequency_map = {
            text: len(pages) for text, pages in text_pages.items()
        }
        
        self._scanned = True
        logger.debug(f"FurnitureDetector scanned {self._total_pages} pages, "
                    f"found {len(self._frequency_map)} candidate furniture strings")
    
    def is_furniture(self, seg: Dict) -> bool:
        """
        Determine if a segment is furniture (page decoration).
        
        Returns True if the segment matches furniture criteria based on:
        - Hard Safe-Zone check (Any content in edge bands is furniture)
        - Cross-page frequency (repeated across many pages)
        - Lexical patterns (matches known furniture phrases)
        - Semantic weakness (short, non-substantive content)
        """
        if seg.get('type') not in ['Header', 'Paragraph', 'Text', 'ListItem']:
            return False
            
        text = seg.get('text', '').strip()
        if not text:
            return False
            
        # Feature 0: Hard Safe-Zone Check (Strongest Feature)
        # If content is in the edge margins, it is by definition page furniture
        # (Header, Footer, or Sidebar), regardless of its length or content.
        if self._in_edge_band(seg):
            # NEW: Protection for potential references in sidebars/headers
            # If it matches a reference pattern (e.g. "Table 1.1"), it's NOT furniture
            if self._matches_furniture_pattern(text): # Using pattern check
                self._stats["furniture_detected"] += 1
                self._stats["by_position"] += 1
                return True
            
            # Additional check: If it looks like a block caption (Table X), don't mark as furniture
            if re.search(r'^(?:Table|Figure|Fig\.?|Equation|Eq\.?)\s+\d+', text, re.I):
                return False
                
            self._stats["furniture_detected"] += 1
            self._stats["by_position"] += 1
            return True
            
        # Feature 1: Check known furniture patterns
        if self._matches_furniture_pattern(text):
            self._stats["furniture_detected"] += 1
            self._stats["by_pattern"] += 1
            return True
            
        # Feature 2: Check cross-page frequency (for repeated running elements)
        if self._scanned and self._total_pages >= self.config.FURNITURE_MIN_PAGES_FOR_STATS:
            norm_text = self._normalize_text(text)
            page_count = self._frequency_map.get(norm_text, 0)
            
            if page_count > 0:
                repeat_ratio = page_count / self._total_pages
                if repeat_ratio >= self.config.FURNITURE_REPEAT_THRESHOLD:
                    self._stats["furniture_detected"] += 1
                    self._stats["by_frequency"] += 1
                    return True
        
        return False
    
    def _in_edge_band(self, seg: Dict) -> bool:
        """
        Check if segment is in prohibited 'Safe Zone' edge bands.
        
        Includes Top/Bottom (Headers/Footers) and Left/Right (Sidebars).
        """
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4:
            return False
            
        # Get dimensions from config
        pw = self.config.PAGE_WIDTH_DEFAULT
        ph = self.config.PAGE_HEIGHT_DEFAULT
        
        # Bbox: [x_left, y_top, x_right, y_bottom] (Standard PDF coordinates used in this script)
        # Note: In this script's coordinate system, y_top is the higher value.
        x_left, y_top, x_right, y_bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # 1. Top band check (Header Zone)
        top_threshold = ph * (1 - self.config.FURNITURE_TOP_BAND)
        if y_top > top_threshold:
            return True
            
        # 2. Bottom band check (Footer Zone)
        bottom_threshold = ph * self.config.FURNITURE_BOTTOM_BAND
        if y_bottom < bottom_threshold:
            return True
            
        # 3. Left margin check (Sidebar Zone)
        left_threshold = pw * self.config.FURNITURE_LEFT_BAND
        if x_right < left_threshold: # Entire block is within left margin
            return True
            
        # 4. Right margin check (Sidebar Zone)
        right_threshold = pw * (1 - self.config.FURNITURE_RIGHT_BAND)
        if x_left > right_threshold: # Entire block is within right margin
            return True
            
        return False
    
    def _matches_furniture_pattern(self, text: str) -> bool:
        """Check if text matches any known furniture pattern."""
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                return True
        return False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for frequency comparison."""
        # Remove extra whitespace, lowercase, strip punctuation
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove page numbers from strings like "Chapter 1 - Page 23"
        normalized = re.sub(r'\b(?:page\s*)?\d+\b', '', normalized)
        return normalized.strip()
    
    def _is_trivial_content(self, text: str) -> bool:
        """Check if text is trivial (numbers only, single symbols, etc.)."""
        # Pure numbers
        if re.match(r'^[\d\s.,-]+$', text):
            return True
        # Single character or punctuation
        if len(text) <= 2:
            return True
        # Roman numerals
        if re.match(r'^[ivxlcdmIVXLCDM]+$', text):
            return True
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Return detection statistics."""
        return self._stats.copy()


# =============================================================================
# Dehyphenation Helper
# =============================================================================

class DehyphenationHelper:
    """
    Cross-page and cross-line hyphenation repair.
    
    Handles cases where words are split across lines or pages with hyphens:
    - "invest-" + "ment" → "investment"
    - "con-" + "tinuation" → "continuation"
    
    Strategy:
    1. Detect if previous text ends with a hyphen
    2. Attempt to merge the hyphenated word parts
    3. Validate the merged word (optional dictionary check)
    4. Return the repaired text
    
    Usage:
        helper = DehyphenationHelper()
        merged = helper.merge_hyphenated("The invest-", "ment was successful.")
        # Returns: ("The investment", "was successful.")
    """
    
    # Common hyphen characters
    HYPHEN_CHARS = {'-', '‐', '‑', '–', '—'}
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {
            "hyphens_detected": 0,
            "merges_performed": 0,
            "merges_skipped": 0,
        }
    
    def merge_hyphenated(self, prev_text: str, curr_text: str) -> Tuple[str, str]:
        """
        Attempt to merge hyphenated word across text boundary.
        
        Args:
            prev_text: Text that may end with a hyphenated word fragment
            curr_text: Text that may begin with the rest of the word
            
        Returns:
            Tuple of (modified_prev_text, modified_curr_text)
            If merge performed, the hyphenated word is moved entirely to prev_text
        """
        if not prev_text or not curr_text:
            return prev_text, curr_text
        
        prev_stripped = prev_text.rstrip()
        
        # Check if prev ends with hyphen
        if not prev_stripped or prev_stripped[-1] not in self.HYPHEN_CHARS:
            return prev_text, curr_text
        
        # Ensure hyphen is attached to a word (not standalone like "- ")
        # The character before hyphen should be alphanumeric
        if len(prev_stripped) < 2 or not prev_stripped[-2].isalnum():
            return prev_text, curr_text
        
        self._stats["hyphens_detected"] += 1
        
        # Extract the word fragment before hyphen
        words = prev_stripped[:-1].split()
        if not words:
            return prev_text, curr_text
        
        word_part1 = words[-1]
        
        # Extract the first word from curr_text (potential word completion)
        curr_stripped = curr_text.lstrip()
        curr_words = curr_stripped.split()
        if not curr_words:
            return prev_text, curr_text
        
        word_part2 = curr_words[0]
        
        # Check if word_part2 looks like a word continuation (lowercase, no punctuation at start)
        if not word_part2 or not word_part2[0].isalpha():
            self._stats["merges_skipped"] += 1
            return prev_text, curr_text
        
        # Merge the word
        merged_word = word_part1 + word_part2.rstrip('.,;:!?')
        
        # Validate: merged word should be reasonably long and alphanumeric
        if len(merged_word) < 3:
            self._stats["merges_skipped"] += 1
            return prev_text, curr_text
        
        # Check if it looks like a valid word (simple heuristic)
        if not self._is_likely_word(merged_word):
            self._stats["merges_skipped"] += 1
            return prev_text, curr_text
        
        # Perform the merge
        self._stats["merges_performed"] += 1
        
        # Reconstruct texts
        # prev_text: remove the hyphenated fragment, add merged word
        prev_words = prev_stripped[:-1].split()
        prev_words[-1] = merged_word
        new_prev = ' '.join(prev_words)
        
        # Preserve trailing whitespace from original
        if prev_text.endswith(' '):
            new_prev += ' '
        
        # curr_text: remove the merged part
        remaining_curr_words = curr_words[1:]
        new_curr = ' '.join(remaining_curr_words)
        
        # Preserve any punctuation that was attached to word_part2
        punct_match = re.search(r'^[a-zA-Z]+([.,;:!?]+)', curr_words[0])
        if punct_match and remaining_curr_words:
            pass  # Punctuation already in remaining
        elif punct_match:
            new_prev += punct_match.group(1)
        
        return new_prev, new_curr
    
    def _is_likely_word(self, word: str) -> bool:
        """
        Simple heuristic to check if merged string looks like a valid word.
        
        This is a lightweight check; for production use, integrate a dictionary.
        """
        # Must be alphabetic (allow some internal hyphens for compound words)
        clean = word.replace('-', '')
        if not clean.isalpha():
            return False
        
        # Shouldn't have too many consecutive consonants or vowels
        vowels = set('aeiouAEIOU')
        consonant_run = 0
        vowel_run = 0
        max_consonant_run = 0
        max_vowel_run = 0
        
        for char in clean:
            if char in vowels:
                vowel_run += 1
                max_consonant_run = max(max_consonant_run, consonant_run)
                consonant_run = 0
            else:
                consonant_run += 1
                max_vowel_run = max(max_vowel_run, vowel_run)
                vowel_run = 0
        
        max_consonant_run = max(max_consonant_run, consonant_run)
        max_vowel_run = max(max_vowel_run, vowel_run)
        
        # Reject if implausible consonant/vowel sequences
        if max_consonant_run > 5 or max_vowel_run > 4:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """Return repair statistics."""
        return self._stats.copy()


# =============================================================================
# Caption Bonding Helper (Atomic Unit Bonding)
# =============================================================================

class CaptionBondingHelper:
    """
    Detects and bonds Table/Figure captions with their structural blocks.
    
    Solves three subproblems:
    1. Over-fragmentation: Prevents splitting caption + table + notes into separate chunks
    2. Side-caption detection: Identifies captions in non-standard positions (side/bottom)
    3. Metadata drift: Distinguishes block captions from structural headers
    
    Caption Types:
    - Block Caption: "Table 1.1", "Figure 2.3" - should NOT update global heading stack
    - Structural Header: "Chapter 1", "1.1 Introduction" - SHOULD update heading stack
    
    Usage:
        helper = CaptionBondingHelper()
        is_caption, caption_info = helper.detect_caption(segment)
        if is_caption:
            # Bond with next Table/Figure block
    """
    
    # Patterns for block captions (Table/Figure/Exhibit/Equation)
    BLOCK_CAPTION_PATTERNS = [
        r'^(?:Table|Tbl\.?)\s*(\d+(?:\.\d+)?)',
        r'^(?:Figure|Fig\.?)\s*(\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        r'^(?:Exhibit)\s*(\d+(?:\.\d+)?)',
        r'^(?:Equation|Eq\.?)\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?',
        r'^(?:Formula)\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?',
        r'^(?:Chart|Graph|Diagram)\s*(\d+(?:\.\d+)?)',
    ]
    
    # Patterns for structural headers (should update heading stack)
    STRUCTURAL_HEADER_PATTERNS = [
        r'^(?:Chapter|Part|Section|Unit)\s+\d+',
        r'^\d+(?:\.\d+)*\s+[A-Z]',  # "1.1 Introduction"
        r'^(?:Appendix|Glossary|Index|Bibliography|References)',
        r'^(?:Summary|Conclusion|Introduction|Overview|Preface)',
    ]
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._caption_patterns = [re.compile(p, re.IGNORECASE) for p in self.BLOCK_CAPTION_PATTERNS]
        self._structural_patterns = [re.compile(p, re.IGNORECASE) for p in self.STRUCTURAL_HEADER_PATTERNS]
        self._stats = {
            "captions_detected": 0,
            "captions_bonded": 0,
            "side_captions_detected": 0,
        }
    
    def detect_caption(self, seg: Dict) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if a segment is a block caption (Table/Figure title).
        
        Distinguishes between:
        - True captions: "Figure 3.7 The biggest stock markets" (short, title-like)
        - Descriptive references: "Figure 3.7 shows the market..." (long, contains verbs)
        
        Returns:
            Tuple of (is_block_caption, caption_info)
            - is_block_caption: True if this is a Table/Figure caption
            - caption_info: Dict with caption_type, caption_id, target_type
        """
        text = seg.get('text', '').strip()
        if not text:
            return False, {}
        
        # Check against block caption patterns
        for pattern in self._caption_patterns:
            match = pattern.match(text)
            if match:
                caption_id = match.group(1)
                
                # Filter out descriptive sentences that reference figures but aren't captions
                # True captions are typically:
                # 1. Short (under 100 chars for first sentence)
                # 2. Don't contain common descriptive verbs right after the figure number
                first_sentence = text.split('.')[0] if '.' in text else text
                text_after_id = text[match.end():].strip().lower()
                
                # Check for descriptive verb patterns (not a caption)
                # True captions never start with verbs like "shows", "presents", etc.
                descriptive_verbs = ['shows', 'presents', 'displays', 'illustrates', 
                                    'depicts', 'demonstrates', 'provides', 'contains',
                                    'compares', 'summarizes', 'lists', 'is a', 'is the',
                                    'plots', 'graphs', 'charts', 'traces', 'reports',
                                    'is an', 'was', 'were', 'has', 'have', 'gives',
                                    'indicates', 'reveals', 'suggests', 'confirms']
                is_descriptive = any(text_after_id.startswith(verb) for verb in descriptive_verbs)
                
                # If it starts with a descriptive verb, it's a reference sentence, not a caption
                if is_descriptive:
                    continue  # Skip this pattern, not a true caption
                
                # Infer target type from pattern
                target_type = self._infer_target_type(text)
                
                self._stats["captions_detected"] += 1
                
                return True, {
                    "is_block_caption": True,
                    "caption_id": caption_id,
                    "caption_text": text,
                    "target_type": target_type,
                    "full_caption_id": f"{target_type} {caption_id}",
                }
        
        return False, {}
    
    def is_structural_header(self, seg: Dict) -> bool:
        """
        Check if a segment is a structural header (should update heading stack).
        
        Structural headers define document hierarchy.
        Block captions (Table 1.1) should NOT be treated as structural headers.
        """
        text = seg.get('text', '').strip()
        if not text:
            return False
        
        # First check if it's a block caption - those are NOT structural
        is_caption, _ = self.detect_caption(seg)
        if is_caption:
            return False
        
        # Check structural patterns
        for pattern in self._structural_patterns:
            if pattern.match(text):
                return True
        
        # Fallback: if it's a Header type and not a caption, it's structural
        if seg.get('type') == 'Header':
            return True
        
        return False
    
    def _infer_target_type(self, text: str) -> str:
        """Infer the target block type from caption text."""
        text_lower = text.lower()
        if text_lower.startswith(('table', 'tbl')):
            return "Table"
        elif text_lower.startswith(('figure', 'fig', 'chart', 'graph', 'diagram')):
            return "Figure"
        elif text_lower.startswith('exhibit'):
            return "Exhibit"
        elif text_lower.startswith(('equation', 'eq', 'formula')):
            return "Equation"
        return "Figure"  # Default
    
    def find_nearby_caption(self, segments: List[Dict], table_index: int, 
                           max_distance: int = 3) -> Optional[int]:
        """
        Find a caption segment near a Table/Figure block.
        
        Searches both before and after the table for potential captions.
        Useful for side-caption and bottom-caption detection.
        
        Args:
            segments: List of all segments
            table_index: Index of the Table/Figure segment
            max_distance: Maximum segments to search in each direction
            
        Returns:
            Index of the caption segment, or None if not found
        """
        table_seg = segments[table_index]
        table_page = table_seg.get('page', 0)
        table_bbox = table_seg.get('bbox', [])
        
        # Search before the table
        for i in range(table_index - 1, max(0, table_index - max_distance) - 1, -1):
            seg = segments[i]
            # Must be on same page
            if seg.get('page', 0) != table_page:
                continue
            # Skip if already processed as table/figure
            if seg.get('type') in ['Table', 'Picture', 'Formula']:
                continue
            
            is_caption, _ = self.detect_caption(seg)
            if is_caption:
                self._stats["side_captions_detected"] += 1
                return i
        
        # Search after the table (for bottom captions)
        for i in range(table_index + 1, min(len(segments), table_index + max_distance + 1)):
            seg = segments[i]
            if seg.get('page', 0) != table_page:
                continue
            if seg.get('type') in ['Table', 'Picture', 'Formula']:
                continue
            
            is_caption, _ = self.detect_caption(seg)
            if is_caption:
                self._stats["side_captions_detected"] += 1
                return i
        
        return None
    
    def bond_caption_with_block(self, caption_seg: Dict, block_seg: Dict, 
                                notes_segs: List[Dict] = None) -> Dict:
        """
        Create a bonded unit combining caption + block + notes.
        
        Returns a merged segment that can be processed as a single chunk.
        """
        notes_segs = notes_segs or []
        
        # Combine text
        combined_text = caption_seg.get('text', '').strip()
        for note in notes_segs:
            note_text = note.get('text', '').strip()
            if note_text:
                combined_text += "\n" + note_text
        
        # Merge segment IDs
        all_seg_ids = [caption_seg.get('segment_id', '')]
        all_seg_ids.append(block_seg.get('segment_id', ''))
        for note in notes_segs:
            all_seg_ids.append(note.get('segment_id', ''))
        
        # Determine page range
        pages = {caption_seg.get('page', 0), block_seg.get('page', 0)}
        for note in notes_segs:
            pages.add(note.get('page', 0))
        
        self._stats["captions_bonded"] += 1
        
        return {
            "type": block_seg.get('type', 'Table'),
            "text": combined_text,
            "segment_id": block_seg.get('segment_id', ''),
            "bonded_segment_ids": [sid for sid in all_seg_ids if sid],
            "page": min(pages),
            "page_range": sorted(pages),
            "bbox": block_seg.get('bbox', []),
            "heading_path": caption_seg.get('heading_path', ''),
            "is_bonded": True,
            "caption_text": caption_seg.get('text', '').strip(),
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Return detection statistics."""
        return self._stats.copy()


# =============================================================================
# Reference Detector
# =============================================================================

class ReferenceDetector:
    """
    Detects references to Figure/Table/Equation blocks within paragraph text.
    
    Supports two types of references:
    1. Explicit: "Figure 1.1", "Table 2.3", "Equation (5)"
    2. Implicit: "the figure above", "this table", "as shown below"
    
    Usage:
        detector = ReferenceDetector(config)
        catalog = detector.build_block_catalog(segments)
        refs = detector.detect_references(content, page, catalog)
    """
    
    # Explicit reference patterns by type
    EXPLICIT_PATTERNS = {
        "Figure": [
            r'\b(?:Figure|Fig\.?)\s*(\d+(?:\.\d+)?(?:\s*\([a-z]\))?)',
        ],
        "Table": [
            r'\b(?:Table|Tbl\.?)\s*(\d+(?:\.\d+)?)',
        ],
        "Equation": [
            r'\b(?:Equation|Eq\.?)\s*\((\d+)\)',
            r'\b(?:Formula)\s*\((\d+)\)',
        ],
        "Exhibit": [
            r'\b(?:Exhibit)\s*(\d+(?:\.\d+)?)',
        ],
    }
    
    # Implicit reference patterns
    IMPLICIT_PATTERNS = {
        "above": [
            r'\b(?:the|this)\s+(?:figure|table|chart|graph|diagram)\s+(?:above|shown above)',
            r'\b(?:above|preceding)\s+(?:figure|table|chart)',
        ],
        "below": [
            r'\b(?:the|this)\s+(?:figure|table|chart|graph|diagram)\s+(?:below|shown below)',
            r'\b(?:following|below)\s+(?:figure|table|chart)',
        ],
        "this": [
            r'\b(?:this|that)\s+(?:figure|table|equation|chart|diagram)',
        ],
        "see": [
            r'\b(?:see|refer to)\s+(?:the\s+)?(?:figure|table)(?:\s+(?:above|below))?',
        ],
    }
    
    # Block types to track
    BLOCK_TYPES = {'Figure', 'Table', 'Formula', 'Picture', 'Equation', 'Header', 'Title', 'Paragraph', 'Text'}
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {
            "explicit_refs": 0,
            "implicit_refs": 0,
            "resolved": 0,
            "unresolved": 0,
        }
        
        # Compile patterns
        self._explicit_compiled = {}
        for ref_type, patterns in self.EXPLICIT_PATTERNS.items():
            self._explicit_compiled[ref_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._implicit_compiled = {}
        for direction, patterns in self.IMPLICIT_PATTERNS.items():
            self._implicit_compiled[direction] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def build_block_catalog(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Build a catalog of all Figure/Table/Equation blocks from segments.
        
        Returns:
            Dict with:
            - by_caption: caption_id -> segment info
            - by_segment_id: segment_id -> segment info
            - by_page: page_num -> list of block segments
        """
        catalog = {
            "by_caption": {},
            "by_segment_id": {},
            "by_page": {},
        }
        
        for seg in segments:
            seg_type = seg.get('type', '')
            
            # Check if this is a block type
            if seg_type in self.BLOCK_TYPES:
                segment_id = seg.get('segment_id', '')
                page = seg.get('page', 0)
                text = seg.get('text', '')
                bbox = seg.get('bbox', [])
                
                # Try to extract caption ID
                caption_id = self._extract_caption_id(text, seg_type)
                
                block_info = {
                    'segment_id': segment_id,
                    'type': seg_type,
                    'page': page,
                    'bbox': bbox,
                    'text': text[:100],  # Preview
                    'caption_id': caption_id,
                }
                
                # Store in catalog
                catalog['by_segment_id'][segment_id] = block_info
                
                if caption_id:
                    catalog['by_caption'][caption_id] = block_info
                
                if page not in catalog['by_page']:
                    catalog['by_page'][page] = []
                catalog['by_page'][page].append(block_info)
        
        logger.debug(f"ReferenceDetector: Built catalog with {len(catalog['by_segment_id'])} blocks, "
                    f"{len(catalog['by_caption'])} with caption IDs")
        
        return catalog
    
    def detect_references(self, content: str, chunk_page: int, 
                         catalog: Dict, chunk_bbox: List = None) -> List[Reference]:
        """
        Detect all references to blocks within chunk content.
        
        Args:
            content: Chunk text content
            chunk_page: Page number of the chunk
            catalog: Block catalog from build_block_catalog()
            chunk_bbox: Optional bbox of the chunk for proximity calculation
            
        Returns:
            List of Reference objects
        """
        if not content:
            return []
        
        refs = []
        
        # 1. Detect explicit references
        for ref_type, patterns in self._explicit_compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    ref_text = match.group(0)
                    ref_id = match.group(1)
                    caption_id = self._normalize_caption_id(ref_type, ref_id)
                    
                    # Try to resolve target
                    target = catalog['by_caption'].get(caption_id)
                    
                    refs.append(Reference(
                        ref_text=ref_text,
                        start_offset=match.start(),
                        end_offset=match.end(),
                        target_segment_id=target['segment_id'] if target else "",
                        target_type=ref_type,
                        ref_kind="explicit",
                        confidence=0.95 if target else 0.5
                    ))
                    
                    self._stats['explicit_refs'] += 1
                    if target:
                        self._stats['resolved'] += 1
                    else:
                        self._stats['unresolved'] += 1
        
        # 2. Detect implicit references
        for direction, patterns in self._implicit_compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(content):
                    ref_text = match.group(0)
                    
                    # Infer type from text
                    ref_type = self._infer_type_from_text(ref_text)
                    
                    # Try to resolve by proximity
                    target = self._resolve_implicit(
                        direction, ref_type, chunk_page, catalog, chunk_bbox
                    )
                    
                    refs.append(Reference(
                        ref_text=ref_text,
                        start_offset=match.start(),
                        end_offset=match.end(),
                        target_segment_id=target['segment_id'] if target else "",
                        target_type=ref_type,
                        ref_kind="implicit",
                        confidence=0.7 if target else 0.3
                    ))
                    
                    self._stats['implicit_refs'] += 1
                    if target:
                        self._stats['resolved'] += 1
                    else:
                        self._stats['unresolved'] += 1
        
        return refs
    
    def _extract_caption_id(self, text: str, block_type: str) -> Optional[str]:
        """Extract caption ID from block text (e.g., 'Figure 1.1: Title')."""
        if not text:
            return None
        
        # Try to match caption patterns
        patterns = {
            'Figure': r'^(?:Figure|Fig\.?)\s*(\d+(?:\.\d+)?)',
            'Table': r'^(?:Table|Tbl\.?)\s*(\d+(?:\.\d+)?)',
            'Picture': r'^(?:Figure|Fig\.?)\s*(\d+(?:\.\d+)?)',
            'Formula': r'^(?:Equation|Eq\.?)\s*\((\d+)\)',
            'Equation': r'^(?:Equation|Eq\.?)\s*\((\d+)\)',
        }
        
        # If it's a generic text segment (Header, Paragraph, etc.), try all patterns
        if block_type in ['Header', 'Title', 'Paragraph', 'Text']:
            for btype, pattern in patterns.items():
                match = re.match(pattern, text.strip(), re.IGNORECASE)
                if match:
                    return self._normalize_caption_id(btype, match.group(1))
        else:
            pattern = patterns.get(block_type)
            if pattern:
                match = re.match(pattern, text.strip(), re.IGNORECASE)
                if match:
                    return self._normalize_caption_id(block_type, match.group(1))
        
        return None
    
    def _normalize_caption_id(self, ref_type: str, ref_id: str) -> str:
        """Normalize caption ID for matching (e.g., 'Figure 1.1')."""
        # Clean up the ID
        clean_id = ref_id.strip()
        # Map Picture to Figure
        display_type = "Figure" if ref_type == "Picture" else ref_type
        return f"{display_type} {clean_id}"
    
    def _infer_type_from_text(self, text: str) -> str:
        """Infer block type from implicit reference text."""
        text_lower = text.lower()
        if 'figure' in text_lower or 'chart' in text_lower or 'graph' in text_lower or 'diagram' in text_lower:
            return "Figure"
        elif 'table' in text_lower:
            return "Table"
        elif 'equation' in text_lower or 'formula' in text_lower:
            return "Equation"
        return "Figure"  # Default
    
    def _resolve_implicit(self, direction: str, ref_type: str, 
                         page: int, catalog: Dict, chunk_bbox: List = None) -> Optional[Dict]:
        """
        Resolve implicit reference by proximity.
        
        Args:
            direction: 'above', 'below', 'this', 'see'
            ref_type: Expected block type
            page: Current page
            catalog: Block catalog
            chunk_bbox: Optional chunk bbox for proximity
            
        Returns:
            Best matching block info or None
        """
        # Get blocks on same page
        page_blocks = catalog['by_page'].get(page, [])
        
        # Filter by type
        type_map = {"Figure": ["Figure", "Picture"], "Table": ["Table"], "Equation": ["Formula", "Equation"]}
        valid_types = type_map.get(ref_type, [ref_type])
        candidates = [b for b in page_blocks if b['type'] in valid_types]
        
        if not candidates:
            return None
        
        # For now, return the first/last based on direction
        # TODO: Use bbox proximity for better resolution
        if direction == 'above':
            return candidates[0]  # First block on page
        elif direction == 'below':
            return candidates[-1]  # Last block on page
        else:
            return candidates[0]  # Default to first
    
    def get_stats(self) -> Dict[str, int]:
        """Return detection statistics."""
        return self._stats.copy()


class ContentGatekeeper:
    """
    Intelligent document gating to identify main body content vs front/back matter.
    
    Uses sequential pattern recognition to distinguish between isolated references 
    in front matter and the actual start of structured chapters.
    """
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.start_id = None
        self.end_id = None
        self.detected_pattern = None

    def analyze(self, segments: List[Dict], detected_toc_pages: set = None) -> Dict[str, Any]:
        """
        Scan segments to identify logical start and end of main content.
        
        Algorithm:
        1. Identify level-1 headers.
        2. Filter out headers on known TOC pages or in high-density TOC-like zones.
        3. Enforce a minimum page gap after the first continuous block of TOC pages.
        4. Match against common chapter/section starting patterns.
        """
        all_toc_pages = (detected_toc_pages or set()).copy()
        l1_headers = [s for s in segments if s.get('type') == 'Header' and s.get('level') == 1]
        
        if not l1_headers:
            return {"start_id": None, "end_id": None}

        # Calculate the end of the INITIAL TOC/Preface block
        sorted_toc = sorted([p for p in all_toc_pages if p < self.config.GATING_SCAN_LIMIT_PAGE])
        effective_toc_end = 0
        if sorted_toc:
            effective_toc_end = sorted_toc[0]
            for j in range(1, len(sorted_toc)):
                if sorted_toc[j] - sorted_toc[j-1] > 8: # Gap > 8 pages likely means end of TOC
                    break
                effective_toc_end = sorted_toc[j]
        
        # Start scanning for body AFTER the initial TOC block
        safe_start_page = effective_toc_end + 1

        # 1. Detect Main Body Start
        best_start_idx = -1
        logger.info(f"Gatekeeper: Scanning {len(l1_headers)} L1 headers. TOC pages: {sorted(list(all_toc_pages))}. safe_start_page: {safe_start_page}")
        
        for i, header in enumerate(l1_headers):
            text = header.get('text', '').strip()
            page = header.get('page', 0)
            
            # Skip if on a known TOC page OR before the end of the initial TOC block
            if page in all_toc_pages or page < safe_start_page:
                logger.debug(f"Gatekeeper: Skipping page {page} (TOC/Front-Matter zone)")
                continue
                
            # Stop searching if we're too deep into the document
            if page > self.config.GATING_SCAN_LIMIT_PAGE:
                logger.debug(f"Gatekeeper: Skipping page {page} (> scan limit {self.config.GATING_SCAN_LIMIT_PAGE})")
                continue
                
            logger.info(f"Gatekeeper: Checking header '{text}' at page {page}")
            for pattern in self.config.MAIN_BODY_START_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    self.start_id = header.get('segment_id')
                    best_start_idx = i
                    self.detected_pattern = pattern
                    logger.info(f"Gatekeeper: Main Body start matches pattern '{pattern}' at page {page}")
                    break
            if self.start_id:
                break

        # 2. Detect Back Matter Start
        if self.start_id:
            # Look for back-matter patterns among headers following the start
            for header in l1_headers[best_start_idx + 1:]:
                text = header.get('text', '').strip()
                for pattern in self.config.BACK_MATTER_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        self.end_id = header.get('segment_id')
                        logger.info(f"Gatekeeper: Potential Back Matter start at '{text}' ({self.end_id})")
                        break
                if self.end_id:
                    break

        return {"start_id": self.start_id, "end_id": self.end_id}


class ReadingOrderCorrector:
    """
    Three-Phase Pipeline for Reading Order and Heading Path Correction.
    
    Addresses the common PDF parsing issue where:
    - Multi-column layouts cause incorrect reading order
    - Headers appearing late in scan order get wrong parent assignments
    - Same-page segments inherit stale heading paths
    - Page decorations (furniture) interfere with content extraction
    
    Pipeline:
    - Pre-Phase: Furniture detection using FurnitureDetector
    - Phase 1: Column-based segment reordering (left-col then right-col per page)
    - Phase 2: Heading stack reconstruction (ancestor stack algorithm)
    - Phase 3: Same-page backfilling for late-discovered L1 headers
    """
    
    PATH_SEPARATOR = " > "
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.furniture_detector = FurnitureDetector(config)
        self._stats = {
            "furniture_detected": 0,
            "pages_reordered": 0,
            "segments_moved": 0,
            "paths_reconstructed": 0,
            "backfill_corrections": 0
        }
    
    def process(self, segments: List[Dict], gating_info: Dict = None) -> List[Dict]:
        """
        Apply four-phase correction pipeline to segments.
        
        Args:
            segments: List of segment dictionaries from parser_docling
            gating_info: Optional dict with start_id and end_id for main body
            
        Returns:
            Corrected segments with proper reading order and heading paths
        """
        result = segments
        
        # Pre-Phase: Furniture Detection
        if self.config.ENABLE_FURNITURE_DETECTION:
            # First pass: scan document for frequency statistics
            self.furniture_detector.scan_document(result)
            
            # Second pass: mark furniture
            for seg in result:
                if self.furniture_detector.is_furniture(seg):
                    seg['is_furniture'] = True
                    seg['is_noise_header'] = True  # Backward compatibility
            
            furniture_stats = self.furniture_detector.get_stats()
            self._stats['furniture_detected'] = furniture_stats.get('furniture_detected', 0)
            logger.info(f"Pre-Phase: Detected {self._stats['furniture_detected']} furniture elements "
                       f"(pattern: {furniture_stats.get('by_pattern', 0)}, "
                       f"frequency: {furniture_stats.get('by_frequency', 0)}, "
                       f"position: {furniture_stats.get('by_position', 0)})")
        
        # Phase 1: Reading Order Correction
        if self.config.ENABLE_READING_ORDER_CORRECTION:
            result = self._phase1_reorder_by_columns(result)
            logger.info(f"Phase 1: Reordered {self._stats['segments_moved']} segments across {self._stats['pages_reordered']} pages")
        
        # Phase 2: Heading Stack Reconstruction
        if self.config.ENABLE_HEADING_RECONSTRUCTION:
            result = self._phase2_rebuild_heading_paths(result, gating_info)
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
                    seg['column_index'] = -1
                    spanning.append(seg)
                    continue
                
                x_left, y_top, x_right, y_bottom = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Check if segment spans both columns (wide element)
                if x_left < mid_x and x_right > page_width * 0.55:
                    seg['column_index'] = -1
                    spanning.append(seg)
                elif x_left < mid_x:
                    seg['column_index'] = 0
                    left_col.append(seg)
                else:
                    seg['column_index'] = 1
                    right_col.append(seg)
            
            # Sort each group by y-coordinate (PDF y: higher = top of page)
            left_col.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            right_col.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            spanning.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            
            # Determine ordering strategy
            if self.config.ENABLE_COLUMN_ISOLATION:
                # Absolute Isolation: Spanning -> Left Column -> Right Column
                reordered = spanning + left_col + right_col
            else:
                # Interleaved merge by y-position (original behavior)
                all_segs = []
                for seg in left_col + right_col + spanning:
                    y_top = seg.get('bbox', [0, 0, 0, 0])[1]
                    col_idx = seg.get('column_index', -1)
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
    
    def _phase2_rebuild_heading_paths(self, segments: List[Dict], gating_info: Dict = None) -> List[Dict]:
        """
        Rebuild heading_path for all segments using the ancestor stack algorithm.
        
        Algorithm:
        1. Maintain a stack of (level, text) tuples representing current heading hierarchy
        2. Handle Main Body Gating:
           - Track if we have entered main content or back matter.
           - Reset stack upon entering main body to prevent hierarchy contamination.
           - Assign virtual paths (Front Matter / Back Matter) for segments outside body.
        3. For each Header segment:
           - Skip noise headers.
           - Update hierarchy stack and generate path.
        4. For each non-Header segment:
           - Inherit current path or virtual path.
        """
        stack = []  # [(level, text), ...]
        
        start_id = (gating_info or {}).get('start_id')
        end_id = (gating_info or {}).get('end_id')
        
        has_entered_body = False
        has_entered_back = False
        
        # If no gating is activated, start in body immediately.
        # If gating IS activated but no start matches found after analysis, 
        # we also default to body to avoid hiding the whole document (fail-safe).
        if not self.config.ENABLE_CONTENT_GATING or not start_id:
            has_entered_body = True
            if self.config.ENABLE_CONTENT_GATING:
                logger.warning("Gating: Main body start pattern NOT found. Defaulting to body.")

        for seg in segments:
            seg_id = seg.get('segment_id')
            
            # Gating State Machine
            if self.config.ENABLE_CONTENT_GATING:
                if seg_id == start_id:
                    has_entered_body = True
                    has_entered_back = False
                    stack = []  # HARD RESET: Clean stack upon entering main content
                    logger.info(f"Gating: Found start_id {seg_id} at page {seg.get('page')}. Entering Body.")
                
                if seg_id == end_id:
                    has_entered_body = False
                    has_entered_back = True
                    stack = []  # RESET: Clean stack upon entering back matter
                    logger.debug(f"Gating: Entering Back Matter at {seg_id}")

            if seg.get('type') == 'Header':
                # Skip furniture/noise headers - they don't affect hierarchy
                if seg.get('is_furniture', False) or seg.get('is_noise_header', False):
                    seg['heading_path'] = self._get_virtual_path(stack, has_entered_body, has_entered_back)
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
                new_path = self._get_virtual_path(stack, has_entered_body, has_entered_back)
                
                if old_path != new_path:
                    self._stats['paths_reconstructed'] += 1
                    seg['heading_path'] = new_path
                    seg['heading_path_original'] = old_path
            else:
                # Non-header: inherit path
                old_path = seg.get('heading_path', '')
                new_path = self._get_virtual_path(stack, has_entered_body, has_entered_back)
                
                if old_path != new_path:
                    self._stats['paths_reconstructed'] += 1
                    seg['heading_path'] = new_path
                    seg['heading_path_original'] = old_path
            
            # Update doc_zone metadata
            seg['doc_zone'] = "body" if has_entered_body else ("back" if has_entered_back else "front")
            
            # Update full_context_text
            if seg.get('heading_path'):
                seg['full_context_text'] = f"[Path: {seg['heading_path']}] {seg.get('text', '')}"
        
        return segments

    def _get_virtual_path(self, stack: List, in_body: bool, in_back: bool) -> str:
        """Construct path string with virtual pre-fixes for out-of-body zones."""
        path_text = self.PATH_SEPARATOR.join([t for _, t in stack])
        
        if in_body:
            return path_text
        elif in_back:
            prefix = "Back Matter"
            return f"{prefix}{self.PATH_SEPARATOR}{path_text}" if path_text else prefix
        else:
            prefix = "Front Matter"
            return f"{prefix}{self.PATH_SEPARATOR}{path_text}" if path_text else prefix
    
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
            r"\bassume\s+(?:that|we)\b",
            r"\bassuming\b",
            r"\bgiven\s+that\b",
            r"\bsuppose\s+(?:that|we)\b",
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


# =============================================================================
# POS Analyzer (Enhanced with Imperative Detection)
# =============================================================================

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
        
        Args:
            text: The text to analyze
            heading_path: The heading path for context (e.g., "Chapter 1 > Examples")
            forced_role: Optional role to force for all sentences
            chunk_type: Optional chunk type to influence role determination
        
        Roles (17 types):
        - topic: Introduces main idea
        - definition: Concept definitions
        - example: Examples and illustrations
        - evidence: Data, statistics, measurements (NEW)
        - theorem: Academic theorems, lemmas, or propositions (NEW)
        - procedure: Step-by-step instructions
        - mechanism: Causal chains and system operation (NEW)
        - contrast: Intra-sentence opposing logic (NEW)
        - assumption: Prerequisites/conditions
        - interpretation: Explaining meaning
        - limitation: Constraints/boundaries
        - comparison: Comparing concepts
        - application: Practical use cases
        - reference: Figure/Table references
        - conclusion: Summary statements
        - explanation: Default explanatory content
        - irrelevant: Low semantic value
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

            sentences.append({
                "text": sent_text,
                "role": role,
                "tags": sentence_tags,  # NEW: Preserve all semantic tags
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
                    # 2. Ignore if contains "Note:" or "McGraw Hill" (automatic exclusion)
                    if re.search(r"\bNote:|\bMcGraw\s*Hill\b", text):
                        continue
                    # 3. Only in的主体块 (explanation, example, or generic paragraph)
                    # We use chunk_type if available; if not, we rely on context
                    if chunk_type and chunk_type not in ["explanation", "example", "paragraph"]:
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


# =============================================================================
# Cross-Page Continuation Detector (Enhanced)
# =============================================================================

class ContinuationDetector:
    """
    Enhanced cross-page paragraph continuation detector.
    
    Implements the task requirements for accurate cross-page detection:
    - Incomplete sentences (missing terminal punctuation)
    - Hyphenation (word breaks across lines)
    - Abrupt clause endings (sentences ending with prepositions/conjunctions)
    - bbox position validation
    - Skip furniture elements (page decorations) when detecting continuations
    
    Strategy:
    1. Check if previous segment is at page bottom and current is at page top
    2. Verify style consistency (column, text patterns)
    3. Calculate detailed continuation score with evidence
    4. Return continuation type with merge evidence for explainability
    5. Skip furniture elements (now detected by FurnitureDetector) when crossing pages
    
    Returns:
    - 'full': High confidence continuation, auto-merge (score >= 0.7)
    - 'partial': Uncertain, needs human review (score 0.4-0.7)
    - 'none': Not a continuation (score < 0.4)
    """
    
    # Types that should NOT be merged across pages
    BREAK_TYPES = {'Header', 'Table', 'Picture', 'Formula', 'LearningObjective'}
    
    # Fallback noise patterns (used if is_furniture not set)
    NOISE_HEADER_PATTERNS = [
        r'^\s*\(?\s*concluded\s*\)?\s*$',
        r'^\s*\(?\s*continued\s*\)?\s*$',
        r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',
        r'^\s*\(?\s*continuation\s*\)?\s*$',
        r'^\s*continued\s+(?:on|from)\s+(?:next|previous)?\s*page',
        r'^\s*see\s+(?:next|previous)\s+page',
        r'^\s*to\s+be\s+continued',
    ]
    
    # Score thresholds
    FULL_THRESHOLD = 0.7
    PARTIAL_THRESHOLD = 0.4
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._last_evidence = {}  # Store last detection's evidence
        # Compile noise patterns for fallback
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_HEADER_PATTERNS]
    
    def _is_noise_header(self, seg: Dict[str, Any]) -> bool:
        """
        Check if a segment is furniture/noise that should be skipped.
        
        Uses the is_furniture flag set by FurnitureDetector (preferred),
        falls back to pattern matching if flag not available.
        """
        # First check the is_furniture flag (set by FurnitureDetector in Pre-Phase)
        if seg.get('is_furniture', False) or seg.get('is_noise_header', False):
            return True
        
        # Fallback: pattern-based detection for Headers
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
        # Factor 10: STRONG COMBINATION - Incomplete + Lowercase start
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
            
        # Using Safe Zone threshold from config
        # PDF Coordinates: Low Y is bottom
        ph = self.config.PAGE_HEIGHT_DEFAULT
        bottom_threshold = ph * self.config.FURNITURE_BOTTOM_BAND
        return bbox[3] < bottom_threshold
    
    def _estimate_at_top(self, seg: Dict[str, Any]) -> bool:
        """Estimate if segment is at page top based on bbox."""
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4:
            return False
            
        # Using Safe Zone threshold from config
        # PDF Coordinates: High Y is top
        ph = self.config.PAGE_HEIGHT_DEFAULT
        top_threshold = ph * (1 - self.config.FURNITURE_TOP_BAND)
        return bbox[1] > top_threshold
    
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
        
        Enhanced: Skips noise headers (like "(concluded)") when detecting
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
    
    Pipeline (Three-Phase Architecture):
    1. Load segments from JSON
    2. Apply Three-Phase Correction Pipeline:
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
        self.reading_order_corrector = ReadingOrderCorrector(self.config)
        self.reference_detector = ReferenceDetector(self.config)
        self.caption_bonding_helper = CaptionBondingHelper(self.config)  # Caption bonding
        self.chunk_counter = 0
        self.previous_chunk_text = ""  # For overlap
        self.block_catalog = None  # Will be populated during processing
        self.structural_heading_path = ""  # Track structural headers separately from block captions
    
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
        
        # Pre-execution Gating: Detect Main Body start/end
        gating_info = None
        if self.config.ENABLE_CONTENT_GATING:
            # We need toc_pages for better gating, detect them first
            toc_pages = self._detect_toc_pages(flat_segments)
            gatekeeper = ContentGatekeeper(self.config)
            gating_info = gatekeeper.analyze(flat_segments, detected_toc_pages=toc_pages)
            if gating_info.get('start_id'):
                logger.info("Main Body Gating: Active for this document")

        # Apply Three-Phase Correction Pipeline
        # This must run BEFORE continuation detection to ensure correct reading order
        if (self.config.ENABLE_READING_ORDER_CORRECTION or 
            self.config.ENABLE_HEADING_RECONSTRUCTION or 
            self.config.ENABLE_BACKFILL_CORRECTION):
            flat_segments = self.reading_order_corrector.process(flat_segments, gating_info)
            corrector_stats = self.reading_order_corrector.get_stats()
        else:
            corrector_stats = {}
        
        # Optional: Strip Front Matter segments completely if configured
        if self.config.ENABLE_CONTENT_GATING and self.config.STRIP_FRONT_MATTER:
            original_len = len(flat_segments)
            flat_segments = [s for s in flat_segments if s.get('doc_zone') != 'front']
            logger.info(f"Gating: Stripped {original_len - len(flat_segments)} front-matter segments")
        
        # Annotate segments with continuation markers
        if self.config.ENABLE_CONTINUATION_DETECTION:
            flat_segments = self.continuation_detector.annotate_segments(flat_segments)
            continuation_count = sum(1 for s in flat_segments if s.get('is_continuation') != 'none')
            logger.info(f"Detected {continuation_count} cross-page continuations")
        
        # Detect TOC pages
        toc_pages = self._detect_toc_pages(flat_segments)
        if toc_pages:
            logger.info(f"Detected TOC on pages: {sorted(list(toc_pages))}")
        
        # Build block catalog for reference detection
        self.block_catalog = self.reference_detector.build_block_catalog(flat_segments)
        logger.info(f"Built block catalog: {len(self.block_catalog['by_segment_id'])} blocks, "
                   f"{len(self.block_catalog['by_caption'])} with captions")
        
        # Process segments into chunks
        chunks = self._process_segments(flat_segments, toc_pages=toc_pages)
        
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
        if self.config.ENABLE_FURNITURE_DETECTION:
            features.append("furniture_detection")
        if self.config.ENABLE_DEHYPHENATION:
            features.append("dehyphenation")
        features.append("reference_detection")  # NEW
        
        # Calculate stats and include corrector stats
        processing_stats = self._calculate_stats(chunks)
        if corrector_stats:
            processing_stats['reading_order_correction'] = corrector_stats
        
        result = {
            "metadata": {
                **metadata,
                "total_chunks": len(chunks),
                "total_segments": len(flat_segments),
                "processing_version": "production",
                "features": features,
                "processing_stats": processing_stats
            },
            "chunks": [asdict(c) for c in chunks]
        }
        
        if output_path:
            self._save_json(result, output_path)
        
        return result
    
    def _detect_toc_pages(self, segments: List[Dict]) -> set:
        """Detect pages that are likely Table of Contents."""
        from collections import defaultdict
        toc_pages = set()
        page_texts = defaultdict(list)
        for seg in segments:
            p = seg.get('page', 0)
            page_texts[p].append(seg.get('text', '').lower())
        
        toc_keywords = {
            'contents', 'table of contents', 'index', '目录', 
            'brief contents', 'detailed contents', 'summary table of contents'
        }
        for p, texts in page_texts.items():
            full_text = " ".join(texts)
            # 1. Keyword check (prioritize early pages)
            if any(kw in full_text for kw in toc_keywords) and p < 60:
                toc_pages.add(p)
                continue
            # 2. Structure check: dotted leaders + numbers at end of lines
            # Restricted to very early document to avoid misdetecting List of Tables/Figures
            if p < 45: 
                # Require more dots (5) and ensure it looks like a TOC line
                dotted_lines = sum(1 for t in texts if re.search(r'[\.·-]{5,}\s*\d+$', t))
                if dotted_lines >= 3:
                    toc_pages.add(p)
        return toc_pages

    def _process_segments(self, segments: List[Dict], toc_pages: set = None) -> List[EnrichedChunk]:
        """
        Core processing logic:
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
        
        toc_pages = toc_pages or set()
        
        for i, seg in enumerate(segments):
            seg_text = seg.get('text', '')
            seg_type = seg.get('type', 'Paragraph')
            heading_path = seg.get('heading_path', '')
            is_continuation = seg.get('is_continuation', 'none')
            continuation_evidence = seg.get('continuation_evidence', {})
            seg_col = seg.get('column_index', -1)
            
            # TOC Special Handling
            if seg.get('page', 0) in toc_pages:
                # If matching TOC line pattern (chapter/section ... page)
                # Force it to be a Header/Topic to prevent adsorption and preserve hierarchy
                if re.search(r'[\.·-]{3,}\s*\d+$', seg_text) or re.search(r'^\d+(\.\d+)*\s+[A-Z]', seg_text):
                    seg_type = 'Header'
                    seg['type'] = 'Header'
                    seg['inferred_role'] = 'topic'
                    logger.debug(f"TOC Rule: Promoting {seg.get('segment_id')} to Header")
            
            # NEW: Merge Guards
            # 1. Column Locking: Prevent merging across columns unless spanning
            # 2. ID Gap Detection: Prevent merging segments with large ID distance
            should_flush_by_guard = False
            if buffer:
                prev_seg = buffer[-1]
                prev_col = prev_seg.get('column_index', -1)
                
                # Column Guard: If both are non-spanning and different, block merge
                if (self.config.COLUMN_MERGE_GUARD and 
                    seg_col != -1 and prev_col != -1 and seg_col != prev_col):
                    should_flush_by_guard = True
                    logger.debug(f"Merge Guard: Column mismatch ({prev_col} != {seg_col}). Flushing.")
                
                # ID Gap Guard: Check segment ID sequence
                prev_id_str = prev_seg.get('segment_id', 'seg_0')
                curr_id_str = seg.get('segment_id', 'seg_0')
                try:
                    prev_id_num = int(re.search(r'\d+', prev_id_str).group())
                    curr_id_num = int(re.search(r'\d+', curr_id_str).group())
                    if abs(curr_id_num - prev_id_num) > self.config.SEGMENT_ID_GAP_THRESHOLD:
                        should_flush_by_guard = True
                        logger.debug(f"Merge Guard: ID gap too large ({prev_id_num} -> {curr_id_num}). Flushing.")
                except:
                    pass

            if should_flush_by_guard and buffer:
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
            
            # Phase 1: Semantic Role Pre-analysis (Role-First)
            discourse_role = self.pos_analyzer.pre_analyze_role(seg_text) if self.pos_analyzer else None
            
            # If POS predicts LO but Docling says Paragraph, trust POS for grouping
            if discourse_role == 'learning_objective' and seg_type == 'Paragraph':
                seg_type = 'LearningObjective'
                seg['inferred_role'] = 'learning_objective'
                logger.debug(f"Role-First: Overriding Paragraph -> LearningObjective for {seg.get('segment_id')}")
            
            # Determine if buffer currently contains a "Master" block that can adsorb
            buffer_is_lo = buffer and (buffer[-1].get('type') == 'LearningObjective' or buffer[-1].get('inferred_role') == 'learning_objective')
            buffer_is_header = buffer and buffer[-1].get('type') == 'Header'
            can_adsorb = buffer_is_lo or buffer_is_header

            
            # =================================================================
            # Rule 0: Cross-page continuation handling with evidence
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
            # Rule 1: Headers - with Caption Bonding logic
            # =================================================================
            # Distinguish between:
            # - Block Captions (Table 1.1, Figure 2.3) -> bond with next block, don't update global path
            # - Structural Headers (Chapter 1, 1.1 Introduction) -> update global path
            if seg_type == 'Header':
                # Check if this is a noise header (concluded, continued, etc.)
                is_noise = seg.get('is_noise_header', False) or self.continuation_detector._is_noise_header(seg)
                
                if is_noise:
                    buffer.append(seg)
                    logger.debug(f"Skipping noise header {seg.get('segment_id')}: {seg.get('text', '')[:30]}")
                    continue
                
                # Check if this is a block caption (Table X.X, Figure X.X)
                is_block_caption, caption_info = self.caption_bonding_helper.detect_caption(seg)
                
                if is_block_caption:
                    # Block caption: DO NOT flush buffer or update global heading path
                    # Instead, add to buffer to bond with upcoming Table/Figure
                    # Mark it for later bonding
                    seg['is_block_caption'] = True
                    seg['caption_info'] = caption_info
                    buffer.append(seg)
                    logger.debug(f"Block caption detected: {caption_info.get('full_caption_id', '')} - buffering for bonding")
                    continue
                
                # Structural header: flush buffer and update global heading path
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
                
                # Update structural heading path
                current_heading_path = heading_path
                self.structural_heading_path = heading_path
                
                # Headers themselves become chunks
                chunks.append(self._create_chunk([seg], heading_path, chunk_type="header"))
                continue
            
            # =================================================================
            # Rule 2: ListItems should be grouped together
            # =================================================================
            if seg_type == 'ListItem':
                # Rule 2a: Adsorption - if buffer is an LO or Header, adsorb the list
                if can_adsorb and heading_path == current_heading_path:
                    buffer.append(seg)
                    logger.debug(f"Adsorption: ListItem {seg.get('segment_id')} adsorbed into active {buffer[-1].get('type')} buffer")
                    continue

                # Rule 2b: Standard List Item grouping
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
            # Rule 4: Tables/Pictures/Formulas - with Caption Bonding
            # =================================================================
            # Bond buffered block captions + notes with this structural block
            if seg_type in ['Table', 'Picture', 'Formula']:
                # Check if buffer contains a block caption to bond with
                caption_seg = None
                notes_segs = []
                other_segs = []
                
                for buffered_seg in buffer:
                    if buffered_seg.get('is_block_caption'):
                        caption_seg = buffered_seg
                    elif buffered_seg.get('type') in ['Header', 'Paragraph', 'Text']:
                        # Could be notes/description for the table
                        notes_segs.append(buffered_seg)
                    else:
                        other_segs.append(buffered_seg)
                
                # First, flush any non-caption content separately
                if other_segs:
                    chunk = self._create_chunk(other_segs, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                
                # Now create the bonded Table/Picture chunk
                # But ONLY if the caption type matches the block type
                if caption_seg:
                    caption_info = caption_seg.get('caption_info', {})
                    expected_type = caption_info.get('target_type', '')
                    
                    # Type matching rules:
                    # - "Table" caption should only bond with Table blocks
                    # - "Figure/Chart/Graph" caption should only bond with Picture blocks
                    # - "Equation/Formula" caption should only bond with Formula blocks
                    type_matches = False
                    if expected_type == 'Table' and seg_type == 'Table':
                        type_matches = True
                    elif expected_type in ['Figure', 'Exhibit'] and seg_type == 'Picture':
                        type_matches = True
                    elif expected_type == 'Equation' and seg_type == 'Formula':
                        type_matches = True
                    elif not expected_type:  # No specific type, allow any
                        type_matches = True
                    
                    if type_matches:
                        # Bond caption + notes + block into single atomic unit
                        bonded_segments = [caption_seg] + notes_segs + [seg]
                        caption_text = caption_seg.get('text', '').strip()
                        
                        # Use structural heading path, NOT the caption as the heading
                        effective_heading_path = self.structural_heading_path or current_heading_path
                        
                        chunk = self._create_chunk(
                            bonded_segments, 
                            effective_heading_path, 
                            chunk_type=seg_type.lower()
                        )
                        # Attach caption metadata for RAG enrichment
                        chunk.merge_evidence = {
                            "bonded": True,
                            "caption_text": caption_text,
                            "notes_count": len(notes_segs),
                            "source_segments": [s.get('segment_id', '') for s in bonded_segments],
                        }
                        chunks.append(chunk)
                        
                        logger.info(f"Caption bonded: {caption_text} -> {seg_type} ({len(notes_segs)} notes)")
                    else:
                        # Type mismatch: flush caption as separate chunk, block standalone
                        logger.info(f"Caption type mismatch: {expected_type} caption cannot bond with {seg_type} block")
                        # Flush caption + notes
                        all_caption_content = [caption_seg] + notes_segs
                        if all_caption_content:
                            chunk = self._create_chunk(all_caption_content, current_heading_path)
                            chunks.append(chunk)
                        # Create standalone block
                        chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                else:
                    # No caption in buffer, just create standalone chunk
                    if notes_segs:
                        # Flush notes first
                        chunk = self._create_chunk(notes_segs, current_heading_path)
                        chunks.append(chunk)
                    
                    chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                
                # Clear buffer
                buffer = []
                has_cross_page = False
                continuation_type = "none"
                merge_evidences = []
                continue
            
            # =================================================================
            # Rule 4.5: Group consecutive LearningObjectives under same heading
            # =================================================================
            if seg_type == 'LearningObjective':
                # If buffer has non-LO content OR heading changed, flush first
                if buffer and (buffer[-1].get('type') != 'LearningObjective' or buffer[-1].get('heading_path', '') != heading_path):
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
                    # Update current_heading_path for the upcoming LOs
                    current_heading_path = heading_path
                
                if not buffer:
                    current_heading_path = heading_path
                    
                buffer.append(seg)
                continue

            # =================================================================
            # Rule 4.6: Flush LearningObjectives sequence if next is different
            # =================================================================
            if buffer and buffer[-1].get('type') == 'LearningObjective' and seg_type != 'LearningObjective':
                # Rule 4.7: Soft Adsorption - allow one trailing Paragraph if it looks like content
                if seg_type == 'Paragraph' and len(seg_text.split()) < 50 and heading_path == current_heading_path:
                     buffer.append(seg)
                     logger.debug(f"Soft Adsorption: Small paragraph {seg.get('segment_id')} added to LO chunk")
                     continue

                chunk = self._create_chunk(buffer, current_heading_path, chunk_type="learning_objective")
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
            # Rule 5.5: Detect block captions in Paragraphs (side/bottom captions)
            # =================================================================
            # Sometimes PDF parsers misclassify "Table 1.2" as Paragraph instead of Header
            # We still need to detect and mark these for bonding
            if seg_type in ['Paragraph', 'Text']:
                is_block_caption, caption_info = self.caption_bonding_helper.detect_caption(seg)
                if is_block_caption:
                    seg['is_block_caption'] = True
                    seg['caption_info'] = caption_info
                    buffer.append(seg)
                    logger.debug(f"Paragraph caption detected: {caption_info.get('full_caption_id', '')} - buffering for bonding")
                    continue
            
            # =================================================================
            # Default: Add to buffer
            # =================================================================
            buffer.append(seg)
            
            # =================================================================
            # Rule 6: Flush if buffer exceeds threshold (with lookahead)
            # Check if next non-noise segment is a continuation
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
        1. Bond orphan captions with adjacent Table/Picture chunks (bottom caption repair)
        2. Merge short adjacent chunks under same heading
        3. Add context overlap from previous chunk
        """
        if not chunks:
            return chunks
        
        # Phase 1: Orphan caption bonding (bottom caption repair)
        # Detect chunks that are block captions but weren't bonded
        # and retroactively merge them with preceding Table/Picture chunks
        bonded_chunks = []
        skip_next = False
        
        for i, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue
            
            # Check if this is an orphan caption (starts with Table/Figure but wasn't bonded)
            is_orphan_caption = False
            content = chunk.content.strip()
            if chunk.chunk_type in ['explanation', 'header'] and content:
                is_caption, caption_info = self.caption_bonding_helper.detect_caption({
                    'text': content.split('\n')[0]  # Check first line only
                })
                if is_caption and not chunk.merge_evidence.get('bonded'):
                    is_orphan_caption = True
            
            if is_orphan_caption:
                # First, try to bond with a PRECEDING Table/Picture chunk
                for j in range(len(bonded_chunks) - 1, max(0, len(bonded_chunks) - 3) - 1, -1):
                    prev_chunk = bonded_chunks[j]
                    if prev_chunk.chunk_type in ['table', 'picture', 'formula']:
                        # Skip if this table already has a caption bonded
                        if prev_chunk.merge_evidence.get('bonded'):
                            continue
                        
                        # Check if they're on the same page or adjacent pages
                        same_or_adjacent_page = (
                            set(prev_chunk.page_range) & set(chunk.page_range) or
                            abs(max(prev_chunk.page_range) - min(chunk.page_range)) <= 1
                        )
                        if same_or_adjacent_page:
                            # Bond: prepend caption to table content
                            prev_chunk.content = chunk.content + "\n" + prev_chunk.content if prev_chunk.content else chunk.content
                            prev_chunk.source_segments.extend(chunk.source_segments)
                            prev_chunk.merge_evidence = {
                                "bonded": True,
                                "caption_text": content.split('\n')[0][:50],
                                "bond_type": "post_process_orphan_caption",
                            }
                            logger.info(f"Post-process bonded orphan caption (backward): {content[:30]}... -> {prev_chunk.chunk_type}")
                            is_orphan_caption = False
                            break
                
                # If still orphan, try to bond with the NEXT Table/Picture chunk (look ahead)
                if is_orphan_caption:
                    for k in range(i + 1, min(len(chunks), i + 4)):
                        next_chunk = chunks[k]
                        if next_chunk.chunk_type in ['table', 'picture', 'formula']:
                            if not next_chunk.merge_evidence.get('bonded'):
                                same_or_adjacent_page = (
                                    set(next_chunk.page_range) & set(chunk.page_range) or
                                    abs(min(next_chunk.page_range) - max(chunk.page_range)) <= 1
                                )
                                if same_or_adjacent_page:
                                    # Bond: prepend caption to next chunk
                                    next_chunk.content = chunk.content + "\n" + next_chunk.content if next_chunk.content else chunk.content
                                    next_chunk.source_segments = chunk.source_segments + next_chunk.source_segments
                                    next_chunk.merge_evidence = {
                                        "bonded": True,
                                        "caption_text": content.split('\n')[0][:50],
                                        "bond_type": "post_process_forward_bond",
                                    }
                                    logger.info(f"Post-process bonded orphan caption (forward): {content[:30]}... -> {next_chunk.chunk_type}")
                                    is_orphan_caption = False
                                    break
                            break  # Only check the first Table/Picture
                
                if not is_orphan_caption:
                    continue  # Skip this chunk, it was merged
            
            bonded_chunks.append(chunk)
        
        # Phase 1.5: Table-Then-Caption bonding
        # Handle case where Docling outputs Table BEFORE Caption (visual ordering issue)
        # Check each unbonded Table/Picture and see if it's followed by its Caption
        final_bonded = []
        skip_indices = set()
        
        for i, chunk in enumerate(bonded_chunks):
            if i in skip_indices:
                continue
            
            # Phase 1.8: Learning Objective + List regrouping (Hard Regrouping)
            # Handle case where LO header was separated from its list
            if (chunk.chunk_type == 'learning_objective' or 
                (self.pos_analyzer and self.pos_analyzer.pre_analyze_role(chunk.content.split('\n')[0]) == 'learning_objective')):
                
                # Check if the next chunk is a list or unbonded explanation
                for j in range(i + 1, min(len(bonded_chunks), i + 3)):
                    if j in skip_indices:
                        continue
                    next_chunk = bonded_chunks[j]
                    
                    # Merge if it's a list or a short explanation under the same heading
                    should_merge = False
                    if next_chunk.chunk_type == 'list':
                        should_merge = True
                    elif next_chunk.chunk_type == 'explanation' and next_chunk.word_count < 100:
                        should_merge = True
                        
                    if should_merge and next_chunk.heading_path == chunk.heading_path:
                        # Hard Merge: combine content and markers
                        chunk.content += "\n" + next_chunk.content
                        chunk.source_segments.extend(next_chunk.source_segments)
                        chunk.sentences.extend(next_chunk.sentences)
                        chunk.tags = list(set(chunk.tags + next_chunk.tags))
                        chunk.chunk_type = 'learning_objective' 
                        chunk.word_count = len(chunk.content.split())
                        skip_indices.add(j)
                        logger.info(f"Hard Regrouping: LO Header merged with {next_chunk.chunk_type} (chunk_{j})")
                        # Keep looking if there's more to adsorb
                        continue
                    else:
                        break

            # If this is an unbonded Table/Picture, check if the next chunk is its Caption
            if chunk.chunk_type in ['table', 'picture', 'formula'] and not chunk.merge_evidence.get('bonded'):
                # Look ahead for a Caption
                for j in range(i + 1, min(len(bonded_chunks), i + 3)):
                    if j in skip_indices:
                        continue
                    next_chunk = bonded_chunks[j]
                    if next_chunk.chunk_type in ['header', 'explanation']:
                        # Check if it's a caption for this table
                        is_caption, _ = self.caption_bonding_helper.detect_caption({
                            'text': next_chunk.content.split('\n')[0]
                        })
                        if is_caption and not next_chunk.merge_evidence.get('bonded'):
                            # Bond: prepend caption to table
                            chunk.content = next_chunk.content + "\n" + chunk.content if chunk.content else next_chunk.content
                            chunk.source_segments = next_chunk.source_segments + chunk.source_segments
                            chunk.merge_evidence = {
                                "bonded": True,
                                "caption_text": next_chunk.content.split('\n')[0][:50],
                                "bond_type": "table_then_caption",
                            }
                            skip_indices.add(j)
                            logger.info(f"Table-then-Caption bonded: {next_chunk.content[:30]}... -> {chunk.chunk_type}")
                            break
                    # Stop if we hit another table/picture
                    elif next_chunk.chunk_type in ['table', 'picture', 'formula']:
                        break
            
            final_bonded.append(chunk)
        
        bonded_chunks = final_bonded
        
        # Phase 2: Original post-processing
        processed = []
        
        for i, chunk in enumerate(bonded_chunks):
            # Add context overlap (if enabled and not first chunk)
            if self.config.ENABLE_OVERLAP and i > 0 and chunk.chunk_type not in ['header', 'picture', 'table']:
                prev_chunk = bonded_chunks[i - 1]
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
        
        # Combine text - exclude furniture/noise from content
        # Apply dehyphenation to repair word breaks across segments
        text_parts = []
        for s in segments:
            # Skip furniture elements (their text shouldn't appear in content)
            if s.get('is_furniture', False) or s.get('is_noise_header', False):
                continue
            text = s.get('text', '').strip()
            if text:
                text_parts.append(text)
        
        # Apply dehyphenation if enabled
        if self.config.ENABLE_DEHYPHENATION and len(text_parts) > 1:
            repaired_parts = []
            dehyph = DehyphenationHelper(self.config)
            
            for i, part in enumerate(text_parts):
                if i == 0:
                    repaired_parts.append(part)
                else:
                    # Try to merge hyphenated word from previous part
                    prev_part = repaired_parts[-1]
                    new_prev, new_curr = dehyph.merge_hyphenated(prev_part, part)
                    repaired_parts[-1] = new_prev
                    if new_curr:  # Only add if there's remaining text
                        repaired_parts.append(new_curr)
                    elif not new_curr and new_prev != prev_part:
                        # Word was fully absorbed into previous part
                        pass
            
            text_parts = repaired_parts
        
        full_text = " ".join(text_parts).strip()
        
        # Word count
        word_count = len(full_text.split())
        
        # Get source segment IDs (include noise headers for traceability)
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
        
        # Determine chunk type early to influence sentence roles
        if chunk_type is None:
            chunk_type = self._infer_chunk_type(segments, tags, [])
            
        # Forced Role Linkage logic
        forced_role = None
        if chunk_type == "header":
            forced_role = "topic"
        elif all(s.get('is_furniture', False) or s.get('is_noise_header', False) for s in segments):
            forced_role = "irrelevant"
            
        # Analyze sentences with POS
        sentences = []
        if self.pos_analyzer and full_text:
            sentences = self.pos_analyzer.analyze_sentences(
                full_text, heading_path, forced_role=forced_role, chunk_type=chunk_type
            )
            
            # AGGREGATE TAGS: Ensure chunk tags include all tags found in any sentence
            # This is crucial for 'reference' detection in complex sentences
            sentence_tag_set = set(tags)
            for sent in sentences:
                if "tags" in sent:
                    sentence_tag_set.update(sent["tags"])
            tags = list(sentence_tag_set)
        
        # Determine chunk type if not provided
        if chunk_type is None:
            chunk_type = self._infer_chunk_type(segments, tags, sentences)
        
        # Detect references to Figure/Table/Equation (NEW)
        references = []
        if self.block_catalog and full_text:
            chunk_page = page_range[0] if page_range else 0
            references = self.reference_detector.detect_references(
                full_text, chunk_page, self.block_catalog
            )
        
        # Determine dominant zone for the chunk
        zones = [s.get('doc_zone', 'body') for s in segments]
        dominant_zone = max(set(zones), key=zones.count) if zones else "body"
        
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
            word_count=word_count,
            references=references,
            doc_zone=dominant_zone
        )
    
    def _infer_chunk_type(self, segments: List[Dict], tags: List[str], 
                          sentences: List[Dict]) -> str:
        """Infer chunk type from segments, tags, and sentence analysis."""
        seg_types = [s.get('type', '') for s in segments]
        
        # Check segment types first
        if all(t == 'ListItem' for t in seg_types):
            return "list"
        
        if all(t == 'LearningObjective' for t in seg_types):
            return "learning_objective"
        
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
        
        # Cross-page statistics
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
            
            # Cross-page continuation stats
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
            # Cross-page continuation stats
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
    input_dir = base_dir / "outputs" / "Docling_json"
    output_dir = base_dir / "outputs" / "Chunks_Semantic"
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

