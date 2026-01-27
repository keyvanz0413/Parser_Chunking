import re
import logging
from typing import List, Dict
from ..config import ChunkingConfig

logger = logging.getLogger(__name__)

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
