import re
import logging
from typing import List, Dict, Any
from ...core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class FurnitureDetector:
    """Multi-feature page decoration (furniture) detector."""
    
    FURNITURE_PHRASES = [
        r'^\s*\(?\s*concluded\s*\)?\s*$',
        r'^\s*\(?\s*continued\s*\)?\s*$',
        r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',
        r'^\s*\(?\s*continuation\s*\)?\s*$',
        r'^\s*continued\s+(?:on|from)\s+(?:next|previous)?\s*page',
        r'^\s*see\s+(?:next|previous)\s+page',
        r'^\s*to\s+be\s+continued',
        r'^(?:Table|Figure|Exhibit)\s+[\d.]+\s*\(?\s*(?:continued|cont\'?d?)\s*\)?',
        r'^\s*(?:Page\s+)?\d+\s*$',  
        r'^\s*\d+\s*/\s*\d+\s*$',    
        r'^\s*[-–—]\s*\d+\s*[-–—]\s*$', 
    ]
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.FURNITURE_PHRASES]
        self._frequency_map: Dict[str, int] = {}  
        self._total_pages: int = 0
        self._scanned: bool = False
        self._stats = {"furniture_detected": 0, "by_position": 0, "by_frequency": 0, "by_pattern": 0}

    def scan_document(self, segments: List[Dict]) -> None:
        from collections import defaultdict
        text_pages: Dict[str, set] = defaultdict(set)
        all_pages: set = set()
        for seg in segments:
            page = seg.get('page', 0)
            all_pages.add(page)
            text = seg.get('text', '').strip()
            if not text or len(text.split()) > self.config.FURNITURE_MAX_WORDS: continue
            if self._in_edge_band(seg):
                norm_text = self._normalize_text(text)
                text_pages[norm_text].add(page)
        self._total_pages = len(all_pages)
        self._frequency_map = {text: len(pages) for text, pages in text_pages.items()}
        self._scanned = True

    def is_furniture(self, seg: Dict) -> bool:
        if seg.get('type') not in ['Header', 'Paragraph', 'Text']: return False
        text = seg.get('text', '').strip()
        if not text: return False
        word_count = len(text.split())
        if self._matches_furniture_pattern(text):
            self._stats["furniture_detected"] += 1
            self._stats["by_pattern"] += 1
            return True
        if self._scanned and self._total_pages >= self.config.FURNITURE_MIN_PAGES_FOR_STATS:
            if self._in_edge_band(seg) and word_count <= self.config.FURNITURE_MAX_WORDS:
                norm_text = self._normalize_text(text)
                page_count = self._frequency_map.get(norm_text, 0)
                if page_count > 0 and (page_count / self._total_pages) >= self.config.FURNITURE_REPEAT_THRESHOLD:
                    self._stats["furniture_detected"] += 1
                    self._stats["by_frequency"] += 1
                    return True
        if word_count <= 2 and self._in_edge_band(seg) and self._is_trivial_content(text):
            self._stats["furniture_detected"] += 1
            self._stats["by_position"] += 1
            return True
        return False

    def _in_edge_band(self, seg: Dict) -> bool:
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4: return False
        page_height = self.config.PAGE_HEIGHT_DEFAULT
        y_top, y_bottom = bbox[1], bbox[3] if len(bbox) > 3 else bbox[1]
        if y_top > page_height * (1 - self.config.FURNITURE_TOP_BAND): return True
        if y_bottom < page_height * self.config.FURNITURE_BOTTOM_BAND: return True
        return False

    def _matches_furniture_pattern(self, text: str) -> bool:
        return any(p.match(text) for p in self._compiled_patterns)

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return re.sub(r'\b(?:page\s*)?\d+\b', '', normalized).strip()

    def _is_trivial_content(self, text: str) -> bool:
        if re.match(r'^[\d\s.,-]+$', text): return True
        if len(text) <= 2: return True
        if re.match(r'^[ivxlcdmIVXLCDM]+$', text): return True
        return False

    def get_stats(self) -> Dict[str, Any]: return self._stats
