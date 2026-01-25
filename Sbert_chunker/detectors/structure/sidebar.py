import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from ...core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class SidebarDetector:
    """Sidebar/Floating Block Detection for Structural Isolation."""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._page_columns = {} # page_num -> list of zones
        self._compiled_sidebar_headers = [re.compile(p) for p in self.config.SIDEBAR_HEADING_PATTERNS]
        self._compiled_lo_patterns = [re.compile(p) for p in self.config.LEARNING_OBJECTIVE_PATTERNS]
        self._stats = {"sidebars_detected": 0, "learning_objectives_detected": 0}

    def scan_document(self, segments: List[Dict]) -> None:
        from collections import defaultdict
        pages = defaultdict(list)
        for seg in segments: pages[seg.get('page', 0)].append(seg)
        for page_num, segs in pages.items():
            self._page_columns[page_num] = self._analyze_page_columns(segs)

    def _analyze_page_columns(self, page_segs: List[Dict]) -> Dict:
        # Simplified column analysis logic
        x_coords = []
        for s in page_segs:
            bbox = s.get('bbox')
            if bbox: x_coords.append(bbox[0])
        if not x_coords: return {"main_zone": (0, 1000), "sidebar_zones": [], "is_multi_column": False}
        
        page_width = self.config.PAGE_WIDTH_DEFAULT
        mid_x = page_width * self.config.COLUMN_DETECTION_THRESHOLD
        
        has_left = any(x < mid_x for x in x_coords)
        has_right = any(x >= mid_x for x in x_coords)
        
        zones = {"main_zone": (mid_x if has_left and has_right else 0, page_width), "sidebar_zones": [], "is_multi_column": has_left and has_right}
        if has_left and has_right:
            zones["sidebar_zones"].append((0, mid_x, 'left'))
        return zones

    def annotate_segments(self, segments: List[Dict]) -> List[Dict]:
        for seg in segments:
            text = seg.get('text', '')
            seg['is_sidebar'] = self.is_sidebar(seg)
            if seg['is_sidebar']:
                seg['sidebar_zone'] = self.get_sidebar_zone(seg)
                if self._is_learning_objective(text):
                    seg['sidebar_type'] = 'learning_objective'
                    self._stats['learning_objectives_detected'] += 1
                elif self._is_sidebar_heading(text):
                    seg['sidebar_type'] = 'sidebar_heading'
                self._stats['sidebars_detected'] += 1
        return segments

    def is_sidebar(self, seg: Dict) -> bool:
        text = seg.get('text', '')
        if self._is_sidebar_heading(text) or self._is_learning_objective(text): return True
        bbox = seg.get('bbox')
        if not bbox: return False
        page_num = seg.get('page', 0)
        zones = self._page_columns.get(page_num, {})
        for x_min, x_max, side in zones.get('sidebar_zones', []):
            if bbox[0] >= x_min and bbox[2] <= x_max: return True
        return False

    def get_sidebar_zone(self, seg: Dict) -> Optional[str]:
        bbox = seg.get('bbox')
        if not bbox: return None
        page_num = seg.get('page', 0)
        zones = self._page_columns.get(page_num, {})
        for x_min, x_max, side in zones.get('sidebar_zones', []):
            if bbox[0] >= x_min and bbox[0] <= x_max: return side
        return None

    def _is_sidebar_heading(self, text: str) -> bool:
        return any(p.match(text) for p in self._compiled_sidebar_headers)

    def _is_learning_objective(self, text: str) -> bool:
        return any(p.match(text) for p in self._compiled_lo_patterns)

    def get_stats(self) -> Dict[str, Any]: return self._stats
