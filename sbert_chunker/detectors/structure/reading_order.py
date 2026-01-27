import logging
from typing import List, Dict, Any, Tuple
from ...core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class ReadingOrderCorrector:
    """Three-Phase Pipeline for Reading Order and Heading Path Correction."""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {'pages_reordered': 0, 'segments_moved': 0, 'paths_rebuilt': 0, 'backfilled_segments': 0}

    def process(self, segments: List[Dict]) -> List[Dict]:
        if not segments: return []
        processed = segments
        if self.config.ENABLE_READING_ORDER_CORRECTION:
            processed = self._phase1_reorder_by_columns(processed)
        if self.config.ENABLE_HEADING_RECONSTRUCTION:
            processed = self._phase2_rebuild_heading_paths(processed)
        if self.config.ENABLE_BACKFILL_CORRECTION:
            processed = self._phase3_backfill_same_page(processed)
        return processed

    def get_stats(self) -> Dict[str, Any]: return self._stats

    def _phase1_reorder_by_columns(self, segments: List[Dict]) -> List[Dict]:
        from collections import defaultdict
        pages = defaultdict(list)
        for seg in segments: pages[seg.get('page', 1)].append(seg)
        result = []
        threshold = self.config.COLUMN_DETECTION_THRESHOLD
        page_width = self.config.PAGE_WIDTH_DEFAULT
        
        for page_num in sorted(pages.keys()):
            page_segs = pages[page_num]
            max_x = max((s.get('bbox', [0, 0, 0, 0])[2] for s in page_segs if s.get('bbox')), default=page_width)
            if max_x > 100: page_width = max_x * 1.1
            mid_x = page_width * threshold
            
            left_col, right_col, spanning = [], [], []
            for seg in page_segs:
                bbox = seg.get('bbox')
                if not bbox or len(bbox) < 4:
                    spanning.append(seg)
                    continue
                x_left, x_right = bbox[0], bbox[2]
                if x_left < mid_x and x_right > page_width * 0.55: spanning.append(seg)
                elif x_left < mid_x: left_col.append(seg)
                else: right_col.append(seg)
                
            for col in [left_col, right_col, spanning]:
                col.sort(key=lambda s: -s.get('bbox', [0, 0, 0, 0])[1])
            
            all_segs = []
            for seg in left_col + right_col + spanning:
                y_top = seg.get('bbox', [0, 0, 0, 0])[1]
                col_idx = 0 if seg in left_col else (1 if seg in right_col else -1)
                sort_key = (-y_top, 0 if col_idx <= 0 else 1)
                all_segs.append((sort_key, seg))
            all_segs.sort(key=lambda x: x[0])
            reordered = [seg for _, seg in all_segs]
            
            if [s.get('segment_id') for s in page_segs] != [s.get('segment_id') for s in reordered]:
                self._stats['pages_reordered'] += 1
            result.extend(reordered)
        return result

    def _phase2_rebuild_heading_paths(self, segments: List[Dict]) -> List[Dict]:
        stack = [] # list of (level, text)
        for seg in segments:
            if seg.get('type') == 'Header':
                level = seg.get('depth', 1)
                text = seg.get('text', '').strip()
                while stack and stack[-1][0] >= level: stack.pop()
                stack.append((level, text))
                path = " > ".join([s[1] for s in stack])
                seg['heading_path'] = path
                self._stats['paths_rebuilt'] += 1
            else:
                seg['heading_path'] = " > ".join([s[1] for s in stack]) if stack else ""
        return segments

    def _phase3_backfill_same_page(self, segments: List[Dict]) -> List[Dict]:
        from collections import defaultdict
        pages = defaultdict(list)
        for seg in segments: pages[seg.get('page', 1)].append(seg)
        for page_num in sorted(pages.keys()):
            page_segs = pages[page_num]
            headers = [s for s in page_segs if s.get('type') == 'Header']
            if not headers: continue
            top_header = min(headers, key=lambda h: h.get('depth', 10))
            path = top_header.get('heading_path', '')
            for seg in page_segs:
                if seg == top_header: break
                if not seg.get('heading_path'):
                    seg['heading_path'] = path
                    self._stats['backfilled_segments'] += 1
        return segments
