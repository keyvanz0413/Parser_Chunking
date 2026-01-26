import logging
from typing import List, Dict
from ..config import ChunkingConfig
from ..detectors.furniture import FurnitureDetector

logger = logging.getLogger(__name__)

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
