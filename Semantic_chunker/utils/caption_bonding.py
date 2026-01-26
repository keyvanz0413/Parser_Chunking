import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from ..config import ChunkingConfig

logger = logging.getLogger(__name__)

class CaptionBondingHelper:
    """
    Detects and bonds Table/Figure captions with their structural blocks.
    
    Solves three subproblems:
    1. Over-fragmentation: Prevents splitting caption + table + notes into separate chunks
    2. Side-caption detection: Identifies captions in non-standard positions (side/bottom)
    3. Metadata drift: Distinguishes block captions from structural headers
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
                # True captions usually provide a title, while references describe function
                descriptive_verbs = [
                    'shows', 'presents', 'displays', 'illustrates', 'depicts', 
                    'demonstrates', 'provides', 'contains', 'compares', 'summarizes', 
                    'lists', 'is a', 'is the', 'is an', 'was', 'were', 'has', 'have', 
                    'gives', 'indicates', 'reveals', 'suggests', 'confirms', 'adjusts',
                    'rearranged', 'expresses', 'calculates', 'defines'
                ]
                
                # Use regex to find verb in the first part of the sentence, 
                # allowing for "Panel A", "below", etc.
                verb_pattern = r'\b(?:' + '|'.join(descriptive_verbs) + r')\b'
                is_descriptive = bool(re.search(verb_pattern, text_after_id[:100], re.I))
                
                # Also, true captions are rarely extremely long sentences (> 40 words)
                word_count = len(text.split())
                if is_descriptive or word_count > 40:
                    continue  # Skip this pattern, not a true caption
                
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
        """
        table_seg = segments[table_index]
        table_page = table_seg.get('page', 0)
        table_bbox = table_seg.get('bbox', [])
        
        # Search before the table
        for i in range(table_index - 1, max(0, table_index - max_distance) - 1, -1):
            seg = segments[i]
            if seg.get('page', 0) != table_page:
                continue
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
            "page_range": list(sorted(pages)),
            "bbox": block_seg.get('bbox', []),
            "heading_path": caption_seg.get('heading_path', ''),
            "is_bonded": True,
            "caption_text": caption_seg.get('text', '').strip(),
        }
    
    def verify_spatial_proximity(self, caption_seg: Dict, block_seg: Dict) -> bool:
        """
        Verify if a caption and a block are spatially close on the same page.
        Coordinates are usually normalized [x_min, y_min, x_max, y_max].
        """
        if not self.config.ENABLE_SPATIAL_BONDING_CHECK:
            return True
        
        cap_bbox = caption_seg.get('bbox', [])
        blk_bbox = block_seg.get('bbox', [])
        
        if not cap_bbox or not blk_bbox:
            return True  # Fallback if no bbox info
            
        # 1. Vertical Distance Check
        # Distance between bottom of one and top of another (order depends on which is above)
        is_caption_above = cap_bbox[3] <= blk_bbox[1]
        is_block_above = blk_bbox[3] <= cap_bbox[1]
        
        v_dist = 0
        if is_caption_above:
            v_dist = blk_bbox[1] - cap_bbox[3]
        elif is_block_above:
            v_dist = cap_bbox[1] - blk_bbox[3]
        else:
            # Overlapping vertically? (Sides)
            v_dist = 0
            
        # Normalize by page height (assume 1000 if not provided in bbox units)
        # Docling usually provides absolute points or normalized 0-1
        # If max value is > 1.0, it's absolute. Normalizing to 0-1 scale.
        height_factor = self.config.PAGE_HEIGHT_DEFAULT if any(v > 1.0 for v in cap_bbox + blk_bbox) else 1.0
        v_dist_norm = v_dist / height_factor
        
        # Horizontal Overlap Check
        i_x_min = max(cap_bbox[0], blk_bbox[0])
        i_x_max = min(cap_bbox[2], blk_bbox[2])
        overlap_width = max(0, i_x_max - i_x_min)
        
        cap_width = cap_bbox[2] - cap_bbox[0]
        blk_width = blk_bbox[2] - blk_bbox[0]
        overlap_ratio = overlap_width / min(cap_width, blk_width) if min(cap_width, blk_width) > 0 else 0
        
        if v_dist_norm > self.config.MAX_CAPTION_VERTICAL_DISTANCE:
            return False
            
        if overlap_width == 0:
            return False
            
        if overlap_ratio < self.config.MAX_CAPTION_HORIZONTAL_OVERLAP:
            return False
            
        return True

    def get_stats(self) -> Dict[str, int]:
        """Return detection statistics."""
        return self._stats.copy()
