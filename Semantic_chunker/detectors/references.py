import re
import logging
from typing import List, Dict, Any, Optional
from ..config import ChunkingConfig
from ..schema import Reference

logger = logging.getLogger(__name__)

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
            chunk_bbox: Optional chunk bbox for proximity calculation
            
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
