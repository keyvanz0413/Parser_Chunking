import re
import logging
from typing import List, Dict, Any, Optional
from ..core.config import ChunkingConfig
from ..core.models import Reference

logger = logging.getLogger(__name__)

class ReferenceDetector:
    """Detects references to Figure/Table/Equation blocks within paragraph text."""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {"refs_detected": 0, "refs_resolved": 0}

    def build_block_catalog(self, segments: List[Dict]) -> Dict:
        catalog = {'by_segment_id': {}, 'by_caption': {}}
        for seg in segments:
            if seg.get('type') in ['Table', 'Picture', 'Formula']:
                catalog['by_segment_id'][seg.get('segment_id')] = seg
        return catalog

    def detect_references(self, content: str, chunk_page: int, catalog: Dict) -> List[Reference]:
        refs = []
        # Simplified reference detection logic
        pattern = r'(?i)(?:Figure|Table|Equation)\s+\d+[\.\d]*'
        for match in re.finditer(pattern, content):
            ref_text = match.group()
            refs.append(Reference(
                ref_text=ref_text,
                start_offset=match.start(),
                end_offset=match.end(),
                target_segment_id="", # Needs resolution logic
                target_type="Block",
                ref_kind="explicit"
            ))
            self._stats["refs_detected"] += 1
        return refs

    def get_stats(self): return self._stats
