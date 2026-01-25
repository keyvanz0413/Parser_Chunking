import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from ...core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class CaptionBondingHelper:
    """Detects and bonds Table/Figure captions with their structural blocks."""
    
    CAPTION_PATTERNS = [
        r'(?i)^(?:Table|Figure|Exhibit|Chart|Graph|Equation|Formula|Exh\.|Fig\.)\s+[\d.]+[A-Za-z]?[:.]?\s+',
        r'(?i)^(?:Table|Figure|Exhibit|Chart|Graph|Equation|Formula|Exh\.|Fig\.)\s+\d+\s+[:.]?\s+',
    ]
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._compiled_patterns = [re.compile(p) for p in self.CAPTION_PATTERNS]
        self._stats = {"captions_detected": 0, "bonded_units": 0}

    def detect_caption(self, seg: Dict) -> Tuple[bool, Dict]:
        text = seg.get('text', '').strip()
        if not text: return False, {}
        
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                caption_type = self._infer_target_type(text)
                self._stats["captions_detected"] += 1
                return True, {"caption_type": caption_type, "target_type": caption_type}
        return False, {}

    def _infer_target_type(self, text: str) -> str:
        text = text.lower()
        if 'table' in text: return 'Table'
        if any(x in text for x in ['figure', 'fig.', 'chart', 'graph', 'exhibit']): return 'Figure'
        if any(x in text for x in ['equation', 'formula']): return 'Equation'
        return 'Block'

    def bond_caption_with_block(self, caption_seg: Dict, block_seg: Dict, notes_segs: List[Dict] = None) -> Dict:
        # Bonding logic moved to pipeline for higher-level control, but helper provides type info
        self._stats["bonded_units"] += 1
        return {"caption": caption_seg, "block": block_seg, "notes": notes_segs or []}
