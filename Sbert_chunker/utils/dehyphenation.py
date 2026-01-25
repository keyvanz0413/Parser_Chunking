import re
import logging
from typing import Tuple, List, Dict
from ..core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class DehyphenationHelper:
    """Cross-page and cross-line hyphenation repair."""
    
    HYPHEN_CHARS = ['-', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014']
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {"hyphens_detected": 0, "merges_performed": 0, "merges_skipped": 0}

    def process_segments(self, segments: List[Dict]) -> List[Dict]:
        """Iteratively repair hyphenation across segments."""
        if len(segments) < 2:
            return segments
            
        for i in range(1, len(segments)):
            prev = segments[i-1]
            curr = segments[i]
            
            # Only attempt if they follow logically (e.g. same column or cross-page)
            if curr.get('is_continuation') in ['full', 'partial']:
                new_prev_text, new_curr_text = self.merge_hyphenated(prev.get('text', ''), curr.get('text', ''))
                if new_prev_text != prev.get('text'):
                    prev['text'] = new_prev_text
                    curr['text'] = new_curr_text
                    
        return segments

    def merge_hyphenated(self, prev_text: str, curr_text: str) -> Tuple[str, str]:
        if not prev_text or not curr_text: 
            return prev_text, curr_text
        
        prev_stripped = prev_text.rstrip()
        if not prev_stripped or prev_stripped[-1] not in self.HYPHEN_CHARS:
            return prev_text, curr_text
        
        # Extract last word part
        match_prev = re.search(r'(\w+)[-‐‑−–—]\s*$', prev_text)
        if not match_prev:
            return prev_text, curr_text
            
        word_part1 = match_prev.group(1)
        
        # Extract first word part of next segment
        curr_stripped = curr_text.lstrip()
        match_curr = re.match(r'^(\w+)', curr_stripped)
        if not match_curr:
            return prev_text, curr_text
            
        word_part2 = match_curr.group(1)
        
        # Validation: check if merged word looks like a real word (basic heuristic)
        merged_word = word_part1 + word_part2
        if len(merged_word) < 3:
            return prev_text, curr_text
            
        # Success: Perform merge
        self._stats["merges_performed"] += 1
        
        # Remove hyphen and join
        new_prev = prev_text[:match_prev.start(0)] + word_part1 + word_part2
        new_curr = curr_text[match_curr.end(0):].lstrip()
        
        # We handle the join by putting the whole word in the first segment 
        # to ensure SBERT sees the full semantics during chunking.
        return new_prev, new_curr

    def get_stats(self):
        return self._stats
