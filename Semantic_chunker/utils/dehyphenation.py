import re
from typing import Tuple, Dict
from ..config import ChunkingConfig

class DehyphenationHelper:
    """
    Cross-page and cross-line hyphenation repair.
    
    Handles cases where words are split across lines or pages with hyphens:
    - "invest-" + "ment" → "investment"
    - "con-" + "tinuation" → "continuation"
    
    Strategy:
    1. Detect if previous text ends with a hyphen
    2. Attempt to merge the hyphenated word parts
    3. Validate the merged word (optional dictionary check)
    4. Return the repaired text
    """
    
    # Common hyphen characters
    HYPHEN_CHARS = {'-', '‐', '‑', '–', '—'}
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._stats = {
            "hyphens_detected": 0,
            "merges_performed": 0,
            "merges_skipped": 0,
        }
    
    def merge_hyphenated(self, prev_text: str, curr_text: str) -> Tuple[str, str]:
        """
        Attempt to merge hyphenated word across text boundary.
        
        Args:
            prev_text: Text that may end with a hyphenated word fragment
            curr_text: Text that may begin with the rest of the word
            
        Returns:
            Tuple of (modified_prev_text, modified_curr_text)
            If merge performed, the hyphenated word is moved entirely to prev_text
        """
        if not prev_text or not curr_text:
            return prev_text, curr_text
        
        prev_stripped = prev_text.rstrip()
        
        # Check if prev ends with hyphen
        if not prev_stripped or prev_stripped[-1] not in self.HYPHEN_CHARS:
            return prev_text, curr_text
        
        # Ensure hyphen is attached to a word (not standalone like "- ")
        # The character before hyphen should be alphanumeric
        if len(prev_stripped) < 2 or not prev_stripped[-2].isalnum():
            return prev_text, curr_text
        
        self._stats["hyphens_detected"] += 1
        
        # Extract the word fragment before hyphen
        words = prev_stripped[:-1].split()
        if not words:
            return prev_text, curr_text
        
        word_part1 = words[-1]
        
        # Extract the first word from curr_text (potential word completion)
        curr_stripped = curr_text.lstrip()
        curr_words = curr_stripped.split()
        if not curr_words:
            return prev_text, curr_text
        
        word_part2 = curr_words[0]
        
        # Check if word_part2 looks like a word continuation (lowercase, no punctuation at start)
        if not word_part2 or not word_part2[0].isalpha():
            self._stats["merges_skipped"] += 1
            return prev_text, curr_text
        
        # Merge the word
        merged_word = word_part1 + word_part2.rstrip('.,;:!?')
        
        # Validate: merged word should be reasonably long and alphanumeric
        if len(merged_word) < 3:
            self._stats["merges_skipped"] += 1
            return prev_text, curr_text
        
        # Check if it looks like a valid word (simple heuristic)
        if not self._is_likely_word(merged_word):
            self._stats["merges_skipped"] += 1
            return prev_text, curr_text
        
        # Perform the merge
        self._stats["merges_performed"] += 1
        
        # Reconstruct texts
        # prev_text: remove the hyphenated fragment, add merged word
        prev_words = prev_stripped[:-1].split()
        prev_words[-1] = merged_word
        new_prev = ' '.join(prev_words)
        
        # Preserve trailing whitespace from original
        if prev_text.endswith(' '):
            new_prev += ' '
        
        # curr_text: remove the merged part
        remaining_curr_words = curr_words[1:]
        new_curr = ' '.join(remaining_curr_words)
        
        # Preserve any punctuation that was attached to word_part2
        punct_match = re.search(r'^[a-zA-Z]+([.,;:!?]+)', curr_words[0])
        if punct_match and remaining_curr_words:
            pass  # Punctuation already in remaining
        elif punct_match:
            new_prev += punct_match.group(1)
        
        return new_prev, new_curr
    
    def _is_likely_word(self, word: str) -> bool:
        """
        Simple heuristic to check if merged string looks like a valid word.
        """
        # Must be alphabetic (allow some internal hyphens for compound words)
        clean = word.replace('-', '')
        if not clean.isalpha():
            return False
        
        # Shouldn't have too many consecutive consonants or vowels
        vowels = set('aeiouAEIOU')
        consonant_run = 0
        vowel_run = 0
        max_consonant_run = 0
        max_vowel_run = 0
        
        for char in clean:
            if char in vowels:
                vowel_run += 1
                max_consonant_run = max(max_consonant_run, consonant_run)
                consonant_run = 0
            else:
                consonant_run += 1
                max_vowel_run = max(max_vowel_run, vowel_run)
                vowel_run = 0
        
        max_consonant_run = max(max_consonant_run, consonant_run)
        max_vowel_run = max(max_vowel_run, vowel_run)
        
        # Reject if implausible consonant/vowel sequences
        if max_consonant_run > 5 or max_vowel_run > 4:
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """Return repair statistics."""
        return self._stats.copy()
