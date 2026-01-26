import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LogicBook:
    """
    Represents the document's logical structure derived from Docling's Markdown output.
    Used to verify and correct the structural hierarchy of segments.
    """
    def __init__(self, markdown_text: str):
        self.headings: List[Dict[str, Any]] = []
        self.text_to_path: Dict[str, str] = {}
        if markdown_text:
            self._parse_markdown(markdown_text)

    def _parse_markdown(self, text: str):
        """Parse markdown headings to build a hierarchical skeleton with aggressive noise filtering."""
        # Pattern to match # Heading, ## Subheading, etc.
        pattern = re.compile(r'^(#+)\s+(.*)$', re.MULTILINE)
        
        # Noise patterns that should NEVER be considered a structural section
        self.noise_regex = re.compile('|'.join([
            r'^\d+[\s\d]+[A-Z]*[\s\d]+',       # Printing strings like "1 2 3 4 5 LWI..."
            r'^[A-Z]$',                         # Single letter index categories
            r'^(Table|Figure|Exhibit|Equation)\s+\d+', # Misclassified captions
            r'^(University of|Boston College|McGraw Hill|Printed in|All rights reserved)', 
            r'^(Index|Name Index|Subject Index|Glossary)$', 
            r'^[0-9\.\s]+$',                     # Just numbers and dots
            r'^(E-INVESTMENTS|EXERCISES|Concept\s+Check|Summary|Key\s+Terms|Solutions|Web\s+Resources)', # Sidebars/End of chapter
            r'^(Source:|Concept Check|Visit\s+us\s+at)', # Attribution noise
        ]), re.IGNORECASE)
        
        raw_headings = []
        for match in pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # AGGRESSIVE FILTERING - Use search instead of match for better coverage
            if self.is_noise(title) or len(title) < 2:
                continue
                
            # Clean up markdown artifacts
            title = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', title)
            title = title.replace('**', '').replace('__', '')
            
            raw_headings.append({
                "level": level,
                "title": title,
                "path": ""
            })

        # Build paths using a stack
        stack = []
        for h in raw_headings:
            while stack and stack[-1]["level"] >= h["level"]:
                stack.pop()
            stack.append(h)
            
            h["path"] = " > ".join([item["title"] for item in stack])
            
            norm_title = self._normalize(h["title"])
            # Only index long enough titles to avoid collision
            if norm_title not in self.text_to_path or h["level"] < 3:
                self.text_to_path[norm_title] = h["path"]

        self.headings = raw_headings
        logger.info(f"LogicBook: Selected {len(self.headings)} structural headings (Filtered out potential noise).")

    def _normalize(self, text: str) -> str:
        """Normalize text for matching (lowercase, strip, remove non-alphanumeric)."""
        if not text: return ""
        text = text.lower().strip()
        return re.sub(r'[^a-z0-9]', '', text)

    def is_noise(self, text: str) -> bool:
        """Check if the text matches any noise heading patterns."""
        if not text: return True
        return bool(self.noise_regex.search(text))

    def get_path(self, segment_text: str) -> Optional[str]:
        """
        Get the logical path for a segment if it matches a known heading.
        """
        norm_text = self._normalize(segment_text)
        # 1. Try exact match on normalized text
        if norm_text in self.text_to_path:
            return self.text_to_path[norm_text]
            
        # 2. Try partial match if the segment starts with a known heading
        # (Handles cases where Parser adds page numbers or noise to header segments)
        for anchor_norm, path in self.text_to_path.items():
            if len(anchor_norm) > 10 and (norm_text.startswith(anchor_norm) or anchor_norm.startswith(norm_text)):
                return path
                
        return None

    def get_level(self, segment_text: str) -> Optional[int]:
        """Get the logical level (1, 2, 3...) of a heading."""
        norm_text = self._normalize(segment_text)
        for h in self.headings:
            if self._normalize(h["title"]) == norm_text:
                return h["level"]
        return None
