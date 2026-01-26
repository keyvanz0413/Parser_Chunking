import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GlossaryDetector:
    """
    Detects and parses Glossary pages to extract Term-Definition pairs.
    """
    def __init__(self):
        # Regex for Glossary entries:
        # Pattern 1: Bold Term followed by definition (relies on font weight if avail, else visual)
        # Pattern 2: Term ending with colon or newline, followed by indented definition
        # "Term \n Definition"
        # We'll use a heuristic for text-only extraction first
        self.term_pattern = re.compile(r'^([A-Z][\w\s\-\(\)]{3,50})[:\n]\s+([A-Z].+)', re.MULTILINE)

    def detect_glossary_pages(self, segments: List[Dict]) -> set:
        """Identify pages belonging to the Glossary."""
        glossary_pages = set()
        for s in segments:
            text = s.get('text', '').strip().lower()
            if text == 'glossary' or text == 'glossary of terms':
                glossary_pages.add(s.get('page', 0))
                # Heuristic: Glossary is usually contiguous at the end. 
                # We assume contiguous pages following this title are also glossary.
        
        if not glossary_pages:
            return set()
            
        # Expand to subsequent pages until we hit Index or End
        start_page = min(glossary_pages)
        max_page = max(s.get('page', 0) for s in segments)
        
        current = start_page
        while current < max_page:
            glossary_pages.add(current)
            current += 1
            # Stop if we see "Index" header on this page
            page_text = " ".join([s.get('text', '').lower() for s in segments if s.get('page') == current])
            if 'index' in page_text[:100]: # Check top of page
                break
                
        return glossary_pages

    def parse(self, segments: List[Dict], glossary_pages: set) -> List[Dict]:
        """
        Extract terms and definitions from glossary pages.
        Returns list of dicts: {'term': str, 'definition': str, 'page': int}
        """
        if not glossary_pages:
            return []
            
        items = []
        glossary_segments = [s for s in segments if s.get('page', 0) in glossary_pages]
        glossary_segments.sort(key=lambda x: (x['page'], x['bbox'][1]))
        
        # Simple finite state machine
        current_term = None
        current_def = []
        
        for seg in glossary_segments:
            text = seg.get('text', '').strip()
            if not text or len(text) < 2: continue
            
            # Skip Headers
            if text.lower() == 'glossary': continue
            if len(text) == 1 and text.isalpha(): continue # A, B, C headers
            
            # Check for Term
            # Layout heuristic: Terms often start lines and are short (< 50 chars)
            # Definitions are longer.
            # OR Term matches regex
            
            # Heuristic: If text contains ": ", split
            if ': ' in text:
                term, definition = text.split(': ', 1)
                items.append({'term': term.strip(), 'definition': definition.strip(), 'page': seg['page']})
                current_term = None
                continue
            
            # Heuristic: Bold term (if style info avail - assuming Docling provides 'bold' in style/font?)
            # Docling segments might not have style info in this simplified dict.
            # Using length heuristic.
            
            if len(text) < 60 and not text.endswith('.'):
                # Likely a term
                # Save previous
                if current_term and current_def:
                    items.append({'term': current_term, 'definition': " ".join(current_def), 'page': seg.get('page', 0)})
                current_term = text
                current_def = []
            else:
                # Likely definition part
                if current_term:
                    current_def.append(text)
        
        # Flush last
        if current_term and current_def:
            items.append({'term': current_term, 'definition': " ".join(current_def), 'page': 0})
            
        logger.info(f"Extracted {len(items)} glossary terms.")
        return items


class IndexParser:
    """
    Parses Index pages to create reference map.
    """
    def parse(self, segments: List[Dict]) -> Dict[str, List[int]]:
        # Index parsing is complex due to multi-column and "term, 12, 14-16" format.
        # Placeholder for now.
        return {}
