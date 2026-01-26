import re
import logging
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Environment tracking for PDFMetadataExtractor
HAS_METADATA_EXTRACTOR = False
try:
    _src_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "synapta-ds" / "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from etl.extractors.pdf_metadata_extractor import PDFMetadataExtractor
    HAS_METADATA_EXTRACTOR = True
except (ImportError, Exception) as e:
    logger.debug(f"Metadata manager: PDFMetadataExtractor integration disabled: {e}")

class MetadataManager:
    """
    Decoupled manager for book-level metadata extraction.
    Strictly enforces ISBN-based API lookup.
    """
    
    def __init__(self):
        self.isbn_patterns = [
            re.compile(r'ISBN(?:-13|-10)?[:\s]+(?:\d{1,5}[-\s]?)?(\d+[-\s]?\d+[-\s]?\d+[-\s]?\d+[-\s]?\d+)', re.IGNORECASE),
            re.compile(r'ISBN\s+([0-9-]{10,25})', re.IGNORECASE)
        ]

    def extract_strict(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Scan segments for ISBN and fetch metadata from authority API.
        Raises RuntimeError if ISBN is missing or API lookup fails.
        """
        if not segments:
            raise RuntimeError("Metadata extraction failed: No segments provided.")

        # 1. Scan for ISBN in the first 30 pages
        front_segments = [s for s in segments if s.get('page', 0) <= 30]
        isbn = self._scan_for_isbn(front_segments)

        if not isbn:
            raise RuntimeError("CRITICAL ERROR: No ISBN detected in the first 30 pages. "
                             "Book metadata cannot be retrieved via API.")

        if not HAS_METADATA_EXTRACTOR:
            raise RuntimeError("CRITICAL ERROR: PDFMetadataExtractor is not available in the environment.")

        # 2. API-Only Lookup
        try:
            logger.info(f"MetadataManager: Targeting API lookup for ISBN: {isbn}")
            extractor = PDFMetadataExtractor(enable_api_lookup=True)
            
            # Use direct API fetcher to avoid visual fallbacks
            api_data = extractor._fetch_from_api(isbn)
            
            if not api_data or not api_data.title:
                raise RuntimeError(f"CRITICAL ERROR: External API (Google Books/Crossref) returned no records for ISBN {isbn}. "
                                 "Manual metadata entry or ISBN verification required.")
            
            logger.info(f"✓ MetadataManager: Success - '{api_data.title}'")
            return api_data.to_dict()

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            raise RuntimeError(f"CRITICAL ERROR: Metadata API lookup failed: {str(e)}")

    def _scan_for_isbn(self, segments: List[Dict]) -> Optional[str]:
        """Scan through segments to find the first valid ISBN."""
        for s in segments:
            raw_text = s.get('text', '').strip()
            text = self.normalize_spaced_text(raw_text)
            for p in self.isbn_patterns:
                match = p.search(text)
                if match:
                    # Clean the ISBN
                    return match.group(1).replace('-', '').replace(' ', '')
        return None

    @staticmethod
    def normalize_spaced_text(text: str) -> str:
        """
        Normalize text that has been parsed with excessive spaces (e.g., from artistic fonts).
        Example: "I N V E S T M E N T S" -> "INVESTMENTS"
        """
        if not text or len(text) < 5:
            return text
            
        if re.search(r'[A-Za-z]\s[A-Za-z]\s[A-Za-z]', text):
            words = text.split()
            if len(words) >= 3:
                single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
                if single_char_ratio > 0.6:
                    # Look for clusters separated by 3 or more spaces (real boundaries)
                    parts = re.split(r'\s{3,}', text)
                    if len(parts) > 1:
                        normalized_parts = ["".join(p.split()) for p in parts]
                        return " ".join(normalized_parts)
                    else:
                        return "".join(words)
        return text
