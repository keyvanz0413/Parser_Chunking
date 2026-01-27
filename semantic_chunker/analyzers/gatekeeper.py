import re
import logging
from typing import List, Dict, Any, Optional
from ..config import ChunkingConfig

logger = logging.getLogger(__name__)

class ContentGatekeeper:
    """
    Intelligent document gating to identify main body content vs front/back matter.
    
    Uses sequential pattern recognition to distinguish between isolated references 
    in front matter and the actual start of structured chapters.
    """
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.start_id = None
        self.end_id = None
        self.detected_pattern = None

    def analyze(self, segments: List[Dict], detected_toc_pages: set = None) -> Dict[str, Any]:
        """
        Scan segments to identify logical start and end of main content.
        
        Algorithm:
        1. Identify level-1 headers.
        2. Filter out headers on known TOC pages or in high-density TOC-like zones.
        3. Enforce a minimum page gap after the first continuous block of TOC pages.
        4. Match against common chapter/section starting patterns.
        """
        all_toc_pages = (detected_toc_pages or set()).copy()
        l1_headers = [s for s in segments if s.get('type') == 'Header' and s.get('level') == 1]
        
        if not l1_headers:
            return {"start_id": None, "end_id": None}

        # Calculate the end of the INITIAL TOC/Preface block
        sorted_toc = sorted([p for p in all_toc_pages if p < self.config.GATING_SCAN_LIMIT_PAGE])
        effective_toc_end = 0
        if sorted_toc:
            effective_toc_end = sorted_toc[0]
            for j in range(1, len(sorted_toc)):
                if sorted_toc[j] - sorted_toc[j-1] > 8: # Gap > 8 pages likely means end of TOC
                    break
                effective_toc_end = sorted_toc[j]
        
        # Start scanning for body AFTER the initial TOC block
        safe_start_page = effective_toc_end + 1

        # 1. Detect Main Body Start
        best_start_idx = -1
        logger.info(f"Gatekeeper: Scanning {len(l1_headers)} L1 headers. TOC pages: {sorted(list(all_toc_pages))}. safe_start_page: {safe_start_page}")
        
        for i, header in enumerate(l1_headers):
            text = header.get('text', '').strip()
            page = header.get('page', 0)
            
            # Skip if on a known TOC page OR before the end of the initial TOC block
            if page in all_toc_pages or page < safe_start_page:
                logger.debug(f"Gatekeeper: Skipping page {page} (TOC/Front-Matter zone)")
                continue
                
            # Stop searching if we're too deep into the document
            if page > self.config.GATING_SCAN_LIMIT_PAGE:
                logger.debug(f"Gatekeeper: Skipping page {page} (> scan limit {self.config.GATING_SCAN_LIMIT_PAGE})")
                continue
                
            logger.info(f"Gatekeeper: Checking header '{text}' at page {page}")
            for pattern in self.config.MAIN_BODY_START_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    self.start_id = header.get('segment_id')
                    best_start_idx = i
                    self.detected_pattern = pattern
                    logger.info(f"Gatekeeper: Main Body start matches pattern '{pattern}' at page {page}")
                    break
            if self.start_id:
                break

        # 2. Detect Back Matter Start
        if self.start_id:
            # Look for back-matter patterns among headers following the start
            for header in l1_headers[best_start_idx + 1:]:
                text = header.get('text', '').strip()
                for pattern in self.config.BACK_MATTER_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        self.end_id = header.get('segment_id')
                        logger.info(f"Gatekeeper: Potential Back Matter start at '{text}' ({self.end_id})")
                        break
                if self.end_id:
                    break

        return {"start_id": self.start_id, "end_id": self.end_id}
