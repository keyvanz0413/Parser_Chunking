import fitz  # PyMuPDF
import re
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BookStructureAnalyzer:
    """
    Unified analyzer for PDF structure using the 3-stage funnel:
    1. Metadata Bookmarks
    2. Heuristic TOC Scanning
    3. Vision Fallback (Mocked)
    
    Plus Heuristic Section Classification (Front/Body/Back matter).
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None
        if os.path.exists(pdf_path):
            self.doc = fitz.open(pdf_path)
            
        self.toc_data = []
        self.structure_map = {}

    def analyze(self) -> Dict[str, Any]:
        """Runs the full analysis pipeline."""
        if not self.doc:
            return {"error": "PDF file not found"}

        # 1. Extract TOC
        toc_result = self._extract_toc()
        self.toc_data = toc_result["data"]
        
        # 2. Classify Sections
        section_map = self._classify_sections(self.toc_data)
        
        return {
            "toc": toc_result,
            "sections": section_map
        }

    def _extract_toc(self, force_stage: Optional[int] = None) -> Dict[str, Any]:
        # Stage 1: Metadata
        if force_stage is None or force_stage == 1:
            toc = self.doc.get_toc()
            if len(toc) > 10:
                logger.info(f"Structure: Metadata TOC hit with {len(toc)} entries.")
                return {
                    "source": "metadata",
                    "data": [{"level": e[0], "title": e[1], "page": int(e[2])} for e in toc]
                }
        
        # Stage 2: Heuristics
        if force_stage is None or force_stage == 2:
            toc = self._stage_2_heuristic()
            if toc:
                logger.info(f"Structure: Heuristic TOC hit with {len(toc)} entries.")
                return {"source": "heuristic", "data": toc}
        
        return {"source": "fallback", "data": []}

    def _stage_2_heuristic(self, max_pages: int = 25) -> List[Dict[str, Any]]:
        toc_entries = []
        page_pattern = re.compile(r"([\u2000-\u200B\.\s\t]{2,}|(?<=\s))([ivxIVX]+|\d+)$")
        toc_keywords = ["contents", "table of contents", "index", "brief contents"]
        
        for page_num in range(min(max_pages, len(self.doc))):
            page = self.doc[page_num]
            blocks = page.get_text("blocks")
            page_width = page.rect.width
            mid_x = page_width / 2
            
            left_col = [b for b in blocks if b[0] < mid_x]
            right_col = [b for b in blocks if b[0] >= mid_x]
            
            left_base = min([b[0] for b in left_col] or [0])
            right_base = min([b[0] for b in right_col] or [mid_x])
            
            sorted_blocks = sorted(left_col, key=lambda x: x[1]) + sorted(right_col, key=lambda x: x[1])
            
            page_text = page.get_text().lower()
            is_toc_page = any(kw in page_text[:1000] for kw in toc_keywords)
            
            page_entries = []
            for b in sorted_blocks:
                text = b[4].strip()
                if not text: continue
                
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                buffer_title = []
                
                for line in lines:
                    match = page_pattern.search(line)
                    is_valid_pnum = False
                    if match:
                        pnum_str = match.group(2)
                        prefix = match.group(1)
                        if '.' in prefix or len(prefix) >= 2 or (pnum_str.isdigit() and len(line) > 10):
                            is_valid_pnum = True

                    if is_valid_pnum:
                        current_title_part = line[:match.start()].strip()
                        pnum = match.group(2)
                        full_title = " ".join(buffer_title + ([current_title_part] if current_title_part else []))
                        buffer_title = [] 
                        
                        base_x = left_base if b[0] < mid_x else right_base
                        indent = b[0] - base_x
                        
                        level = 1
                        if indent > 30: level = 3
                        elif indent > 10: level = 2
                        
                        page_entries.append({
                            "level": level,
                            "title": full_title,
                            "page": pnum
                        })
                    else:
                        if len(line) > 2 and not re.match(r"^\d+$", line):
                            buffer_title.append(line)
            
            if is_toc_page or len(page_entries) > 3:
                toc_entries.extend(page_entries)
                
        return toc_entries

    def _classify_sections(self, entries: List[Dict]) -> Dict[str, Any]:
        """Heuristically classifies PDF into sections (Front, Body, Back)."""
        body_start_keywords = [r"^chapter\s+1\b", r"^part\s+i\b", r"^1\s+introduction", r"^1\.1\b"]
        back_matter_keywords = [r"appendix", r"glossary", r"index", r"bibliography", r"references"]
        
        body_start_page = 1
        back_matter_start_page = 9999
        glossary_range = None
        found_body = False
        
        # Sort entries by page to be safe
        # Some pages might be roman numerals, we need a way to sort them
        # For simplicity, we assume the TOC entries are mostly in order
        
        for i, entry in enumerate(entries):
            title = entry["title"].lower().strip()
            # Convert roman page to int if possible for comparison
            try:
                page = int(entry["page"])
            except:
                continue # Skip roman numerals for boundary detection
            
            if not found_body and any(re.search(pat, title) for pat in body_start_keywords):
                if page < 150:
                    body_start_page = page
                    found_body = True

            if found_body and page > body_start_page + 100:
                if any(re.search(pat, title) for pat in back_matter_keywords):
                    if entry.get("level", 1) == 1 and page < back_matter_start_page:
                        back_matter_start_page = page

            if "glossary" in title:
                glossary_range = {"start": page, "end": page + 15}
                for next_entry in entries[i+1:]:
                    if next_entry.get("level", 1) == 1:
                        try:
                            glossary_range["end"] = int(next_entry["page"]) - 1
                            break
                        except: continue

        return {
            "front_matter": [1, body_start_page - 1],
            "body": [body_start_page, back_matter_start_page - 1],
            "back_matter": [back_matter_start_page, len(self.doc) if self.doc else 2000],
            "glossary": [glossary_range["start"], glossary_range["end"]] if glossary_range else None
        }

    def get_heading_path(self, page_num: int, context_text: str = "") -> str:
        """
        Returns the gold heading path for a given page based on TOC.
        """
        if not self.toc_data:
            return ""
            
        stack = {}
        for entry in self.toc_data:
            try:
                e_page = int(entry['page'])
            except: continue
            
            if e_page <= page_num:
                lvl = entry['level']
                stack[lvl] = entry['title']
                for k in list(stack.keys()):
                    if k > lvl: del stack[k]
            else:
                break
        
        return " > ".join([stack[i] for i in sorted(stack.keys())])
