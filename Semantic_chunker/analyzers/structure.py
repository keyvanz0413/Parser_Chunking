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
        
        # Stage 2: Heuristic Scanning (Text-based)
        if force_stage is None or force_stage == 2:
            logger.info("Structure: Attempting Stage 2 (Heuristic Text Scan)...")
            heuristic_entries = self._stage_2_heuristic()
            if len(heuristic_entries) > 5:
                logger.info(f"Structure: Heuristic TOC hit with {len(heuristic_entries)} entries.")
                return {
                    "source": "heuristic_text",
                    "data": heuristic_entries
                }

        # Stage 3: Vision Fallback / Artifact Capture
        if force_stage is None or force_stage == 3:
            logger.info("Structure: Attempting Stage 3 (Vision Fallback / Artifact Capture)...")
            artifacts = self._capture_toc_candidates()
            if artifacts:
                return {
                    "source": "vision_pending", 
                    "data": [], 
                    "artifacts": artifacts,
                    "message": "Automatic extraction failed. Candidate TOC pages captured for Vision AI/Manual review."
                }
        
        return {"source": "fallback", "data": [], "message": "No TOC found in any stage."}

    def _capture_toc_candidates(self, max_pages: int = 40) -> List[str]:
        """
        Identifies likely TOC pages and saves them as images for manual/AI review.
        Returns paths to saved images.
        """
        candidates = []
        toc_keywords = ["contents", "table of contents", "index", "brief contents"]
        
        artifact_dir = Path("outputs/TOC_Artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        for page_num in range(min(max_pages, len(self.doc))):
            page = self.doc[page_num]
            text = page.get_text().lower()
            
            # Simple scoring for TOC candidate
            score = 0
            if any(kw in text[:1000] for kw in toc_keywords): score += 5
            if len(re.findall(r"[\.·\d]{5,}", text)) > 3: score += 5
            
            if score >= 5:
                img_path = artifact_dir / f"toc_candidate_p{page_num+1}.png"
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Zoom for better readability
                pix.save(str(img_path))
                candidates.append(str(img_path))
                logger.info(f"Structure: Captured TOC candidate image: {img_path}")
                
        return candidates

    def _stage_2_heuristic(self, max_pages: int = 35) -> List[Dict[str, Any]]:
        """
        Stage 2: Heuristic Scanning for TOC pages.
        Uses enhanced regex and multiline buffer logic from sandbox.
        """
        toc_entries = []
        # Robust TOC pattern: Title followed by optional dots/spaces and a page number
        # Supports special whitespace characters and varying dot patterns
        toc_pattern = re.compile(r"^(.*?)(?:[\s\.·\u00A0\u2001\u2003]{3,})?\s*([ivxIVX\d]+)$", re.MULTILINE)
        toc_keywords = ["contents", "table of contents", "index", "brief contents", "detailed contents"]
        
        current_buffer = []
        
        for page_num in range(min(max_pages, len(self.doc))):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Check if this looks like a TOC page
            page_text_lower = text.lower()
            is_toc_indicator = any(kw in page_text_lower[:1000] for kw in toc_keywords)
            
            # Find lines that match the pattern
            matches = toc_pattern.findall(text)
            
            if len(matches) > 5 or (is_toc_indicator and len(matches) > 2):
                logger.debug(f"Structure: Detected potential TOC on page {page_num+1} ({len(matches)} entries)")
                
                for title, target_page in matches:
                    clean_title = title.strip()
                    if not clean_title or len(clean_title) < 2:
                        continue
                        
                    # Basic depth inference based on leading whitespace in original line
                    # (PyMuPDF's get_text() with default flags might lose indentation, 
                    # but we can try to find the line in the text)
                    depth = 1
                    if clean_title.startswith("   ") or clean_title.startswith("\t"):
                        depth = 2
                    
                    toc_entries.append({
                        "level": depth,
                        "title": clean_title,
                        "page": target_page, # Keep original (could be roman)
                        "source_page": page_num + 1
                    })
                    
        return toc_entries

    def _classify_sections(self, entries: List[Dict]) -> Dict[str, Any]:
        """Heuristically classifies PDF into sections and extracts top-level chapter ranges."""
        body_start_keywords = [r"^chapter\s+1\b", r"^part\s+i\b", r"^1\s+introduction", r"^1\.1\b"]
        back_matter_keywords = [r"appendix", r"glossary", r"index", r"bibliography", r"references", r"name index", r"subject index"]
        
        body_start_page = 1
        back_matter_start_page = 9999
        glossary_range = None
        found_body = False
        
        chapters = []
        
        for i, entry in enumerate(entries):
            title = entry["title"]
            title_lower = title.lower().strip()
            
            try:
                page = int(entry["page"])
            except:
                continue 
            
            # Detect Body Start
            if not found_body and any(re.search(pat, title_lower) for pat in body_start_keywords):
                if page < 150:
                    body_start_page = page
                    found_body = True

            # Detect Back Matter Start
            if found_body and page > body_start_page + 100:
                if any(re.search(pat, title_lower) for pat in back_matter_keywords):
                    if entry.get("level", 1) == 1 and page < back_matter_start_page:
                        back_matter_start_page = page

            # Track top-level chapters for boundary logic
            if entry.get("level") == 1:
                chapters.append({"title": title, "start_page": page})

            if "glossary" in title_lower:
                glossary_range = {"start": page, "end": page + 15}
                for next_entry in entries[i+1:]:
                    if next_entry.get("level", 1) == 1:
                        try:
                            glossary_range["end"] = int(next_entry["page"]) - 1
                            break
                        except: continue

        # Calculate end pages for chapters
        for i in range(len(chapters)):
            if i < len(chapters) - 1:
                chapters[i]["end_page"] = chapters[i+1]["start_page"] - 1
            else:
                chapters[i]["end_page"] = len(self.doc) if self.doc else 2000

        return {
            "front_matter": [1, body_start_page - 1],
            "body": [body_start_page, back_matter_start_page - 1],
            "back_matter": [back_matter_start_page, len(self.doc) if self.doc else 2000],
            "glossary": [glossary_range["start"], glossary_range["end"]] if glossary_range else None,
            "chapters": chapters
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
