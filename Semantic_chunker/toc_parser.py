import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TOCEntry:
    level: int
    title: str
    page: int
    raw_text: str

class TOCParser:
    """
    Parses Table of Contents from document segments to build a structural skeleton.
    """
    def __init__(self):
        self.toc_entries: List[TOCEntry] = []
    
    def parse(self, segments: List[Dict], toc_pages: set) -> List[TOCEntry]:
        """
        Parse TOC entries from the raw segments of detected TOC pages.
        """
        if not toc_pages:
            return []
            
        logger.info(f"Parsing TOC from pages: {sorted(list(toc_pages))}")
        entries = []
        
        # Sort segments by page then reading order
        toc_segments = [s for s in segments if s.get('page', 0) in toc_pages]
        toc_segments.sort(key=lambda x: (x.get('page', 0), x['bbox'][1])) # roughly top to bottom
        
        # 1. Regex Patterns
        # Line ending in number: "Chapter 1 Introduction ... 5"
        # Or: "1. Introduction 5"
        # Dotted leaders are strong signal: "...... 5"
        line_pattern = re.compile(r'^(.*?)(?:[\.·\-_]{3,}|[\s]{2,})(\d+)$')
        
        for seg in toc_segments:
            text = seg.get('text', '').strip()
            # Clean up furniture artifacts
            if len(text) < 3: continue
            
            match = line_pattern.search(text)
            if match:
                title_part = match.group(1).strip()
                page_part = match.group(2)
                try:
                    page_num = int(page_part)
                except ValueError:
                    continue
                
                # Infer Level
                level = self._infer_level(title_part, seg)
                
                # Create Entry
                entry = TOCEntry(
                    level=level,
                    title=title_part,
                    page=page_num,
                    raw_text=text
                )
                entries.append(entry)
        
        self.toc_entries = entries
        logger.info(f"Parsed {len(entries)} TOC entries.")
        return entries

    def _infer_level(self, title: str, segment: Dict) -> int:
        """Infer hierarchy level specific to TOC indendation/styling."""
        # 1. Check direct numbering: "PART I", "CHAPTER 1", "2.1"
        title_upper = title.upper()
        if title_upper.startswith("PART") or title_upper.startswith("MODULE"):
            return 0
        if title_upper.startswith("CHAPTER") or re.match(r'^\d+\.?\s+[A-Z]', title):
            return 1
        
        # 2. Check Visual Indentation (bbox x0)
        x0 = segment['bbox'][0]
        # Heuristic: < 60 is level 0, 60-90 level 1, >90 level 2
        # This is document specific, might need normalization
        if x0 < 60: return 0
        if x0 < 90: return 1
        return 2

    def detect_pages(self, segments: List[Dict]) -> set:
        """Detect pages that are likely Table of Contents."""
        toc_pages = set()
        page_texts = defaultdict(list)
        for seg in segments:
            p = seg.get('page', 0)
            page_texts[p].append(seg.get('text', '').lower())
        
        toc_keywords = {
            'contents', 'table of contents', 'index', '目录', 
            'brief contents', 'detailed contents', 'summary table of contents'
        }
        for p, texts in page_texts.items():
            full_text = " ".join(texts)
            if any(kw in full_text for kw in toc_keywords) and p < 60:
                toc_pages.add(p)
                continue
            if p < 45: 
                dotted_lines = sum(1 for t in texts if re.search(r'[\.·-]{5,}\s*\d+$', t))
                if dotted_lines >= 3:
                    toc_pages.add(p)
        return toc_pages

    def apply_skeleton(self, segments: List[Dict], toc_entries: List[TOCEntry]) -> List[Dict]:
        """
        Enforce TOC hierarchy on segments based on page ranges.
        """
        if not toc_entries:
            return segments
            
        sorted_entries = sorted(toc_entries, key=lambda x: x.page)
        page_to_context = {}
        
        for i, entry in enumerate(sorted_entries):
            start_page = entry.page
            if i < len(sorted_entries) - 1:
                end_page = sorted_entries[i+1].page - 1
            else:
                end_page = 99999 
            
            if start_page > end_page: end_page = start_page 
            context_path = entry.title
            
            for p in range(start_page, end_page + 1):
                if entry.level <= 1:
                    page_to_context[p] = context_path
        
        updates = 0
        for seg in segments:
            p = seg.get('page', 0)
            if p in page_to_context:
                toc_context = page_to_context[p]
                current_path = seg.get('heading_path', '')
                if toc_context not in current_path:
                    if not current_path:
                        seg['heading_path'] = toc_context
                    else:
                        seg['heading_path'] = f"{toc_context} > {current_path}"
                    updates += 1
        
        logger.info(f"TOC Seeding: Updated hierarchy for {updates} segments.")
        return segments

    def get_skeleton(self) -> List[Dict]:
        """Convert parsed entries into a hierarchical dictionary skeleton."""
        return [vars(e) for e in self.toc_entries]
