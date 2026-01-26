import re
import logging
from typing import List, Dict, Any, Optional
from ..schema import EnrichedChunk
from ..utils.dehyphenation import DehyphenationHelper

logger = logging.getLogger(__name__)

class ChunkFactory:
    """
    Handles the creation and enrichments of chunks from raw segments.
    Decouples chunk assembly, type inference, and structural analysis.
    """
    
    def __init__(self, config, tag_detector, pos_analyzer=None, reference_detector=None):
        self.config = config
        self.tag_detector = tag_detector
        self.pos_analyzer = pos_analyzer
        self.reference_detector = reference_detector
        self.chunk_counter = 0
        self.block_catalog = None

    def set_block_catalog(self, catalog):
        self.block_catalog = catalog

    def create_chunk(self, segments: List[Dict], heading_path: str, 
                     chunk_type: Optional[str] = None) -> EnrichedChunk:
        """Create an enriched chunk from a list of segments."""
        self.chunk_counter += 1
        chunk_id = f"chunk_{self.chunk_counter:04d}"
        
        # Combine text - exclude furniture/noise from content
        text_parts = []
        for s in segments:
            if s.get('is_furniture', False) or s.get('is_noise_header', False):
                continue
            text = s.get('text', '').strip()
            if text:
                text_parts.append(text)
        
        # Apply dehyphenation if enabled
        if self.config.ENABLE_DEHYPHENATION and len(text_parts) > 1:
            repaired_parts = []
            dehyph = DehyphenationHelper(self.config)
            
            for i, part in enumerate(text_parts):
                if i == 0:
                    repaired_parts.append(part)
                else:
                    prev_part = repaired_parts[-1]
                    new_prev, new_curr = dehyph.merge_hyphenated(prev_part, part)
                    repaired_parts[-1] = new_prev
                    if new_curr:
                        repaired_parts.append(new_curr)
            
            text_parts = repaired_parts
        
        full_text = " ".join(text_parts).strip()
        word_count = len(full_text.split())
        source_ids = [s.get('segment_id', '') for s in segments if s.get('segment_id')]
        
        pages = [s.get('page', 0) for s in segments if s.get('page')]
        page_range = [min(pages), max(pages)] if pages else []
        depth = segments[0].get('depth', 0) if segments else 0
        
        # Detect tags
        tags = self.tag_detector.detect_tags(full_text)
        
        # Check for imperative start
        if self.tag_detector.detect_imperative(full_text):
            if "procedure" not in tags:
                tags.append("procedure")
        
        # Determine chunk type early
        if chunk_type is None:
            chunk_type = self.infer_chunk_type(segments, tags, [])
            
        forced_role = None
        if chunk_type == "header":
            forced_role = "topic"
        elif all(s.get('is_furniture', False) or s.get('is_noise_header', False) for s in segments):
            forced_role = "irrelevant"
            
        # Analyze sentences with POS
        sentences = []
        if self.pos_analyzer and full_text:
            sentences = self.pos_analyzer.analyze_sentences(
                full_text, heading_path, forced_role=forced_role, chunk_type=chunk_type
            )
            
            sentence_tag_set = set(tags)
            for sent in sentences:
                if "tags" in sent:
                    sentence_tag_set.update(sent["tags"])
            tags = list(sentence_tag_set)
        
        # Final type inference
        if chunk_type is None:
            chunk_type = self.infer_chunk_type(segments, tags, sentences)
        
        # Detect references
        references = []
        if self.reference_detector and self.block_catalog and full_text:
            chunk_page = page_range[0] if page_range else 0
            references = self.reference_detector.detect_references(
                full_text, chunk_page, self.block_catalog
            )
        
        # Zones and BBoxes
        zones = [s.get('doc_zone', 'body') for s in segments]
        dominant_zone = max(set(zones), key=zones.count) if zones else "body"
        
        page_bboxes = {}
        for s in segments:
            if s.get('is_furniture', False) or s.get('is_noise_header', False):
                continue
            p = s.get('page', 0)
            bbox = s.get('bbox', [])
            if p and bbox and len(bbox) == 4:
                if p not in page_bboxes:
                    page_bboxes[p] = list(bbox)
                else:
                    curr = page_bboxes[p]
                    page_bboxes[p] = [
                        min(curr[0], bbox[0]), min(curr[1], bbox[1]),
                        max(curr[2], bbox[2]), max(curr[3], bbox[3])
                    ]
        
        combined_bbox = []
        if page_bboxes:
            all_bboxes = list(page_bboxes.values())
            combined_bbox = [
                min(b[0] for b in all_bboxes), min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes), max(b[3] for b in all_bboxes)
            ]

        return EnrichedChunk(
            chunk_id=chunk_id,
            heading_path=heading_path,
            chunk_type=chunk_type,
            content=full_text,
            context_prefix="",
            sentences=sentences,
            tags=tags,
            source_segments=source_ids,
            page_range=page_range,
            depth=depth,
            word_count=word_count,
            references=references,
            bbox=combined_bbox,
            page_bboxes=page_bboxes,
            doc_zone=dominant_zone
        )
    
    def infer_chunk_type(self, segments: List[Dict], tags: List[str], 
                          sentences: List[Dict]) -> str:
        """Infer chunk type from segments, tags, and sentence analysis."""
        seg_types = [s.get('type', '') for s in segments]
        
        if all(t == 'ListItem' for t in seg_types): return "list"
        if all(t == 'LearningObjective' for t in seg_types): return "learning_objective"
        if "theorem" in tags: return "theorem"
        if "proof" in tags: return "proof"
        
        if sentences:
            imperative_count = sum(1 for s in sentences if s.get('is_imperative', False))
            if imperative_count > 0 and imperative_count >= len(sentences) / 2:
                return "procedure"
        
        priority = ["definition", "procedure", "example", "exercise", "formula", "summary"]
        for tag in priority:
            if tag in tags:
                return tag
        
        return "explanation"

    def infer_from_tags(self, tags: List[str]) -> str:
        """Infer chunk type from tags only."""
        priority = ["theorem", "proof", "definition", "procedure", "example", 
                    "exercise", "formula", "summary"]
        for tag in priority:
            if tag in tags: return tag
        return "explanation"
