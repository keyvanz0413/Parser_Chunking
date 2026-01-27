import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .config import ChunkingConfig
from .models import EnrichedChunk, Reference
from ..analyzers.semantic import SemanticAnalyzer
from ..analyzers.discourse import POSAnalyzer
from ..analyzers.tagging import TagDetector
from ..detectors.structure.furniture import FurnitureDetector
from ..detectors.structure.sidebar import SidebarDetector
from ..detectors.structure.reading_order import ReadingOrderCorrector
from ..detectors.logic.continuation import ContinuationDetector
from ..detectors.logic.bonding import CaptionBondingHelper
from ..utils.dehyphenation import DehyphenationHelper
from ..utils.ref_resolver import ReferenceDetector

logger = logging.getLogger(__name__)

class LogicSegmenter:
    """Main Orchestrator for the Sbert Chunking Pipeline."""
    
    def __init__(self, use_pos: bool = True, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        
        # Analyzers
        self.tag_detector = TagDetector()
        self.pos_analyzer = POSAnalyzer(config=self.config) if use_pos else None
        self.semantic_analyzer = SemanticAnalyzer(self.config) if self.config.ENABLE_SEMANTIC_ANALYSIS else None
        
        # Detectors
        self.furniture_detector = FurnitureDetector(self.config)
        self.sidebar_detector = SidebarDetector(self.config)
        self.reading_order_corrector = ReadingOrderCorrector(self.config)
        self.continuation_detector = ContinuationDetector(self.config, semantic_analyzer=self.semantic_analyzer)
        self.caption_bonding_helper = CaptionBondingHelper(self.config)
        
        # Utils
        self.reference_detector = ReferenceDetector(self.config)
        self.dehyphenation_helper = DehyphenationHelper(self.config)
        
        self.chunk_counter = 0
        self.block_catalog = None

    def process_file(self, input_path: str, output_path: str = None) -> Dict:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = data.get('flat_segments', [])
        
        # 1. Structure Analysis
        if self.config.ENABLE_FURNITURE_DETECTION:
            self.furniture_detector.scan_document(segments)
            segments = [s for s in segments if not self.furniture_detector.is_furniture(s)]
            
        if self.config.ENABLE_SIDEBAR_DETECTION:
            self.sidebar_detector.scan_document(segments)
            segments = self.sidebar_detector.annotate_segments(segments)
            
        segments = self.reading_order_corrector.process(segments)
        
        # 2. Logic Annotation
        segments = self.continuation_detector.annotate_segments(segments)
        
        # --- NEW: Dehyphenation Layer ---
        # Run dehyphenation AFTER continuation tagging to ensure we know which segments connect
        self.dehyphenation_helper.process_segments(segments)
        
        self.block_catalog = self.reference_detector.build_block_catalog(segments)
        
        # 3. Chunking logic (Enhanced with Stitching)
        chunks = self._group_into_chunks(segments)
        
        # 4. Finalizing
        result = {
            "metadata": {
                "total_chunks": len(chunks),
                "stats": self._get_combined_stats()
            },
            "chunks": [asdict(c) for c in chunks]
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def _group_into_chunks(self, segments: List[Dict]) -> List[EnrichedChunk]:
        """Main Loop for grouping segments into logical chunks."""
        chunks = []
        buffer = []
        
        for i in range(len(segments)):
            seg = segments[i]
            if not buffer:
                buffer.append(seg)
                continue
            
            prev_seg = buffer[-1]
            is_continuation = seg.get('is_continuation') in ['full', 'partial']
            
            # --- Hard Break Detection ---
            h_break = False
            h_reason = ""
            
            # 1. Heading Path Transition (REDUCED PRIORITY IF CONTINUATION)
            if seg.get('heading_path') != prev_seg.get('heading_path'):
                if is_continuation:
                    # Stitching: Ignore heading change if physical continuation is strong
                    h_break = False 
                else:
                    h_break = True
                    h_reason = "heading_change"
                
            # 2. Sidebar Transition
            if not h_break:
                if seg.get('is_sidebar') != prev_seg.get('is_sidebar'):
                    h_break = True
                    h_reason = "sidebar_isolation"
                elif seg.get('is_sidebar') and seg.get('sidebar_type') != prev_seg.get('sidebar_type'):
                    h_break = True
                    h_reason = "sidebar_type_change"
                
            # 3. Structural Isolation (Tables/Images/Formulas)
            if not h_break and not is_continuation:
                if seg.get('type') in ['Table', 'Picture', 'Formula']:
                    h_break = True
                    h_reason = f"structure_{seg.get('type').lower()}_start"
                elif prev_seg.get('type') in ['Table', 'Picture', 'Formula']:
                    h_break = True
                    h_reason = f"structure_{prev_seg.get('type').lower()}_end"

            # --- Soft Break (Semantic) ---
            s_break = False
            if not h_break and not is_continuation and len(buffer) >= 1:
                if self.config.ENABLE_SEMANTIC_CHUNKING and self.semantic_analyzer:
                    prev_text = prev_seg.get('text', '')
                    curr_text = seg.get('text', '')
                    if prev_text.strip() and curr_text.strip():
                        sim = self.semantic_analyzer.compute_cross_page_semantic_score(prev_text, curr_text)
                        if sim < self.config.SEMANTIC_SIMILARITY_THRESHOLD:
                            s_break = True

            if h_break:
                chunks.append(self._create_chunk_from_buffer(buffer, split_reason=h_reason))
                buffer = [seg]
            elif s_break:
                chunks.append(self._create_chunk_from_buffer(buffer, split_reason="semantic_break"))
                buffer = [seg]
            elif is_continuation:
                # MARKING JOIN POINT FOR SPACY
                # Insert a [PAGE_JOIN] marker between segments if they are strong continuations
                buffer.append(seg)
            elif len(buffer) < self.config.MAX_BUFFER_SEGMENTS:
                buffer.append(seg)
            else:
                chunks.append(self._create_chunk_from_buffer(buffer, split_reason="max_buffer"))
                buffer = [seg]
        
        if buffer:
            chunks.append(self._create_chunk_from_buffer(buffer))
        return chunks

    def _create_chunk_from_buffer(self, buffer: List[Dict], split_reason: str = "none") -> EnrichedChunk:
        self.chunk_counter += 1
        
        # Stitch text with markers if they are continuations
        content_parts = []
        for i in range(len(buffer)):
            if i > 0 and buffer[i].get('is_continuation') in ['full', 'partial']:
                content_parts.append("[PAGE_JOIN]")
            content_parts.append(buffer[i].get('text', ''))
            
        content = " ".join(content_parts)
        # Clean up double joins and spaces
        content = re.sub(r' +', ' ', content).strip()
        
        # Metadata extraction
        page_range = [min(s.get('page', 0) for s in buffer), max(s.get('page', 0) for s in buffer)]
        
        # POS & Semantic Enrichment
        sentences = []
        if self.pos_analyzer:
            sentences = self.pos_analyzer.analyze_sentences(
                content, 
                semantic_analyzer=self.semantic_analyzer
            )
            
        coherence = 1.0
        if self.semantic_analyzer:
            coherence = self.semantic_analyzer.compute_chunk_coherence([s['text'] for s in sentences])

        # Tags detection
        tags = self.tag_detector.detect_tags(content)
        
        # Calculate role distribution
        from collections import Counter
        roles = [s.get('role') for s in sentences]
        role_dist = dict(Counter(roles))

        return EnrichedChunk(
            chunk_id=f"chunk_{self.chunk_counter:04d}",
            heading_path=buffer[0].get('heading_path', ''),
            chunk_type=self._infer_chunk_type(buffer, tags, sentences),
            content=content,
            sentences=sentences,
            tags=tags,
            page_range=page_range,
            semantic_coherence=round(coherence, 3),
            is_sidebar=buffer[0].get('is_sidebar', False),
            sidebar_type=buffer[0].get('sidebar_type'),
            role_distribution=role_dist,
            split_reason=split_reason,
            word_count=len(content.split())
        )

    def _infer_chunk_type(self, buffer: List[Dict], tags: List[str], sentences: List[Dict]) -> str:
        """从片段类型、标签和句子角色分片中推断分块类型"""
        # 0. Sidebar prioritized
        if any(s.get('is_sidebar') for s in buffer):
            sidebar_type = buffer[0].get('sidebar_type')
            if sidebar_type == 'learning_objective': 
                return 'learning_objective'
            return 'sidebar'

        # 1. 结构化优先
        seg_types = [s.get('type') for s in buffer]
        if 'Table' in seg_types: return 'table'
        if 'Picture' in seg_types: return 'picture'
        if 'Formula' in seg_types: return 'formula'
        if 'ListItem' in seg_types: return 'list'
        
        # 2. 标签/角色次之
        if 'definition' in tags: return 'definition'
        if 'procedure' in tags: return 'procedure'
        
        # 3. 句子角色分布
        roles = [s.get('role') for s in sentences]
        if not roles: return 'explanation'
        
        from collections import Counter
        counts = Counter(roles)
        
        # Priority mapping for chunk types
        if counts.get('example', 0) > 0: return 'example'
        if counts.get('question', 0) > 0: return 'question'
        if counts.get('definition', 0) > 0: return 'definition'
        
        most_common = counts.most_common(1)[0][0]
        return most_common if most_common != 'explanation' else 'explanation'

    def _get_combined_stats(self) -> Dict:
        return {
            "reading_order": self.reading_order_corrector.get_stats(),
            "furniture": self.furniture_detector.get_stats(),
            "sidebar": self.sidebar_detector.get_stats()
        }
