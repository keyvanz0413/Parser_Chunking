import json
import re
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from pathlib import Path

from .config import ChunkingConfig
from .schema import EnrichedChunk, SentenceRole, Reference, Edge, EdgeType
from .detectors.tags import TagDetector
from .detectors.continuation import ContinuationDetector
from .detectors.references import ReferenceDetector
from .analyzers.pos_analyzer import POSAnalyzer
from .analyzers.reading_order import ReadingOrderCorrector
from .analyzers.gatekeeper import ContentGatekeeper
from .utils.caption_bonding import CaptionBondingHelper
from .utils.caption_bonding import CaptionBondingHelper
from .utils.dehyphenation import DehyphenationHelper
from .utils.metadata_manager import MetadataManager
from .toc_parser import TOCParser
from .special_blocks import GlossaryDetector
from .analyzers.kg_linker import KGLinker
from .analyzers.chunk_factory import ChunkFactory
from .analyzers.logic_book import LogicBook
from .utils.metrics import MetricsCollector

from .analyzers.structure import BookStructureAnalyzer

logger = logging.getLogger(__name__)

class LogicSegmenter:
    """
    Main segmenter that processes flat_segments from parser_docling.py.
    """
    
    def __init__(self, use_pos: bool = True, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.tag_detector = TagDetector()
        self.pos_analyzer = POSAnalyzer() if use_pos else None
        self.continuation_detector = ContinuationDetector(self.config)
        self.reading_order_corrector = ReadingOrderCorrector(self.config)
        self.reference_detector = ReferenceDetector(self.config)
        self.caption_bonding_helper = CaptionBondingHelper(self.config)
        self.toc_parser = TOCParser()
        self.glossary_detector = GlossaryDetector()
        self.metadata_manager = MetadataManager()
        
        # New decoupled factory
        self.chunk_factory = ChunkFactory(
            self.config, self.tag_detector, self.pos_analyzer, self.reference_detector
        )
        
        self.structural_heading_path = ""
        self.structure_analyzer = None
    
    def process_file(self, input_json_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a parser_docling.py output file.
        """
        logger.info(f"Processing: {input_json_path}")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flat_segments = data.get('flat_segments', [])
        metadata = data.get('metadata', {})
        source_file = metadata.get('source_file', '')
        
        # Initialize Structure Analyzer if PDF exists
        # Navigate from Semantic_chunker/segmenter.py up to Parser_Chunking/inputs/
        base_path = Path(__file__).resolve().parent.parent
        pdf_path = base_path / "inputs" / source_file
        
        if pdf_path.exists():
            pdf_path_str = str(pdf_path)
            self.structure_analyzer = BookStructureAnalyzer(pdf_path_str)
        else:
            self.structure_analyzer = None
            logger.warning(f"PDF not found at {pdf_path}. Skipping structure analysis.")

        if (self.config.ENABLE_READING_ORDER_CORRECTION or 
            self.config.ENABLE_HEADING_RECONSTRUCTION or 
            self.config.ENABLE_BACKFILL_CORRECTION):
            flat_segments = self.reading_order_corrector.process(flat_segments, None)
            corrector_stats = self.reading_order_corrector.get_stats()
        else:
            corrector_stats = {}

        # 4.5 Structure Analysis & TOC Seeding (AUTHORITATIVE)
        glossary_pages = set()
        if self.structure_analyzer:
            self.structure = self.structure_analyzer.analyze()
            logger.info(f"Structure analysis complete for {source_file}")
            
            # Apply TOC Seeding via heading_path enrichment
            # Logic: TOC Path is the master context. 
            # To avoid noise and excessive length, we cap the hierarchy:
            # - If TOC is at Section level (e.g. 3.2), we stop there for standard chunks.
            # - We strictly filter out sidebars and known noise from the parser's path.
            for seg in flat_segments:
                page = seg.get('page', 1)
                gold_path = self.structure_analyzer.get_heading_path(page)
                if gold_path:
                    current_path = seg.get('heading_path', '')
                    
                    # 1. Clean current_path using aggressive logic
                    gold_parts = [p.strip() for p in gold_path.split(' > ')]
                    curr_parts = [p.strip() for p in current_path.split(' > ')]
                    
                    noise_pats = [
                        r'E-INVESTMENTS', r'EXERCISES', r'Concept Check', r'CONCEPT', 
                        r'Source:', r'Summary', r'Key Terms', r'\d+[\s\d]+LWI'
                    ]
                    
                    clean_curr = []
                    gold_norms = [re.sub(r'[^a-zA-Z0-9]', '', p.lower()) for p in gold_parts]
                    for p in curr_parts:
                        p_norm = re.sub(r'[^a-zA-Z0-9]', '', p.lower())
                        # Filter noise
                        if any(re.search(pat, p, re.IGNORECASE) for pat in noise_pats):
                            continue
                        # Filter redundancy with gold path
                        if p_norm in gold_norms or len(p_norm) < 3:
                            continue
                        clean_curr.append(p)

                    # 2. Decision Logic for Length
                    if len(gold_parts) >= 3:
                        # We are already at Section level (Part > Chap > Sect)
                        # Only append ONE local heading if it exists and looks structural
                        if clean_curr and seg.get('type') == 'Header':
                            seg['heading_path'] = f"{gold_path} > {clean_curr[-1]}"
                        else:
                            seg['heading_path'] = gold_path
                    else:
                        # Shallow TOC, can afford 1-2 local headings
                        local_tail = " > ".join(clean_curr[-1:]) if clean_curr else ""
                        seg['heading_path'] = f"{gold_path} > {local_tail}" if local_tail else gold_path
                    
                    # Update context text
                    seg['full_context_text'] = f"[Path: {seg['heading_path']}] {seg.get('text', '')}"
            
            front_range = self.structure["sections"]["front_matter"]
            back_range = self.structure["sections"]["back_matter"]
            if self.structure["sections"]["glossary"]:
                glossary_pages = set(range(self.structure["sections"]["glossary"][0], self.structure["sections"]["glossary"][1] + 1))
            
            logger.info(f"Structure: Applied TOC seeding to {len(flat_segments)} segments.")
        else:
            self.structure = None
        
        # 5. Build Block Catalog (Reference Detection)
        block_catalog = self.reference_detector.build_block_catalog(flat_segments)
        self.chunk_factory.set_block_catalog(block_catalog)
        
        # 5.5 Initialize LogicBook (Markdown Structure)
        logic_book = None
        markdown_text = data.get('full_markdown', '')
        if markdown_text:
            logic_book = LogicBook(markdown_text)
        
        # 6. Process Segments into Chunks
        chunks = self._process_segments(
            flat_segments, 
            toc_pages=set(), # Handled by gold_path now
            exclude_pages=glossary_pages,
            logic_book=logic_book
        )
        
        # 7. Process Glossary (Specialized Path)
        if glossary_pages:
            glossary_items = self.glossary_detector.parse(flat_segments, glossary_pages)
            for item in glossary_items:
                chunk = self.chunk_factory.create_chunk(
                    [item], # Glossary uses artificial segments or just items
                    "Glossary", 
                    chunk_type="definition"
                )
                chunks.append(chunk)
        
        # 7. Post-process (Merge, Overlap)
        chunks = self._post_process_chunks(chunks)
        
        # 8. Enrich & Link (KG Edges)
        enriched_chunks = KGLinker.link(chunks)
        
        # 9. Book Metadata (Strict API)
        try:
            book_metadata = self.metadata_manager.extract_strict(flat_segments)
        except Exception as e:
            logger.warning(f"MetadataManager: Strict extraction failed ({e}). Using falling back toDocling metadata.")
            book_metadata = {}

        # 10. Generate Stats & Output
        processing_stats = MetricsCollector.calculate(enriched_chunks)
        if corrector_stats:
            processing_stats['reading_order_correction'] = corrector_stats
            
        # Flatten and merge metadata
        # We merge book_metadata fields into the top-level metadata
        # Existing metadata from parser_docling (like source_file) is preserved
        final_metadata = {**metadata}
        for k, v in book_metadata.items():
            if v and (not final_metadata.get(k) or final_metadata.get(k) == ""):
                final_metadata[k] = v
        
        # Add processing-specific fields
        final_metadata.update({
            "total_chunks": len(enriched_chunks),
            "total_segments": len(flat_segments),
            "processing_version": "production",
            "features": [
                "context_overlap", "imperative_detection", "theorem_tagging", 
                "length_control", "reading_order_correction", "heading_reconstruction",
                "backfill_correction", "cross_page_continuation", "furniture_detection",
                "dehyphenation", "reference_detection", "kg_edges", 
                "strict_toc_seeding", "structure_analysis_v2", "heuristic_sectioning"
            ],
            "processing_stats": processing_stats
        })

        result = {
            "metadata": final_metadata,
            "chunks": [asdict(c) for c in enriched_chunks]
        }
        
        if output_path:
            self._save_json(result, output_path)
        
        return result



    def _process_segments(self, segments: List[Dict], toc_pages: set = None, 
                          exclude_pages: set = None, logic_book: LogicBook = None) -> List[EnrichedChunk]:
        """
        Core processing logic:
        1. Handle cross-page continuations with evidence tracking
        2. Group related segments (lists, procedures)
        3. Create enriched chunks with continuation metadata and evidence
        4. (NEW) Use logic_book for Markdown-verified hierarchy
        """
        chunks = []
        buffer = []
        current_heading_path = ""
        has_cross_page = False  # Track if current buffer spans pages
        continuation_type = "none"  # Track continuation confidence
        merge_evidences = []  # Collect all evidence for this buffer
        
        toc_pages = toc_pages or set()
        exclude_pages = exclude_pages or set()
        
        for i, seg in enumerate(segments):
            # Skip excluded pages (e.g. Glossary)
            if seg.get('page', 0) in exclude_pages:
                continue
                
            seg_text = seg.get('text', '')
            seg_type = seg.get('type', 'Paragraph')
            heading_path = seg.get('heading_path', '')
            is_continuation = seg.get('is_continuation', 'none')
            continuation_evidence = seg.get('continuation_evidence', {})
            seg_col = seg.get('column_index', -1)
            
            # 0.5 LogicBook Alignment: Primary Structural Authority
            logic_path = None
            if logic_book:
                # NEW: Clean Parser noise (institutional info, printing artifacts)
                if logic_book.is_noise(seg_text) and seg_type == 'Header':
                    logger.debug(f"LogicBook: Demoting noise header '{seg_text[:30]}' to Paragraph")
                    seg_type = 'Paragraph'
                    seg['type'] = 'Paragraph'

                logic_path = logic_book.get_path(seg_text)
                if logic_path:
                    # Upgrade to Header if LogicBook confirms it's a heading
                    # Use authoritative path, but don't lose the TOC depth if we already have it
                    if not heading_path or len(logic_path) > len(heading_path):
                        heading_path = logic_path
                        seg['heading_path'] = logic_path

            # TOC Special Handling
            if seg.get('page', 0) in toc_pages:
                # If matching TOC line pattern (chapter/section ... page)
                # Force it to be a Header/Topic to prevent adsorption and preserve hierarchy
                if re.search(r'[\.·-]{3,}\s*\d+$', seg_text) or re.search(r'^\d+(\.\d+)*\s+[A-Z]', seg_text):
                    seg_type = 'Header'
                    seg['type'] = 'Header'
                    seg['inferred_role'] = 'topic'
                    logger.debug(f"TOC Rule: Promoting {seg.get('segment_id')} to Header")
            
            # NEW: Merge Guards
            # 1. Column Locking: Prevent merging across columns unless spanning
            # 2. ID Gap Detection: Prevent merging segments with large ID distance
            should_flush_by_guard = False
            if buffer:
                prev_seg = buffer[-1]
                prev_col = prev_seg.get('column_index', -1)
                
                # Column Guard: If both are non-spanning and different, block merge
                if (self.config.COLUMN_MERGE_GUARD and 
                    seg_col != -1 and prev_col != -1 and seg_col != prev_col):
                    should_flush_by_guard = True
                    logger.debug(f"Merge Guard: Column mismatch ({prev_col} != {seg_col}). Flushing.")
                
                # ID Gap Guard: Check segment ID sequence
                prev_id_str = prev_seg.get('segment_id', 'seg_0')
                curr_id_str = seg.get('segment_id', 'seg_0')
                try:
                    prev_id_num = int(re.search(r'\d+', prev_id_str).group())
                    curr_id_num = int(re.search(r'\d+', curr_id_str).group())
                    if abs(curr_id_num - prev_id_num) > self.config.SEGMENT_ID_GAP_THRESHOLD:
                        should_flush_by_guard = True
                        logger.debug(f"Merge Guard: ID gap too large ({prev_id_num} -> {curr_id_num}). Flushing.")
                except:
                    pass

            if should_flush_by_guard and buffer:
                chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                chunk.is_cross_page = has_cross_page
                chunk.continuation_type = continuation_type
                chunk.needs_review = (continuation_type == 'partial')
                chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                chunks.append(chunk)
                buffer = []
                has_cross_page = False
                continuation_type = "none"
                merge_evidences = []
            
            # Phase 1: Semantic Role Pre-analysis (Role-First)
            discourse_role = self.pos_analyzer.pre_analyze_role(seg_text) if self.pos_analyzer else None
            
            # If POS predicts LO but Docling says Paragraph, trust POS for grouping
            if discourse_role == 'learning_objective' and seg_type == 'Paragraph':
                seg_type = 'LearningObjective'
                seg['inferred_role'] = 'learning_objective'
                logger.debug(f"Role-First: Overriding Paragraph -> LearningObjective for {seg.get('segment_id')}")
            
            # Determine if buffer currently contains a "Master" block that can adsorb
            buffer_is_lo = buffer and (buffer[-1].get('type') == 'LearningObjective' or buffer[-1].get('inferred_role') == 'learning_objective')
            buffer_is_header = buffer and buffer[-1].get('type') == 'Header'
            can_adsorb = buffer_is_lo or buffer_is_header

            
            # =================================================================
            # Rule 0: Cross-page continuation handling with evidence
            # =================================================================
            # If this segment is marked as a continuation, do NOT flush buffer
            # Instead, continue accumulating to preserve paragraph integrity
            if is_continuation in ['full', 'partial']:
                # Track cross-page status
                has_cross_page = True
                if is_continuation == 'partial':
                    continuation_type = 'partial'
                elif continuation_type != 'partial':
                    continuation_type = 'full'
                
                # Collect evidence for explainability
                if continuation_evidence:
                    merge_evidences.append(continuation_evidence)
                
                # Add to buffer and continue (skip other rules)
                buffer.append(seg)
                
                # Log for debugging
                score = continuation_evidence.get('final_score', 0)
                logger.debug(f"Cross-page continuation: adding {seg.get('segment_id')} to buffer "
                           f"(type: {is_continuation}, score: {score:.2f})")
                
                # Still check buffer size limit
                if len(buffer) >= self.config.MAX_BUFFER_SEGMENTS * 2:  # Allow 2x for continuations
                    chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                continue
            
            # =================================================================
            # Rule 1: Headers - with Caption Bonding logic
            # =================================================================
            # Distinguish between:
            # - Block Captions (Table 1.1, Figure 2.3) -> bond with next block, don't update global path
            # - Structural Headers (Chapter 1, 1.1 Introduction) -> update global path
            if seg_type == 'Header':
                # Check if this is a noise header (concluded, continued, etc.)
                is_noise = seg.get('is_noise_header', False) or self.continuation_detector._is_noise_header(seg)
                
                if is_noise:
                    buffer.append(seg)
                    logger.debug(f"Skipping noise header {seg.get('segment_id')}: {seg.get('text', '')[:30]}")
                    continue
                
                # Check if this is a block caption (Table X.X, Figure X.X)
                is_block_caption, caption_info = self.caption_bonding_helper.detect_caption(seg)
                
                if is_block_caption:
                    # Block caption: DO NOT flush buffer or update global heading path
                    # Instead, add to buffer to bond with upcoming Table/Figure
                    # Mark it for later bonding
                    seg['is_block_caption'] = True
                    seg['caption_info'] = caption_info
                    buffer.append(seg)
                    logger.debug(f"Block caption detected: {caption_info.get('full_caption_id', '')} - buffering for bonding")
                    continue
                
                # Structural header: flush buffer and update global heading path
                if buffer:
                    chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                
                # Update structural heading path
                current_heading_path = heading_path
                self.structural_heading_path = heading_path
                
                # Headers themselves become chunks
                chunks.append(self.chunk_factory.create_chunk([seg], heading_path, chunk_type="header"))
                continue
            
            # =================================================================
            # Rule 2: ListItems should be grouped together
            # =================================================================
            if seg_type == 'ListItem':
                # Rule 2a: Adsorption - if buffer is an LO or Header, adsorb the list
                if can_adsorb and heading_path == current_heading_path:
                    buffer.append(seg)
                    logger.debug(f"Adsorption: ListItem {seg.get('segment_id')} adsorbed into active {buffer[-1].get('type')} buffer")
                    continue

                # Rule 2b: Standard List Item grouping
                # If buffer has non-list items, flush first
                if buffer and buffer[-1].get('type') != 'ListItem':
                    chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                buffer.append(seg)
                continue
            
            # =================================================================
            # Rule 3: If we have ListItems in buffer and current is not ListItem, flush
            # =================================================================
            if buffer and buffer[-1].get('type') == 'ListItem' and seg_type != 'ListItem':
                chunk = self.chunk_factory.create_chunk(buffer, current_heading_path, chunk_type="list")
                chunk.is_cross_page = has_cross_page
                chunk.continuation_type = continuation_type
                chunk.needs_review = (continuation_type == 'partial')
                chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                chunks.append(chunk)
                buffer = []
                has_cross_page = False
                continuation_type = "none"
                merge_evidences = []
            
            # =================================================================
            # Rule 4: Tables/Pictures/Formulas - with Caption Bonding
            # =================================================================
            # Bond buffered block captions + notes with this structural block
            if seg_type in ['Table', 'Picture', 'Formula']:
                # Check if buffer contains a block caption to bond with
                caption_seg = None
                notes_segs = []
                other_segs = []
                
                for buffered_seg in buffer:
                    if buffered_seg.get('is_block_caption'):
                        caption_seg = buffered_seg
                    elif buffered_seg.get('type') in ['Header', 'Paragraph', 'Text']:
                        # Could be notes/description for the table
                        notes_segs.append(buffered_seg)
                    else:
                        other_segs.append(buffered_seg)
                
                # First, flush any non-caption content separately
                if other_segs:
                    chunk = self.chunk_factory.create_chunk(other_segs, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                
                # Now create the bonded Table/Picture chunk
                # But ONLY if the caption type matches the block type
                if caption_seg:
                    caption_info = caption_seg.get('caption_info', {})
                    expected_type = caption_info.get('target_type', '')
                    
                    # Relaxed type matching rules to handle parser misclassifications:
                    # - Allow "Table" caption to bond with "Picture" (often tables are images)
                    # - Allow "Figure" caption to bond with "Formula" or vice-versa
                    # - Allow "Exhibit" to bond with anything
                    type_matches = False
                    if expected_type == seg_type:
                        type_matches = True
                    elif expected_type == 'Table' and seg_type in ['Table', 'Picture', 'Formula']:
                        type_matches = True
                    elif expected_type in ['Figure', 'Exhibit'] and seg_type in ['Picture', 'Table', 'Formula']:
                        type_matches = True
                    elif expected_type == 'Equation' and seg_type in ['Formula', 'Picture']:
                        type_matches = True
                    elif not expected_type:
                        type_matches = True
                    
                    if type_matches:
                        # Spatial Proximity Check: Security Layer
                        if not self.caption_bonding_helper.verify_spatial_proximity(caption_seg, seg):
                             # Spatial Rejection
                             logger.info(f"Bonding rejected by spatial check (Distance/Overlap): {caption_info.get('full_caption_id')} far from {seg_type}. Flushing caption separately.")
                             
                             # Flush caption + notes
                             all_caption_content = [caption_seg] + notes_segs
                             if all_caption_content:
                                 chunk = self.chunk_factory.create_chunk(all_caption_content, current_heading_path)
                                 chunks.append(chunk)
                             # Create standalone block
                             chunks.append(self.chunk_factory.create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                        else:
                            # SUCCESS: Bond caption + notes + block into single atomic unit
                            bonded_segments = [caption_seg] + notes_segs + [seg]
                            caption_text = caption_seg.get('text', '').strip()
                            effective_heading_path = self.structural_heading_path or current_heading_path
                            
                            chunk = self.chunk_factory.create_chunk(
                                bonded_segments, 
                                effective_heading_path, 
                                chunk_type=seg_type.lower()
                            )
                            chunk.merge_evidence = {
                                "bonded": True,
                                "caption_text": caption_text,
                                "notes_count": len(notes_segs),
                                "source_segments": [s.get('segment_id', '') for s in bonded_segments],
                            }
                            chunks.append(chunk)
                            logger.info(f"Caption bonded: {caption_text[:50]}... -> {seg_type} ({len(notes_segs)} notes)")
                    else:
                        # LOGICAL TYPE MISMATCH
                        logger.info(f"Caption type mismatch: {expected_type} caption cannot bond with {seg_type} block by logic rules.")
                        # Flush caption + notes
                        all_caption_content = [caption_seg] + notes_segs
                        if all_caption_content:
                            chunk = self.chunk_factory.create_chunk(all_caption_content, current_heading_path)
                            chunks.append(chunk)
                        # Create standalone block
                        chunks.append(self.chunk_factory.create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                else:
                    # No caption in buffer, just create standalone chunk
                    if notes_segs:
                        # Flush notes first
                        chunk = self.chunk_factory.create_chunk(notes_segs, current_heading_path)
                        chunks.append(chunk)
                    
                    chunks.append(self.chunk_factory.create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                
                # Clear buffer
                buffer = []
                has_cross_page = False
                continuation_type = "none"
                merge_evidences = []
                continue
            
            # =================================================================
            # Rule 4.5: Group consecutive LearningObjectives under same heading
            # =================================================================
            if seg_type == 'LearningObjective':
                # If buffer has non-LO content OR heading changed, flush first
                if buffer and (buffer[-1].get('type') != 'LearningObjective' or buffer[-1].get('heading_path', '') != heading_path):
                    chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                    # Update current_heading_path for the upcoming LOs
                    current_heading_path = heading_path
                
                if not buffer:
                    current_heading_path = heading_path
                    
                buffer.append(seg)
                continue

            # =================================================================
            # Rule 4.6: Flush LearningObjectives sequence if next is different
            # =================================================================
            if buffer and buffer[-1].get('type') == 'LearningObjective' and seg_type != 'LearningObjective':
                # Rule 4.7: Soft Adsorption - allow one trailing Paragraph if it looks like content
                if seg_type == 'Paragraph' and len(seg_text.split()) < 50 and heading_path == current_heading_path:
                     buffer.append(seg)
                     logger.debug(f"Soft Adsorption: Small paragraph {seg.get('segment_id')} added to LO chunk")
                     continue

                chunk = self.chunk_factory.create_chunk(buffer, current_heading_path, chunk_type="learning_objective")
                chunk.is_cross_page = has_cross_page
                chunk.continuation_type = continuation_type
                chunk.needs_review = (continuation_type == 'partial')
                chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                chunks.append(chunk)
                buffer = []
                has_cross_page = False
                continuation_type = "none"
                merge_evidences = []
            
            # =================================================================
            # Rule 5: Check for theorem/proof block starters
            # =================================================================
            text = seg.get('text', '')
            if re.match(r'^(?:Theorem|Lemma|Proposition|Corollary|Proof)\s*\d*', text, re.IGNORECASE):
                if buffer:
                    chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
                buffer.append(seg)
                continue
            
            # =================================================================
            # Rule 5.5: Detect block captions in Paragraphs (side/bottom captions)
            # =================================================================
            # Sometimes PDF parsers misclassify "Table 1.2" as Paragraph instead of Header
            # We still need to detect and mark these for bonding
            if seg_type in ['Paragraph', 'Text']:
                is_block_caption, caption_info = self.caption_bonding_helper.detect_caption(seg)
                if is_block_caption:
                    seg['is_block_caption'] = True
                    seg['caption_info'] = caption_info
                    buffer.append(seg)
                    logger.debug(f"Paragraph caption detected: {caption_info.get('full_caption_id', '')} - buffering for bonding")
                    continue
            
            # =================================================================
            # Default: Add to buffer
            # =================================================================
            buffer.append(seg)
            
            # =================================================================
            # Rule 6: Flush if buffer exceeds threshold (with lookahead)
            # Check if next non-noise segment is a continuation
            #           If so, delay flush to preserve paragraph integrity
            # =================================================================
            if len(buffer) >= self.config.MAX_BUFFER_SEGMENTS:
                # Lookahead: check if upcoming segments form a continuation chain
                should_delay_flush = False
                for lookahead_idx in range(i + 1, min(i + 4, len(segments))):  # Look ahead up to 3 segments
                    future_seg = segments[lookahead_idx]
                    # Skip noise headers in lookahead
                    if future_seg.get('is_noise_header', False):
                        continue
                    # If we find a continuation, delay the flush
                    if future_seg.get('is_continuation') in ['full', 'partial']:
                        should_delay_flush = True
                        logger.debug(f"Delaying flush: upcoming {future_seg.get('segment_id')} is a continuation")
                    break  # Only check the first non-noise segment
                
                if not should_delay_flush:
                    chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
                    chunk.is_cross_page = has_cross_page
                    chunk.continuation_type = continuation_type
                    chunk.needs_review = (continuation_type == 'partial')
                    chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
                    chunks.append(chunk)
                    buffer = []
                    has_cross_page = False
                    continuation_type = "none"
                    merge_evidences = []
        
        # Flush remaining buffer
        if buffer:
            chunk = self.chunk_factory.create_chunk(buffer, current_heading_path)
            chunk.is_cross_page = has_cross_page
            chunk.continuation_type = continuation_type
            chunk.needs_review = (continuation_type == 'partial')
            chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
            chunks.append(chunk)
        
        # Apply Structure-based Tagging (Zone & Type overrides)
        for chunk in chunks:
            self._tag_chunk_by_structure(chunk)
            
        return chunks

    def _tag_chunk_by_structure(self, chunk: EnrichedChunk):
        """Overrides chunk_type and doc_zone based on PDF structure maps."""
        if not self.structure:
            return
            
        page = chunk.page_range[0]
        boundaries = self.structure["sections"]
        
        # 1. Zone Classification
        if page <= boundaries["front_matter"][1]:
            chunk.doc_zone = "front"
            # User request: front matter chunks get special type
            if chunk.chunk_type not in ["header"]:
                chunk.chunk_type = "front_matter"
        elif page >= boundaries["back_matter"][0]:
            chunk.doc_zone = "back"
            # Glossary check
            is_glossary = False
            if boundaries.get("glossary"):
                if boundaries["glossary"][0] <= page <= boundaries["glossary"][1]:
                    is_glossary = True
            
            if is_glossary:
                chunk.chunk_type = "glossary"
            else:
                chunk.chunk_type = "back_matter"
        else:
            chunk.doc_zone = "body"

    
    def _compile_merge_evidence(self, evidences: List[Dict]) -> Dict[str, Any]:
        """
        Compile multiple evidence dicts into a summary for explainability.
        
        Returns a dict with:
        - merge_count: Number of cross-page merges in this chunk
        - avg_confidence: Average continuation score
        - key_indicators: Most common triggering rules
        - details: List of individual evidence entries (summarized)
        """
        if not evidences:
            return {}
        
        # Calculate summary statistics
        scores = [e.get('final_score', 0) for e in evidences]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Collect all triggered rules
        all_triggers = []
        for e in evidences:
            all_triggers.extend(e.get('rules_triggered', []))
        
        # Count rule frequency
        rule_counts = {}
        for rule in all_triggers:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Get top indicators
        top_indicators = sorted(rule_counts.items(), key=lambda x: -x[1])[:5]
        
        # Summarize individual merges
        details = []
        for e in evidences:
            detail = {
                "from": e.get('prev_segment_id', ''),
                "to": e.get('curr_segment_id', ''),
                "score": e.get('final_score', 0),
                "decision": e.get('decision', 'none'),
            }
            # Add key score factors
            breakdown = e.get('score_breakdown', {})
            if breakdown:
                high_scorers = [(k, v['score']) for k, v in breakdown.items() 
                               if isinstance(v, dict) and v.get('score', 0) > 0]
                detail["contributing_factors"] = [f"{k}: {v:.2f}" for k, v in high_scorers]
            details.append(detail)
        
        return {
            "merge_count": len(evidences),
            "avg_confidence": round(avg_score, 3),
            "key_indicators": [f"{k}: {v}x" for k, v in top_indicators],
            "details": details
        }
    
    def _post_process_chunks(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """
        Post-processing:
        1. Bond orphan captions with adjacent Table/Picture chunks (bottom caption repair)
        2. Merge short adjacent chunks under same heading
        3. Add context overlap from previous chunk
        """
        if not chunks:
            return chunks
        
        # Phase 1: Orphan caption bonding (bottom caption repair)
        # Detect chunks that are block captions but weren't bonded
        # and retroactively merge them with preceding Table/Picture chunks
        bonded_chunks = []
        skip_next = False
        
        for i, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue
            
            # Check if this is an orphan caption (starts with Table/Figure but wasn't bonded)
            is_orphan_caption = False
            content = chunk.content.strip()
            if chunk.chunk_type in ['explanation', 'header'] and content:
                is_caption, caption_info = self.caption_bonding_helper.detect_caption({
                    'text': content.split('\n')[0]  # Check first line only
                })
                if is_caption and not chunk.merge_evidence.get('bonded'):
                    is_orphan_caption = True
            
            if is_orphan_caption:
                # First, try to bond with a PRECEDING Table/Picture chunk
                for j in range(len(bonded_chunks) - 1, max(0, len(bonded_chunks) - 3) - 1, -1):
                    prev_chunk = bonded_chunks[j]
                    if prev_chunk.chunk_type in ['table', 'picture', 'formula']:
                        # Skip if this table already has a caption bonded
                        if prev_chunk.merge_evidence.get('bonded'):
                            continue
                        
                        # Check if they're on the same page or adjacent pages
                        same_or_adjacent_page = (
                            set(prev_chunk.page_range) & set(chunk.page_range) or
                            abs(max(prev_chunk.page_range) - min(chunk.page_range)) <= 1
                        )
                        if same_or_adjacent_page:
                            # Bond: prepend caption to table content
                            prev_chunk.content = chunk.content + "\n" + prev_chunk.content if prev_chunk.content else chunk.content
                            prev_chunk.source_segments.extend(chunk.source_segments)
                            
                            # Update bboxes
                            prev_chunk.bbox = [
                                min(prev_chunk.bbox[0], chunk.bbox[0]) if prev_chunk.bbox and chunk.bbox else (prev_chunk.bbox[0] if prev_chunk.bbox else (chunk.bbox[0] if chunk.bbox else 0)),
                                min(prev_chunk.bbox[1], chunk.bbox[1]) if prev_chunk.bbox and chunk.bbox else (prev_chunk.bbox[1] if prev_chunk.bbox else (chunk.bbox[1] if chunk.bbox else 0)),
                                max(prev_chunk.bbox[2], chunk.bbox[2]) if prev_chunk.bbox and chunk.bbox else (prev_chunk.bbox[2] if prev_chunk.bbox else (chunk.bbox[2] if chunk.bbox else 0)),
                                max(prev_chunk.bbox[3], chunk.bbox[3]) if prev_chunk.bbox and chunk.bbox else (prev_chunk.bbox[3] if prev_chunk.bbox else (chunk.bbox[3] if chunk.bbox else 0)),
                            ]
                            for p, b in chunk.page_bboxes.items():
                                if p not in prev_chunk.page_bboxes:
                                    prev_chunk.page_bboxes[p] = b
                                else:
                                    pb = prev_chunk.page_bboxes[p]
                                    prev_chunk.page_bboxes[p] = [min(pb[0], b[0]), min(pb[1], b[1]), max(pb[2], b[2]), max(pb[3], b[3])]

                            prev_chunk.merge_evidence = {
                                "bonded": True,
                                "caption_text": content.split('\n')[0][:50],
                                "bond_type": "post_process_orphan_caption",
                            }
                            logger.info(f"Post-process bonded orphan caption (backward): {content[:30]}... -> {prev_chunk.chunk_type}")
                            is_orphan_caption = False
                            break
                
                # If still orphan, try to bond with the NEXT Table/Picture chunk (look ahead)
                if is_orphan_caption:
                    for k in range(i + 1, min(len(chunks), i + 4)):
                        next_chunk = chunks[k]
                        if next_chunk.chunk_type in ['table', 'picture', 'formula']:
                            if not next_chunk.merge_evidence.get('bonded'):
                                same_or_adjacent_page = (
                                    set(next_chunk.page_range) & set(chunk.page_range) or
                                    abs(min(next_chunk.page_range) - max(chunk.page_range)) <= 1
                                )
                                if same_or_adjacent_page:
                                    # Bond: prepend caption to next chunk
                                    next_chunk.content = chunk.content + "\n" + next_chunk.content if next_chunk.content else chunk.content
                                    next_chunk.source_segments = chunk.source_segments + next_chunk.source_segments
                                    
                                    # Update bboxes
                                    next_chunk.bbox = [
                                        min(next_chunk.bbox[0], chunk.bbox[0]) if next_chunk.bbox and chunk.bbox else (next_chunk.bbox[0] if next_chunk.bbox else (chunk.bbox[0] if chunk.bbox else 0)),
                                        min(next_chunk.bbox[1], chunk.bbox[1]) if next_chunk.bbox and chunk.bbox else (next_chunk.bbox[1] if next_chunk.bbox else (chunk.bbox[1] if chunk.bbox else 0)),
                                        max(next_chunk.bbox[2], chunk.bbox[2]) if next_chunk.bbox and chunk.bbox else (next_chunk.bbox[2] if next_chunk.bbox else (chunk.bbox[2] if chunk.bbox else 0)),
                                        max(next_chunk.bbox[3], chunk.bbox[3]) if next_chunk.bbox and chunk.bbox else (next_chunk.bbox[3] if next_chunk.bbox else (chunk.bbox[3] if chunk.bbox else 0)),
                                    ]
                                    for p, b in chunk.page_bboxes.items():
                                        if p not in next_chunk.page_bboxes:
                                            next_chunk.page_bboxes[p] = b
                                        else:
                                            pb = next_chunk.page_bboxes[p]
                                            next_chunk.page_bboxes[p] = [min(pb[0], b[0]), min(pb[1], b[1]), max(pb[2], b[2]), max(pb[3], b[3])]

                                    next_chunk.merge_evidence = {
                                        "bonded": True,
                                        "caption_text": content.split('\n')[0][:50],
                                        "bond_type": "post_process_forward_bond",
                                    }
                                    logger.info(f"Post-process bonded orphan caption (forward): {content[:30]}... -> {next_chunk.chunk_type}")
                                    is_orphan_caption = False
                                    break
                            break  # Only check the first Table/Picture
                
                if not is_orphan_caption:
                    continue  # Skip this chunk, it was merged
            
            bonded_chunks.append(chunk)
        
        # Phase 1.5: Table-Then-Caption bonding
        # Handle case where Docling outputs Table BEFORE Caption (visual ordering issue)
        # Check each unbonded Table/Picture and see if it's followed by its Caption
        final_bonded = []
        skip_indices = set()
        
        for i, chunk in enumerate(bonded_chunks):
            if i in skip_indices:
                continue
            
            # Phase 1.8: Learning Objective + List regrouping (Hard Regrouping)
            # Handle case where LO header was separated from its list
            if (chunk.chunk_type == 'learning_objective' or 
                (self.pos_analyzer and self.pos_analyzer.pre_analyze_role(chunk.content.split('\n')[0]) == 'learning_objective')):
                
                # Check if the next chunk is a list or unbonded explanation
                for j in range(i + 1, min(len(bonded_chunks), i + 3)):
                    if j in skip_indices:
                        continue
                    next_chunk = bonded_chunks[j]
                    
                    # Merge if it's a list or a short explanation under the same heading
                    should_merge = False
                    if next_chunk.chunk_type == 'list':
                        should_merge = True
                    elif next_chunk.chunk_type == 'explanation' and next_chunk.word_count < 100:
                        should_merge = True
                        
                    if should_merge and next_chunk.heading_path == chunk.heading_path:
                        # Hard Merge: combine content and markers
                        chunk.content += "\n" + next_chunk.content
                        chunk.source_segments.extend(next_chunk.source_segments)
                        chunk.sentences.extend(next_chunk.sentences)
                        chunk.tags = list(set(chunk.tags + next_chunk.tags))
                        chunk.chunk_type = 'learning_objective' 
                        chunk.word_count = len(chunk.content.split())
                        
                        # Update bboxes
                        chunk.bbox = [
                            min(chunk.bbox[0], next_chunk.bbox[0]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[0] if chunk.bbox else (next_chunk.bbox[0] if next_chunk.bbox else 0)),
                            min(chunk.bbox[1], next_chunk.bbox[1]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[1] if chunk.bbox else (next_chunk.bbox[1] if next_chunk.bbox else 0)),
                            max(chunk.bbox[2], next_chunk.bbox[2]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[2] if chunk.bbox else (next_chunk.bbox[2] if next_chunk.bbox else 0)),
                            max(chunk.bbox[3], next_chunk.bbox[3]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[3] if chunk.bbox else (next_chunk.bbox[3] if next_chunk.bbox else 0)),
                        ]
                        for p, b in next_chunk.page_bboxes.items():
                            if p not in chunk.page_bboxes:
                                chunk.page_bboxes[p] = b
                            else:
                                pb = chunk.page_bboxes[p]
                                chunk.page_bboxes[p] = [min(pb[0], b[0]), min(pb[1], b[1]), max(pb[2], b[2]), max(pb[3], b[3])]

                        skip_indices.add(j)
                        logger.info(f"Hard Regrouping: LO Header merged with {next_chunk.chunk_type} (chunk_{j})")
                        # Keep looking if there's more to adsorb
                        continue
                    else:
                        break

            # If this is an unbonded Table/Picture, check if the next chunk is its Caption
            if chunk.chunk_type in ['table', 'picture', 'formula'] and not chunk.merge_evidence.get('bonded'):
                # Look ahead for a Caption
                for j in range(i + 1, min(len(bonded_chunks), i + 3)):
                    if j in skip_indices:
                        continue
                    next_chunk = bonded_chunks[j]
                    if next_chunk.chunk_type in ['header', 'explanation']:
                        # Check if it's a caption for this table
                        is_caption, _ = self.caption_bonding_helper.detect_caption({
                            'text': next_chunk.content.split('\n')[0]
                        })
                        if is_caption and not next_chunk.merge_evidence.get('bonded'):
                            # Bond: prepend caption to table
                            chunk.content = next_chunk.content + "\n" + chunk.content if chunk.content else next_chunk.content
                            chunk.source_segments = next_chunk.source_segments + chunk.source_segments
                            
                            # Update bboxes
                            chunk.bbox = [
                                min(chunk.bbox[0], next_chunk.bbox[0]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[0] if chunk.bbox else (next_chunk.bbox[0] if next_chunk.bbox else 0)),
                                min(chunk.bbox[1], next_chunk.bbox[1]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[1] if chunk.bbox else (next_chunk.bbox[1] if next_chunk.bbox else 0)),
                                max(chunk.bbox[2], next_chunk.bbox[2]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[2] if chunk.bbox else (next_chunk.bbox[2] if next_chunk.bbox else 0)),
                                max(chunk.bbox[3], next_chunk.bbox[3]) if chunk.bbox and next_chunk.bbox else (chunk.bbox[3] if chunk.bbox else (next_chunk.bbox[3] if next_chunk.bbox else 0)),
                            ]
                            for p, b in next_chunk.page_bboxes.items():
                                if p not in chunk.page_bboxes:
                                    chunk.page_bboxes[p] = b
                                else:
                                    pb = chunk.page_bboxes[p]
                                    chunk.page_bboxes[p] = [min(pb[0], b[0]), min(pb[1], b[1]), max(pb[2], b[2]), max(pb[3], b[3])]

                            chunk.merge_evidence = {
                                "bonded": True,
                                "caption_text": next_chunk.content.split('\n')[0][:50],
                                "bond_type": "table_then_caption",
                            }
                            skip_indices.add(j)
                            logger.info(f"Table-then-Caption bonded: {next_chunk.content[:30]}... -> {chunk.chunk_type}")
                            break
                    # Stop if we hit another table/picture
                    elif next_chunk.chunk_type in ['table', 'picture', 'formula']:
                        break
            
            final_bonded.append(chunk)
        
        bonded_chunks = final_bonded
        
        # Phase 2: Original post-processing
        processed = []
        
        for i, chunk in enumerate(bonded_chunks):
            # Add context overlap (if enabled and not first chunk)
            if self.config.ENABLE_OVERLAP and i > 0 and chunk.chunk_type not in ['header', 'picture', 'table']:
                prev_chunk = bonded_chunks[i - 1]
                if prev_chunk.content and prev_chunk.chunk_type not in ['header', 'picture', 'table']:
                    # Get last N sentences from previous chunk
                    if self.pos_analyzer:
                        overlap_text = self.pos_analyzer.get_last_n_sentences(
                            prev_chunk.content, 
                            self.config.OVERLAP_SENTENCES
                        )
                    else:
                        # Fallback: take last 100 chars
                        overlap_text = prev_chunk.content[-100:] if len(prev_chunk.content) > 100 else ""
                    
                    # Only add if meaningful
                    if len(overlap_text.split()) > 5:
                        chunk.context_prefix = f"[Previous context: {overlap_text}]"
            
            processed.append(chunk)
        
        # Merge very short chunks (optional second pass)
        merged = self._merge_short_chunks(processed)
        
        return merged
    
    def _merge_short_chunks(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """Merge consecutive short paragraphs under the same heading."""
        if len(chunks) < 2:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Skip non-mergeable types
            if current.chunk_type in ['header', 'picture', 'table', 'list', 'formula']:
                merged.append(current)
                i += 1
                continue
            
            # Check if current is short and can be merged with next
            if (current.word_count < self.config.SHORT_PARAGRAPH_WORDS and 
                i + 1 < len(chunks)):
                next_chunk = chunks[i + 1]
                
                # Merge if same heading and next is also short enough
                if (next_chunk.heading_path == current.heading_path and
                    next_chunk.chunk_type in ['explanation', 'example'] and
                    current.word_count + next_chunk.word_count < self.config.MAX_CHUNK_WORDS):
                    
                    # Create merged chunk
                    merged_content = current.content + " " + next_chunk.content
                    merged_tags = list(set(current.tags + next_chunk.tags))
                    merged_sources = current.source_segments + next_chunk.source_segments
                    
                    # Merge bboxes
                    merged_bbox = [
                        min(current.bbox[0], next_chunk.bbox[0]) if current.bbox and next_chunk.bbox else (current.bbox[0] if current.bbox else (next_chunk.bbox[0] if next_chunk.bbox else 0)),
                        min(current.bbox[1], next_chunk.bbox[1]) if current.bbox and next_chunk.bbox else (current.bbox[1] if current.bbox else (next_chunk.bbox[1] if next_chunk.bbox else 0)),
                        max(current.bbox[2], next_chunk.bbox[2]) if current.bbox and next_chunk.bbox else (current.bbox[2] if current.bbox else (next_chunk.bbox[2] if next_chunk.bbox else 0)),
                        max(current.bbox[3], next_chunk.bbox[3]) if current.bbox and next_chunk.bbox else (current.bbox[3] if current.bbox else (next_chunk.bbox[3] if next_chunk.bbox else 0)),
                    ]
                    merged_page_bboxes = dict(current.page_bboxes)
                    for p, b in next_chunk.page_bboxes.items():
                        if p not in merged_page_bboxes:
                            merged_page_bboxes[p] = b
                        else:
                            pb = merged_page_bboxes[p]
                            merged_page_bboxes[p] = [min(pb[0], b[0]), min(pb[1], b[1]), max(pb[2], b[2]), max(pb[3], b[3])]

                    merged_chunk = EnrichedChunk(
                        chunk_id=current.chunk_id,
                        heading_path=current.heading_path,
                        chunk_type=self.chunk_factory.infer_from_tags(merged_tags),
                        content=merged_content,
                        context_prefix=current.context_prefix,
                        sentences=current.sentences + next_chunk.sentences,
                        tags=merged_tags,
                        source_segments=merged_sources,
                        page_range=[
                            min(current.page_range[0] if current.page_range else 0, 
                                next_chunk.page_range[0] if next_chunk.page_range else 0),
                            max(current.page_range[1] if len(current.page_range) > 1 else 0,
                                next_chunk.page_range[1] if len(next_chunk.page_range) > 1 else 0)
                        ],
                        depth=current.depth,
                        word_count=len(merged_content.split()),
                        bbox=merged_bbox,
                        page_bboxes=merged_page_bboxes
                    )
                    merged.append(merged_chunk)
                    i += 2  # Skip both chunks
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    
    def _save_json(self, data: Dict, path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved enriched chunks to: {path}")
