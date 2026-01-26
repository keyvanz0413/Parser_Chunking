import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from pathlib import Path

from .config import ChunkingConfig
from .schema import EnrichedChunk
from .detectors.tags import TagDetector
from .detectors.continuation import ContinuationDetector
from .detectors.references import ReferenceDetector
from .analyzers.pos_analyzer import POSAnalyzer
from .analyzers.reading_order import ReadingOrderCorrector
from .analyzers.gatekeeper import ContentGatekeeper
from .utils.caption_bonding import CaptionBondingHelper
from .utils.dehyphenation import DehyphenationHelper

logger = logging.getLogger(__name__)

class LogicSegmenter:
    """
    Main segmenter that processes flat_segments from parser_docling.py.
    
    Pipeline (Three-Phase Architecture):
    1. Load segments from JSON
    2. Apply Three-Phase Correction Pipeline:
       - Phase 1: Column-based reading order correction
       - Phase 2: Heading stack reconstruction
       - Phase 3: Same-page backfilling for L1 headers
    3. Annotate segments with continuation markers
    4. Group segments by logical rules (lists, procedures, cross-page)
    5. Analyze with POS tagging (enhanced with imperative detection)
    6. Apply context overlap for RAG continuity
    7. Enrich with tags (20+ including Theorem/Lemma/Proof)
    8. Apply length constraints (min/max word counts)
    9. Output standardized chunks with continuation metadata
    """
    
    def __init__(self, use_pos: bool = True, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.tag_detector = TagDetector()
        self.pos_analyzer = POSAnalyzer() if use_pos else None
        self.continuation_detector = ContinuationDetector(self.config)
        self.reading_order_corrector = ReadingOrderCorrector(self.config)
        self.reference_detector = ReferenceDetector(self.config)
        self.caption_bonding_helper = CaptionBondingHelper(self.config)  # Caption bonding
        self.chunk_counter = 0
        self.previous_chunk_text = ""  # For overlap
        self.block_catalog = None  # Will be populated during processing
        self.structural_heading_path = ""  # Track structural headers separately from block captions
    
    def process_file(self, input_json_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a parser_docling.py output file.
        
        Args:
            input_json_path: Path to the JSON from parser_docling
            output_path: Optional path to save enriched output
            
        Returns:
            Dictionary with enriched chunks and statistics
        """
        logger.info(f"Processing: {input_json_path}")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flat_segments = data.get('flat_segments', [])
        metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(flat_segments)} segments")
        
        # Pre-execution Gating: Detect Main Body start/end
        gating_info = None
        if self.config.ENABLE_CONTENT_GATING:
            # We need toc_pages for better gating, detect them first
            toc_pages = self._detect_toc_pages(flat_segments)
            gatekeeper = ContentGatekeeper(self.config)
            gating_info = gatekeeper.analyze(flat_segments, detected_toc_pages=toc_pages)
            if gating_info.get('start_id'):
                logger.info("Main Body Gating: Active for this document")

        # Apply Three-Phase Correction Pipeline
        # This must run BEFORE continuation detection to ensure correct reading order
        if (self.config.ENABLE_READING_ORDER_CORRECTION or 
            self.config.ENABLE_HEADING_RECONSTRUCTION or 
            self.config.ENABLE_BACKFILL_CORRECTION):
            flat_segments = self.reading_order_corrector.process(flat_segments, gating_info)
            corrector_stats = self.reading_order_corrector.get_stats()
        else:
            corrector_stats = {}
        
        # Optional: Strip Front Matter segments completely if configured
        if self.config.ENABLE_CONTENT_GATING and self.config.STRIP_FRONT_MATTER:
            original_len = len(flat_segments)
            flat_segments = [s for s in flat_segments if s.get('doc_zone') != 'front']
            logger.info(f"Gating: Stripped {original_len - len(flat_segments)} front-matter segments")
        
        # Annotate segments with continuation markers
        if self.config.ENABLE_CONTINUATION_DETECTION:
            flat_segments = self.continuation_detector.annotate_segments(flat_segments)
            continuation_count = sum(1 for s in flat_segments if s.get('is_continuation') != 'none')
            logger.info(f"Detected {continuation_count} cross-page continuations")
        
        # Detect TOC pages
        toc_pages = self._detect_toc_pages(flat_segments)
        if toc_pages:
            logger.info(f"Detected TOC on pages: {sorted(list(toc_pages))}")
        
        # Build block catalog for reference detection
        self.block_catalog = self.reference_detector.build_block_catalog(flat_segments)
        logger.info(f"Built block catalog: {len(self.block_catalog['by_segment_id'])} blocks, "
                   f"{len(self.block_catalog['by_caption'])} with captions")
        
        # Process segments into chunks
        chunks = self._process_segments(flat_segments, toc_pages=toc_pages)
        
        # Post-process: merge short chunks and apply overlap
        chunks = self._post_process_chunks(chunks)
        
        # Build feature list
        features = ["context_overlap", "imperative_detection", "theorem_tagging", "length_control"]
        if self.config.ENABLE_READING_ORDER_CORRECTION:
            features.append("reading_order_correction")
        if self.config.ENABLE_HEADING_RECONSTRUCTION:
            features.append("heading_reconstruction")
        if self.config.ENABLE_BACKFILL_CORRECTION:
            features.append("backfill_correction")
        if self.config.ENABLE_CONTINUATION_DETECTION:
            features.append("cross_page_continuation")
        if self.config.ENABLE_FURNITURE_DETECTION:
            features.append("furniture_detection")
        if self.config.ENABLE_DEHYPHENATION:
            features.append("dehyphenation")
        features.append("reference_detection")  # NEW
        
        # Calculate stats and include corrector stats
        processing_stats = self._calculate_stats(chunks)
        if corrector_stats:
            processing_stats['reading_order_correction'] = corrector_stats
        
        result = {
            "metadata": {
                **metadata,
                "total_chunks": len(chunks),
                "total_segments": len(flat_segments),
                "processing_version": "production",
                "features": features,
                "processing_stats": processing_stats
            },
            "chunks": [asdict(c) for c in chunks]
        }
        
        if output_path:
            self._save_json(result, output_path)
        
        return result
    
    def _detect_toc_pages(self, segments: List[Dict]) -> set:
        """Detect pages that are likely Table of Contents."""
        from collections import defaultdict
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
            # 1. Keyword check (prioritize early pages)
            if any(kw in full_text for kw in toc_keywords) and p < 60:
                toc_pages.add(p)
                continue
            # 2. Structure check: dotted leaders + numbers at end of lines
            # Restricted to very early document to avoid misdetecting List of Tables/Figures
            if p < 45: 
                # Require more dots (5) and ensure it looks like a TOC line
                dotted_lines = sum(1 for t in texts if re.search(r'[\.·-]{5,}\s*\d+$', t))
                if dotted_lines >= 3:
                    toc_pages.add(p)
        return toc_pages

    def _process_segments(self, segments: List[Dict], toc_pages: set = None) -> List[EnrichedChunk]:
        """
        Core processing logic:
        1. Handle cross-page continuations with evidence tracking
        2. Group related segments (lists, procedures)
        3. Create enriched chunks with continuation metadata and evidence
        """
        chunks = []
        buffer = []
        current_heading_path = ""
        has_cross_page = False  # Track if current buffer spans pages
        continuation_type = "none"  # Track continuation confidence
        merge_evidences = []  # Collect all evidence for this buffer
        
        toc_pages = toc_pages or set()
        
        for i, seg in enumerate(segments):
            seg_text = seg.get('text', '')
            seg_type = seg.get('type', 'Paragraph')
            heading_path = seg.get('heading_path', '')
            is_continuation = seg.get('is_continuation', 'none')
            continuation_evidence = seg.get('continuation_evidence', {})
            seg_col = seg.get('column_index', -1)
            
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
                chunk = self._create_chunk(buffer, current_heading_path)
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
                    chunk = self._create_chunk(buffer, current_heading_path)
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
                    chunk = self._create_chunk(buffer, current_heading_path)
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
                chunks.append(self._create_chunk([seg], heading_path, chunk_type="header"))
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
                    chunk = self._create_chunk(buffer, current_heading_path)
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
                chunk = self._create_chunk(buffer, current_heading_path, chunk_type="list")
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
                    chunk = self._create_chunk(other_segs, current_heading_path)
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
                                 chunk = self._create_chunk(all_caption_content, current_heading_path)
                                 chunks.append(chunk)
                             # Create standalone block
                             chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                        else:
                            # SUCCESS: Bond caption + notes + block into single atomic unit
                            bonded_segments = [caption_seg] + notes_segs + [seg]
                            caption_text = caption_seg.get('text', '').strip()
                            effective_heading_path = self.structural_heading_path or current_heading_path
                            
                            chunk = self._create_chunk(
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
                            chunk = self._create_chunk(all_caption_content, current_heading_path)
                            chunks.append(chunk)
                        # Create standalone block
                        chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                else:
                    # No caption in buffer, just create standalone chunk
                    if notes_segs:
                        # Flush notes first
                        chunk = self._create_chunk(notes_segs, current_heading_path)
                        chunks.append(chunk)
                    
                    chunks.append(self._create_chunk([seg], heading_path, chunk_type=seg_type.lower()))
                
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
                    chunk = self._create_chunk(buffer, current_heading_path)
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

                chunk = self._create_chunk(buffer, current_heading_path, chunk_type="learning_objective")
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
                    chunk = self._create_chunk(buffer, current_heading_path)
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
                    chunk = self._create_chunk(buffer, current_heading_path)
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
            chunk = self._create_chunk(buffer, current_heading_path)
            chunk.is_cross_page = has_cross_page
            chunk.continuation_type = continuation_type
            chunk.needs_review = (continuation_type == 'partial')
            chunk.merge_evidence = self._compile_merge_evidence(merge_evidences)
            chunks.append(chunk)
        
        return chunks
    
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
                    
                    merged_chunk = EnrichedChunk(
                        chunk_id=current.chunk_id,
                        heading_path=current.heading_path,
                        chunk_type=self._infer_chunk_type_from_tags(merged_tags),
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
                        word_count=len(merged_content.split())
                    )
                    merged.append(merged_chunk)
                    i += 2  # Skip both chunks
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _create_chunk(self, segments: List[Dict], heading_path: str, 
                      chunk_type: Optional[str] = None) -> EnrichedChunk:
        """Create an enriched chunk from a list of segments."""
        self.chunk_counter += 1
        chunk_id = f"chunk_{self.chunk_counter:04d}"
        
        # Combine text - exclude furniture/noise from content
        # Apply dehyphenation to repair word breaks across segments
        text_parts = []
        for s in segments:
            # Skip furniture elements (their text shouldn't appear in content)
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
                    # Try to merge hyphenated word from previous part
                    prev_part = repaired_parts[-1]
                    new_prev, new_curr = dehyph.merge_hyphenated(prev_part, part)
                    repaired_parts[-1] = new_prev
                    if new_curr:  # Only add if there's remaining text
                        repaired_parts.append(new_curr)
                    elif not new_curr and new_prev != prev_part:
                        # Word was fully absorbed into previous part
                        pass
            
            text_parts = repaired_parts
        
        full_text = " ".join(text_parts).strip()
        
        # Word count
        word_count = len(full_text.split())
        
        # Get source segment IDs (include noise headers for traceability)
        source_ids = [s.get('segment_id', '') for s in segments if s.get('segment_id')]
        
        # Get page range
        pages = [s.get('page', 0) for s in segments if s.get('page')]
        page_range = [min(pages), max(pages)] if pages else []
        
        # Get depth
        depth = segments[0].get('depth', 0) if segments else 0
        
        # Detect tags
        tags = self.tag_detector.detect_tags(full_text)
        
        # Check for imperative start
        if self.tag_detector.detect_imperative(full_text):
            if "procedure" not in tags:
                tags.append("procedure")
        
        # Determine chunk type early to influence sentence roles
        if chunk_type is None:
            chunk_type = self._infer_chunk_type(segments, tags, [])
            
        # Forced Role Linkage logic
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
            
            # AGGREGATE TAGS: Ensure chunk tags include all tags found in any sentence
            # This is crucial for 'reference' detection in complex sentences
            sentence_tag_set = set(tags)
            for sent in sentences:
                if "tags" in sent:
                    sentence_tag_set.update(sent["tags"])
            tags = list(sentence_tag_set)
        
        # Determine chunk type if not provided
        if chunk_type is None:
            chunk_type = self._infer_chunk_type(segments, tags, sentences)
        
        # Detect references to Figure/Table/Equation (NEW)
        references = []
        if self.block_catalog and full_text:
            chunk_page = page_range[0] if page_range else 0
            references = self.reference_detector.detect_references(
                full_text, chunk_page, self.block_catalog
            )
        
        # Determine dominant zone for the chunk
        zones = [s.get('doc_zone', 'body') for s in segments]
        dominant_zone = max(set(zones), key=zones.count) if zones else "body"
        
        return EnrichedChunk(
            chunk_id=chunk_id,
            heading_path=heading_path,
            chunk_type=chunk_type,
            content=full_text,
            context_prefix="",  # Will be filled in post-processing
            sentences=sentences,
            tags=tags,
            source_segments=source_ids,
            page_range=page_range,
            depth=depth,
            word_count=word_count,
            references=references,
            doc_zone=dominant_zone
        )
    
    def _infer_chunk_type(self, segments: List[Dict], tags: List[str], 
                          sentences: List[Dict]) -> str:
        """Infer chunk type from segments, tags, and sentence analysis."""
        seg_types = [s.get('type', '') for s in segments]
        
        # Check segment types first
        if all(t == 'ListItem' for t in seg_types):
            return "list"
        
        if all(t == 'LearningObjective' for t in seg_types):
            return "learning_objective"
        
        # Check for theorem/proof (highest priority for academic content)
        if "theorem" in tags:
            return "theorem"
        if "proof" in tags:
            return "proof"
        
        # Check for imperative sentences (indicates procedure)
        if sentences:
            imperative_count = sum(1 for s in sentences if s.get('is_imperative', False))
            if imperative_count > 0 and imperative_count >= len(sentences) / 2:
                return "procedure"
        
        # Check other tags
        if "definition" in tags:
            return "definition"
        if "procedure" in tags:
            return "procedure"
        if "example" in tags:
            return "example"
        if "exercise" in tags:
            return "exercise"
        if "formula" in tags:
            return "formula"
        if "summary" in tags:
            return "summary"
        
        return "explanation"
    
    def _infer_chunk_type_from_tags(self, tags: List[str]) -> str:
        """Infer chunk type from tags only (for merged chunks)."""
        priority = ["theorem", "proof", "definition", "procedure", "example", 
                    "exercise", "formula", "summary"]
        for tag in priority:
            if tag in tags:
                return tag
        return "explanation"
    
    def _calculate_stats(self, chunks: List[EnrichedChunk]) -> Dict[str, Any]:
        """Calculate processing statistics including cross-page metrics."""
        type_counts = {}
        tag_counts = {}
        word_counts = []
        overlap_count = 0
        imperative_count = 0
        
        # Cross-page statistics
        cross_page_count = 0
        full_continuation_count = 0
        partial_continuation_count = 0
        needs_review_count = 0
        
        for chunk in chunks:
            # Count types
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
            # Count tags
            for tag in chunk.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            # Word counts
            word_counts.append(chunk.word_count)
            # Overlap
            if chunk.context_prefix:
                overlap_count += 1
            # Imperative sentences
            for sent in chunk.sentences:
                if sent.get('is_imperative', False):
                    imperative_count += 1
            
            # Cross-page continuation stats
            if chunk.is_cross_page:
                cross_page_count += 1
            if chunk.continuation_type == 'full':
                full_continuation_count += 1
            elif chunk.continuation_type == 'partial':
                partial_continuation_count += 1
            if chunk.needs_review:
                needs_review_count += 1
        
        return {
            "chunk_types": type_counts,
            "tag_distribution": tag_counts,
            "avg_sentences_per_chunk": sum(len(c.sentences) for c in chunks) / len(chunks) if chunks else 0,
            "avg_words_per_chunk": sum(word_counts) / len(word_counts) if word_counts else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "chunks_with_overlap": overlap_count,
            "imperative_sentences_detected": imperative_count,
            # Cross-page continuation stats
            "cross_page_chunks": cross_page_count,
            "full_continuations": full_continuation_count,
            "partial_continuations": partial_continuation_count,
            "chunks_needing_review": needs_review_count
        }
    
    def _save_json(self, data: Dict, path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved enriched chunks to: {path}")
