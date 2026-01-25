import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from ...core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class ContinuationDetector:
    """Enhanced cross-page paragraph continuation detector."""
    
    # Thresholds for decision making
    FULL_THRESHOLD = 0.70    
    PARTIAL_THRESHOLD = 0.45 
    
    # Types that shouldn't be merged
    BREAK_TYPES = ['Table', 'Picture', 'Formula', 'LearningObjective']
    
    def __init__(self, config: ChunkingConfig = None, semantic_analyzer=None):
        self.config = config or ChunkingConfig()
        self.semantic_analyzer = semantic_analyzer
        self._last_evidence = {}
        self.HYPHEN_CHARS = ['-', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014']

    def detect_continuation(self, prev_seg: Dict[str, Any], curr_seg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        evidence = {
            "prev_segment_id": prev_seg.get('segment_id', ''),
            "curr_segment_id": curr_seg.get('segment_id', ''),
            "rules_triggered": [],
            "rules_failed": [],
            "score_breakdown": {},
            "final_score": 0.0,
            "decision": "none",
        }
        
        if not self.config.ENABLE_CONTINUATION_DETECTION:
            return 'none', evidence
        
        prev_type = prev_seg.get('type', 'Paragraph')
        curr_type = curr_seg.get('type', 'Paragraph')
        if curr_type in self.BREAK_TYPES or prev_type in self.BREAK_TYPES:
            return 'none', evidence

        # Score calculation
        position_valid = self._check_position(prev_seg, curr_seg)
        column_consistent = self._check_columns(prev_seg, curr_seg)
        
        score, breakdown = self._calculate_style_score(prev_seg, curr_seg, position_valid, column_consistent)
        
        # Add semantic score if available (NEW)
        if self.semantic_analyzer and self.config.ENABLE_SEMANTIC_ANALYSIS:
            sem_score = self.semantic_analyzer.compute_cross_page_semantic_score(
                prev_seg.get('text', ''), curr_seg.get('text', '')
            )
            score = (score * 0.7) + (sem_score * 0.3)
            breakdown["semantic_continuity"] = {"score": round(sem_score * 0.3, 3), "original": sem_score}
            
        evidence["score_breakdown"] = breakdown
        evidence["final_score"] = round(score, 3)
        
        if score >= self.FULL_THRESHOLD:
            decision = 'full'
        elif score >= self.PARTIAL_THRESHOLD:
            decision = 'partial'
        else:
            decision = 'none'
            
        return decision, evidence

    def _check_position(self, prev: Dict, curr: Dict) -> bool:
        # Same page allowed now (multi-column), but must be physically separated
        if prev.get('page') == curr.get('page'):
            # Check if prev is bottom of one column and curr is top of another
            # For simplicity, returning True if they are consecutive in logical flow
            return True
        return abs(curr.get('page', 0) - prev.get('page', 0)) <= 1

    def _check_columns(self, prev: Dict, curr: Dict) -> bool:
        prev_col = prev.get('column_index', 0)
        curr_col = curr.get('column_index', 0)
        if self.config.CONTINUATION_COLUMN_STRICT:
            if prev.get('page') == curr.get('page'):
                return prev_col != curr_col # Cross column same page
            return prev_col == curr_col # Same column cross page
        return True

    def _calculate_style_score(self, prev_seg: Dict, curr_seg: Dict, pos_valid: bool, col_valid: bool) -> Tuple[float, Dict]:
        # Implementation of style scoring (shortened for clarity)
        score = 0.0
        breakdown = {}
        
        text1 = prev_seg.get('text', '').strip()
        text2 = curr_seg.get('text', '').strip()
        
        if text1.endswith(tuple(self.HYPHEN_CHARS)):
            score += 0.3
            breakdown["hyphenation"] = 0.3
        
        if text2 and text2[0].islower():
            score += 0.3
            breakdown["lowercase_start"] = 0.3
            
        if not re.search(r'[.!?]$', text1):
            score += 0.2
            breakdown["incomplete_sentence"] = 0.2
            
        if pos_valid: score += 0.1
        if col_valid: score += 0.1
        
        return score, breakdown

    def annotate_segments(self, segments: List[Dict]) -> List[Dict]:
        """Optimized version with dynamic scoring and special handling for formulas/lists."""
        if not segments: return segments
        
        # 1. Structural and Style Pre-analysis
        candidates = [] 
        texts_to_encode = [] 
        mapping = [] 
        
        for i in range(1, len(segments)):
            prev_seg = segments[i-1]
            curr_seg = segments[i]
            
            # Filter
            prev_type = prev_seg.get('type', 'Paragraph')
            curr_type = curr_seg.get('type', 'Paragraph')
            if not self.config.ENABLE_CONTINUATION_DETECTION:
                segments[i]['is_continuation'] = 'none'
                continue
            
            # --- Dynamic Scoring (Probabilistic approach) ---
            score = 0.0
            breakdown = {}
            
            text1 = prev_seg.get('text', '').strip()
            text2 = curr_seg.get('text', '').strip()
            
            # Rule 1: Open_End (No punctuation at end of segment)
            if text1 and not re.search(r'[.!?:]$', text1):
                score += 0.4
                breakdown["open_end"] = 0.4
            
            # Rule 2: Lowercase Start
            if text2 and text2[0].islower():
                score += 0.3
                breakdown["lowercase_start"] = 0.3
                
            # Rule 3: Hyphenation
            if text1.endswith(tuple(self.HYPHEN_CHARS)):
                score += 0.3
                breakdown["hyphenation"] = 0.3
            
            # Rule 4: List Item Continuation (Special for Investments textbook)
            if (prev_type == 'ListItem' or re.match(r'^[a-z]\)\s+', text1)) and \
               re.match(r'^[a-z0-9]\)\s+', text2[:5]):
                # This might be next list item, but if text1 is open, it's more likely a split
                pass 
            
            # Rule 5: Formula Protection
            if prev_type == 'Formula':
                # Formulas are almost always followed by continuations like "where..."
                score += 0.5
                breakdown["formula_protection"] = 0.5

            # Layout checks
            pos_valid = self._check_position(prev_seg, curr_seg)
            col_valid = self._check_columns(prev_seg, curr_seg)
            if pos_valid: score += 0.1
            if col_valid: score += 0.1
            
            candidates.append((i, score, breakdown))
            
            # Prepare for semantic check
            if self.semantic_analyzer and self.config.ENABLE_SEMANTIC_ANALYSIS:
                # Use slightly larger context for better vector alignment
                sample1 = text1[-300:] if len(text1) > 300 else text1
                sample2 = text2[:300] if len(text2) > 300 else text2
                
                pos1 = len(texts_to_encode)
                texts_to_encode.append(sample1)
                pos2 = len(texts_to_encode)
                texts_to_encode.append(sample2)
                mapping.append((i, pos1, pos2))

        # 2. Batch Semantic Encoding
        semantic_scores = {}
        if texts_to_encode:
            embeddings = self.semantic_analyzer.encode_sentences(texts_to_encode)
            if embeddings is not None:
                for idx, p1, p2 in mapping:
                    sim = self.semantic_analyzer.compute_similarity(embeddings[p1], embeddings[p2])
                    semantic_scores[idx] = max(0.0, float(sim))

        # 3. Final Computation & Decision
        for idx, style_score, breakdown in candidates:
            sem_score = semantic_scores.get(idx, 0.5)
            # Weighted combine (higher weight on style for physical breaks)
            final_score = (style_score * 0.6) + (sem_score * 0.4)
            breakdown["semantic_continuity"] = {"score": round(sem_score * 0.4, 3), "original": sem_score}
            
            if final_score >= self.FULL_THRESHOLD:
                decision = 'full'
            elif final_score >= self.PARTIAL_THRESHOLD:
                decision = 'partial'
            else:
                decision = 'none'
                
            segments[idx]['is_continuation'] = decision
            segments[idx]['continuation_evidence'] = {
                "score_breakdown": breakdown,
                "final_score": round(final_score, 3),
                "decision": decision
            }
            
        return segments

    def _is_noise_header(self, seg: Dict) -> bool:
        # Check against patterns like (continued)
        return False 
