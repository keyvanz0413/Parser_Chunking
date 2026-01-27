import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from ..config import ChunkingConfig

logger = logging.getLogger(__name__)

class ContinuationDetector:
    """
    Enhanced cross-page paragraph continuation detector.
    
    Implements the task requirements for accurate cross-page detection:
    - Incomplete sentences (missing terminal punctuation)
    - Hyphenation (word breaks across lines)
    - Abrupt clause endings (sentences ending with prepositions/conjunctions)
    - bbox position validation
    - Skip furniture elements (page decorations) when detecting continuations
    
    Strategy:
    1. Check if previous segment is at page bottom and current is at page top
    2. Verify style consistency (column, text patterns)
    3. Calculate detailed continuation score with evidence
    4. Return continuation type with merge evidence for explainability
    5. Skip furniture elements (now detected by FurnitureDetector) when crossing pages
    
    Returns:
    - 'full': High confidence continuation, auto-merge (score >= 0.7)
    - 'partial': Uncertain, needs human review (score 0.4-0.7)
    - 'none': Not a continuation (score < 0.4)
    """
    
    # Types that should NOT be merged across pages
    BREAK_TYPES = {'Header', 'Table', 'Picture', 'Formula', 'LearningObjective'}
    
    # Fallback noise patterns (used if is_furniture not set)
    NOISE_HEADER_PATTERNS = [
        r'^\s*\(?\s*concluded\s*\)?\s*$',
        r'^\s*\(?\s*continued\s*\)?\s*$',
        r'^\s*\(?\s*cont\'?d?\s*\)?\s*$',
        r'^\s*\(?\s*continuation\s*\)?\s*$',
        r'^\s*continued\s+(?:on|from)\s+(?:next|previous)?\s*page',
        r'^\s*see\s+(?:next|previous)\s+page',
        r'^\s*to\s+be\s+continued',
    ]
    
    # Score thresholds
    FULL_THRESHOLD = 0.7
    PARTIAL_THRESHOLD = 0.4
    
    # Prepositions and conjunctions that shouldn't end a sentence
    OPEN_CLAUSE_WORDS = {
        # Prepositions
        'of', 'to', 'for', 'with', 'in', 'on', 'at', 'by', 'from', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between',
        'under', 'over', 'about', 'against', 'among', 'around', 'without',
        # Conjunctions
        'and', 'or', 'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        # Articles (often indicate continuation)
        'the', 'a', 'an',
        # Relative pronouns / determiners
        'that', 'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
        # Other incomplete indicators
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'as', 'if', 'whether', 'although', 'because', 'since', 'unless',
        'such', 'than', 'these', 'those', 'this', 'its',
        'real', 'financial', 'more', 'most', 'less', 'least', 'other', 'another'
    }

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._last_evidence = {}  # Store last detection's evidence
        # Compile noise patterns for fallback
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_HEADER_PATTERNS]
    
    def _is_noise_header(self, seg: Dict[str, Any]) -> bool:
        """
        Check if a segment is furniture/noise that should be skipped.
        
        Uses the is_furniture flag set by FurnitureDetector (preferred),
        falls back to pattern matching if flag not available.
        """
        # First check the is_furniture flag (set by FurnitureDetector in Pre-Phase)
        if seg.get('is_furniture', False) or seg.get('is_noise_header', False):
            return True
        
        # Fallback: pattern-based detection for Headers
        if seg.get('type') != 'Header':
            return False
        
        text = seg.get('text', '').strip()
        if not text:
            return False
        
        for pattern in self._noise_patterns:
            if pattern.match(text):
                return True
        
        return False
    
    def detect_continuation(self, prev_seg: Dict[str, Any], curr_seg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Detect if curr_seg is a continuation of prev_seg.
        
        Returns:
            Tuple of (continuation_type, evidence_dict)
            - continuation_type: 'full', 'partial', or 'none'
            - evidence_dict: Detailed reasons for the decision
        """
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
            evidence["rules_failed"].append("continuation_detection_disabled")
            self._last_evidence = evidence
            return 'none', evidence
        
        prev_page = prev_seg.get('page', 0)
        curr_page = curr_seg.get('page', 0)
        prev_type = prev_seg.get('type', 'Paragraph')
        curr_type = curr_seg.get('type', 'Paragraph')
        
        # =====================================================================
        # Rule 1: Break types never continue
        # =====================================================================
        if curr_type in self.BREAK_TYPES or prev_type in self.BREAK_TYPES:
            evidence["rules_failed"].append(f"break_type: prev={prev_type}, curr={curr_type}")
            self._last_evidence = evidence
            return 'none', evidence
        
        # =====================================================================
        # Rule 2: Must be on different pages (cross-page only)
        # =====================================================================
        if curr_page == prev_page:
            evidence["rules_failed"].append(f"same_page: {curr_page}")
            self._last_evidence = evidence
            return 'none', evidence
        
        # =====================================================================
        # Rule 3: Must be consecutive pages
        # =====================================================================
        if curr_page != prev_page + 1:
            evidence["rules_failed"].append(f"non_consecutive: prev={prev_page}, curr={curr_page}")
            self._last_evidence = evidence
            return 'none', evidence
        
        evidence["rules_triggered"].append("consecutive_pages")
        
        # =====================================================================
        # Rule 4: Check page position (bottom -> top)
        # =====================================================================
        prev_at_bottom = prev_seg.get('at_page_bottom', False)
        curr_at_top = curr_seg.get('at_page_top', False)
        
        # If position info is missing, estimate from bbox
        if 'at_page_bottom' not in prev_seg:
            prev_at_bottom = self._estimate_at_bottom(prev_seg)
        if 'at_page_top' not in curr_seg:
            curr_at_top = self._estimate_at_top(curr_seg)
        
        if not prev_at_bottom:
            evidence["rules_failed"].append("prev_not_at_bottom")
        else:
            evidence["rules_triggered"].append("prev_at_page_bottom")
            
        if not curr_at_top:
            evidence["rules_failed"].append("curr_not_at_top")
        else:
            evidence["rules_triggered"].append("curr_at_page_top")
        
        # Position check is a soft requirement - low score if fails
        position_valid = prev_at_bottom and curr_at_top
        
        # =====================================================================
        # Rule 5: Check column consistency
        # =====================================================================
        prev_col = prev_seg.get('column_index', 0)
        curr_col = curr_seg.get('column_index', 0)
        
        column_consistent = True
        if self.config.CONTINUATION_COLUMN_STRICT:
            # Spanning items (-1) always consistent
            if prev_col >= 0 and curr_col >= 0 and prev_col != curr_col:
                column_consistent = False
                evidence["rules_failed"].append(f"column_mismatch: prev={prev_col}, curr={curr_col}")
            else:
                evidence["rules_triggered"].append("column_consistent")
        
        # =====================================================================
        # Calculate comprehensive style score with breakdown
        # =====================================================================
        score, score_breakdown = self._calculate_enhanced_style_score(
            prev_seg, curr_seg, position_valid, column_consistent
        )
        
        evidence["score_breakdown"] = score_breakdown
        evidence["final_score"] = round(score, 3)
        
        # =====================================================================
        # Make decision based on score
        # =====================================================================
        if score >= self.FULL_THRESHOLD:
            decision = 'full'
            evidence["decision"] = "full (high confidence merge)"
        elif score >= self.PARTIAL_THRESHOLD:
            decision = 'partial'
            evidence["decision"] = "partial (needs review)"
        else:
            decision = 'none'
            evidence["decision"] = "none (score too low)"
        
        self._last_evidence = evidence
        
        if decision != 'none':
            logger.debug(f"Cross-page continuation: {prev_seg.get('segment_id')} -> "
                        f"{curr_seg.get('segment_id')} = {decision} (score: {score:.2f})")
        
        return decision, evidence
    
    def _calculate_enhanced_style_score(self, prev_seg: Dict[str, Any], curr_seg: Dict[str, Any],
                                        position_valid: bool, column_consistent: bool) -> Tuple[float, Dict]:
        """
        Calculate comprehensive continuation score with detailed breakdown.
        
        Enhanced scoring using task requirements:
        - Hyphenation detection (high weight)
        - Open clause endings (high weight)
        - Sentence completeness
        - Start pattern analysis
        """
        breakdown = {}
        total_score = 0.0
        
        prev_hints = prev_seg.get('style_hints') or self._detect_text_style(prev_seg)
        curr_hints = curr_seg.get('style_hints') or self._detect_text_style(curr_seg)
        
        # =====================================================================
        # Factor 1: Position validity (weight: 0.15)
        # =====================================================================
        weight = 0.15
        if position_valid:
            total_score += weight
            breakdown["position_valid"] = {"score": weight, "max": weight, "detail": "bottom->top verified"}
        else:
            breakdown["position_valid"] = {"score": 0, "max": weight, "detail": "position check failed"}
        
        # =====================================================================
        # Factor 2: Column consistency (weight: 0.10)
        # =====================================================================
        weight = 0.10
        if column_consistent:
            total_score += weight
            breakdown["column_consistent"] = {"score": weight, "max": weight, "detail": "same column"}
        else:
            breakdown["column_consistent"] = {"score": 0, "max": weight, "detail": "column mismatch"}
        
        # =====================================================================
        # Factor 3: Hyphenation - STRONG INDICATOR (weight: 0.20)
        # =====================================================================
        weight = 0.20
        if prev_hints.get('ends_with_hyphen', False):
            total_score += weight
            breakdown["hyphenation"] = {"score": weight, "max": weight, "detail": "word break at line end"}
        else:
            breakdown["hyphenation"] = {"score": 0, "max": weight, "detail": "no hyphenation"}
        
        # =====================================================================
        # Factor 4: Open clause ending - STRONG INDICATOR (weight: 0.15)
        # =====================================================================
        weight = 0.15
        if prev_hints.get('ends_with_open_clause', False):
            total_score += weight
            # Extract last word for evidence
            prev_text = prev_seg.get('text', '')
            last_word = prev_text.split()[-1] if prev_text.split() else ''
            breakdown["open_clause"] = {"score": weight, "max": weight, 
                                        "detail": f"ends with '{last_word}'"}
        else:
            breakdown["open_clause"] = {"score": 0, "max": weight, "detail": "complete clause"}
        
        # =====================================================================
        # Factor 5: Fragment pattern (weight: 0.10)
        # =====================================================================
        weight = 0.10
        if prev_hints.get('has_fragment_pattern', False):
            total_score += weight
            breakdown["fragment_pattern"] = {"score": weight, "max": weight, 
                                             "detail": "comma/ellipsis/equation ending"}
        else:
            breakdown["fragment_pattern"] = {"score": 0, "max": weight, "detail": "no fragment pattern"}
        
        # =====================================================================
        # Factor 6: Continuation start pattern (weight: 0.15)
        # =====================================================================
        weight = 0.15
        if curr_hints.get('starts_like_continuation', False):
            total_score += weight
            curr_text = curr_seg.get('text', '')
            first_word = curr_text.split()[0] if curr_text.split() else ''
            breakdown["continuation_start"] = {"score": weight, "max": weight,
                                               "detail": f"starts with '{first_word}'"}
        elif curr_hints.get('starts_lowercase', False):
            # Partial credit for lowercase start
            partial = weight * 0.7
            total_score += partial
            breakdown["continuation_start"] = {"score": partial, "max": weight,
                                               "detail": "starts lowercase"}
        else:
            breakdown["continuation_start"] = {"score": 0, "max": weight, "detail": "normal start"}
        
        # =====================================================================
        # Factor 7: Same heading path (weight: 0.05)
        # =====================================================================
        weight = 0.05
        if prev_seg.get('heading_path') == curr_seg.get('heading_path'):
            total_score += weight
            breakdown["same_heading"] = {"score": weight, "max": weight, "detail": "same section"}
        else:
            breakdown["same_heading"] = {"score": 0, "max": weight, "detail": "different sections"}
        
        # =====================================================================
        # Factor 8: Same segment type (weight: 0.05)
        # =====================================================================
        weight = 0.05
        if prev_seg.get('type') == curr_seg.get('type'):
            total_score += weight
            breakdown["same_type"] = {"score": weight, "max": weight, 
                                      "detail": prev_seg.get('type', 'unknown')}
        else:
            breakdown["same_type"] = {"score": 0, "max": weight, 
                                      "detail": f"{prev_seg.get('type')} -> {curr_seg.get('type')}"}
        
        # =====================================================================
        # Factor 9: Incomplete sentence (weight: 0.40) - DECISIVE FACTOR
        # =====================================================================
        weight = 0.40  # Massive increase: if it's not done, it's not done.
        if prev_hints.get('ends_incomplete', False) and not prev_hints.get('has_terminal_punctuation', True):
            total_score += weight
            breakdown["incomplete_sentence"] = {"score": weight, "max": weight, 
                                                "detail": "no terminal punctuation"}
        else:
            breakdown["incomplete_sentence"] = {"score": 0, "max": weight, 
                                                "detail": "complete sentence"}
        
        # =====================================================================
        # Factor 10: STRONG COMBINATION - Incomplete + Lowercase start
        # =====================================================================
        weight = 0.30  # Increased from 0.20
        prev_incomplete = prev_hints.get('ends_incomplete', False) and not prev_hints.get('has_terminal_punctuation', True)
        curr_lowercase = curr_hints.get('starts_lowercase', False)
        
        if prev_incomplete and curr_lowercase:
            total_score += weight
            breakdown["strong_continuation_signal"] = {
                "score": weight, "max": weight, 
                "detail": "incomplete sentence + lowercase start (strong signal)"
            }
        else:
            breakdown["strong_continuation_signal"] = {
                "score": 0, "max": weight, 
                "detail": "no strong continuation pattern"
            }

        # =====================================================================
        # Factor 11: List Item Continuation (weight: 0.50)
        # =====================================================================
        # New structural rule: If prev is ListItem and looks unfinished, 
        # and curr is ListItem, almost certainly a continuation.
        weight = 0.50
        prev_is_list = prev_seg.get('type') == 'ListItem'
        curr_is_list = curr_seg.get('type') == 'ListItem'
        
        # Check if prev list item ended abruptly (no punctuation or just comma/semicolon)
        prev_text = prev_seg.get('text', '').strip()
        list_incomplete = False
        if prev_text and prev_text[-1] not in ['.', '!', '?']:
            list_incomplete = True
            
        if prev_is_list and curr_is_list and list_incomplete:
             total_score += weight
             breakdown["list_continuation"] = {
                 "score": weight, "max": weight,
                 "detail": "split list item detected"
             }
        else:
             breakdown["list_continuation"] = {
                 "score": 0, "max": weight,
                 "detail": "no list split"
             }
        
        # Calculate summary
        breakdown["total"] = round(total_score, 3)
        breakdown["max_possible"] = 1.0
        
        return total_score, breakdown
    
    def _estimate_at_bottom(self, seg: Dict[str, Any]) -> bool:
        """Estimate if segment is at page bottom based on bbox."""
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4:
            return False
            
        # Using Safe Zone threshold from config
        # PDF Coordinates: Low Y is bottom
        ph = self.config.PAGE_HEIGHT_DEFAULT
        bottom_threshold = ph * self.config.FURNITURE_BOTTOM_BAND
        return bbox[3] < bottom_threshold
    
    def _estimate_at_top(self, seg: Dict[str, Any]) -> bool:
        """Estimate if segment is at page top based on bbox."""
        bbox = seg.get('bbox')
        if not bbox or len(bbox) < 4:
            return False
            
        # Using Safe Zone threshold from config
        # PDF Coordinates: High Y is top
        ph = self.config.PAGE_HEIGHT_DEFAULT
        top_threshold = ph * (1 - self.config.FURNITURE_TOP_BAND)
        return bbox[1] > top_threshold
    
    def get_last_evidence(self) -> Dict[str, Any]:
        """Return the evidence from the last detection call."""
        return self._last_evidence
    
    def _find_previous_content_segment(self, segments: List[Dict], current_index: int) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Find the previous content segment, skipping noise headers.
        
        Returns:
            Tuple of (previous_content_segment, list_of_skipped_noise_segments)
            Returns (None, []) if no valid previous segment found.
        """
        skipped = []
        for j in range(current_index - 1, -1, -1):
            seg = segments[j]
            # Skip noise headers AND regular headers when looking for paragraph flow
            # (unless the current segment is also a header, but paragraph continuation 
            # is our primary concern here)
            if seg.get('type') == 'Header' or self._is_noise_header(seg):
                skipped.append(seg)
                continue
            # Found a non-noise, non-header segment
            return seg, skipped
        return None, skipped
    
    def _detect_text_style(self, seg: Dict[str, Any]) -> Dict[str, Any]:
        """Detect text style hints for continuation matching if not provided."""
        text = seg.get('text', '')
        seg_type = seg.get('type', 'Paragraph')
        text_stripped = text.rstrip() if text else ""
        
        # Terminal punctuation check
        has_terminal = text_stripped.endswith(('.', '?', '!', ':', ';')) if text_stripped else False
        
        # Open clause check
        words = text_stripped.rstrip('.,;:!?').split()
        last_word = words[-1].lower() if words else ''
        ends_with_open_clause = last_word in self.OPEN_CLAUSE_WORDS
        
        # Lowercase start
        starts_lowercase = text.strip()[0].islower() if text.strip() else False
        
        # Starts like continuation
        starts_like_continuation = False
        if text.strip():
            first_word = text.strip().split()[0].lower().rstrip('.,;:')
            continuation_starters = {
                'and', 'or', 'but', 'nor', 'so', 'yet', 'also', 'however',
                'therefore', 'thus', 'hence', 'moreover', 'furthermore',
                'which', 'that', 'who', 'where', 'when', 'while',
                'because', 'since', 'although', 'though', 'unless',
                'as', 'if', 'whether', 'whereas',
            }
            starts_like_continuation = first_word in continuation_starters or starts_lowercase

        return {
            "starts_lowercase": starts_lowercase,
            "ends_incomplete": text and not has_terminal,
            "ends_with_open_clause": ends_with_open_clause,
            "starts_like_continuation": starts_like_continuation,
            "has_terminal_punctuation": has_terminal,
            "ends_with_hyphen": text_stripped.endswith('-'),
            "has_fragment_pattern": text_stripped.endswith(',') or text_stripped.endswith('...')
        }

    def annotate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Annotate all segments with continuation markers and evidence.
        
        Enhanced: Skips noise headers (like "(concluded)") when detecting
        cross-page continuations. This allows paragraphs to be correctly merged
        even when page decoration elements appear between them.
        
        Adds to each segment:
        - 'is_continuation': 'full', 'partial', or 'none'
        - 'continuation_evidence': Dict with detailed reasons
        - 'skipped_noise_headers': List of noise headers that were skipped (if any)
        """
        if len(segments) < 2:
            for seg in segments:
                seg['is_continuation'] = 'none'
                seg['continuation_evidence'] = {}
            return segments
        
        # First segment is never a continuation
        segments[0]['is_continuation'] = 'none'
        segments[0]['continuation_evidence'] = {}
        
        for i in range(1, len(segments)):
            curr_seg = segments[i]
            
            # Skip if current segment is a noise header
            if self._is_noise_header(curr_seg):
                curr_seg['is_continuation'] = 'none'
                curr_seg['continuation_evidence'] = {"skipped": "noise_header"}
                curr_seg['is_noise_header'] = True
                continue
            
            # Find previous content segment, skipping noise headers
            prev_seg, skipped_noise = self._find_previous_content_segment(segments, i)
            
            if prev_seg is None:
                curr_seg['is_continuation'] = 'none'
                curr_seg['continuation_evidence'] = {}
                continue
            
            # Detect continuation
            continuation, evidence = self.detect_continuation(prev_seg, curr_seg)
            
            # Record skipped noise headers in evidence
            if skipped_noise:
                evidence['skipped_noise_headers'] = [s.get('segment_id') for s in skipped_noise]
                evidence['skipped_noise_texts'] = [s.get('text', '')[:50] for s in skipped_noise]
            
            curr_seg['is_continuation'] = continuation
            curr_seg['continuation_evidence'] = evidence
            
            if continuation != 'none':
                skip_info = f" (skipped {len(skipped_noise)} noise headers)" if skipped_noise else ""
                logger.debug(f"Cross-page continuation: {prev_seg.get('segment_id')} -> "
                           f"{curr_seg.get('segment_id')} ({continuation}, score: {evidence.get('final_score', 0):.2f}){skip_info}")
        
        return segments
