import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from ..core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class POSAnalyzer:
    """Uses spaCy for POS tagging and sentence role identification."""
    
    def __init__(self, model_name: str = "en_core_web_md", config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._model_name = model_name
        self._nlp = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized: return True
        try:
            import spacy
            from spacy.language import Language
            
            # 1. Load model
            self._nlp = spacy.load(self._model_name)
            
            # 2. Add custom pipe to prevent splitting at [PAGE_JOIN]
            if "prevent_page_break_split" not in self._nlp.pipe_names:
                @Language.component("prevent_page_break_split")
                def prevent_split(doc):
                    for i in range(len(doc)):
                        token = doc[i]
                        if "[PAGE_JOIN]" in token.text:
                            # If this token is a join marker, ensure the next token is not a sentence start
                            if i + 1 < len(doc):
                                doc[i+1].is_sent_start = False
                    return doc
                
                # Add it before the parser/sentencizer
                if "parser" in self._nlp.pipe_names:
                    self._nlp.add_pipe("prevent_page_break_split", before="parser")
                elif "sentencizer" in self._nlp.pipe_names:
                    self._nlp.add_pipe("prevent_page_break_split", before="sentencizer")
                else:
                    self._nlp.add_pipe("prevent_page_break_split", first=True)
            
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load spaCy model {self._model_name}: {e}")
            return False

    def analyze_sentences(self, text: str, heading_path: str = "", is_caption: bool = False, semantic_analyzer=None) -> List[Dict]:
        is_spacy_loaded = self._lazy_init()
        
        # Clean up markers for actual sentence splitting but keep them for POS logic if needed
        # Actually, the pipe handles it, so we leave it in 'text' and let SpaCy process it.
        
        sentences_data = []
        texts_to_refine = []
        initial_roles = []
        
        if is_spacy_loaded:
            doc = self._nlp(text)
            sent_items = [sent for sent in doc.sents]
        else:
            # Fallback
            split_texts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            sent_items = [{"text": s} for s in split_texts]

        for sent in sent_items:
            if is_spacy_loaded:
                text_str = sent.text.strip()
                # Clean up the marker for final output
                text_str = text_str.replace("[PAGE_JOIN]", " ").replace("  ", " ")
                pos_tags = [token.pos_ for token in sent]
                is_imp = self._is_imperative(sent)
            else:
                text_str = sent["text"]
                pos_tags = []
                is_imp = False
                
            if not text_str: continue
            
            # 2. 角色初步判定 (正则/关键词硬匹配)
            role = "explanation"
            text_lower = text_str.lower()
            
            if is_caption: 
                role = "interpretation"
            elif is_imp:
                role = "procedure"
            elif "?" in text_str:
                role = "question"
            elif "for example" in text_lower or "e.g." in text_lower or re.match(r'^example\s+\d+', text_lower):
                role = "example"
            elif any(kw in text_lower for kw in ["defined as", "refers to", "is a measure of", "is the proportion of"]):
                role = "definition"
            elif any(kw in text_lower for kw in ["assume", "suppose", "given that", "let us", "assuming"]):
                role = "assumption"
            elif any(kw in text_lower for kw in ["limit", "caveat", "fails to", "not account for", "valid only", "not applicable"]):
                role = "limitation"
            elif any(kw in text_lower for kw in ["conclude", "therefore", "summary", "takeaway", "consequently", "thus", "in summary", "overall"]):
                role = "conclusion"
            elif (any(kw in text_lower for kw in ["%", "table", "figure", "exhibit"]) or (is_spacy_loaded and any(token.like_num for token in sent))) and len(text_str) > 20:
                role = "evidence"
            
            sentences_data.append({
                "text": text_str,
                "role": role,
                "is_imperative": is_imp,
                "pos_tags": pos_tags,
                "semantic_confidence": 1.0
            })
            texts_to_refine.append(text_str)
            initial_roles.append(role)
            
        # 3. 语义精化 (批量处理)
        if semantic_analyzer and self.config.ENABLE_PROTOTYPE_MATCHING:
            refined_results = semantic_analyzer.batch_refine_roles(texts_to_refine, initial_roles)
            change_count = 0
            for i, (new_role, confidence) in enumerate(refined_results):
                if new_role != sentences_data[i]["role"]:
                    change_count += 1
                sentences_data[i]["role"] = new_role
                sentences_data[i]["semantic_confidence"] = confidence
            if change_count > 0:
                logger.info(f"Refined {change_count} sentence roles in chunk.")
                
        return sentences_data

    def _is_imperative(self, sent) -> bool:
        """简单的祈使句检测：动词原形开头且无主语"""
        if len(sent) > 0 and sent[0].pos_ == "VERB" and sent[0].dep_ == "ROOT":
            return True
        return False

    def get_last_n_sentences(self, text: str, n: int = 2) -> str:
        if not self._lazy_init(): return text[-100:]
        doc = self._nlp(text)
        sentences = list(doc.sents)
        if not sentences: return ""
        last_n = sentences[-n:]
        return " ".join([s.text.strip() for s in last_n])
