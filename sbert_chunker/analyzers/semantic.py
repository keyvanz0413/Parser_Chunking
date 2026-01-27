import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from ..core.config import ChunkingConfig

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Semantic analysis layer using Sentence-Transformers (SBERT)."""
    
    ROLE_PROTOTYPES = {
        'definition': [
            "A bond is a financial instrument that represents a loan made by an investor.",
            "The yield to maturity is defined as the internal rate of return of the bond.",
            "Duration measures the sensitivity of bond price to interest rate changes.",
        ],
        'assumption': [
            "Assume that the interest rate remains constant over the investment period.",
            "Let us suppose that the market is efficient and all information is reflected in prices.",
            "Given that inflation is zero, the nominal rate equals the real rate.",
        ],
        'limitation': [
            "This model does not account for transaction costs or taxes.",
            "The analysis is limited to the case where default risk is negligible.",
            "One caveat is that liquidity may vary significantly across different securities.",
        ],
        'mechanism': [
            "When interest rates rise, bond prices fall because the present value decreases.",
            "The arbitrage mechanism ensures that prices converge to their fair value.",
            "Duration works by measuring the weighted average time of cash flows.",
        ],
        'procedure': [
            "First, calculate the present value of each cash flow. Then sum all values.",
            "To compute the yield, solve the equation iteratively until convergence.",
            "Follow these steps: enter the data, run the model, and interpret results.",
        ],
        'evidence': [
            "The correlation coefficient was 0.85, indicating a strong positive relationship.",
            "Returns increased by 15% over the sample period from 2010 to 2020.",
            "Statistical analysis shows p < 0.05, rejecting the null hypothesis.",
        ],
        'example': [
            "For instance, consider a 10-year bond with a 5% coupon rate.",
            "An example would be the 2008 financial crisis when credit spreads widened.",
            "To illustrate, suppose an investor purchases $1,000 worth of Treasury bonds.",
        ],
        'interpretation': [
            "This means that investors are willing to accept lower returns for safety.",
            "In other words, the market is pricing in a higher probability of default.",
            "The implication is that portfolio diversification reduces overall risk.",
        ],
        'conclusion': [
            "In conclusion, duration matching is an effective immunization strategy.",
            "Therefore, we recommend a barbell strategy for this interest rate environment.",
            "The key takeaway is that credit risk dominates during economic downturns.",
        ],
        'comparison': [
            "Unlike stocks, bonds provide fixed income streams to investors.",
            "Compared to Treasury bonds, corporate bonds offer higher yields but more risk.",
            "The longer-duration portfolio outperformed the shorter-duration one.",
        ],
    }
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self._model = None
        self._role_embeddings = {}  
        self._initialized = False
        
    def _lazy_init(self):
        if self._initialized: return True
        if not self.config.ENABLE_SEMANTIC_ANALYSIS: return False
            
        try:
            import os
            import json
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer(self.config.SBERT_MODEL_NAME)
            self._np = np
            
            # Load expanded prototypes from JSON if configured
            if hasattr(self.config, 'ROLE_PROTOTYPES_PATH') and os.path.exists(self.config.ROLE_PROTOTYPES_PATH):
                try:
                    with open(self.config.ROLE_PROTOTYPES_PATH, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'prototypes' in data:
                            # Update ROLE_PROTOTYPES with external data
                            self.ROLE_PROTOTYPES = {k: v.get('examples', []) for k, v in data['prototypes'].items()}
                            logger.info(f"Successfully loaded {len(self.ROLE_PROTOTYPES)} roles from prototype library.")
                except Exception as e:
                    logger.warning(f"Failed to load prototypes from {self.config.ROLE_PROTOTYPES_PATH}: {e}")

            if self.config.ENABLE_PROTOTYPE_MATCHING:
                self._compute_role_embeddings()
            self._initialized = True
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed.")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize SemanticAnalyzer: {e}")
            return False
    
    def _compute_role_embeddings(self):
        for role, sentences in self.ROLE_PROTOTYPES.items():
            embeddings = self._model.encode(sentences)
            self._role_embeddings[role] = embeddings
    
    def encode_sentences(self, sentences: List[str]) -> Optional[Any]:
        if not self._lazy_init(): return None
        if not sentences: return self._np.array([])
        return self._model.encode(sentences)
    
    def compute_similarity(self, embedding1, embedding2) -> float:
        if not self._lazy_init(): return 0.0
        norm1 = self._np.linalg.norm(embedding1)
        norm2 = self._np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return float(self._np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def find_semantic_breakpoints(self, sentences: List[str]) -> List[Dict[str, Any]]:
        if not self._lazy_init() or len(sentences) < 2: return []
        embeddings = self.encode_sentences(sentences)
        if embeddings is None or len(embeddings) < 2: return []
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.compute_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        breakpoints = []
        window_size = min(self.config.SEMANTIC_WINDOW_SIZE, len(similarities))
        threshold = self.config.SEMANTIC_SIMILARITY_THRESHOLD
        
        for i in range(len(similarities)):
            start_idx = max(0, i - window_size)
            window = similarities[start_idx:i] if i > 0 else [similarities[0]]
            avg_sim = sum(window) / len(window) if window else similarities[i]
            current_sim = similarities[i]
            drop_ratio = current_sim / avg_sim if avg_sim > 0 else 1.0
            if current_sim < threshold and drop_ratio < 0.7:
                breakpoints.append({
                    "index": i + 1,
                    "similarity": round(current_sim, 3),
                    "avg_similarity": round(avg_sim, 3),
                    "drop_ratio": round(drop_ratio, 3),
                })
        return breakpoints
    
    def refine_role_by_semantics(self, sentence: str, current_role: str) -> Tuple[str, float]:
        results = self.batch_refine_roles([sentence], [current_role])
        return results[0]

    def batch_refine_roles(self, sentences: List[str], current_roles: List[str]) -> List[Tuple[str, float]]:
        """Batch version of role refinement to reduce SBERT overhead."""
        if not self._lazy_init() or not self.config.ENABLE_PROTOTYPE_MATCHING:
            return [(r, 0.0) for r in current_roles]
            
        # Identify sentences that need semantic check (usually 'explanation')
        refine_indices = [i for i, r in enumerate(current_roles) if r in ('explanation', 'irrelevant')]
        if not refine_indices:
            return [(r, 1.0) for r in current_roles]
            
        # Initialize results with current values
        results = [(r, 1.0) for r in current_roles]
        
        # Batch encode only the necessary sentences
        refine_texts = [sentences[i] for i in refine_indices]
        sentence_embeddings = self._model.encode(refine_texts)
        
        for i, idx in enumerate(refine_indices):
            sentence_embedding = sentence_embeddings[i]
            best_role = current_roles[idx]
            best_score = 0.0
            
            for role, support_matrix in self._role_embeddings.items():
                # Find maximum similarity against the support set (prototypes)
                similarities = [self.compute_similarity(sentence_embedding, e) for e in support_matrix]
                max_sim = max(similarities) if similarities else 0.0
                if max_sim > best_score:
                    best_score = max_sim
                    best_role = role
            
            if best_score >= self.config.ROLE_PROTOTYPE_SIMILARITY_THRESHOLD:
                results[idx] = (best_role, round(best_score, 3))
            else:
                results[idx] = (current_roles[idx], round(best_score, 3))
                
        return results
    
    def compute_chunk_coherence(self, sentences: List[str]) -> float:
        if not self._lazy_init() or len(sentences) < 2: return 1.0
        embeddings = self.encode_sentences(sentences)
        if embeddings is None or len(embeddings) < 2: return 1.0
        total_sim, count = 0.0, 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                total_sim += self.compute_similarity(embeddings[i], embeddings[j])
                count += 1
        return total_sim / count if count > 0 else 1.0

    def compute_cross_page_semantic_score(self, prev_text: str, curr_text: str) -> float:
        """New: Explicitly scoring continuity between pages/columns."""
        if not self._lazy_init(): return 0.5
        prev_sample = prev_text[-200:] if len(prev_text) > 200 else prev_text
        curr_sample = curr_text[:200] if len(curr_text) > 200 else curr_text
        embeddings = self.encode_sentences([prev_sample, curr_sample])
        if embeddings is None or len(embeddings) < 2: return 0.5
        return max(0.0, self.compute_similarity(embeddings[0], embeddings[1]))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "model_name": self.config.SBERT_MODEL_NAME if self._initialized else None,
            "prototype_count": sum(len(v) for v in self._role_embeddings.values()),
        }
