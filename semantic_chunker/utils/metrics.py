import logging
from typing import List, Dict, Any
from ..schema import EnrichedChunk

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects and calculates processing statistics for the chunking pipeline.
    """
    
    @staticmethod
    def calculate(chunks: List[EnrichedChunk]) -> Dict[str, Any]:
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
            "cross_page_chunks": cross_page_count,
            "full_continuations": full_continuation_count,
            "partial_continuations": partial_continuation_count,
            "chunks_needing_review": needs_review_count
        }
