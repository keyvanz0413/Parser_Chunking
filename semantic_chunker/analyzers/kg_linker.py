import logging
from typing import List, Dict, Any
from ..schema import EnrichedChunk, Edge, EdgeType

logger = logging.getLogger(__name__)

class KGLinker:
    """
    Analyzes processed chunks to generate Knowledge Graph edges.
    1. NEXT: Sequential flow
    2. PART_OF: Hierarchical structure
    3. REFERENCES: Explicit citations
    """
    
    @staticmethod
    def link(chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """
        Generate Knowledge Graph edges between chunks.
        """
        if not chunks:
            return chunks

        # 1. Build Index Maps
        segment_to_chunk = {}
        heading_path_to_id = {}
        
        for c in chunks:
            # Map all source segments to this chunk
            for seg_id in c.source_segments:
                segment_to_chunk[seg_id] = c.chunk_id
            
            # Map heading path to header chunk ID (if it's a header)
            if c.chunk_type == 'header':
                # Map the unique path string identifies this header concept
                heading_path_to_id[c.heading_path] = c.chunk_id

        # 2. Generate Edges
        for i, chunk in enumerate(chunks):
            # A. NEXT Edge (Sequence)
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                chunk.edges.append(Edge(target_id=next_chunk.chunk_id, edge_type=EdgeType.NEXT))

            # B. PART_OF Edge (Hierarchy)
            if chunk.chunk_type != 'header':
                parent_id = heading_path_to_id.get(chunk.heading_path)
                if parent_id and parent_id != chunk.chunk_id:
                     chunk.edges.append(Edge(target_id=parent_id, edge_type=EdgeType.PART_OF))
            else:
                 # If this IS a header, it is PART_OF its parent header
                 if " > " in chunk.heading_path:
                     parent_path = chunk.heading_path.rsplit(" > ", 1)[0]
                     parent_id = heading_path_to_id.get(parent_path)
                     if parent_id:
                         chunk.edges.append(Edge(target_id=parent_id, edge_type=EdgeType.PART_OF))

            # C. REFERENCES Edge (Explicit Citations)
            for ref in chunk.references:
                if ref.target_segment_id:
                    target_chunk_id = segment_to_chunk.get(ref.target_segment_id)
                    if target_chunk_id and target_chunk_id != chunk.chunk_id:
                        # Avoid duplicate edges
                        if not any(e.target_id == target_chunk_id and e.edge_type == EdgeType.REFERENCES for e in chunk.edges):
                            chunk.edges.append(Edge(
                                target_id=target_chunk_id, 
                                edge_type=EdgeType.REFERENCES,
                                metadata={'ref_text': ref.ref_text, 'confidence': ref.confidence}
                            ))

        return chunks
