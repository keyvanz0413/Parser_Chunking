from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SentenceRole:
    """Represents a sentence with its identified role."""
    text: str
    role: str  # topic, definition, example, conclusion, evidence, procedural, imperative
    pos_tags: List[str] = field(default_factory=list)
    is_imperative: bool = False


@dataclass
class Reference:
    """
    Represents a reference to a Figure/Table/Equation within a chunk.
    
    Used for cross-linking paragraph chunks with visual/structural blocks.
    """
    ref_text: str              # Original reference text (e.g., "Figure 1.1")
    start_offset: int          # Start position in chunk content
    end_offset: int            # End position in chunk content
    target_segment_id: str     # Segment ID of the referenced block (or None)
    target_type: str           # "Figure", "Table", "Equation", "Formula"
    ref_kind: str              # "explicit" or "implicit"
    confidence: float = 0.0    # Confidence score (0.0-1.0)


@dataclass
class EnrichedChunk:
    """
    Core Schema - The standardized output format with references support.
    
    Fields:
    - chunk_id: Unique identifier
    - heading_path: Breadcrumb context (e.g., "Chapter 1 > Section 1.1")
    - chunk_type: Primary type (definition, procedure, explanation, list, theorem, etc.)
    - content: Full text content
    - context_prefix: Overlap text from previous chunk (for RAG continuity)
    - sentences: List of sentences with roles
    - tags: Semantic tags from 20+ taxonomy
    - source_segments: Original segment IDs for traceability
    - page_range: Page numbers covered by this chunk
    - word_count: Approximate word count for length analysis
    
    Cross-page continuation:
    - is_cross_page: True if chunk spans multiple pages
    - continuation_type: 'none', 'full' (confident), 'partial' (needs review)
    - needs_review: True if continuation detection was uncertain
    - merge_evidence: Dict with reasons for merge decision (explainability)
    
    NEW: Figure/Table/Equation references:
    - references: List of detected references to visual/structural blocks
    """
    chunk_id: str
    heading_path: str
    chunk_type: str
    content: str
    context_prefix: str = ""  # Overlap from previous chunk
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source_segments: List[str] = field(default_factory=list)
    page_range: List[int] = field(default_factory=list)
    depth: int = 0
    word_count: int = 0
    # Cross-page continuation fields
    is_cross_page: bool = False
    continuation_type: str = "none"  # none, full, partial
    needs_review: bool = False
    merge_evidence: Dict[str, Any] = field(default_factory=dict)
    # Traceability: Bounding boxes
    bbox: List[float] = field(default_factory=list)      # Combined bbox [x1, y1, x2, y2]
    page_bboxes: Dict[int, List[float]] = field(default_factory=dict) # Per-page bboxes
    # NEW: Reference detection
    references: List[Reference] = field(default_factory=list)
    doc_zone: str = "body"      # "front", "body", "back"
    # NEW: Knowledge Graph Edges
    edges: List["Edge"] = field(default_factory=list)

@dataclass
class Edge:
    """
    Represents a directed relationship between this chunk and another entity.
    Used for building Knowledge Graphs (KG) for RAG.
    """
    target_id: str      # ID of the target chunk or entity
    edge_type: str      # Enum from EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., {'confidence': 0.9}

class EdgeType:
    """Standardized edge types for RAG KGs."""
    PART_OF = "part_of"       # Hierarchical: Section -> Chapter
    NEXT = "next"             # Sequential: Para 1 -> Para 2
    REFERENCES = "references" # Explicit: "See Figure 1" -> Figure 1
    DEFINES = "defines"       # Semantic: "ROI is..." -> "ROI" (Concept)
    CONTAINS = "contains"     # Inverse of PART_OF
