# Synapta Parser & Chunker

This project implements a high-precision PDF parsing, structural reconstruction, and semantic segmentation pipeline. It transforms raw PDF documents into a strict, page-aware hierarchy (**Topic → Section → Subsection → Objects**) where every chunk is traceable and linkable for RAG and Knowledge Graph (KG) workflows.

## 🏗️ Project Structure

The project follows a decoupled, component-based architecture:

1.  **`Semantic_chunker/`**: The core rule-based segmentation engine.
    *   `main.py`: Entry point for batch processing.
    *   `segmenter.py`: Orchestrates the pipeline (`LogicSegmenter`) by coordinating specialized analyzers.
    *   `config.py`: Centralized configuration (`ChunkingConfig`).
    *   `schema.py`: Shared data models (`EnrichedChunk`, `Edge`, `Reference`).
    *   **`analyzers/`**:
        *   **`structure.py`**: **Advanced TOC Funnel**. Implements the 3-stage analysis (Metadata, Heuristics, Vision) and document boundary mapping.
        *   `chunk_factory.py`: Atomic chunk assembly, text repair, and semantic type inference.
        *   `kg_linker.py`: Knowledge Graph topology generator (Next, PartOf, Reference edges).
        *   `pos_analyzer.py`: Functional role analysis using spaCy and POS tagging.
        *   `reading_order.py`: Multicolumn flow correction and heading reconstruction.
        *   `gatekeeper.py`: Document zone analysis (Front/Body/Back matter).
    *   **`detectors/`**: Specialized logic for tags, references, and cross-page continuations.
    *   **`utils/`**:
        *   `metadata_manager.py`: **Strict Authority API**. Handles ISBN-triggered metadata lookups.
        *   `metrics.py`: Pipeline performance and quality metrics collector.
        *   `caption_bonding.py`: Visual/Spatial logic for block-caption association.
        *   `dehyphenation.py`: Linguistic repair for line-break artifacts.

2.  **`Sbert_chunker/`**: An enhanced experimental version using Sentence-BERT for semantic coherence (optional/advanced usage).

## 🧠 Semantic Chunking Strategy

The standard pipeline follows a "Pre-Enhance -> Stream-Chunk -> Topologize" flow:

### 1. Advanced TOC Analysis (The 3-Stage Funnel)
The pipeline uses a "Funnel" strategy (implemented in `BookStructureAnalyzer`) to establish the document's structural "North Star":
*   **Stage 1: Metadata Outlines**: Direct extraction of PDF bookmarks for 100% accuracy.
*   **Stage 2: Heuristic Fingerprinting**: Robust regex scanning for TOC patterns (`Title ... Page`) when metadata is missing.
*   **Stage 3: Vision Fallback (Artifact Capture)**: Automatic detection and high-res capture of candidate TOC pages for Vision-LLM or manual review.

**Outcome**: Every chunk is calibrated against this authoritative skeleton, ensuring precise `heading_path` generation even when layout analysis fails.

### 2. Structural Boundary Mapping
The system automatically classifies the document into functional zones:
*   **Front Matter**: Identified via TOC indicators (Preface, Brief Contents).
*   **Body**: Precise chapter-level boundaries and page ranges.
*   **Back Matter & Glossary**: Heuristic detection of end-of-book sections, with specialized tagging for glossary definitions.

### 3. Functional Role Identification (SRI) & Object Detection
The `POSAnalyzer` and `TagDetector` collaborate to identify specialized "Text Objects" beyond standard paragraphs:
*   **Learning Objectives**: Header + bullet list patterns.
*   **Concepts & Definitions**: Key terms (bold/italic) + colon/dash definition patterns.
*   **Theorems & Rules**: Callouts like "Theorem", "Lemma", "Principle", or "Key Takeaways".
*   **Lists & Procedures**: Systematic detection of bullets and "Step 1/2/3" sequences.

**Domain-Specific Semantic Roles:**
*   `Mechanism`: Logic-heavy sentences (e.g., *transmission*, *arbitrage*).
*   `Contrast`: Intra-sentence opposing logic (*however*, *despite*).
*   `Assumption`: Hypothetical scenarios (*Assume*, *Suppose*).
*   `Evidence`: Data-driven statements (percentages, financial figures).

### 4. Reading Order & Spatial Bonding
*   **Column Correction**: Handles complex multi-column flows using spatial relative positioning.
*   **Caption Bonding**: Uses the `CaptionBondingHelper` to verify physical proximity (vertical distance and horizontal overlap) between captions (Table/Figure) and blocks.

### 5. Cross-Page Continuity
The `ContinuationDetector` prevents artificial splits at page ends by checking:
*   **Syntactic Openings**: Sentences ending in conjunctions or prepositions.
*   **Spatial Proximity**: Distance between the last block of one page and the first of the next.
*   **Dehyphenation**: Linguistic repair for words split across pages.

### 6. KG Topology & Reference Linking
*   **KGLinker**: Converts linear chunks into a graph using stable IDs (`chunk_xxxx`).
*   **Cross-Reference Detection**: Automatically captures patterns like *"see Figure 2.3"*, *"Table 5.1"*, or *"Eq. (7.4)"* as lightweight `Reference` markers.
*   **Relationship Types**: Generates `next` (context), `child_of` (taxonomy), and `references` (cross-object) edges.
*   **MetricsCollector**: Provides granular stats on chunk density and role distribution.

## 📊 Output Schema
The resulting JSON follows a strict schema for downstream compatibility:
*   **`metadata`**: Authoritative book data (Title, Authors, ISBN, Publisher) + Processing Stats.
*   **`chunks`**: Enriched objects containing:
    *   `segment_id`: Stable unique identifier.
    *   `chunk_type`: Taxonomy type (`header`, `list`, `definition`, `table`, etc.).
    *   `heading_path`: Full breadcrumb (e.g., `Part I > Chapter 1 > 1.1`).
    *   `page_range`: [start_page, end_page].
    *   `bbox`: [x0, y0, x1, y1] spatial coordinates.
    *   `text`: Cleaned content.
    *   `tags`: Semantic labels (e.g., `evidence`, `procedure`).
    *   `edges`: Typed KG links to other chunks or references.
