# Synapta Parser & Chunker

This project implements a high-precision PDF parsing, structural reconstruction, and semantic segmentation pipeline. It transforms raw PDF documents into context-enriched, semantically tagged chunks suitable for Retrieval-Augmented Generation (RAG) and LLM-based applications.

## 🏗️ Project Structure

The project follows a decoupled, component-based architecture:

1.  **`Semantic_chunker/`**: The core rule-based segmentation engine.
    *   `main.py`: Entry point for batch processing.
    *   `segmenter.py`: Orchestrates the pipeline (`LogicSegmenter`) by coordinating specialized analyzers.
    *   `toc_parser.py`: **Authority Structural Seeding**. Detects and parses Table of Contents to enforce document hierarchy.
    *   `config.py`: Centralized configuration (`ChunkingConfig`).
    *   `schema.py`: Shared data models (`EnrichedChunk`, `Edge`, `Reference`).
    *   **`analyzers/`**:
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

### 1. Structural Seeding (TOC-First)
The process begins with the `TOCParser`. It identifies TOC pages, extracts the hierarchical index, and "seeds" every segment in the document with its authoritative section path. This ensures that even if a header is misidentified during layout analysis, the chunk retains its correct structural context.

### 2. Strict Metadata & Gating
*   **Authority API Lookup**: The `MetadataManager` scans for an ISBN. If found, it fetches verified book metadata via Google Books API. **No ISBN means no metadata**, ensuring 100% accuracy.
*   **Content Gating**: The `Gatekeeper` identifies the transition from front matter to the actual main body to optimize RAG relevance.

### 3. Functional Role Identification (SRI)
The `POSAnalyzer` assigns roles to sentences using spaCy. This guides the `ChunkFactory` in determining if a chunk represents a `definition`, `theorem`, `procedure`, or `explanation`.

**Domain-Specific Optimization (Financial/Technical):**
*   `Mechanism`: Logic-heavy sentences (e.g., *arbitrage*, *volatility*).
*   `Assumption`: Hypothetical scenarios (*Assume*, *Suppose*).
*   `Assumption`, `Limitation`, `Conclusion`, etc.

### 4. Reading Order & Spatial Bonding
*   **Column Correction**: Handles complex multi-column flows using spatial relative positioning.
*   **Caption Bonding**: Uses the `CaptionBondingHelper` to verify physical proximity (vertical distance and horizontal overlap) between captions (Table/Figure) and blocks.

### 5. Cross-Page Continuity
The `ContinuationDetector` prevents artificial splits at page ends by checking:
*   **Syntactic Openings**: Sentences ending in conjunctions or prepositions.
*   **Spatial Proximity**: Distance between the last block of one page and the first of the next.
*   **Dehyphenation**: Linguistic repair for words split across pages.

### 6. KG Topology & Metrics
*   **KGLinker**: Converts the linear list of chunks into a relative graph. Chunks are linked to their parents (headers) and targets (resolved references).
*   **MetricsCollector**: Provides granular stats on chunk density, role distribution, and cross-page merge confidence.

## 🛠️ Usage

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Running the Pipeline
```bash
# Processes documents from parser output to semantic chunks
python -m Semantic_chunker.main
```

## 📊 Output Schema
The resulting JSON includes:
*   `metadata`: Authoratitive book data + processing metrics.
*   `chunks`: Enriched objects containing `content`, `heading_path`, `tags`, `bbox`, and `edges` (KG links).
