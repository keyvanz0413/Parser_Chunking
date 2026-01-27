# Synapta Parser & Chunker

This project implements a high-precision PDF parsing, structural reconstruction, and semantic segmentation pipeline. It transforms raw PDF documents into a strict, page-aware hierarchy (**Topic → Section → Subsection → Objects**) where every chunk is traceable and linkable for RAG and Knowledge Graph (KG) workflows.

## 🏗️ Project Structure

The project follows a decoupled, component-based architecture:

```text
Parser_Chunking/
├── run_pipeline.py          # Unified entry point for Parsing + Chunking
├── semantic_chunker/        # Core Rule-Based Segmentation Engine
│   ├── parser.py            # Enhanced Parser (Docling + Hierarchy & Metadata)
│   ├── segmenter.py         # Orchestrator (LogicSegmenter)
│   ├── config.py            # Central Configuration (ChunkingConfig)
│   ├── analyzers/           # Specialized Analysis Modules
│   │   ├── structure.py     # Authoritative Structure Analysis (TOC Funnel)
│   │   ├── pos_analyzer.py  # Function & Role Analysis (NLP/SPAcy)
│   │   └── ...
│   └── ...
├── sbert_chunker/           # Experimental SBERT-based Segmenter
└── inputs/ & outputs/       # I/O Directories
```

## Quick Start
Run the full pipeline (Parsing -> Chunking) with a single command:
```bash
python run_pipeline.py
```
This will process all PDFs in `inputs/` and generate enriched JSON in `outputs/Chunks_Semantic/`.

## 🚀 Key Features

### 1. Authoritative Header Enforcement (Pre-Chunking)
We solve the "Toxic Header" problem by enforcing a strict **Authoritative TOC Filter**:
*   **Source of Truth**: The `BookStructureAnalyzer` extracts a "Gold Standard" TOC from metadata or heuristic scanning.
*   **Enforcement**: Before chunking, every `Header` detected by the OCR/Parser is cross-referenced against this TOC list.
*   **Purification**: Unauthorized headers (e.g., long sentences misclassified as headers) are aggressively **downgraded to Paragraphs**. This ensures clean, concise `heading_path` breadcrumbs.

### 2. Enhanced Semantic Role Identification (POS++)
The `POSAnalyzer` uses a composite strategy to assign granular roles to every sentence:
*   **Keyword Overrides**: Explicit markers force specific roles to boost RAG performance:
    *   `"For example..."` → **Example**
    *   `"In summary..."` → **Conclusion**
*   **NER Salvation**: Prevents critical named entities (e.g., "Carl Icahn") from being discarded as "noise" even if the sentence is short.
*   **Fragment Merging**: Automatically heals broken sentence fragments (OCR artifacts) by merging them into the preceding context.

### 3. Advanced TOC Analysis (The 3-STAGE Funnel)
The pipeline uses a "Funnel" strategy to establish the document's structural "North Star":
*   **Stage 1: Metadata Outlines**: Direct extraction of PDF bookmarks for 100% accuracy.
*   **Stage 2: Heuristic Fingerprinting**: Robust regex scanning for TOC patterns (`Title ... Page`) when metadata is missing.
*   **Stage 3: Vision Fallback**: Automatic capture of TOC pages for potential downstream Vision correction.

### 4. Cross-Page Continuity
The `ContinuationDetector` prevents artificial splits at page ends by checking:
*   **Syntactic Openings**: Sentences ending in conjunctions or prepositions.
*   **Spatial Proximity**: Distance between the last block of one page and the first of the next.
*   **Dehyphenation**: Linguistic repair for words split across pages.

## 📊 Output Schema
The resulting JSON follows a strict schema for downstream compatibility:
*   **`metadata`**: Authoritative book data (Title, Authors, ISBN, Publisher) + Processing Stats.
*   **`chunks`**: Enriched objects containing:
    *   `segment_id`: Stable unique identifier.
    *   `chunk_type`: Taxonomy type (`header`, `list`, `definition`, `formula`, etc.).
    *   `heading_path`: Full breadcrumb (e.g., `Part I > Chapter 1 > 1.1`).
    *   `text`: Cleaned content.
    *   `tags`: Semantic labels (e.g., `evidence`, `procedure`).
    *   `edges`: Typed KG links to other chunks or references.
