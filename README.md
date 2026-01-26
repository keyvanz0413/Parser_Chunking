# Synapta Parser & Chunker

This project implements a high-precision PDF parsing, structural reconstruction, and semantic segmentation pipeline. It transforms raw PDF documents into context-enriched, semantically tagged chunks suitable for Retrieval-Augmented Generation (RAG) and LLM-based applications.

## 🏗️ Project Structure

The project is modularized into two main packages:

1.  **`Semantic_chunker/`**: The core rule-based segmentation engine.
    *   `main.py`: Entry point for batch processing.
    *   `segmenter.py`: Orchestrates the chunking logic (`LogicSegmenter`).
    *   `config.py`: Configuration parameters (`ChunkingConfig`).
    *   `schema.py`: Data models (`EnrichedChunk`, `Reference`).
    *   `detectors/`: Modules for detecting page furniture, tags, references, and continuations.
    *   `analyzers/`: Modules for POS analysis, reading order correction, and content gating.
    *   `utils/`: Helpers for caption bonding and dehyphenation.

2.  **`Sbert_chunker/`**: An enhanced experimental version using Sentence-BERT for semantic coherence (optional/advanced usage).

## 🚀 Key Features

*   **Structural Reconstruction**: Rebuilds the document hierarchy (headings, lists, tables) before chunking.
*   **Semantic Segmentation**: Groups text not just by length, but by logical coherence (e.g., keeping a procedure together, spotting theorem/proof blocks).
*   **Cross-Page Integrity**: Detects and merges paragraphs that span across page boundaries using linguistic and spatial cues.
*   **Rich Metadata**: Enriches chunks with:
    *   `heading_path`: The full breadcrumb trail (e.g., "Chapter 1 > 1.2 Asset Classes").
    *   `tags`: Semantic tags (e.g., `definition`, `theorem`, `procedure`).
    *   `chunk_type`: Functional classification (e.g., `explanation`, `list`, `table`).
    *   `doc_zone`: Identifies if content is from the Front Matter, Body, or Back Matter.

## 🧠 Semantic Chunking Strategy

The `LogicSegmenter` uses a multi-phase strategy to create meaningful chunks:

### 1. Pre-Processing & Corrections
*   **Reading Order Correction**: Reorders multi-column text and reconstructs the correct flow.
*   **Furniture Detection**: Removes headers, footers, and page numbers to clean the text.
*   **Gating**: Identifies the true start of the main body (skipping TOCs, Prefaces) using `ContentGatekeeper`.

### 2. Structural Bonding
*   **Caption Bonding (Enhanced)**: Automatically bonds captions (e.g., "Table 1.1") to their corresponding blocks.
    *   **Relaxed Type Matching**: Handles parser misclassifications (e.g., bonding a "Table" caption to a "Picture" block).
    *   **Spatial Verification (New)**: Uses bounding box coordinates to verify physical proximity (vertical distance and horizontal overlap) ensuring accurate bonding even in complex layouts.
*   **List Adsorption**: Groups list items together or adsorbs them into the preceding introductory paragraph.
*   **Learning Objectives**: Groups extraction of learning objectives into single coherent blocks.

### 3. Sentence Role Identification (SRI)
The `POSAnalyzer` uses spaCy to assign a functional role to every sentence. This granular analysis guides the chunking decision (e.g., a "Definition" starts a new chunk).

**Supported Sentence Roles:**
*   **Cognitive**: 
    *   `Mechanism`: Causal/Systemic logic (Optimized for **Financial Domain** with keywords like *arbitrage, rate, liquidity*).
    *   `Assumption`: Hypothetical scenarios (Optimized for **Hypothesis Detection** using *Suppose/Assume/Imagine* while filtering out *supposed to*).
    *   `Topic`, `Definition`, `Explanation`, `Interpretation`, `Conclusion`, `Limitation`, `Contrast`, `Comparison`.
*   **Data-Driven**: `Evidence` (stats/data), `Formula`, `Reference`, `Theorem`.
*   **Instructional**: `Procedure` (imperative steps), `Example`, `Application`, `Caution`.
*   **Structural**: `Question`, `Transition`.

*Note: Roles are determined via a priority cascade guided by POS tags and domain-specific heuristics.*

### 4. Cross-Page Continuity
The `ContinuationDetector` analyzes the end of one page and the start of the next to determine if a paragraph continues. It uses:
*   **Syntactic Analysis**: Checks for open clauses (e.g., ending with "the", "of").
*   **Dehyphenation**: Reconstructs words split across lines/pages (e.g., "continu-" + "ation").

### 5. Reference Resolution (New)
The `ReferenceDetector` connects text to specific structural blocks:
*   **Explicit**: Resolves "Figure 1.1" or "Table 2" in text to the specific block ID of that figure/table.
*   **Implicit**: Attempts to resolve "the table below" or "this figure" based on proximity and page layout.

### 6. Composite Semantic Recognition: "Primary Role + Attributes"
To solve "semantic loss" when a sentence serves multiple functions, the system uses a **Composite Architecture**:

*   **Primary Role (Quantitative)**: Standard classification (e.g., `question`).
*   **Attributes (Qualitative / Sub-tags)**: Secondary semantics stored in `secondary_attributes`.

#### Example Case: Question + Reference
When a question requires data from a specific table:
*   **Role**: `question`
*   **Secondary Attributes**:
    *   `has_reference`: `true`
    *   `ref_target`: `"Table 1.1"`

**RAG Benefit**: Enables reverse-linking questions to the data sources they rely on (e.g., "Find all questions that use Table 1.1").

### 7. Core Segmentation Rules
The `LogicSegmenter` applies the following rules in order of priority:

1.  **Cross-Page Continuation (Rule 0)**: High-confidence cross-page merges (detected by `ContinuationDetector`) always take precedence, preventing artificial splits at page boundaries.
2.  **Header Isolation (Rule 1)**: Structural headers force a new chunk, updating the global `heading_path`.
3.  **List Handling (Rule 2 & 3)**:
    *   **Adsorption**: Short Introductory paragraphs can "adsorb" the subsequent list into a single chunk.
    *   **Grouping**: Consecutive list items are grouped into a single `list` chunk.
4.  **Block Bonding (Rule 4)**:
    *   **Captions**: Captions (e.g., "Table 1") are bonded to their structural block (Table/Image).
    *   **Notes**: Descriptions immediately following a table/image are bonded to it.
5.  **Academic Blocks (Rule 5)**: Sentences starting with "Theorem", "Lemma", or "Proof" trigger a dedicated chunk to preserve mathematical rigor.
6.  **Learning Objectives**: Consecutive LOs are grouped; LO headers are merged with their content.
7.  **Buffer Limit (Rule 6)**: If no logical break is found, chunks are split semantically when the buffer exceeds `MAX_CHUNK_WORDS`.

## 🛠️ Usage

### Installation
Ensure you have the required dependencies installed (including `spacy` and `en_core_web_md`):
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Running the Chunker
To run the standard semantic chunker on processed JSON files (from `parser_docling`):

```bash
# From the project root (Parser_Chunking/)
python -m semantic_chunker.main
```

This will:
1.  Read JSON inputs from `outputs/Docling_json/`.
2.  Process each file using the `LogicSegmenter`.
3.  Save enriched chunks to `outputs/Chunks_Semantic/`.

### Configuration
You can tune the chunking behavior in `semantic_chunker/config.py`:
*   `MIN_CHUNK_WORDS`: Merge chunks smaller than this.
*   `MAX_CHUNK_WORDS`: Split chunks larger than this.
*   `ENABLE_OVERLAP`: Add previous context to chunks for RAG.
*   `ENABLE_DEHYPHENATION`: Turn linguistic repair on/off.

## 📊 Statistics
The system provides detailed processing stats in the output JSON, including:
*   Number of chunks generated.
*   Count of cross-page continuations detected.
*   Distribution of chunk types and tags.
