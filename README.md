# Technical Report: Structural-Semantic Fusion for High-Precision PDF Parsing
**Version**: 2.4 | **Project**: Synapta Parser & Logic Segmenter

---

## 1. Abstract
This report details the implementation of a high-performance, rule-based pipeline designed to transform large-scale, complex PDF documents (1000+ page financial and academic textbooks) into context-enriched, semantically tagged chunks. By integrating a multi-phase structural reconstruction algorithm with a state-aware sentence classification engine, the system achieves approximately **85% classification accuracy** and maintains full structural lineage without the high latency or cost associated with Large Language Model (LLM) pre-processing.

## 2. Introduction
In typical RAG (Retrieval-Augmented Generation) systems, PDF parsing often suffers from "Contextual Blindness"—where chunks are separated from their parent headings, or "Structural Fragmentation"—where page breaks and column layouts disrupt the natural flow. Our approach, **Structural-Semantic Fusion**, addresses these issues by reconstructing the document's logical architecture before executing semantic segmentation.

---

## 3. Methodology: Five-Phase Chunking Architecture
The system employs a deterministic five-phase pipeline to ensure data integrity and structural continuity.

### 3.1 Phase 1: Reading Order & Noise Suppression
*   **Geometric-Sequential Correction**: The parser identifies bounding box (bbox) coordinates to correct reading order in multi-column layouts, ensuring text flows logically from the bottom of the left column to the top of the right column.
*   **Furniture Detection (De-noising)**: A frequency-based spatial algorithm identifies "Furniture" elements (running headers, footers, page numbers). By scanning cross-page repetitions, the system prevents non-content metadata from polluting the semantic chunks.

### 3.2 Phase 2: Structural Skeleton Reconstruction
*   **Stateful Heading Stacking**: The system maintains a hierarchy stack (L1–L4). As it traverses the document, every fragment inherits a `heading_path` (e.g., `Chapter 2 > 2.1 The Asset Market`).
*   **Backfill Corrections**: A "look-ahead" logic detects primary headers that appear unexpectedly in the middle of a page and retroactively updates the context for preceding segments on that page.

### 3.3 Phase 3: Dynamic Semantic Windowing
Chunks are not split by arbitrary character counts but by **Semantic Density**.
*   **Boundary Constraints**: Chunks are optimized for a target size of **~200 words**, with a strict floor (30 words) to prevent orphaned context and a ceiling (500 words) for embedding model compatibility.
*   **Type-Specific Handling**: Tables, Lists, and Learning Objectives are preserved as atomic units to ensure structural cohesion.

### 3.4 Phase 4: Cross-Page Paragraph Recovery
To maintain sentence integrity across physical boundaries, the **Continuation Detector** evaluates:
*   **Linguistic Cues**: Detects hyphenated word breaks (e.g., `distribu-` followed by `tion`) and open-clause markers (sentences ending in prepositions).
*   **Spatial Transitions**: Validates if a break occurs at the geometric bottom of one page and continues at the top of the next.

### 3.5 Phase 5: Sentence-Level Enrichment
Final chunks are post-processed for RAG optimization:
*   **Contextual Overlap**: Final sentences of a chunk are injected into the following chunk as a `context_prefix`, preserving local discourse flow.
*   **Positional Tagging**: Each sentence is tagged with its relative position within the paragraph (First, Middle, Last).

---

## 4. Sentence Role Identification (SRI)
The core intelligence of the system lies in the SRI engine, which classifies sentences into 20+ functional roles.

### 4.1 Layered Detection Logic
1.  **Linguistic Layer**: Utilizing **spaCy en_core_web_md**, the system performs Part-of-Speech (POS) tagging to identify imperative structures (Commands/Procedures) and statistical markers.
2.  **Discourse Transition Layer (The "State Machine")**: Inspired by discourse analysis, the system utilizes the `prev_role` to influence the `current_role`. For example, a declarative sentence following an `example` trigger is prioritized as a `mechanism` or `interpretation`.
3.  **Negative Rule Layer**: To mitigate common classification noise (e.g., mistaking "10-year period" for a mathematical `formula`), the system applies a comprehensive exclusion library.

### 4.2 Classification Taxonomy Highlights
*   **Cognitive Roles**: `Topic`, `Definition`, `Explanation`, `Mechanism`, `Interpretation`.
*   **Data-Driven Roles**: `Evidence` (statistical data), `Formula` (mathematical expressions).
*   **Instructional Roles**: `Procedure`, `Example`, `Caution`, `Assumption`.

---

## 5. Evaluation and Results
The system’s performance was validated through **Stratified Sampling Audit** on a 1073-page textbook (*Investments*). 
*   **Accuracy**: Achieved **~85% precision** across complex roles.
*   **Efficiency**: Total processing time for 1,000+ pages averaged **12 minutes** on local hardware (MPS accelerated).
*   **Reliability**: Significant reduction in "Hallucinated Context" during RAG retrieval due to 100% path coverage.

---

## 6. Conclusion
The Synapta Parser & Logic Segmenter demonstrates that high-precision document chunking can be achieved through a fusion of structural heuristics and linguistic pattern matching. By treating the document as a logical tree rather than a flat string, the system provides a robust foundation for advanced RAG applications in high-stakes domains like finance and education.
