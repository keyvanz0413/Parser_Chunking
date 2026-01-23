# Technical Report: Structural-Semantic Fusion for High-Precision PDF Parsing

---

## 1. Abstract
This report details the implementation of a high-performance, rule-based pipeline designed to transform large-scale, complex PDF documents into context-enriched, semantically tagged chunks. 

**Currently, the system supports two distinct processing schemes:**
1.  **Rule-Based Scheme** (`semantic_chunker.py`): Optimized for speed and deterministic performance using linguistic heuristics.
2.  **SBERT-Enhanced Scheme** (`sbert_chunker.py`): Optimized for high precision using Sentence-BERT for semantic refinement and breakpoint detection.

By integrating a multi-phase structural reconstruction algorithm with these segmentation engines, the system achieves an **LLM-verified classification accuracy of ~94.5%** for the SBERT-enhanced version.

## 2. Introduction
In typical RAG (Retrieval-Augmented Generation) systems, PDF parsing often suffers from "Contextual Blindness"—where chunks are separated from their parent headings, or "Structural Fragmentation"—where page breaks and column layouts disrupt the natural flow. Our approach, **Structural-Semantic Fusion**, addresses these issues by reconstructing the document's logical architecture before executing semantic segmentation.

---

## 2.1 Scheme Comparison
| Feature | Rule-Based (`semantic_chunker.py`) | SBERT-Enhanced (`sbert_chunker.py`) |
| :--- | :--- | :--- |
| **Logic** | Heuristics & POS Tagging | SBERT Embedding + Prototypical Matching |
| **Speed** | Ultra-Fast (~5 mins / 1k pages) | Fast (~12 mins / 1k pages) |
| **Hardware** | CPU Optimized | GPU/MPS Optimized |
| **Precision** | High (~82-85%) | Extreme (~94.5%) |
| **Use Case** | Fast ingestion / Pre-tagging | Advanced RAG / High-precision training data |

---

## 3. Methodology: Five-Phase Chunking Architecture
The system employs a deterministic five-phase pipeline to ensure data integrity and structural continuity. Phases 1, 2, and 4 are foundational to both schemes.

### 3.1 Phase 1: Reading Order & Noise Suppression
*   **Geometric-Sequential Correction**: The parser identifies bounding box (bbox) coordinates to correct reading order in multi-column layouts, ensuring text flows logically from the bottom of the left column to the top of the right column.
*   **Furniture Detection (De-noising)**: A frequency-based spatial algorithm identifies "Furniture" elements (running headers, footers, page numbers). By scanning cross-page repetitions, the system prevents non-content metadata from polluting the semantic chunks.

### 3.2 Phase 2: Structural Skeleton Reconstruction
*   **Stateful Heading Stacking**: The system maintains a hierarchy stack (L1–L4). As it traverses the document, every fragment inherits a `heading_path` (e.g., `Chapter 2 > 2.1 The Asset Market`).
*   **Backfill Corrections**: A "look-ahead" logic detects primary headers that appear unexpectedly in the middle of a page and retroactively updates the context for preceding segments on that page.

### 3.3 Phase 3: Dynamic Semantic Windowing
Chunks are not split by arbitrary character counts but by **Semantic Density**.

#### Option A: Heuristic splitting (`semantic_chunker.py`)
*   Uses paragraph boundaries and target word counts (~200 words) to create stable, uniform chunks.

#### Option B: AI-Enhanced splitting (`sbert_chunker.py`)
*   **Semantic Breakpoint Detection**: Utilizing SBERT embeddings to identify "topical shifts" within paragraphs, ensuring that a single chunk does not span multiple unrelated concepts even if size constraints are met.
*   **Coherence Optimization**: Strictly enforces semantic similarity thresholds within a chunk.

*Both options maintain Phase 2 structural tags (headers, lists, tables) as atomic units.*

### 3.4 Phase 4: Cross-Page Paragraph Recovery
To maintain sentence integrity across physical boundaries, the **Continuation Detector** evaluates:
*   **Linguistic Cues**: Detects hyphenated word breaks (e.g., `distribu-` followed by `tion`) and open-clause markers (sentences ending in prepositions).
*   **Spatial Transitions**: Validates if a break occurs at the geometric bottom of one page and continues at the top of the next.

### 3.5 Phase 5: Sentence-Level Enrichment
Final chunks are post-processed for RAG optimization:
*   **Contextual Overlap**: Final sentences of a chunk are injected into the following chunk as a `context_prefix`.
*   **Positional Tagging**: Each sentence is tagged with its relative position (First, Middle, Last).
*   **Coherence Scoring (SBERT Scheme only)**: Each chunk is assigned a **Semantic Coherence Score** (0.0 - 1.0) using SBERT to validate that all sentences within a chunk are semantically aligned.

---

## 4. Sentence Role Identification (SRI)
The core intelligence of the system lies in the SRI engine, which classifies sentences into 20+ functional roles.

### 4.1 Layered Detection Logic
1.  **Linguistic Layer**: Utilizing **spaCy en_core_web_md**, the system performs Part-of-Speech (POS) tagging to identify imperative structures (Commands/Procedures) and statistical markers.
2.  **Discourse Transition Layer (The "State Machine")**: Inspired by discourse analysis, the system utilizes the `prev_role` to influence the `current_role`. For example, a declarative sentence following an `example` trigger is prioritized as a `mechanism` or `interpretation`.
3.  **Funnel-Style Priority Cascading**: Unlike flat if-statements, roles are matched via a prioritized cascade where low-level roles (e.g., `explanation`) are progressively refined into specialized roles.
4.  **Negative Rule Layer (Global Filter)**: A comprehensive exclusion library (e.g., preventing hyphenated words like "day-to-day" from triggering mathematical `formula` tags) is applied as a final global filter, ensuring all candidate roles pass through a noise-suppression gate.
5.  **Glossary-Style Pattern Recognition**: Specifically handles "Noun [Space] Definition" patterns typical in textbook glossaries, which lack explicit system verbs or connecting phrases like "is defined as."

### 4.2 Classification Taxonomy Highlights
*   **Cognitive Roles**: `Topic`, `Definition`, `Explanation`, `Mechanism`, `Interpretation`.
*   **Data-Driven Roles**: `Evidence` (statistical data), `Formula` (mathematical expressions).
*   **Instructional Roles**: `Procedure`, `Example`, `Caution`, `Assumption`.

### 4.3 Semantic Refinement & Prototype Matching (v2.0 Upgrade)
To handle cases where rule-based detection is ambiguous (e.g., distinguishing between a general `explanation` and a specific `mechanism`), the system integrates a **Sentence-BERT (SBERT)** refinement layer:
*   **Max-Similarity Prototypical Matching**: The system maintains a curated library of "Role Prototypes" (130+ samples across Finance & Academic domains). Each candidate sentence is embedded and compared against this support set.
*   **Nearest-Neighbor Classification**: Instead of a simple centroid comparison, the system uses a winner-take-all similarity approach against all prototype samples, allowing for high-variance linguistic expressions within the same role.
*   **Probabilistic Smoothing**: If no rule is triggered, the SBERT engine provides a "soft assignment," ensuring 100% coverage of the document text with functional metadata.

---

## 5. Evaluation and Results
The system’s performance was validated through **LLM-Based Stratified Audit** on a 1073-page textbook (*Investments*). 
*   **Accuracy**: Achieved **~94.5% precision** as verified by an LLM-based semantic review (200 random samples). The SBERT refinement layer specifically resolved ~12% of previously ambiguous `explanation` tags into high-value cognitive roles (`mechanism`, `assumption`, `limitation`).
*   **Rule Robustness**: Significant improvement in "Scenario Setup" detection (e.g., correctly tagging "Suppose your client..." as `assumption` despite numerical content) and "Glossary Style" definitions.
*   **Efficiency**: Total processing time for 1,000+ pages averaged **12 minutes** on local hardware (MPS accelerated), including SBERT embedding generation.
*   **Reliability**: Significant reduction in "Hallucinated Context" during RAG retrieval due to 100% path coverage.

---

## 6. Conclusion
The Synapta Parser & Logic Segmenter demonstrates that high-precision document chunking can be achieved through a fusion of structural heuristics and linguistic pattern matching. By treating the document as a logical tree rather than a flat string, the system provides a robust foundation for advanced RAG applications in high-stakes domains like finance and education.
