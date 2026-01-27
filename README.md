# Technical Protocol: Semantic Structure Parsing & Chunking

**Version:** 2.1  
**Date:** 2026-01-27  
**Scope:** `semantic_chunker` Architecture & Logic  

---

## 1. Overview

This document outlines the technical architecture and methodological protocols implemented in the **Synapta Parser & Chunker**. The system is designed to transform unstructured PDF documents into highly structured, semantically enriched chunks optimized for Retrieval-Augmented Generation (RAG). Unlike standard flat chunking (sliding window), this pipeline enforces strict document structure and semantic coherence.

## 2. Pipeline Architecture

The processing pipeline follows a linear orchestration flow (`run_pipeline.py`), comprising four distinct stages:

1.  **Metadata & Structure Analysis**
2.  **Layout Parsing (Docling)**
3.  **Semantic Chunking (Logic Segmenter)**
4.  **Semantic Enrichment (Role Analysis)**

---

## 3. Methodology Details

### 3.1 Metadata & Structure Discovery
Before parsing content, the system establishes a "ground truth" for the document's identity and structure.

*   **Book Metadata**: We utilize `MetadataManager` to extract high-fidelity metadata (ISBN, Authors, Publication Year) via cross-referencing internal PDF metadata with external APIs (Google Books, Crossref).
*   **TOC Extraction (The "Skeleton")**: 
    *   **Dual-Stage Strategy**: First attempts to extract the Table of Contents (TOC) from PDF metadata. If absent, it falls back to a **Heuristic Text Scanner** (`structure.py`) that identifies TOC pages based on indentation patterns, dotted leaders (`...... 5`), and keywords.
    *   **Authoritative Hierarchy**: This extracted TOC serves as the immutable skeleton for the entire document, defining valid `Heading Paths` (e.g., `Chapter 1 > Section 1.2`).

### 3.2 Pre-Chunking Sanitization (Header Enforcement)
A critical step to prevent RAG hallucinations effectively.

*   **Problem**: PDF parsers often misclassify long, bold sentences or sidebar text as "Headers," polluting the retrieval context.
*   **Solution**: **Strict Header Enforcement**. 
    *   Before the chunker runs, every segment labeled as `Header` contains a lookup against the Authoritative TOC.
    *   **Action**: If a header is not found in the TOC, it is aggressively downgraded to a `Paragraph`. This eliminates "toxic" heading paths.

### 3.3 Semantic Chunking (LogicSegmenter)
The `LogicSegmenter` transforms a flat list of parsed elements into coherent chunks using structural logic rather than token counts.

*   **Structural Boundaries**: Chunks are strictly bounded by document sections. A chunk never crosses a `Header` boundary unless it is explicitly adsorbing a sub-header.
*   **Cross-Page Continuation**: 
    *   **Algorithm**: Detects paragraphs split across pages by analyzing sentence completeness (e.g., ends with no punctuation, starts with lowercase). 
    *   **Result**: Merges split segments into a single semantic unit, ensuring that a sentence is never broken in the vector space.
*   **Caption Bonding**: 
    *   **Logic**: Captions (e.g., "Table 1.1: Returns") are physically bonded to their associated structural block (Table, Picture, Formula).
    *   **Benefit**: Prevents captions from becoming "orphan" chunks that retrieve without their data.
*   **Fragment Repair**: 
    *   **Logic**: Automatically detects short artifacts (< 5 words) that look like `explanation` but are likely line-break fragments. These are merged retroactively into the preceding sentence.

### 3.4 Semantic Enrichment (POSAnalyzer)
Once a chunk is formed, we analyze its content to assign Semantic Roles. This is crucial for filtering and re-ranking in RAG.

#### A. Role Taxonomy
We use a rule-based NLP engine (`spaCy`) to classify sentences into roles:
*   `definition`: Defines a term.
*   `procedure`: Step-by-step instructions.
*   `assumption`: Hypothetical statements (e.g., "Suppose that...").
*   `explanation`: General explanatory text.
*   `evidence`: Data-heavy statements.

#### B. RAG Optimization (Keyword Overrides)
To specifically improve retrieval for common user intent patterns ("Give me an example", "Summarize this"), we implemented **Force-Overrides**:
*   **Examples**: Sentences starting with `For example`, `For instance`, `e.g.` are forced to role `example`.
*   **Conclusions**: Sentences starting with `In summary`, `Therefore`, `Thus`, `Consequently` are forced to role `conclusion`.

#### C. Noise Filtering (NER Salvation)
*   **Problem**: Heuristic cleaning often deletes very short sentences (assumed to be page numbers or noise).
*   **Solution**: **NER Salvation**.
    *   Before deletion, we check for Named Entities (`PERSON`, `ORG`, `GPE`).
    *   **Effect**: A short segment like "Carl Icahn" (2 words) is preserved because it contains a critical entity, whereas "Page 2" is discarded.

---

## 4. Output Data Structure

The final output is a JSON catalog where each chunk contains:
*   `content`: Cleaned text.
*   `heading_path`: Validated hierarchical path (e.g., `Investments > Chapter 5 > Risk`).
*   `roles`: Distribution of sentence roles (e.g., `{"definition": 20%, "example": 80%}`).
*   `metadata`: Page numbers, bounding boxes, and continued_status.

---
**Confidentiality**: Internal Technical Report for Synapta Team.
