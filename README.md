# Synapta Parser & Chunker

A robust, structure-aware PDF parsing and semantic chunking pipeline designed for high-quality RAG (Retrieval-Augmented Generation) applications.

## 🚀 Key Features

### 1. Structure-Aware Parsing (Enhanced Parser)
- **Docling Integration**: Leverages Docling for accurate layout analysis.
- **Hierarchical Tree**: Builds a strict `Root > Chapter > Section` tree to preserve document structure.
- **Authoritative TOC**: Extracts the Table of Contents (TOC) to serve as the ground truth for document hierarchy.

### 2. Semantic Chunking (Logic Segmenter)
This project uses a sophisticated "Logic Segmenter" that goes beyond simple sliding windows:

#### 🧠 Intelligent Role Detection
- **POS & Rule-Based Analysis**: Identifies sentence roles (`explanation`, `definition`, `procedure`, `hypothesis`, etc.).
- **Keyword Overrides**:
  - `For example...` -> Forces `example` role (improves RAG retrieval for "give me examples of...").
  - `In summary...` / `Therefore...` -> Forces `conclusion` role.
- **NER "Salvation"**: Prevents discarding short sentences if they contain important Named Entities (e.g., "Carl Icahn").

#### 🛡️ Heading Path Sanitization (Pre-chunking)
- **TOC Enforcement**: Before chunking, every identified "Header" is validated against the authoritative TOC.
- **Toxic Header Removal**: Headers not found in the TOC are downgraded to Paragraphs, preventing "dirty" heading paths (e.g., extremely long sentences mistakenly identified as headers) from polluting chunks.

#### 🧩 Advanced Merging & Repair
- **Fragment Merging**: Automatically detects and merges distinct sentence fragments (< 5 words) caused by PDF line breaks or OCR errors.
- **Cross-Page Continuation**: intelligenty detects paragraphs spanning across pages and merges them.
- **Caption Bonding**: "Table 1.1" captions are physically bonded to their corresponding Table/Image blocks, preventing them from becoming orphan chunks.

## 📂 Project Structure

```text
Parser_Chunking/
├── run_pipeline.py          # ⚡️ Main Entry Point
├── inputs/                  # Place PDF files here
├── outputs/                 # Results
│   ├── Docling_json/        # Intermediate parsed JSON
│   └── Chunks_Semantic/     # Final semantic chunks (Ready for RAG)
├── semantic_chunker/        # Core Logic Package
│   ├── parser.py            # Enhanced Parser module
│   ├── segmenter.py         # Main chunking logic
│   ├── config.py            # Configuration settings
│   └── analyzers/           # NLP analyzers (POS, Structure, etc.)
└── sbert_chunker/           # (Optional) SBERT-based alternative
```

## 🛠️ Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    ```

2.  **Add PDFs**:
    Place your PDF files in the `inputs/` directory.

3.  **Run Pipeline**:
    ```bash
    python run_pipeline.py
    ```

4.  **Check Results**:
    Parsed chunks will be in `outputs/Chunks_Semantic/*.json`.

## ⚙️ Configuration

You can tweak chunking behavior in `semantic_chunker/config.py`:

```python
class ChunkingConfig:
    # ...
    ENABLE_STRICT_HEADER_ENFORCEMENT = True  # Toggle TOC-based header filtering
    ENABLE_CONTINUATION_DETECTION = True      # Toggle cross-page logic
    # ...
```
