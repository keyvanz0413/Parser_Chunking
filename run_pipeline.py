import sys
import logging
import time
from pathlib import Path

# Ensure semantic_chunker is importable
base_dir = Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.append(str(base_dir))

from semantic_chunker.parser import EnhancedParser
from semantic_chunker.segmenter import LogicSegmenter
from semantic_chunker.config import ChunkingConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode='w')
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("Pipeline")

    input_dir = base_dir / "inputs"
    docling_output_dir = base_dir / "outputs" / "Docling_json"
    chunk_output_dir = base_dir / "outputs" / "Chunks_Semantic"

    docling_output_dir.mkdir(parents=True, exist_ok=True)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files.")

    # 1. Initialize Parser and Segmenter
    logger.info("Initializing Parser and Segmenter...")
    
    # Initialize Parser
    # Note: OCR disabled by default for speed, set do_ocr=True if needed
    parser = EnhancedParser(do_ocr=False) 
    
    # Initialize Segmenter
    chunking_config = ChunkingConfig()
    try:
        segmenter = LogicSegmenter(use_pos=True, config=chunking_config)
    except Exception as e:
        logger.warning(f"POS Segmenter init failed ({e}), falling back to simple mode.")
        segmenter = LogicSegmenter(use_pos=False, config=chunking_config)

    # 2. Process Loop
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        
        # Step A: Parse (PDF -> JSON)
        json_name = pdf_path.stem + ".json"
        json_path = docling_output_dir / json_name
        
        parse_successful = False
        try:
            if json_path.exists():
                logger.info(f"  - Parsing: Skipped (JSON exists). Remove {json_path.name} to force re-parse.")
                parse_successful = True
            else:
                logger.info(f"  - Parsing PDF...")
                parser.parse_pdf(str(pdf_path), str(json_path))
                parse_successful = True
        except Exception as e:
            logger.error(f"  Parser failed for {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()

        if not parse_successful:
            continue

        # Step B: Chunk (JSON -> JSON)
        final_output_path = chunk_output_dir / json_name
        try:
            logger.info(f"  - Chunking segments...")
            segmenter.process_file(str(json_path), str(final_output_path))
            logger.info(f"  ✓ Chunking complete. saved to {final_output_path}")
            
            # Verify stats
            # We can verify output stats if needed
            
        except Exception as e:
            logger.error(f"  Chunker failed for {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("-" * 50)
    logger.info("All tasks completed.")

if __name__ == "__main__":
    main()
