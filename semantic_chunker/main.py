import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path to ensure we can import if running directly
# This helps when running as `python semantic_chunker/main.py` vs `python -m semantic_chunker.main`
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))

from .config import ChunkingConfig
from .segmenter import LogicSegmenter

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Base directory for the project (Parser_Chunking)
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = base_dir / "outputs" / "Docling_json"
    output_dir = base_dir / "outputs" / "Chunks_Semantic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for JSON files from parser_docling
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please run parser_docling.py first to generate input JSONs.")
        sys.exit(1)

    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        print("Please run parser_docling.py first.")
        sys.exit(0)
    
    print(f"Found {len(json_files)} files to process in {input_dir}")
    
    # Initialize segmenter
    config = ChunkingConfig()
    config.ENABLE_OVERLAP = True
    config.OVERLAP_SENTENCES = 2
    
    # You might want to allow enabling/disabling features here or via args
    try:
        segmenter = LogicSegmenter(use_pos=True, config=config)
    except RuntimeError as e:
        if "Could not load spaCy" in str(e):
            logging.warning(f"SpaCy load failed ({e}). Falling back to use_pos=False.")
            segmenter = LogicSegmenter(use_pos=False, config=config)
        else:
            raise
    
    for input_path in json_files:
        output_path = output_dir / input_path.name
        
        print(f"\n{'='*70}")
        print(f"Processing: {input_path.name}")
        print(f"Output: {output_path}")
        print(f"{'='*70}")
        
        try:
            result = segmenter.process_file(str(input_path), str(output_path))
            
            # Print summary for this file
            metadata = result.get('metadata', {})
            stats = metadata.get('processing_stats', {})
            print(f"✓ Done - Generated {metadata.get('total_chunks', 0)} chunks")
            if stats:
                print(f"  Avg words/chunk: {stats.get('avg_words_per_chunk', 0):.1f}")
                print(f"  Cross-page chunks: {stats.get('cross_page_chunks', 0)}")
            
        except Exception as e:
            print(f"✗ Error processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Processing complete.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
