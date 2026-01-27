import os
import argparse
import logging
from .core.pipeline import LogicSegmenter
from .core.config import ChunkingConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    
    # Base directory for the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "outputs", "Docling_json")
    output_dir = os.path.join(base_dir, "outputs", "Chunks_Sbert")
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan for JSON files
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
        
    print(f"Found {len(json_files)} files to process in {input_dir}")
    
    config = ChunkingConfig()
    try:
        segmenter = LogicSegmenter(use_pos=True, config=config)
    except RuntimeError as e:
        if "Could not load spaCy" in str(e) or "unable to infer type" in str(e):
            logging.warning(f"SpaCy/POS init failed ({e}). Falling back to use_pos=False.")
            segmenter = LogicSegmenter(use_pos=False, config=config)
        else:
            raise
    
    for filename in json_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\nProcessing: {filename}")
        try:
            result = segmenter.process_file(input_path, output_path)
            print(f"Success! Created {result['metadata']['total_chunks']} chunks.")
            print(f"Output saved to: {output_path}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
