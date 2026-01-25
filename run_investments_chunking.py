import sys
import os
import logging

# Add current directory to path so Sbert_chunker can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Sbert_chunker.core.pipeline import LogicSegmenter
from Sbert_chunker.core.config import ChunkingConfig

def main():
    logging.basicConfig(level=logging.INFO)
    
    input_file = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/outputs/Docling_json/Investments.json"
    output_file = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/outputs/Chunks_Refactored/Investments.json"
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    config = ChunkingConfig()
    # For speed in this initial test, you might want to disable heavy models if not needed,
    # but the user asked to use SBERT chunking, so we keep defaults.
    
    segmenter = LogicSegmenter(config=config)
    
    print(f"Processing {input_file}...")
    try:
        result = segmenter.process_file(input_file, output_file)
        print(f"Success! Generated {result['metadata']['total_chunks']} chunks.")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
