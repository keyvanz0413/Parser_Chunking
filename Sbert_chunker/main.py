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
    
    # 示例路径
    input_file = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/outputs/Investments_flat.json"
    output_file = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/outputs/Sbert_refactored_output.json"
    
    config = ChunkingConfig()
    segmenter = LogicSegmenter(config=config)
    
    print(f"Starting refactored chunking pipeline...")
    result = segmenter.process_file(input_file, output_file)
    print(f"Success! Created {result['metadata']['total_chunks']} chunks.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
