import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Sbert_chunker.analyzers.semantic import SemanticAnalyzer
from Sbert_chunker.core.config import ChunkingConfig

def test_semantic():
    config = ChunkingConfig()
    analyzer = SemanticAnalyzer(config)
    
    test_sentences = [
        "A bond is a financial instrument that represents a loan.",
        "To calculate the yield, divide the coupon by the price.",
        "This model does not work in high volatility markets.",
        "The correlation coefficient was 0.95."
    ]
    
    roles = ["explanation"] * len(test_sentences)
    results = analyzer.batch_refine_roles(test_sentences, roles)
    
    for sent, (role, score) in zip(test_sentences, results):
        print(f"Sent: {sent}")
        print(f"Role: {role} (Score: {score})")
        print("-" * 20)

if __name__ == "__main__":
    test_semantic()
