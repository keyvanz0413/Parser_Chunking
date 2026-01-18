import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import DoclingDocument

logger = logging.getLogger(__name__)

# Import metadata extractor for rich metadata
try:
    import sys
    from pathlib import Path as _Path
    _src_dir = str(_Path(__file__).parent.parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from etl.extractors.pdf_metadata_extractor import PDFMetadataExtractor
    HAS_METADATA_EXTRACTOR = True
except ImportError as e:
    logger.debug(f"PDFMetadataExtractor not available: {e}")
    HAS_METADATA_EXTRACTOR = False

class EnhancedParser:
    """
    Enhanced PDF parser based on Docling.
    
    Design principles:
    1. State machine switching: Identify "main switches" (chapter titles) via regex to enforce subsequent content hierarchy.
    2. Tree-based recursive model: Strictly follows Root > Header > Children structure, optimized for RAG retrieval depth.
    3. TOC pruning: Removes meaningless short titles, numeric-only titles, and empty titles to maintain a clean table of contents.
    4. Structural Skeleton: Every chunk carries heading_path (e.g., "Ch1 > Sec 1.2") for context-aware embedding.
    """
    
    # Core recognition labels
    HEADING_LABELS = {"heading", "header", "title", "section_header"}
    NOISE_LABELS = {"page_header", "page_footer", "footnote", "caption"}
    
    # Heading path separator
    PATH_SEPARATOR = " > "

    def __init__(self, do_ocr: bool = False):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        
        # ===== Fast Mode: Detect positions only, no content parsing =====
        # Tables: Detect position and structure, but no deep cell matching
        pipeline_options.do_table_structure = True
        
        # Formulas: Disable VLM formula enrichment (very slow), position info is still preserved
        pipeline_options.do_formula_enrichment = False
        
        # Pictures: Disable VLM picture description, position info is still preserved
        pipeline_options.do_picture_description = False
        pipeline_options.do_picture_classification = False
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info(f"EnhancedParser initialized (Fast Mode: positions only, OCR: {do_ocr})")

    def parse_pdf(self, pdf_path: str, output_path: Optional[str] = None, 
                   extract_rich_metadata: bool = True) -> Dict[str, Any]:
        """
        Parse PDF and extract structured data with rich metadata.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save JSON output
            extract_rich_metadata: If True, use PDFMetadataExtractor for rich metadata
            
        Returns:
            Dictionary with hierarchy, toc, flat_segments, and rich metadata
        """
        logger.info(f"Parsing PDF with Hierarchical Engine: {pdf_path}")
        conversion_result = self.converter.convert(pdf_path)
        doc = conversion_result.document
        
        hierarchy = self._build_tree(doc)
        flat_segments = self._extract_flat_segments(doc)
        
        # Basic metadata
        basic_metadata = {
            "source_file": Path(pdf_path).name,
            "title": getattr(doc, 'name', ""),
            "total_pages": len(doc.pages) if hasattr(doc, 'pages') else 0,
            "extraction_timestamp": datetime.now().isoformat(),
        }
        
        # Extract rich metadata if available
        rich_metadata = {}
        if extract_rich_metadata and HAS_METADATA_EXTRACTOR:
            rich_metadata = self._extract_rich_metadata(pdf_path, flat_segments)
        
        # Merge metadata (rich metadata takes precedence)
        final_metadata = {**basic_metadata, **rich_metadata}
        
        result_data = {
            "metadata": final_metadata,
            "hierarchy": hierarchy,
            "toc": self._extract_toc(hierarchy),
            "full_markdown": doc.export_to_markdown(),
            "flat_segments": flat_segments
        }
        
        if output_path:
            self._save_json(result_data, output_path)
            
        return result_data
    
    def _extract_rich_metadata(self, pdf_path: str, flat_segments: List[Dict]) -> Dict[str, Any]:
        """
        Extract rich metadata using PDFMetadataExtractor.
        
        This provides:
        - title, authors, publisher
        - ISBN (10 and 13)
        - edition, language
        - subject_categories
        - extraction_method and confidence_score
        """
        try:
            extractor = PDFMetadataExtractor(
                enable_api_lookup=True,  # Try Google Books / Crossref
                max_pages_to_scan=25
            )
            
            # Prepare enhanced_data format for the extractor
            enhanced_data = {
                'flat_segments': flat_segments,
                'metadata': {'title': Path(pdf_path).stem}
            }
            
            metadata = extractor.extract_metadata(
                pdf_path=pdf_path,
                enhanced_data=enhanced_data
            )
            
            if metadata:
                result = metadata.to_dict()
                logger.info(f"Rich metadata extracted: {result.get('title')} by {result.get('authors')}")
                return result
            
        except Exception as e:
            logger.warning(f"Could not extract rich metadata: {e}")
        
        return {}

    def _get_item_type(self, item: Any, label: str, heading_path: str = "") -> str:
        """Identify diverse data types to facilitate linking.
        
        Args:
            item: The Docling item object
            label: The item's label/text
            heading_path: Current heading path context (for LO detection)
        """
        # Check for Learning Objectives context
        # LO items are ListItems under "CHAPTER OBJECTIVES" or "LEARNING OBJECTIVES" headings
        heading_upper = heading_path.upper() if heading_path else ""
        if ("OBJECTIVES" in heading_upper or "LEARNING OBJECTIVE" in heading_upper):
            name = type(item).__name__.lower()
            if "list" in name:
                return "LearningObjective"
        
        if any(h in label for h in self.HEADING_LABELS) and not any(n in label for n in self.NOISE_LABELS):
            return "Header"
        
        name = type(item).__name__.lower()
        if "table" in name: return "Table"
        if "list" in name: return "ListItem"
        if "formula" in name or "equation" in name: return "Formula"
        if "picture" in name or "image" in name: return "Picture"
        
        return "Paragraph"

    def _infer_level(self, text: str) -> int:
        """Core inference logic remains unchanged"""
        t = text.strip()
        if len(t) <= 4 and (t.isdigit() or re.match(r'^[IVXLC]+$', t)):
            return -1
        
        # Match Level 1 sections: Chapter 1, PART I, or long ALL-CAPS titles
        if re.match(r'^(Chapter|Part|Unit|Section|Appendix)\s+([0-9]+|[IVXLC]+)', t, re.I):
            return 1
        if t.isupper() and len(t) > 15:
            return 1
            
        # Match Level 2 cascades: 1.1, 1.2
        if re.match(r'^\d+\.\d+\s', t):
            return 2
        
        # Default returns 0, indicating "normal title", depth decided by builder based on context
        return 0

    def _build_heading_path(self, stack: List[Dict[str, Any]], include_current: str = None) -> str:
        """
        Build the heading path string from the current stack.
        
        Args:
            stack: Current hierarchy stack
            include_current: Optional current node text to append
            
        Returns:
            Heading path string like "Chapter 1 > Section 1.1 > Subsection"
        """
        parents = [n["text"] for n in stack if n["type"] != "Root"]
        if include_current:
            parents.append(include_current)
        return self.PATH_SEPARATOR.join(parents)
    
    def _build_full_context_text(self, heading_path: str, text: str, tags: List[str] = None) -> str:
        """
        Build the full context text for embedding.
        
        Format: [Path: {heading_path}] [Tags: {tags}] {text}
        
        Args:
            heading_path: The hierarchical path string
            text: The actual content text
            tags: Optional semantic tags (for future use)
            
        Returns:
            Context-enriched text ready for embedding
        """
        parts = []
        if heading_path:
            parts.append(f"[Path: {heading_path}]")
        if tags:
            parts.append(f"[Tags: {', '.join(tags)}]")
        parts.append(text)
        return " ".join(parts)

    def _build_tree(self, doc: DoclingDocument) -> Dict[str, Any]:
        """
        Enhanced tree construction with structural skeleton support.
        
        Features:
        - heading_path: Full hierarchy path (e.g., "Chapter 1 > Section 1.1")
        - depth: Nesting depth from root (0 = root, 1 = chapter, 2 = section, etc.)
        - full_context_text: Context-enriched text for RAG embedding
        - bbox: Bounding box coordinates
        """
        root = {"type": "Root", "text": "Document Root", "level": 0, "depth": 0, "children": []}
        stack = [root]
        
        for item, level in doc.iterate_items():
            label = str(getattr(item, 'label', "")).lower()
            text = str(getattr(item, 'text', "")).strip()
            
            # Special case: Pictures and Formulas should be kept even if text is empty
            item_type = self._get_item_type(item, label)
            if not text and item_type not in ["Picture", "Formula", "Table"]:
                continue
            
            inferred = self._infer_level(text)
            
            page_no = item.prov[0].page_no if item.prov else 1
            # Get coordinates (xmin, ymin, xmax, ymax) -> Docling output is usually l, t, r, b
            bbox = None
            if item.prov and item.prov[0].bbox:
                b = item.prov[0].bbox
                bbox = [round(b.l, 2), round(b.t, 2), round(b.r, 2), round(b.b, 2)]
            
            if item_type == "Header" and inferred != -1:
                # Determine real level
                if inferred >= 1:
                    real_level = inferred
                else:
                    # If no explicit Chapter start recognized but inside a Chapter, treat as L2
                    has_l1 = any(n.get("level") == 1 for n in stack)
                    real_level = 2 if has_l1 else 1
            
                # Adjust stack: Pop all equal or deeper levels to ensure proper nesting
                while len(stack) > 1 and stack[-1].get("level", 999) >= real_level:
                    stack.pop()
                
                # Build heading path including this header
                heading_path = self._build_heading_path(stack, include_current=text)
                current_depth = len(stack)  # Depth = number of ancestors
                
                node = {
                    "type": "Header",
                    "text": text,
                    "page": page_no,
                    "bbox": bbox,
                    "level": real_level,
                    "depth": current_depth,
                    "heading_path": heading_path,
                    "full_context_text": self._build_full_context_text(heading_path, text),
                    "children": []
                }
                
                stack[-1]["children"].append(node)
                stack.append(node)
            else:
                # Content node (Paragraph, Table, Picture, Formula, LearningObjective, etc.)
                heading_path = self._build_heading_path(stack)
                current_depth = len(stack)  # Depth under current header
                
                # Re-evaluate item_type with heading context (for LO detection)
                item_type = self._get_item_type(item, label, heading_path)
                
                node = {
                    "type": item_type,
                    "text": text,
                    "page": page_no,
                    "bbox": bbox,
                    "depth": current_depth,
                    "heading_path": heading_path,
                    "full_context_text": self._build_full_context_text(heading_path, text)
                }
                if "children" not in stack[-1]: 
                    stack[-1]["children"] = []
                stack[-1]["children"].append(node)
                
        return root

    def _extract_toc(self, hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract ultra-clean TOC with structural skeleton metadata.
        
        Each TOC entry includes:
        - text: The heading text
        - page: Page number
        - level: Semantic level (1=chapter, 2=section, etc.)
        - depth: Nesting depth in tree
        - heading_path: Full path from root
        - children: Nested child entries
        """
        toc = []

        def traverse(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if node["type"] == "Header":
                # Noise pruning: Short titles with no children are usually misidentifications
                if len(node["text"]) < 5 and not node.get("children"):
                    return None
                
                entry = {
                    "text": node["text"],
                    "page": node["page"],
                    "level": node["level"],
                    "depth": node.get("depth", 1),
                    "heading_path": node.get("heading_path", node["text"]),
                    "children": []
                }
                for child in node.get("children", []):
                    child_entry = traverse(child)
                    if child_entry: 
                        entry["children"].append(child_entry)
                return entry
                
            elif node["type"] == "Root":
                for child in node.get("children", []):
                    child_entry = traverse(child)
                    if child_entry: 
                        toc.append(child_entry)
            return None

        traverse(hierarchy)
        return toc

    def _extract_flat_segments(self, doc: DoclingDocument) -> List[Dict[str, Any]]:
        """
        Extract flat segments with full structural skeleton metadata.
        
        Each segment includes:
        - segment_id: Unique identifier (seg_0001, seg_0002, ...)
        - type: Header, Paragraph, Table, Picture, Formula, etc.
        - text: The content text
        - page: Page number
        - bbox: Bounding box coordinates
        - depth: Nesting depth from root
        - heading_path: Full hierarchy path (e.g., "Chapter 1 > Section 1.1")
        - full_context_text: Context-prefixed text ready for embedding
        - tags: Placeholder for semantic tags (to be filled by MultiTagScanner)
        """
        segments = []
        hierarchy = self._build_tree(doc)
        segment_counter = [0]  # Use list to allow mutation in nested function
        
        def flatten(node: Dict[str, Any]):
            if node["type"] != "Root":
                segment_counter[0] += 1
                seg = {
                    "segment_id": f"seg_{segment_counter[0]:04d}",
                    "type": node["type"],
                    "text": node["text"],
                    "page": node.get("page", 1),
                    "bbox": node.get("bbox"),
                    "depth": node.get("depth", 0),
                    "heading_path": node.get("heading_path", ""),
                    "full_context_text": node.get("full_context_text", node["text"]),
                    "tags": []  # Placeholder for MultiTagScanner (Step 2)
                }
                # Include level for Header nodes
                if node["type"] == "Header":
                    seg["level"] = node.get("level", 1)
                segments.append(seg)
                
            for child in node.get("children", []):
                flatten(child)
        
        flatten(hierarchy)
        return segments

    def _save_json(self, data: Dict[str, Any], path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Clean hierarchical JSON saved to {path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    input_pdf = Path("testFloder/inputs/An Introduction to Derivatives and Risk Management.pdf")
    output_dir = Path("testFloder/outputs/docling_json")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_pdf.exists():
        print(f"Error: Input file not found: {input_pdf}")
        exit(1)
    
    output_name = input_pdf.stem + ".json"
    output_path = output_dir / output_name
    
    print(f"Processing: {input_pdf.name}")
    print(f"Output: {output_path}")
    
    parser = EnhancedParser()
    
    try:
        result = parser.parse_pdf(str(input_pdf), str(output_path))
        print(f"✓ Done - Generated {len(result.get('flat_segments', []))} segments")
        print(f"  Hierarchy depth: {len(result.get('toc', []))} top-level entries")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
