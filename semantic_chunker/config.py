# Configuration for chunking behavior.

class ChunkingConfig:
    """Configuration for chunking behavior."""
    # Token limits (approximate, using word count as proxy)
    MIN_CHUNK_WORDS = 30          # Merge if below this
    MAX_CHUNK_WORDS = 500         # Split if above this
    TARGET_CHUNK_WORDS = 200      # Ideal chunk size
    
    # Overlap settings
    OVERLAP_SENTENCES = 2         # Number of sentences to carry over
    ENABLE_OVERLAP = True         # Toggle overlap feature
    
    # Buffer settings
    MAX_BUFFER_SEGMENTS = 15       # Max segments before forced flush
    
    # Short paragraph merge threshold
    SHORT_PARAGRAPH_WORDS = 50    # Paragraphs shorter than this may be merged
    
    # Cross-page continuation settings
    ENABLE_CONTINUATION_DETECTION = True  # Enable cross-page continuation
    CONTINUATION_STYLE_THRESHOLD = 0.6    # Minimum style similarity score for continuation
    CONTINUATION_COLUMN_STRICT = True     # Require same column for continuation
    
    # Pipeline Phase settings
    ENABLE_FURNITURE_DETECTION = True     # Pre-Phase: Furniture detection
    ENABLE_READING_ORDER_CORRECTION = True   # Phase 1: Column-based reordering
    ENABLE_HEADING_RECONSTRUCTION = True     # Phase 2: Stack-based hierarchy rebuild
    ENABLE_BACKFILL_CORRECTION = True        # Phase 3: Same-page L1 backfilling
    
    # Column detection thresholds
    COLUMN_DETECTION_THRESHOLD = 0.45        # x < page_width * threshold => left column
    ENABLE_COLUMN_ISOLATION = True          # Force spanning -> left -> right order
    COLUMN_MERGE_GUARD = True               # Prevents merging segments from different columns
    SEGMENT_ID_GAP_THRESHOLD = 8            # Max ID difference for merging
    PAGE_WIDTH_DEFAULT = 612.0               # Default PDF page width (8.5" × 72dpi)
    PAGE_HEIGHT_DEFAULT = 792.0              # Default PDF page height (11" × 72dpi)
    
    # Furniture detection settings (Safe Zone)
    FURNITURE_TOP_BAND = 0.08                # Page top 8% considered header zone
    FURNITURE_BOTTOM_BAND = 0.08             # Page bottom 8% considered footer zone
    FURNITURE_LEFT_BAND = 0.05               # Left margin 5% (Sidebars)
    FURNITURE_RIGHT_BAND = 0.05              # Right margin 5% (Sidebars)
    FURNITURE_REPEAT_THRESHOLD = 0.20        # String appearing on 20%+ pages is "repeated"
    FURNITURE_MAX_WORDS = 8                  # Short text threshold for furniture
    FURNITURE_MIN_PAGES_FOR_STATS = 10       # Minimum pages to compute frequency stats
    
    # Dehyphenation settings
    ENABLE_DEHYPHENATION = True              # Enable cross-page hyphen repair

    # Content Gating settings (New)
    ENABLE_CONTENT_GATING = True             # Enable main body vs front/back matter gating
    STRIP_FRONT_MATTER = False               # If True, discard front matter instead of labeling
    
    # Gating Patterns (Regex)
    MAIN_BODY_START_PATTERNS = [
        r'^(?:Chapter|CHAPTER|Section|SECTION)\s*(?:1|I|ONE)\b',
        r'^1\.\s+[A-Z\u4e00-\u9fa5]',
        r'^第[一1]章'
    ]
    BACK_MATTER_PATTERNS = [
        r'^(?:Index|Bibliography|References|Appendix|Appendices)\b',
        r'^(?:索引|参考文献|附录|后记)'
    ]
    GATING_SCAN_LIMIT_PAGE = 100             # Max page to search for Main Body start
    GATING_MIN_BODY_OFFSET = 2               # Min pages after last TOC before body start
    GATING_L1_DENSITY_THRESHOLD = 3          # Headers in 5 pages window to consider as TOC
    
    # Spatial Caption Bonding settings
    ENABLE_SPATIAL_BONDING_CHECK = True      # Enable bbox proximity verification
    MAX_CAPTION_VERTICAL_DISTANCE = 0.25     # Max 25% of page height between caption and block
    MAX_CAPTION_HORIZONTAL_OVERLAP = 0.3     # Min 30% horizontal overlap required
    
    # Header Enforcement
    ENABLE_STRICT_HEADER_ENFORCEMENT = True  # Pre-chunking: Downgrade non-TOC headers to Paragraphs
