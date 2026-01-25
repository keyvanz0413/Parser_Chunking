import re

class ChunkingConfig:
    """Configuration for chunking behavior."""
    # Token limits
    MIN_CHUNK_WORDS = 30          
    MAX_CHUNK_WORDS = 500         
    TARGET_CHUNK_WORDS = 200      
    
    # Overlap settings
    OVERLAP_SENTENCES = 2         
    ENABLE_OVERLAP = True         
    
    # Buffer settings
    MAX_BUFFER_SEGMENTS = 5       
    
    # Short paragraph merge threshold
    SHORT_PARAGRAPH_WORDS = 50    
    
    # Cross-page continuation settings
    ENABLE_CONTINUATION_DETECTION = True  
    CONTINUATION_STYLE_THRESHOLD = 0.6    
    CONTINUATION_COLUMN_STRICT = True     
    
    # Pipeline Phase settings
    ENABLE_FURNITURE_DETECTION = True     
    ENABLE_READING_ORDER_CORRECTION = True   
    ENABLE_HEADING_RECONSTRUCTION = True     
    ENABLE_BACKFILL_CORRECTION = True        
    
    # Column detection thresholds
    COLUMN_DETECTION_THRESHOLD = 0.45        
    PAGE_WIDTH_DEFAULT = 612.0               
    PAGE_HEIGHT_DEFAULT = 792.0              
    
    # Furniture detection settings
    FURNITURE_TOP_BAND = 0.10                
    FURNITURE_BOTTOM_BAND = 0.10             
    FURNITURE_REPEAT_THRESHOLD = 0.20        
    FURNITURE_MAX_WORDS = 5                  
    FURNITURE_MIN_PAGES_FOR_STATS = 10       
    
    # Dehyphenation settings
    ENABLE_DEHYPHENATION = True              
    
    # Sidebar Detection Settings
    ENABLE_SIDEBAR_DETECTION = True           
    SIDEBAR_WIDTH_RATIO_MAX = 0.35            
    SIDEBAR_MIN_X_GAP = 30.0                  
    SIDEBAR_ISOLATION_STRICT = True           
    
    SIDEBAR_HEADING_PATTERNS = [
        r'(?i)^(?:chapter\s+)?objectives?',
        r'(?i)^learning\s+objectives?',
        r'(?i)^key\s+(?:terms?|points?|takeaways?|concepts?)',
        r'(?i)^summary',
        r'(?i)^highlights?',
        r'(?i)^(?:quick\s+)?review',
        r'(?i)^(?:in\s+)?this\s+chapter',
        r'(?i)^(?:margin(?:al)?|side)\s*(?:note|box)?',
    ]
    
    LEARNING_OBJECTIVE_PATTERNS = [
        r'(?i)^(?:describe|explain|understand|identify|define|compare|calculate|evaluate|analyze|apply|discuss|list|summarize|provide|address|reacquaint|outline|review|introduce|show|demonstrate|determine|examine|evaluate|survey)\s+',
        r'(?i)^(?:LO|LOS|Learning\s+(?:Objective|Outcome))\s*\d*',
    ]
    
    # Semantic Analysis Settings
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"    
    ENABLE_SEMANTIC_ANALYSIS = True          
    SEMANTIC_SIMILARITY_THRESHOLD = 0.5      
    SEMANTIC_WINDOW_SIZE = 3                 
    ENABLE_SEMANTIC_CHUNKING = True          
    ROLE_PROTOTYPE_SIMILARITY_THRESHOLD = 0.75  
    ENABLE_PROTOTYPE_MATCHING = True          
    ROLE_PROTOTYPES_PATH = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/Sbert_chunker/role_prototypes.json"
    
    CHUNK_START_ROLES = {'topic', 'question'}     
    CHUNK_CONTINUE_ROLES = {'interpretation', 'conclusion', 'evidence'}  
    
    # Block-level Role Smoothing Settings
    ENABLE_ROLE_SMOOTHING = True                  
    SMOOTHING_MAJORITY_THRESHOLD = 0.80           
    SMOOTHING_LOW_CONFIDENCE_THRESHOLD = 0.6      
    SMOOTHING_WINDOW_SIZE = 1                     
    SMOOTHING_NEIGHBOR_CONFIDENCE = 0.5           
    
    ROLE_FLOW_PATHS = {
        'topic': {'definition', 'assumption', 'mechanism', 'example', 'explanation'},
        'definition': {'example', 'mechanism', 'application', 'limitation', 'explanation'},
        'assumption': {'mechanism', 'procedure', 'example', 'explanation'},
        'mechanism': {'example', 'application', 'evidence', 'interpretation', 'explanation'},
        'procedure': {'example', 'evidence', 'interpretation', 'conclusion', 'explanation'},
        'example': {'interpretation', 'evidence', 'conclusion', 'comparison', 'explanation'},
        'evidence': {'interpretation', 'conclusion', 'comparison', 'explanation'},
        'interpretation': {'conclusion', 'application', 'comparison', 'explanation'},
        'comparison': {'conclusion', 'interpretation', 'explanation'},
        'conclusion': {'application', 'explanation'},
        'application': {'example', 'conclusion', 'explanation'},
        'limitation': {'example', 'conclusion', 'explanation'},
        'explanation': {'definition', 'example', 'mechanism', 'procedure', 'evidence', 
                       'interpretation', 'conclusion', 'comparison', 'application', 'explanation'},
    }
    
    CHUNK_TYPE_ROLE_AFFINITY = {
        'procedure': {'procedure', 'example', 'evidence'},
        'example': {'example', 'evidence', 'interpretation'},
        'definition': {'definition', 'mechanism', 'explanation'},
        'theorem': {'definition', 'mechanism', 'explanation'},
        'proof': {'procedure', 'mechanism', 'evidence'},
        'list': {'procedure', 'example', 'evidence'},
    }
