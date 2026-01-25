import json
import re
import os
from typing import List, Dict

class ChunkCorrector:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.data = {}
        self.chunks = []

    def load(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            self.chunks = self.data.get('chunks', [])

    def save(self):
        self.data['chunks'] = self.chunks
        # Update metadata stats
        self.data['metadata']['total_chunks'] = len(self.chunks)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def fix_dehyphenation(self):
        """修复跨块连字符断词问题"""
        text_types = ['explanation', 'example', 'sidebar', 'procedure', 'summary', 'learning_objective', 'definition']
        i = 0
        while i < len(self.chunks) - 1:
            curr = self.chunks[i]
            if curr.get('chunk_type') not in text_types or not curr.get('content'):
                i += 1
                continue

            content = curr['content'].strip()
            if content.endswith('-'):
                # 寻找下一个是以小写字母开头的 textual chunk
                next_idx = i + 1
                found_next = False
                while next_idx < min(i + 10, len(self.chunks)): # 向后看 10 个块
                    nxt = self.chunks[next_idx]
                    nc = nxt.get('content', '').lstrip()
                    if nxt.get('chunk_type') in text_types and nc:
                        if nc[0].islower():
                            found_next = True
                            break
                        # 如果是像 "KEY TERMS" 这样的大写开头块，继续往后看
                        # 但如果是 HEADER 类型且字数较多，通常是新章节，停止寻找
                        if nxt.get('chunk_type') == 'header' and len(nc) > 20:
                            break
                    next_idx += 1
                
                if found_next:
                    nxt = self.chunks[next_idx]
                    next_content = nxt.get('content', '').lstrip()
                    first_word_match = re.match(r'^(\w+)', next_content)
                    if first_word_match:
                        suffix = first_word_match.group(1)
                        print(f"Fixing dehyphenation: {content[-10:]} + {suffix} in {curr['chunk_id']} (using {nxt['chunk_id']})")
                        
                        curr['content'] = content.rstrip('-') + suffix + next_content[len(suffix):]
                        
                        if curr['sentences'] and nxt['sentences']:
                            if curr['sentences'][-1]['text'].strip().endswith('-'):
                                curr['sentences'][-1]['text'] = curr['sentences'][-1]['text'].rstrip('-') + suffix
                                nxt['sentences'][0]['text'] = nxt['sentences'][0]['text'][len(suffix):].lstrip()
                                if not nxt['sentences'][0]['text'].strip():
                                    nxt['sentences'].pop(0)

                        if len(nxt['content'].strip()) < 15 or not nxt['sentences']:
                            self.merge_chunks(i, next_idx)
                            continue
            i += 1

    def merge_chunks(self, idx1: int, idx2: int):
        """合并两个 Chunk"""
        c1 = self.chunks[idx1]
        c2 = self.chunks[idx2]
        
        c1['content'] = (c1['content'].strip() + " " + c2['content'].strip()).strip()
        if 'sentences' in c1 and 'sentences' in c2:
            c1['sentences'].extend(c2['sentences'])
        if 'source_segments' in c1 and 'source_segments' in c2:
            c1['source_segments'] = list(set(c1['source_segments'] + c2['source_segments']))
        
        self.chunks.pop(idx2)

    def clean_header_noise(self):
        """清理无意义的标题噪点"""
        new_chunks = []
        for chunk in self.chunks:
            if chunk.get('chunk_type') == 'header':
                content = chunk.get('content', '').strip().replace('\xa0', ' ')
                if re.search(r'LWI\s+\d+', content) or \
                   re.match(r'^[\d\s]+$', content) or \
                   re.match(r'^[A-Z]$', content) or \
                   (len(content) < 5 and any(c.isdigit() for c in content)):
                    
                    if re.match(r'^[A-Z]$', content):
                        chunk['chunk_type'] = 'index_marker'
                    else:
                        print(f"Removing header noise: {content}")
                        continue
            new_chunks.append(chunk)
        self.chunks = new_chunks

    def filter_formula_fragments(self):
        """过滤被错误识别为正文的数学公式碎片"""
        new_chunks = []
        for chunk in self.chunks:
            keep = True
            if chunk.get('chunk_type') == 'explanation':
                content = chunk.get('content', '').strip()
                if content.count('_') > 2 or \
                   re.match(r'^[0-9\(\)\+\s\.\,*/=_\-≈≤≥αβγδσεστφ]+$', content):
                    if len(content) < 40:
                        print(f"Filtering formula fragment: {content}")
                        keep = False
            if keep:
                new_chunks.append(chunk)
        self.chunks = new_chunks

    def merge_orphans(self):
        """合并孤立的小片段"""
        i = 0
        while i < len(self.chunks):
            chunk = self.chunks[i]
            content = chunk.get('content', '').strip().replace('\xa0', ' ')
            
            is_orphan = False
            if len(content) < 20: 
                cleaned = content.lower().rstrip(',.:')
                if cleaned in ['therefore', 'however', 'thus', 'chapter', 'preface', 'consequence', 'strictly']:
                    is_orphan = True
                elif re.match(r'^\d+(\.\d+)*$', cleaned): 
                    is_orphan = True
            
            if is_orphan:
                if i + 1 < len(self.chunks):
                    if self.chunks[i+1].get('chunk_type') in ['explanation', 'header', 'example', 'sidebar']:
                        print(f"Merging orphan '{content}' forward into {self.chunks[i+1]['chunk_id']}")
                        self.merge_chunks(i, i + 1)
                        continue
                if i > 0:
                    target_idx = i - 1
                    print(f"Merging orphan '{content}' backward into {self.chunks[target_idx]['chunk_id']}")
                    self.merge_chunks(target_idx, i)
                    i -= 1
                    continue
            i += 1

    def run(self):
        print(f"Starting correction for {self.input_path}...")
        self.load()
        initial_count = len(self.chunks)
        print(f"Initial chunk count: {initial_count}")
        
        self.fix_dehyphenation()
        print(f"After dehyphenation: {len(self.chunks)}")
        
        self.clean_header_noise()
        print(f"After header noise cleanup: {len(self.chunks)}")
        
        self.filter_formula_fragments()
        print(f"After formula filter: {len(self.chunks)}")
        
        self.merge_orphans()
        print(f"After orphan merger: {len(self.chunks)}")
        
        final_count = len(self.chunks)
        print(f"Correction finished. Total chunks: {final_count} (Reduced by {initial_count - final_count})")
        self.save()

if __name__ == "__main__":
    input_file = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/outputs/Chunks_sbert/Investments.json"
    output_file = "/Users/keyvanzhuo/Documents/CodeProjects/Synapta/Parser_Chunking/outputs/Chunks_sbert/Investments_fixed.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    corrector = ChunkCorrector(input_file, output_file)
    corrector.run()
