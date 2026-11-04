"""验证语义切分效果 — 与原有硬切分对比"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.demo.fulltextSplitDemo.semantic_splitter import semantic_split, find_semantic_split

def load_long_paragraphs(jsonl_path):
    """从 chunks JSONL 中提取被切分的完整段落"""
    paragraphs = []
    fixed_parent_ids = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line.strip())
            if c['chunkType'] == 'fixed_chunk':
                fixed_parent_ids.add(c['parentParaId'])

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line.strip())
            if c['chunkType'] == 'paragraph' and c['chunkId'] in fixed_parent_ids:
                paragraphs.append(c)

    return paragraphs


def demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(demo_dir, 'output', 'chunks_b85b160c5f0c9778f85fbab812bf1050.jsonl')

    paragraphs = load_long_paragraphs(jsonl_path)
    print(f'测试段落: {len(paragraphs)} 个')

    for p in paragraphs:
        section = p.get('sectionPath', '?')[:35]
        text = p['content']
        text_len = len(text)
        print(f'\n{"="*60}')
        print(f'[{section}]  原文 {text_len} 字符')
        print(f'{"="*60}')

        chunks = semantic_split(text, max_chars=1500, overlap_chars=200, search_window=200)
        print(f'  切分块数: {len(chunks)}')
        for i, ch in enumerate(chunks):
            content = ch['content']
            start = ch['para_start']
            end = ch['para_end']
            boundary_tail = content[-30:].replace('\n', ' ')
            boundary_head = content[:30].replace('\n', ' ')

            # 判断切分点质量
            quality = ''
            if len(content) >= 1400:
                tail = content[-50:].replace('\n', ' ')
                if any(tail.rstrip().endswith(c) for c in ['. ', '! ', '? ']):
                    quality = '[句子结尾]'
                elif any(tail.rstrip().endswith(c) for c in [', ', '; ', ': ']):
                    quality = '[从句边界]'
                elif tail.rstrip()[-1:] in ' .!?':
                    quality = '[单词边界]'
                else:
                    quality = '[硬切分]'

            print(f'    [{i}] {start}-{end} ({len(content)}字) {quality}')
            print(f'         ...{boundary_tail}')
            if i < len(chunks) - 1:
                print(f'         {boundary_head}...')


if __name__ == '__main__':
    demo()
