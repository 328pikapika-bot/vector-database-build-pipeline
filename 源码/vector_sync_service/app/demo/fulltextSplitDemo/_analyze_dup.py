import json, os

chunks_path = os.path.join(os.path.dirname(__file__), 'output', 'chunks_b85b160c5f0c9778f85fbab812bf1050.jsonl')
paragraphs = []
fixed = []

with open(chunks_path, 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line.strip())
        if c['chunkType'] == 'paragraph':
            paragraphs.append(c)
        elif c['chunkType'] == 'fixed_chunk':
            fixed.append(c)

total_para_chars = sum(len(p['content']) for p in paragraphs)
total_fixed_chars = sum(len(f['content']) for f in fixed)

parent_ids = set(f['parentParaId'] for f in fixed)
split_paras = [p for p in paragraphs if p['chunkId'] in parent_ids]
normal_paras = [p for p in paragraphs if p['chunkId'] not in parent_ids]

print('当前存储结构:')
print(f'  section_header: 15 条')
print(f'  paragraph: {len(paragraphs)} 条')
print(f'    其中未切分段落: {len(normal_paras)} 条 (内容 < 1500 字符)')
print(f'    其中被切分段落: {len(split_paras)} 条 (内容 > 1500 字符)')
print(f'  fixed_chunk:  {len(fixed)} 条')

print()
print('数据量统计:')
print(f'  paragraph 总字符: {total_para_chars:,}')
print(f'  fixed_chunk 总字符: {total_fixed_chars:,}')

split_para_chars = sum(len(p['content']) for p in split_paras)
split_fixed_chars = sum(len(f['content']) for f in fixed)
print(f'\n  被切分段落(完整版): {split_para_chars:,} 字符')
print(f'  对应固定块(含重叠): {split_fixed_chars:,} 字符')
print(f'  冗余: {split_para_chars + split_fixed_chars:,} 字符 = {split_para_chars/(split_para_chars+split_fixed_chars)*100:.1f}% 是重复内容')

print()
print('被切分段落明细:')
for p in sorted(split_paras, key=lambda x: len(x['content']), reverse=True):
    pid = p['chunkId']
    fcs = [f for f in fixed if f['parentParaId'] == pid]
    para_len = len(p['content'])
    fcs_len = sum(len(f['content']) for f in fcs)
    overlap_len = fcs_len - para_len
    section = p.get('sectionPath', '?')[:30]
    print(f'  [{section}] 段落={para_len}字, {len(fcs)}个固定块={fcs_len}字, 重叠={overlap_len}字')

print()
print(f'JSONL 文件总大小: {os.path.getsize(chunks_path)/1024:.1f} KB')
