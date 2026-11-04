import json

with open('output/chunks_b85b160c5f0c9778f85fbab812bf1050.jsonl', 'r', encoding='utf-8') as f:
    chunks = [json.loads(l) for l in f if l.strip()]

fixed = [c for c in chunks if c['chunkType'] == 'fixed_chunk']
print('fixed_chunk total:', len(fixed))

good = 0
clause = 0
word_cut = 0
other = 0

for i, fc in enumerate(fixed):
    content = fc['content']
    tail = content[-80:].replace('\n', ' ')
    head = content[:60].replace('\n', ' ')
    start = fc['paraStartIdx']
    end = fc['paraEndIdx']

    clean = tail.rstrip()
    quality = ''
    if clean and clean[-1] in '.!?':
        quality = 'SENTENCE'
        good += 1
    elif clean and clean[-1] in ',;:':
        quality = 'CLAUSE'
        clause += 1
    elif clean and clean[-1].isalpha():
        quality = 'WORD-CUT'
        word_cut += 1
    else:
        quality = 'OTHER'
        other += 1

    section = fc.get('sectionPath', '')[:25]
    print(f'[{i}] {section:25s} | {start:5d}-{end:5d} | {quality:10s}')
    print(f'      tail: ...{tail[-60:]}')
    print(f'      head: {head[:60]}...')
    print()

print(f'\nSummary: SENTENCE={good}, CLAUSE={clause}, WORD-CUT={word_cut}, OTHER={other}')
print(f'Quality: {(good+clause)/len(fixed)*100:.0f}% acceptable boundaries')
