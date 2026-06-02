#!/usr/bin/env python3
from pathlib import Path
import sys, re, json
bad = [
 r'production-ready', r'no accuracy loss', r'lossless', r'vLLM replacement', r'TensorRT replacement',
 r'benchmark reproduced', r'paper results reproduced'
]
hits=[]
for p in list(Path('.').glob('README.md')) + list(Path('docs').rglob('*.md')):
    txt=p.read_text(errors='ignore')
    for b in bad:
        if re.search(b, txt, re.I): hits.append({'file':str(p),'pattern':b})
print(json.dumps({'hits':hits}, indent=2))
sys.exit(1 if hits else 0)
