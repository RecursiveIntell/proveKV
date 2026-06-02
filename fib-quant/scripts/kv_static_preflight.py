#!/usr/bin/env python3
from pathlib import Path
import sys, json
root = Path.cwd()
errors=[]; warnings=[]
required = ['Cargo.toml','README.md','LICENSE','CITATION.cff','src/lib.rs']
for r in required:
    if not (root/r).exists(): errors.append(f'missing {r}')
if not (root/'src/kv').exists(): warnings.append('src/kv does not exist yet; expected before final')
# public root clutter warnings
for p in ['01_CODEX_MASTER_PROMPT.md','OPERATOR_PASTE_FIRST.md','overlays','.agents','.codex']:
    if (root/p).exists(): warnings.append(f'root workbench artifact remains: {p}')
# current z.py may exist internally, but cargo package must exclude it
if (root/'z.py').exists(): warnings.append('z.py exists in repo context; ensure cargo package excludes it')
out={'errors':errors,'warnings':warnings}
print(json.dumps(out, indent=2))
sys.exit(1 if errors else 0)
