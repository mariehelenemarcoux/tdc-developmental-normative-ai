from pathlib import Path
import hashlib, sys
root=Path(__file__).resolve().parents[1]
sums=root/'manifests'/'SHA256SUMS.txt'
failed=[]; checked=0
for line in sums.read_text(encoding='utf-8').splitlines():
    if not line.strip(): continue
    expected, rel=line.split('  ',1)
    p=root/rel
    if not p.exists():
        failed.append((rel,'MISSING')); continue
    actual=hashlib.sha256(p.read_bytes()).hexdigest(); checked+=1
    if actual!=expected: failed.append((rel,actual))
print(f'checked={checked} failed={len(failed)}')
for x in failed: print(x)
sys.exit(1 if failed else 0)
