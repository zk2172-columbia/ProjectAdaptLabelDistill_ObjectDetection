"""Fetch a public immutable SAM3 baseline and apply the reviewable ALD training patch."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='f6e51f59500a87c576c2df2323ce56b9fd7a12de'
URL='https://github.com/facebookresearch/sam3.git'
target=ROOT/'SAM3'
patch=ROOT/'patches/sam3-image-training.patch'
if target.exists():
    raise SystemExit(f'Refusing to modify existing checkout: {target}. Move it aside before a fresh setup.')
subprocess.run(['git','init',str(target)],check=True)
subprocess.run(['git','-C',str(target),'fetch','--depth=1',URL,BASE],check=True)
subprocess.run(['git','-C',str(target),'checkout','--detach','FETCH_HEAD'],check=True)
actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
if actual!=BASE:raise SystemExit('SAM3 revision mismatch')
subprocess.run(['git','-C',str(target),'apply','--check',str(patch)],check=True)
subprocess.run(['git','-C',str(target),'apply',str(patch)],check=True)
(target/'.ald-source.json').write_text(json.dumps({'upstream':URL,'base_sha':BASE,'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest()},indent=2))
print('SAM3 source prepared; install with: python -m pip install -e ./SAM3')
