"""Load a trusted local SAM3 checkpoint and run one synthetic-image text prompt on CPU."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.output.resolve()
if root.exists():raise SystemExit('Choose a fresh smoke output directory')
if not args.checkpoint.is_file():raise SystemExit('Local checkpoint is required; no download is performed')
root.mkdir(parents=True);os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
import torch
from PIL import Image,ImageDraw
torch.set_num_threads(4)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline/pseudo_labeling'))
from run_inference import build_model_and_processor,infer_image
image=root/'synthetic.png';im=Image.new('RGB',(128,128),(60,70,80));ImageDraw.Draw(im).rectangle((40,40,90,90),fill=(220,190,150));im.save(image)
start=time.monotonic()
model,processor=build_model_and_processor({'sam3':{'device':'cpu','checkpoint_path':str(args.checkpoint.resolve()),'resolution':1008,'confidence_threshold':0.0}})
detections=infer_image(image,processor,[{'id':0,'name':'object','text_prompt':'object'}],'bfloat16','cpu',False,{})
if any(not 0 <= d['score'] <= 1 for d in detections):raise RuntimeError('Invalid teacher confidence')
h=hashlib.sha256()
with args.checkpoint.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
(root/'status.json').write_text(json.dumps({'status':'passed','scope':'CPU local base SAM3 load and one synthetic image/text forward; not adaptation training','parameters':sum(p.numel() for p in model.parameters()),'detections':len(detections),'elapsed_seconds':time.monotonic()-start,'checkpoint_sha256':h.hexdigest()},indent=2))
print('Teacher smoke passed')
