"""CPU-only one-epoch smoke using generated images, no pretrained weights or network."""
import argparse
import json
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=args.output.absolute()
if root.exists():raise SystemExit(f'Refusing to reuse smoke output: {root}')
root.mkdir(parents=True)
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WANDB_MODE']='disabled'
os.environ['YOLO_CONFIG_DIR']=str(root/'settings')
os.environ['OMP_NUM_THREADS']='2'
import torch
import yaml
from PIL import Image,ImageDraw
from ultralytics import settings
settings.update({'wandb':False,'mlflow':False,'clearml':False,'comet':False})
torch.set_num_threads(2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline/student_training'))
from trainer import Phase2DetectionTrainer,Phase2Spec
from train import run_experiment
metadata={};annotations=[];images=[]
for split in ['gt','pl','val']:
    for sub in ['images','labels']:(root/split/sub).mkdir(parents=True)
    for i in range(4):
        im=Image.new('RGB',(64,64),(30+i*20,30,30));ImageDraw.Draw(im).rectangle((20,20,44,44),fill=(230,180,100))
        path=root/split/'images'/f'{split}{i}.jpg';im.save(path)
        (root/split/'labels'/f'{split}{i}.txt').write_text('0 0.5 0.5 0.375 0.375\n')
        if split=='pl':metadata[str(path)]=[dict(class_id=0,cx=.5,cy=.5,w=.375,h=.375,w_sam3=.8 if i%2 else .2)]
        if split=='val':
            images.append(dict(id=i+1,file_name=path.name,width=64,height=64))
            annotations.append(dict(id=i+1,image_id=i+1,category_id=1,bbox=[20,20,24,24],area=576,iscrowd=0))
weights=root/'confidence.json';weights.write_text(json.dumps(metadata))
dataset=root/'data.yaml';dataset.write_text(yaml.safe_dump(dict(path=str(root),train=[str(root/'gt/images'),str(root/'pl/images')],val=str(root/'val/images'),test=str(root/'val/images'),names={0:'object'})))
gt=root/'gt.json';gt.write_text(json.dumps(dict(info={},images=images,annotations=annotations,categories=[dict(id=1,name='object')])) )
run_experiment('smoke',dict(model='yolo11n.yaml',pretrained=False,data=str(dataset),epochs=1,batch=2,imgsz=64,
    device='cpu',workers=0,amp=False,plots=False,save=True,project=str(root/'runs'),name='smoke',exist_ok=False,
    optimizer='SGD',lr0=.001,deterministic=True,seed=42,mosaic=1.,mixup=0.,cutmix=0.,copy_paste=0.,close_mosaic=0,
    weights_json_path=str(weights),pseudo_label_paths=[str(root/'pl/images')],lambda1=.5,zerogate=.5,
    spec=Phase2Spec(per_box=True,per_image=False,loc_weight_src='box',cls_weight_src='box')),
    weights_dir=root/'promoted',wandb_cfg={'enabled':False},
    eval_cfg={'enabled':True,'gt_json':str(gt),'dataset_yaml':str(dataset),'split':'test','imgsz':64,'batch':2,'device':'cpu'})
metrics=root/'runs/smoke/coco_metrics.json'
if not metrics.is_file():raise SystemExit('Smoke failed: no held-out metrics')
(root/'smoke_status.json').write_text(json.dumps({'status':'passed','scope':'one epoch synthetic CPU train, portable checkpoint export, held-out scoring','metrics':json.loads(metrics.read_text())},indent=2))
print('CPU smoke passed:',root)
