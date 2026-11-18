"""Exercise real stage CLIs with synthetic COCO and cached synthetic teacher detections."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.output.resolve();repo=Path(__file__).resolve().parents[1]
if root.exists():raise SystemExit('Choose a fresh smoke output directory')
root.mkdir(parents=True)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['WANDB_MODE']='disabled';os.environ['OMP_NUM_THREADS']='2';os.environ['YOLO_CONFIG_DIR']=str(root/'settings')
import yaml
from PIL import Image,ImageDraw
split=root/'synthetic';classes=root/'classes.yaml'
classes.write_text(yaml.safe_dump({'classes':[{'id':0,'name':'object','text_prompt':'object'}]}))
raw=[]
for name in ('train','label','val','eval'):
    d=split/name;d.mkdir(parents=True)
    doc={'info':{},'images':[],'annotations':[],'categories':[{'id':1,'name':'object'}]}
    for i in range(4):
        f=d/f'{name}{i}.jpg';im=Image.new('RGB',(64,64),(20+20*i,30,50));ImageDraw.Draw(im).rectangle((20,20,44,44),fill=(240,190,100));im.save(f)
        doc['images'].append(dict(id=i+1,file_name=f.name,width=64,height=64))
        doc['annotations'].append(dict(id=i+1,image_id=i+1,category_id=1,bbox=[20,20,24,24],area=576,iscrowd=0))
        if name=='label':raw.append(dict(image_path=str(f),class_id=0,class_name='object',bbox_xyxy=[20,20,44,44],score=.8 if i%2 else .3,image_w=64,image_h=64))
    (d/'_annotations.coco.json').write_text(json.dumps(doc))
work=root/'work';config=root/'config'
subprocess.run([sys.executable,str(repo/'scripts/configure_dataset.py'),'--split-root',str(split),'--classes-file',str(classes),'--output',str(config),'--work-root',str(work),'--teacher-checkpoint',str(root/'not-used.pt')],check=True)
student=config/'student_training.yaml';cfg=yaml.safe_load(student.read_text())
cfg['model']['architecture']='yolo11n.yaml';cfg['training'].update(epochs=1,batch=2,imgsz=64,workers=0,device='cpu',amp=False,plots=False,optimizer='SGD',lr0=.001,pretrained=False,close_mosaic=0)
cfg['evaluation'].update(device='cpu',imgsz=64,batch=2)
student.write_text(yaml.safe_dump(cfg,sort_keys=False))
pseudo=config/'pseudo_labeling.yaml';cfg=yaml.safe_load(pseudo.read_text())
for step in cfg['steps']:
    if step['name']=='run_inference' or 'inference.py' in step['script']:step['enabled']=False
pseudo.write_text(yaml.safe_dump(cfg,sort_keys=False))
run=work/'smoke';run.mkdir(parents=True);(run/'raw_predictions.json').write_text(json.dumps(raw))
subprocess.run([sys.executable,str(repo/'pipeline/run_pipeline.py'),'--config',str(config/'pipeline.yaml'),'--run-name','smoke'],check=True,timeout=180)
metrics=run/'training_runs/gt_pl_perbox-zerogate0.5/coco_metrics.json'
if not metrics.is_file():raise SystemExit('Missing pipeline metrics')
(root/'status.json').write_text(json.dumps({'status':'passed','scope':'real Label conversion, fixed evaluation export, GT preparation, weighted Distill and COCOeval CLIs; synthetic teacher detections, no real teacher forward','metrics_path':str(metrics)},indent=2))
print('Pipeline CLI smoke passed')
