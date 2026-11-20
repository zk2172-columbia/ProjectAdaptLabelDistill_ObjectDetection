"""Check CLI imports in separate processes, without data access or model downloads."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=['pipeline/run_pipeline.py','pipeline/teacher_adaptation/run_teacher_adaptation.py','pipeline/teacher_adaptation/generate_ft_config.py','pipeline/teacher_adaptation/run_finetune.py','pipeline/teacher_adaptation/merge_checkpoint.py','pipeline/pseudo_labeling/run_pseudo_labeling.py','pipeline/pseudo_labeling/run_inference.py','pipeline/pseudo_labeling/filter_and_convert.py','pipeline/pseudo_labeling/mask_low_confidence.py','pipeline/student_training/run_student_training.py','pipeline/student_training/prepare_data.py','pipeline/student_training/train.py','pipeline/evaluation/yolo/eval_coco.py','pipeline/evaluation/sam3/run_eval.py','pipeline/baselines/stac/run_stac.py','pipeline/baselines/stac/train_student.py','pipeline/baselines/stac/pseudolabels.py','pipeline/train_baseline_yolo.py','scripts/configure_dataset.py','scripts/run_ald.py','scripts/run_sam3_training.py','scripts/install_environment.py']
results=[]
for path in paths:
    p=subprocess.run([sys.executable,str(root/path),'--help'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','WANDB_MODE':'disabled','HF_HUB_OFFLINE':'1'},capture_output=True,text=True,timeout=30)
    row={'entrypoint':path,'status':'passed' if p.returncode==0 else 'failed'}
    if p.returncode:row['error']=p.stderr[-3000:]
    results.append(row)
print(json.dumps(results,indent=2))
raise SystemExit(any(x['status']!='passed' for x in results))
