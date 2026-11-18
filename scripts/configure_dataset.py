"""Create a self-contained local recipe from four pre-partitioned COCO splits."""
import argparse
import json
from pathlib import Path
import re
import yaml

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split-root',type=Path,required=True,help='Contains train/label/val/eval with _annotations.coco.json and images')
    p.add_argument('--classes-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Fresh configuration directory')
    p.add_argument('--work-root',type=Path,required=True,help='Generated artifacts, separate from source data')
    p.add_argument('--teacher-checkpoint',type=Path,required=True,help='Trusted local adapted teacher checkpoint for Label; may be produced by Adapt')
    args=p.parse_args()
    split=args.split_root.resolve();classes=args.classes_file.resolve();out=args.output.resolve();work=args.work_root.resolve()
    if out.exists():p.error('Configuration output already exists; choose a fresh directory')
    if not classes.is_file():p.error('Missing class metadata')
    if work==split or work.is_relative_to(split):p.error('Work directory must be outside source splits')
    seen=set()
    for name in ('train','label','val','eval'):
        source=split/name/'_annotations.coco.json'
        if not source.is_file():p.error(f'Missing {source}')
        doc=json.loads(source.read_text());names={x['file_name'] for x in doc['images']}
        if seen & names:p.error(f'Overlapping split filenames in {name}')
        seen |= names
        if any(not (split/name/x).is_file() for x in names):p.error(f'Missing source images in {name}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',split.name):p.error('Dataset directory name must be a safe path component')
    def read(stage):return yaml.safe_load((ROOT/'pipeline'/stage/'config.yaml').read_text())
    teacher=read('teacher_adaptation');pseudo=read('pseudo_labeling');student=read('student_training')
    run=work/'{run_name}'
    teacher['dataset']=split.name
    teacher['sam3']['repo_root']=str(ROOT/'SAM3')
    teacher['sam3']['template_config']=str(ROOT/'experiment_files/sam3_train_config/pipeline_sam3_ft_config.yaml')
    teacher['overrides']['paths.bpe_path']=str(ROOT/'SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    teacher['overrides']['paths.roboflow_vl_100_root']=str(split.parent)
    teacher['overrides']['trainer.logging.wandb_writer.should_log']=False
    teacher['output'].update(experiment_dir=str(run/'teacher'),merged_checkpoint_path=str(run/'teacher/merged_checkpoint.pt'),keep_raw_checkpoint=True)
    pseudo['classes_file']=str(classes)
    pseudo['data']={'image_dir':str(split/'label'),'coco_json':str(split/'label/_annotations.coco.json')}
    pseudo['sam3']['checkpoint_path']=str(args.teacher_checkpoint.resolve())
    pseudo['output'].update(raw_predictions_path=str(run/'raw_predictions.json'),yolo_labels_dir=str(run/'label/labels'),weights_json_path=str(run/'label/confidence_weights.json'),dataset_yaml_path=str(run/'label/dataset.yaml'))
    pseudo['inference']['use_visual_prompting']=False
    student['classes_file']=str(classes)
    student['pl_data']={'dataset_yaml':str(run/'label/dataset.yaml'),'weights_json_path':str(run/'label/confidence_weights.json')}
    student['gt']={f'{name}_{key}':str(split/name/('_annotations.coco.json' if key=='coco_json' else '')) for name in ('train','val','eval') for key in ('coco_json','image_dir')}
    student['gt'].update(output_dir=str(run/'gt'),dataset_yaml=str(run/'gt/gt_dataset.yaml'))
    student['combined_dataset_yaml']=None;student['combined_dataset_yaml_output']=str(run/'combined_dataset.yaml')
    student['output']={'project':str(run/'training_runs'),'weights_dir':str(run/'weights')}
    student['evaluation'].update(dataset_yaml=str(work/'evaluation'/split.name/'dataset.yaml'),gt_json=str(work/'evaluation'/split.name/'eval/annotations.coco.json'),source_splits=str(split),export_root=str(work/'evaluation'),dataset_name=split.name)
    student['steps'].insert(0,{'name':'prepare_evaluation','script':str(ROOT/'scripts/prepare_evaluation.py'),'args':[],'enabled':True})
    for stage,cfg in [('teacher_adaptation',teacher),('pseudo_labeling',pseudo),('student_training',student)]:
        cfg['shutdown_on_completion']=False
        for step in cfg['steps']:
            if not Path(step['script']).is_absolute():step['script']=str(ROOT/'pipeline'/stage/step['script'])
        if 'wandb' in cfg:cfg['wandb']={'enabled':False}
        serialized=yaml.safe_dump(cfg,sort_keys=False)
        # Optional masking paths are not part of the default original recipe; still resolve their templates.
        serialized=serialized.replace('<dataset>',split.name)
        cfg=yaml.safe_load(serialized)
        if stage=='pseudo_labeling':
            for step in cfg['steps']:
                if step['name']=='mask_low_confidence':step['enabled']=False
        out.mkdir(parents=True,exist_ok=True)
        (out/f'{stage}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'pipeline.yaml').write_text(yaml.safe_dump({'stages':[{'name':s,'config':f'{s}.yaml'} for s in ('pseudo_labeling','student_training')],'shutdown_on_completion':False},sort_keys=False))
    print(f'Adapt: python {ROOT}/pipeline/teacher_adaptation/run_teacher_adaptation.py --config {out}/teacher_adaptation.yaml --run-name teacher-01')
    print(f'Label + Distill: python {ROOT}/pipeline/run_pipeline.py --config {out}/pipeline.yaml --run-name ald-01')
    print('These recipes preserve the default original loader; they are not proof of fresh paper reproduction.')
if __name__=='__main__':main()
