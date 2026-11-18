"""Pipeline adapter for the fixed COCO/YOLO evaluation export."""
import argparse
from pathlib import Path
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline/data'))
from prepare_ground_truth import main as prepare
p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text());ev=cfg['evaluation']
prepare(['--splits-root',ev['source_splits'],'--out-dir',ev['export_root'],'--dataset',ev['dataset_name'],'--classes-file',cfg['classes_file']])
