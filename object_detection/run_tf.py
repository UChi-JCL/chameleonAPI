import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import subprocess
import pandas as pd
import os
os.environ['WL_EVAL'] = "False"
os.environ['NEW'] = "False"
os.environ['CC'] = "False"


all_configs = json.load(open("configs/config.json"))
for i in all_configs.keys():
    app_name = i
    os.environ['WL_EVAL'] = "False"
    subprocess.run(['python', 'test.py', 
                    '--b', "4", "--model", "fasterrcnn_mobilenet_v3_large_fpn",
                    '--split', 'test', 
                    '--extra_input', f'new_data/split_{app_name}_val.csv',
                    '--wl_path', f"configs/wl_mapping_{app_name}.csv",
                    '--our_loss', 'False', 
                    '--pretrained', 
                    '--data-path', 'coco',
                    '--wl_path_test', f"configs/wl_mapping_{app_name}.csv",
                    '--th', str(all_configs[i]['th3']),
                    '--app', app_name,
                    '--method', 'pretrained',
        ])
    subprocess.run(['python', 'test.py', 
                    '--b', "4", "--model", "fasterrcnn_mobilenet_v3_large_fpn",
                    '--split', 'test', 
                    '--extra_input', f'new_data/split_{app_name}_val.csv',
                    '--wl_path', f"configs/wl_mapping_{app_name}.csv",
                    '--our_loss', 'False', 
                    '--pretrained', 
                    '--resume', f"checkpoints/{app_name}_base.pth",
                    '--data-path', 'coco',
                    '--wl_path_test', f"configs/wl_mapping_{app_name}.csv",
                    '--th', str(all_configs[i]['th2']),
                    '--app', app_name,
                    '--method', 'baseline',
        ])
    subprocess.run(['python', 'test.py', 
                    '--b', "4", "--model", "fasterrcnn_mobilenet_v3_large_fpn",
                    '--split', 'test', 
                    '--extra_input', f'new_data/split_{app_name}_val.csv',
                    '--wl_path', f"configs/wl_mapping_{app_name}.csv",
                    '--our_loss', 'False', 
                    '--pretrained', 
                    '--resume', f"checkpoints/{app_name}.pth",
                    '--data-path', 'coco',
                    '--wl_path_test', f"configs/wl_mapping_{app_name}.csv",
                    '--th', str(all_configs[i]['th1']),
                    '--app', app_name,
                    '--method', 'our',
        ])
    if "pretrained" in all_configs[i]['specialized_path']:
        os.environ['WL_EVAL'] = "False"
        subprocess.run(['python', 'test.py', 
                    '--b', "4", "--model", "fasterrcnn_mobilenet_v3_large_fpn",
                    '--split', 'test', 
                    '--extra_input', f'new_data/split_{app_name}_val.csv',
                    '--wl_path', f"configs/wl_mapping_{app_name}.csv",
                    '--our_loss', 'False', 
                    '--pretrained', 
                    '--data-path', 'coco',
                    '--wl_path_test', f"configs/wl_mapping_{app_name}.csv",
                    '--th', str(all_configs[i]['th3']),
                    '--app', app_name,
                    '--method', 'spec',
        ])
    else:
        os.environ['WL_EVAL'] = "True"
        subprocess.run(['python', 'test.py', 
                    '--b', "4", "--model", "fasterrcnn_mobilenet_v3_large_fpn",
                    '--split', 'test', 
                    '--extra_input', f'new_data/split_{app_name}_val.csv',
                    '--wl_path', f"configs/wl_mapping_{all_configs[i]['specialized_class']}.csv",
                    '--our_loss', 'False', 
                    '--pretrained', 
                    '--data-path', 'coco',
                    '--resume', all_configs[i]['specialized_path'],
                    '--wl_path_test', f"configs/wl_mapping_{app_name}.csv",
                    '--th', str(all_configs[i]['th4']),
                    '--app', app_name,
                    '--method', 'spec',
        ])
for file in ['our', 'pretrained','baseline',]:
    p = pd.read_csv(f"results/results_{file}.csv")
    print(p['precision'].mean())