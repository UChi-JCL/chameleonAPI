import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import subprocess
import pandas as pd

all_configs = pd.read_csv("configs/chameleonAPI_MS.csv")
for i in range(len(all_configs)):
    app_name = all_configs.iloc[i]['App name'].lower()
    
    print(app_name )
    subprocess.run(['python', 'infer_multi_select.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][1]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][2]),
        '--checkpoint_path', f"all_checkpoints/{app_name}.ckpt",
        '--th', str(all_configs.iloc[i][3]),
        '--method', 'our',
        '--app', app_name,
        '--app_type', 'multi_select'
        ])
    subprocess.run(['python', 'infer_multi_select.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][4]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][5]),
        '--checkpoint_path', f"all_checkpoints/{app_name}_base.ckpt",
        '--th', str(all_configs.iloc[i][6]),
        '--method', 'baseline',
        '--app', app_name,
        '--app_type', 'multi_select'
        ])
    subprocess.run(['python', 'infer_multi_select.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][7]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][8]),
        '--checkpoint_path', f"all_checkpoints/pretrained.ckpt",
        '--th', str(all_configs.iloc[i][9]),
        '--method', 'pretrained',
        '--app', app_name,
        '--app_type', 'multi_select'
        ])

app_names =  all_configs['App name'].tolist()
spec_configs = json.load(open("configs/specialized_MS.json"))
for i in range(len(all_configs)):
    app_name = app_names[i]
    
    if "pretrained.ckpt" in str(spec_configs[app_name]["weights"]):
        subprocess.run(['python', 'infer_multi_select.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(spec_configs[app_name]["top"]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(spec_configs[app_name]["split"]),
        '--checkpoint_path', f"all_checkpoints/pretrained.ckpt",
        '--th',str(spec_configs[app_name]["th"]),
         '--method', 'spec',
        '--app', app_name,
        '--app_type', 'multi_select'
        ])
    else:
        subprocess.run(['python', 'infer_specialized_multi_select.py', '--input_size', '448', 
            '--wl_path', f"configs/wl_{app_name}.csv",  
            '--validation_data_file', f"data/{app_name}.csv", 
            '--top',str(spec_configs[app_name]["top"]), 
            '--model_name', 'tresnet_l', 
            '--model_path', 'Open_ImagesV6_TRresNet_L_448.pth',
            '--pic_path', 'test',
            '--split', str(spec_configs[app_name]["split"]),
            '--th', str(spec_configs[app_name]["th"]),
            '--training_wl',str(spec_configs[app_name]["wl"]),
            '--path',str(spec_configs[app_name]["weights"]),
            '--method', 'spec',
            '--app', app_name,
            '--app_type', 'multi_select'
            ])
        
        