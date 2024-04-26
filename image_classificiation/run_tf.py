import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import subprocess
import pandas as pd
# Remove a directory 
all_configs = pd.read_csv("configs/chameleonAPI_TF.csv")
for i in range(len(all_configs)):
    app_name = all_configs.iloc[i]['App name'].lower()
    
    print(app_name, str(all_configs.iloc[i][1]),str(all_configs.iloc[i][2]) )
    subprocess.run(['python', 'pred.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][1]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'all_checkpoints/Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][2]),
        '--checkpoint_path', f"all_checkpoints/{app_name}.ckpt",
        '--th', str(all_configs.iloc[i][3]),
        '--method', 'our',
        '--app', app_name,
        '--app_type', 'TF'
        ])
    subprocess.run(['python', 'pred.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][4]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'all_checkpoints/Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][5]),
        '--checkpoint_path', f"all_checkpoints/{app_name}_base.ckpt",
        '--th', str(all_configs.iloc[i][6]),
        '--method', 'baseline',
        '--app', app_name,
        '--app_type', 'TF'
        ])
    print(str(all_configs.iloc[i][9]))
    subprocess.run(['python', 'pred.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][7]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'all_checkpoints/Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][8]),
        '--checkpoint_path', f"all_checkpoints/pretrained.ckpt",
        '--th', str(all_configs.iloc[i][9]),
        '--method', 'pretrained',
        '--app', app_name,
        '--app_type', 'TF'
        ])

app_names =  all_configs['App name'].tolist()
spec_configs = json.load(open("configs/specialized_TF.json"))
for i in range(len(all_configs)):
    app_name = app_names[i]
    
    if "pretrained.ckpt" in str(spec_configs[app_name]["weights"]):
        subprocess.run(['python', 'pred.py', '--input_size', '448', 
        '--wl_path', f"configs/wl_{app_name}.csv",  
        '--validation_data_file', f"data/{app_name}.csv", 
        '--top', str(all_configs.iloc[i][7]), 
        '--model_name', 'tresnet_l', 
        '--model_path', 'all_checkpoints/Open_ImagesV6_TRresNet_L_448.pth',
        '--pic_path', 'test',
        '--split', str(all_configs.iloc[i][8]),
        '--checkpoint_path', f"all_checkpoints/pretrained.ckpt",
        '--th', str(all_configs.iloc[i][9]),
         '--method', 'spec',
        '--app', app_name,
        '--app_type', 'TF'
        ])
    else:
        subprocess.run(['python', 'infer_specialized.py', '--input_size', '448', 
            '--wl_path', f"configs/wl_{app_name}.csv",  
            '--validation_data_file', f"data/{app_name}.csv", 
            '--top',str(spec_configs[app_name]["top"]), 
            '--model_name', 'tresnet_l', 
            '--model_path', 'all_checkpoints/Open_ImagesV6_TRresNet_L_448.pth',
            '--pic_path', 'test',
            '--split', str(spec_configs[app_name]["split"]),
            '--th', str(spec_configs[app_name]["th"]),
            '--training_wl',str(spec_configs[app_name]["wl"]),
            '--path',str(spec_configs[app_name]["weights"]),
            '--method', 'spec',
            '--app', app_name,
            '--app_type', 'TF'
            ])
        
        

for index in ["pretrained", "baseline", "our", "spec"]:
    df = pd.read_csv(f"results/results_TF_{index}.csv")
    print(index, df["precision"].mean())        