import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import subprocess
import pandas as pd

all_configs = json.load(open("config_kick.json"))

for app_name in all_configs.keys():
    # Pretrained model inference
    subprocess.run(["python", "pred.py", 
                "--app", app_name, 
                "--th", str(all_configs[app_name]["th3"]),
                "--method", "kick",
                "--app_type", "tf"
                ]  )
    
    
