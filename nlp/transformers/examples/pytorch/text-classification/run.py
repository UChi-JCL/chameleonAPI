import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import subprocess
import pandas as pd

all_configs = json.load(open("config.json"))

for app_name in all_configs.keys():
    # Pretrained model inference
    # subprocess.run(["python", "pred.py", 
    #             "--app", app_name, 
    #             "--th", str(all_configs[app_name]["th3"]),
    #             "--method", "pretrained",
    #             "--app_type", "tf"
    #             ]  )
    
    # if "specialized_model" in all_configs[app_name]:
    #     subprocess.run(["python", "pred_specialized.py", 
    #                 "--app", app_name, 
    #                 "--th", str(all_configs[app_name]["th4"]),
    #                 "--path_to_model", all_configs[app_name]["specialized_model"],
    #                 "--num_label", all_configs[app_name]["specialized_labels"],
    #                 "--spec_class", all_configs[app_name]["specialized_class"],
    #                 "--method", "spec",
    #                 "--app_type", "tf"
    #                 ]  )
    # else:
    #     subprocess.run(["python", "pred_specialized.py", 
    #                 "--app", app_name, 
    #                 "--th", str(all_configs[app_name]["th3"]),
    #                  "--method", "spec",
    #                  "--app_type", "tf"
    #                 ]  )
    # subprocess.run(["python", "pred.py", 
    #             "--app", app_name, 
    #             "--th", str(all_configs[app_name]["th2"]),
    #             "--path_to_model", f"checkpoints/{app_name}_baseline.pt",
    #              "--method", "baseline",
    #              "--app_type", "tf"]  )
   
    
    subprocess.run(["python", "pred.py", 
                "--app", app_name, 
                "--th", str(all_configs[app_name]["th1"]),
                "--path_to_model", f"checkpoints/{app_name}.pt",
                "--method", "our",
                "--app_type", "tf"]  )

