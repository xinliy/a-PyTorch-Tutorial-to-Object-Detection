import torch
import json

data_folder='custom_data'
static_data_folder='custom_data/static_test'
eval_data_folder='custom_data/static_test'
keep_difficult = True  # use objects considered difficult to detect?
depth_mode=True
device = torch.device("cuda")
record_file_name='record.csv'
flag_eval_plot_result=True
with open("custom_data\\label_map.json",'r') as j:
    n_classes=len(json.load(j))