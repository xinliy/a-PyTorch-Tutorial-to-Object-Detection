import json
import os
import glob
import numpy as np


# Read list of static test file


# Read label

print(os.getcwd())
f = open(glob.glob("custom_data/**/class.txt")[0], 'r')
label_dict = {}


label_dict['background'] = '0'
for idx, _ in enumerate(f):
    label_dict[_.replace("\n", "")] = str(idx + 1)

with open("custom_data/label_map.json","w") as j:
    json.dump(label_dict,j)

image_data=open(glob.glob("custom_data/**/**/train.txt")[0],'r')
lines=image_data.read().splitlines()
image_paths=[]
image_labels=[]
ind_list=[]
folders=os.listdir('custom_data')
color_folder=[i for i in folders if 'Color' in i][0]
color_path_in_order=sorted(os.listdir(os.path.join(os.getcwd(),'custom_data',color_folder)))
num_image=len(color_path_in_order)
test_percentage=0.3
num_test=int(num_image*test_percentage)
all_index_list=np.arange(num_image)
test_index_list=np.random.choice(all_index_list,size=num_test,replace=False)
test_image_paths=[]
test_ind_list=[]
test_image_labels=[]

for i,l in enumerate(lines):
    # print(l)
    l=l.split(" ")
    color_image_folder=os.path.dirname(l[0])
    image_dict={}
    ind=color_path_in_order.index(os.path.basename(os.path.normpath(l[0])))
    image_dict['boxes']=[list(map(int,i[:-2].split(","))) for i in l[1:]]
    image_dict['labels']=[int(i[-1])+1 for i in l[1:]]
    image_dict['difficulties']=[0]*len(l[1:])
    # For depth path
    if i not in test_index_list:
        image_paths.append(os.path.join("custom_data",l[0]))
        ind_list.append(ind)

        image_labels.append(image_dict)
    else:
        test_image_paths.append(os.path.join("custom_data",l[0]))
        test_ind_list.append(ind)
        test_image_labels.append(image_dict)


depth_image_folder=os.path.join(os.getcwd(),'custom_data',color_image_folder).replace("Color","Depth")
depth_img_path=os.listdir(depth_image_folder)
print(len(depth_img_path))
print(len(ind_list))
train_depth_img_path=[depth_img_path[i] for i in ind_list]
test_depth_img_path=[depth_img_path[i] for i in test_ind_list]
train_depth_img_path=[os.path.join('custom_data',color_image_folder.replace("Color","Depth"),i) for i in train_depth_img_path]
test_depth_img_path=[os.path.join('custom_data',color_image_folder.replace("Color","Depth"),i) for i in test_depth_img_path]
# pprint.pprint(image_labels)
with open("custom_data/TRAIN_images.json","w") as j:
    json.dump(image_paths,j)
with open("custom_data/TRAIN_objects.json",'w') as j:
    json.dump(image_labels,j)
with open("custom_data/TRAIN_depth_images.json","w") as j:
    json.dump(train_depth_img_path,j)
with open("custom_data/TEST_images.json","w") as j:
    json.dump(test_image_paths,j)
with open("custom_data/TEST_objects.json",'w') as j:
    json.dump(test_image_labels,j)
with open("custom_data/TEST_depth_images.json","w") as j:
    json.dump(test_depth_img_path,j)
