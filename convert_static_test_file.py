import json
import os
import glob

# This convert file is used only for compare duplicate impact
# Put the filtered dataset into the static_test folder


static_test_image_name_list=[]
static_test_depth_image_name_list=[]
static_test_folders=os.listdir('custom_data/static_test')
print(static_test_folders)
color_folder_name=[i for i in static_test_folders if "Color" in i][0]
depth_folder_name=color_folder_name.replace("Color","Depth")
with open("custom_data/static_test/TEST_images.json","r") as j:
    data=json.load(j)
    new_path_list=[]
    for i in data:
        new_path=os.path.join('custom_data','static_test',color_folder_name,os.path.basename(os.path.normpath(i)))
        new_path_list.append(new_path)
        static_test_image_name_list.append(os.path.basename(os.path.normpath(i)))
with open("custom_data/static_test/TEST_images.json","w") as j:
    json.dump(new_path_list,j)
with open("custom_data/static_test/TEST_Depth_images.json","r") as j:
    data=json.load(j)
    new_path_list=[]
    for i in data:
        new_path=os.path.join('custom_data','static_test',depth_folder_name,os.path.basename(os.path.normpath(i)))
        new_path_list.append(new_path)
        static_test_depth_image_name_list.append(os.path.basename(os.path.normpath(i)))
with open("custom_data/static_test/TEST_Depth_images.json","w") as j:
    json.dump(new_path_list,j)

print("length of static test",len(static_test_image_name_list))
print(os.getcwd())
# print(glob.glob("custom_data/**/class.txt"))
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
folders=os.listdir('custom_data')
color_folder=[i for i in folders if 'Color' in i][0]
color_path_in_order=sorted(os.listdir(os.path.join(os.getcwd(),'custom_data',color_folder)))
for i,l in enumerate(lines):
    # print(l)
    l=l.split(" ")
    if os.path.basename(os.path.normpath(l[0])) in static_test_image_name_list:
        continue
    color_image_folder=os.path.dirname(l[0])
    image_dict={}
    ind=color_path_in_order.index(os.path.basename(os.path.normpath(l[0])))
    image_dict['boxes']=[list(map(int,i[:-2].split(","))) for i in l[1:]]
    image_dict['labels']=[int(i[-1])+1 for i in l[1:]]
    image_dict['difficulties']=[0]*len(l[1:])
    # For depth path
    image_paths.append(os.path.join("custom_data",l[0]))
    image_labels.append(image_dict)

depth_image_folder=os.path.join(os.getcwd(),'custom_data',color_image_folder).replace("Color","Depth")
depth_img_path=os.listdir(depth_image_folder)
train_depth_img_path = [i for i in depth_img_path if i not in static_test_depth_image_name_list]
train_depth_img_path=[os.path.join('custom_data',color_image_folder.replace("Color","Depth"),i) for i in train_depth_img_path]
print(len(image_paths))
print(len(train_depth_img_path))
with open("custom_data/TRAIN_images.json","w") as j:
    json.dump(image_paths,j)
with open("custom_data/TRAIN_objects.json",'w') as j:
    json.dump(image_labels,j)
with open("custom_data/TRAIN_depth_images.json","w") as j:
    json.dump(train_depth_img_path,j)


