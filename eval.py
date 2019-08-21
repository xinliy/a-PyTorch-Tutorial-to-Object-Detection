from utils import *
from datasets import ImageDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import torch
from model_params import record_file_name,keep_difficult,eval_data_folder,flag_eval_plot_result,device,depth_mode

# Load record csv
if os.path.isfile(record_file_name):
    df=pd.read_csv(record_file_name)
else:
    df=pd.DataFrame()
record_dict={}


# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = eval_data_folder
# keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size=1 if flag_eval_plot_result is True else 16
workers = 4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# depth_mode=True
if depth_mode is True:
    record_dict['depth']=1
    checkpoint = './BEST_RGBD_checkpoint_ssd300.pth.tar'
else:
    record_dict['depth']=0
    checkpoint = './BEST_checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = ImageDataset(data_folder,
                            split='test',
                            keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
record_dict['num_test']=len(test_dataset)

def evaluate(test_loader, model,df):
    print(df)
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, depth_images,boxes, labels, difficulties,origin_images) in enumerate(tqdm(test_loader, desc='Evaluating')):

            for t in difficulties:
                t[t==1]=0
            # img=images[0].permute(1,2,0)
            img=origin_images[0]

            images = images.to(device)  # (N, 3, 300, 300)
            if depth_mode is True:
                depth_images=depth_images.to(device)
                images=torch.cat((images,depth_images),dim=1)


            # Forward prop.
            predicted_locs, predicted_scores = model(images)


            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.35,
                                                                                       top_k=10)
            print(labels)
            print(det_labels_batch)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos
            top_n=len([i for i in det_scores_batch[0] if i >0.5])
            print(det_scores_batch[0])
            print(det_labels_batch[0])

            if batch_size==1:
                fig,ax=plt.subplots(1)
                ax.imshow(img)
                for bounding_box in det_boxes_batch[0][:top_n]:


                    x=300*bounding_box[0]
                    y=300*bounding_box[1]
                    width=300*(bounding_box[2]-bounding_box[0])
                    height=300*(bounding_box[3]-bounding_box[1])
                    rect=patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',fill=False)
                    ax.add_patch(rect)

                # for true_box in boxes[0]:
                #     x=300*true_box[0]
                #     y=300*true_box[1]
                #     width=300*(true_box[2]-true_box[0])
                #     height=300*(true_box[3]-true_box[1])
                #     rect=patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='b',fill=False)
                #     ax.add_patch(rect)

                plt.show()
            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    record_dict.update(APs)
    record_dict['mAP']=mAP
    for key in record_dict.keys():
        if key not in df.columns:
            df[key]=""
    for column_name in df.columns:
        if column_name not in record_dict.keys():
            record_dict[column_name]=-1
    df=df.append(record_dict,ignore_index=True)
    df.to_csv(record_file_name,index=False)


if __name__ == '__main__':
    evaluate(test_loader, model,df)
