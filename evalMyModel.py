from datasets import ImageDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from myModel import *


data_folder='custom_data'
batch_size=1
workers=4
device=torch.device("cuda")
checkpoint='./mymodel'

model=Net()
model.load_state_dict(torch.load('mymodel'))
model=model.to(device)
model.eval()
test_dataset = ImageDataset(data_folder,
                            split='test',
                            keep_difficult=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


if __name__=="__main__":
    with torch.no_grad():
        for i,(images,boxes,labels,d,_) in enumerate(tqdm(test_loader,desc='Evaluating')):
            img=images[0].permute(1,2,0).numpy()
            images=images.to(device)

            locs,scores=model(images)
            box,label,score=model.detect_objects(locs,scores,min_match_score=0.01,max_overlap_toleration=0.45,top_k=5)
            top_n=len([i for i in score[0] if i>0.1])
            print(label)
            print(score)
            fig,ax=plt.subplots(1)
            ax.imshow(img)
            for bounding_box in box[0][:top_n]:
                x=300*bounding_box[0]
                y=300*bounding_box[1]
                width=300*(bounding_box[2]-bounding_box[0])
                height=300*(bounding_box[3]-bounding_box[1])
                rect=patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',fill=False)
                ax.add_patch(rect)
            plt.show()
