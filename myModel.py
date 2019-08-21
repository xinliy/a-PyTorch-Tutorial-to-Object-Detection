import torch
from torch import nn
import math
import torch.nn.functional as F
from datasets import ImageDataset
import torch.utils.data
import torchvision
from model import MultiBoxLoss
import json

device=torch.device("cuda")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def find_intersection(set1,set2):
    set1=set1.to(device)
    set2=set2.to(device)
    lower_bounds=torch.max(set1[...,:2].unsqueeze(1),set2[...,:2].unsqueeze(0))
    upper_bounds=torch.min(set1[...,2:].unsqueeze(1),set2[...,2:].unsqueeze(0))
    intersection_length=torch.clamp(upper_bounds-lower_bounds,min=0)
    return intersection_length[...,0]*intersection_length[...,1]

def cxcy_to_gcxgcy(cxcy,priors_cxcy):
    cxcy=cxcy.to(device)
    priors_cxcy=priors_cxcy.to(device)
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h
def xy_to_cxcy(xy):
    return torch.cat([(xy[...,2:]+xy[...,:2])/2,
                      xy[...,2:]-xy[...,:2]],1)

def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

def find_IOU(set1,set2):
    set1=set1.to(device)
    set2=set2.to(device)
    intersection=find_intersection(set1,set2)

    areas_set1=(set1[...,2]-set1[...,0])*(set1[...,3]-set1[...,1])
    areas_set2=(set2[...,2]-set2[...,0])*(set2[...,3]-set2[...,1])

    union=areas_set1.unsqueeze(1)+areas_set2.unsqueeze(0)-intersection
    return intersection/union

def gcxgcy_to_cxcy(gcxgcy,priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def gcxgcy_to_cxcy2(gcxgcy_prediction,cxcy_prior):
    gcxgcy_prediction=gcxgcy_prediction.to(device)
    cxcy_prior=cxcy_prior.to(device)
    n_boxes=cxcy_prior.shape[0]
    g_cx=gcxgcy_prediction[...,0]
    g_cy=gcxgcy_prediction[...,1]
    g_w=gcxgcy_prediction[...,2]
    g_h=gcxgcy_prediction[...,3]

    cx=cxcy_prior[...,0]
    cy=cxcy_prior[...,1]
    w=cxcy_prior[...,2]
    h=cxcy_prior[...,3]

    result=torch.zeros((n_boxes,4))
    result[...,0]=g_cx*w/10+cx
    result[...,1]=g_cy*h/10+cy
    result[...,2]=torch.exp(g_w/5)*w
    result[...,3]=torch.exp(g_h/5)*h

    return result

def cxcy_to_xy2(cxcy):
    cx=cxcy[...,0]
    cy=cxcy[...,1]
    w=cxcy[...,2]
    h=cxcy[...,3]

    xmin=cx-w/2
    xmax=cx+w/2
    ymin=cy-h/2
    ymax=cy+h/2
    result=torch.empty((cxcy.shape[0],4))
    result[...,0]=xmin
    result[...,1]=ymin
    result[...,2]=xmax
    result[...,3]=ymax
    return result

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.n_boxes=5
        self.n_classes=4
        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1)

        self.conv1_2=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3_3=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.conv4_1=nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_3=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv5_1=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_2=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_3=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv6=nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)
        self.conv7=nn.Conv2d(1024,1024,kernel_size=1)

        self.load_pretrained_layers()
        self.loc_conv=nn.Conv2d(in_channels=512,out_channels=4*self.n_boxes,kernel_size=3,padding=1)
        self.conf_conv=nn.Conv2d(in_channels=512,out_channels=self.n_classes*self.n_boxes,kernel_size=3,padding=1)
        self.priors_cxcy=self.create_prior()

    def load_pretrained_layers(self):
        state_dict=self.state_dict()
        param_names=list(state_dict.keys())

        pretrained_state_dict=torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names=list(pretrained_state_dict.keys())


        for i,param in enumerate(param_names[:-4]):
            state_dict[param]=pretrained_state_dict[pretrained_param_names[i]]
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        self.load_state_dict(state_dict)
    def forward(self,x):
        x=F.relu(self.conv1_1(x))
        x=F.relu(self.conv1_2(x))
        x=self.pool1(x)
        x=F.relu(self.conv2_1(x))
        x=F.relu(self.conv2_2(x))
        x=self.pool2(x)
        x=F.relu(self.conv3_1(x))
        x=F.relu(self.conv3_2(x))
        x=F.relu(self.conv3_3(x))
        x=self.pool3(x)
        x=F.relu(self.conv4_1(x))
        x=F.relu(self.conv4_2(x))
        x=F.relu(self.conv4_3(x))

        location=self.loc_conv(x).permute(0,2,3,1).contiguous()
        location=location.view(x.shape[0],-1,4)
        confidence=self.conf_conv(x).permute(0,2,3,1).contiguous()
        confidence=confidence.view(x.shape[0],-1,self.n_classes)
        # print("x shape:",x.shape)
        # print("location shape:",location.shape)
        # print("confidence shape",confidence.shape)
        return location,confidence

    def create_prior(self):
        width=height=38
        prior_boxes=[]
        aspect_ratio=[1.,2.]
        scale=0.1
        for i in range(38):
            for j in range(38):
                for ratio in aspect_ratio:
                    cx=(0.5+i)/width
                    cy=(0.5+j)/height
                    w=scale*math.sqrt(ratio)
                    h=scale/math.sqrt(ratio)
                    prior_boxes.append([cx,cy,w,h])
        return torch.FloatTensor(prior_boxes)


    def detect_objects(self,predicted_locs,predicted_scores,top_k,min_match_score,max_overlap_toleration):
        """

        :param predicted_locs: [n,n_boxes,4]
        :param predicted_scores: [n,n_boxes,n_classes]
        :param top_k:
        :param min_match_score:
        :param max_overlap_toleration:
        :return:
        """

        batch_size=predicted_locs.shape[0]
        predicted_scores=F.softmax(predicted_scores,dim=2)

        batch_image_boxes=[]
        batch_image_labels=[]
        batch_image_scores=[]

        for i in range(batch_size): # for each image in the batch
            image_predicted_locs=predicted_locs[i]
            image_predicted_scores=predicted_scores[i]
            # Convert all predicted gxgy to xmin ymin
            image_decoded_locs=cxcy_to_xy2(gcxgcy_to_cxcy2(image_predicted_locs,self.priors_cxcy))

            image_boxes=list()
            image_labels=list()
            image_scores=list()

            # for each box get the max score with the label
            max_score,best_label=image_predicted_scores.max(dim=1)

            # check for each class
            for c in range(1,self.n_classes):
                image_class_scores=image_predicted_scores[...,c] # (n_boxes)
                index_above_min_score=image_class_scores>min_match_score
                n_above_min_score=index_above_min_score.sum().item()
                if n_above_min_score==0:
                    continue
                qualified_class_scores=image_class_scores[index_above_min_score] # (n_qualified)
                qualified_decoded_locs=image_decoded_locs[index_above_min_score] # (n_qualified,4)

                sorted_class_scores,sort_index=qualified_class_scores.sort(dim=0,descending=True)
                sorted_decoded_locs=qualified_decoded_locs[sort_index]

                overlap=find_IOU(sorted_decoded_locs,sorted_decoded_locs).to(device)
                suppress=torch.zeros((n_above_min_score),dtype=torch.uint8).to(device)


                for box in range(sorted_decoded_locs.shape[0]):
                    if suppress[box]==1:
                        continue
                    suppress=torch.max(suppress,overlap[box]>max_overlap_toleration).to(device)

                    suppress[box]=0
                image_boxes.append(qualified_decoded_locs[1-suppress])
                image_labels.append(torch.LongTensor((1-suppress).sum().item()*[c]))
                image_scores.append(qualified_class_scores[1-suppress])

            if len(image_boxes)==0:
                image_boxes.append(torch.FloatTensor([[0.,0.,1.,1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes=torch.cat(image_boxes,dim=0)
            image_labels=torch.cat(image_labels,dim=0)
            image_scores=torch.cat(image_scores,dim=0)
            n_objects=image_scores.shape[0]

            if n_objects>top_k:
                image_scores,sort_ind=image_scores.sort(dim=0,descending=True)
                image_scores=image_scores[:top_k]
                image_boxes=image_boxes[sort_ind][:top_k]
                image_labels=image_labels[sort_ind][:top_k]


            batch_image_boxes.append(image_boxes)
            batch_image_scores.append(image_scores)
            batch_image_labels.append(image_labels)
        return batch_image_boxes,batch_image_labels,batch_image_scores

# class MultiBoxLoss(nn.Module):
#     def __init__(self,priors_cxcy,threshold=0.5,neg_pos_ratio=3,alpha=1.):
#         super(MultiBoxLoss,self).__init__()
#         self.priors_cxcy=priors_cxcy
#         self.priors_xy=cxcy_to_xy2(priors_cxcy)
#         self.threshold=threshold
#         self.neg_pos_ratio=neg_pos_ratio
#         self.alpha=alpha
#         self.smooth_l1=nn.L1Loss()
#         self.cross_entropy=nn.CrossEntropyLoss(reduce=False)
#
#     def forward(self,predicted_locs,predicted_scores,boxes,labels):
#         batch_size=predicted_locs.shape[0]
#         n_priors=self.priors_xy.shape[0]
#         n_classes=predicted_scores.size(2)
#
#         true_locs=torch.zeros((batch_size,n_priors,4),dtype=torch.float).to(device)
#         true_classes=torch.zeros((batch_size,n_priors),dtype=torch.long).to(device)
#
#         for i in range(batch_size):
#             num_true_objects=boxes[i].shape[0]
#
#             overlap=find_IOU(boxes[i],self.priors_xy)
#
#             overlap_each_prior,object_each_prior=overlap.max(dim=0)
#
#             _,prior_each_object=overlap.max(dim=1)
#
#             object_each_prior[prior_each_object]=torch.LongTensor(range(num_true_objects)).to(device)
#             overlap_each_prior[prior_each_object]=1.
#
#             label_each_prior=labels[i][object_each_prior]
#             label_each_prior[overlap_each_prior<self.threshold]=0
#
#             true_classes[i]=label_each_prior
#             true_locs[i]=cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_each_prior]),self.priors_cxcy)
#         positive_priors=true_classes!=0
#
#         loc_loss=self.smooth_l1(predicted_locs[positive_priors],true_locs[positive_priors])
#
#         n_positives=positive_priors.sum(dim=1)
#         n_hard_negatives=self.neg_pos_ratio*n_positives
#
#         conf_loss_all=self.cross_entropy(predicted_scores.view(-1,n_classes),true_classes.view(-1))
#         conf_loss_all=conf_loss_all.view(batch_size,n_priors)
#
#         conf_loss_pos=conf_loss_all[positive_priors]
#         conf_loss_neg=conf_loss_all.clone()
#         conf_loss_neg[positive_priors]=0.
#         conf_loss_neg,_=conf_loss_neg.sort(dim=1,descending=True)
#         hardness_ranks=torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
#         hard_negatives=hardness_ranks<n_hard_negatives.unsqueeze(1)
#         conf_loss_hard_neg=conf_loss_neg[hard_negatives]
#
#         conf_loss=(conf_loss_hard_neg.sum()+conf_loss_pos.sum())/n_positives.sum().float()
#         print("conf_loss:",conf_loss)
#         print("loc_loss:",loc_loss)
#
#         return conf_loss+self.alpha*loc_loss

def train(train_loader,model,criterion,optimizer):
    model.train()

    for i,(images,boxes,labels,_) in enumerate(train_loader):
        images=images.to(device)
        boxes=[b.to(device) for b in boxes]
        labels=[l.to(device) for l in labels]
        # print(images.shape)

        prediceted_locs,predicted_scores=model(images)

        loss=criterion(prediceted_locs,predicted_scores,boxes,labels)
        print("loss",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Done")
        del prediceted_locs,predicted_scores,images,boxes,labels

def validate(val_loader,model,criterion):
    model.eval()
    losses=AverageMeter()
    with torch.no_grad():
        for i,(images,boxes,labels,d) in enumerate(val_loader):
            images=images.to(device)
            boxes=[b.to(device) for b in boxes]
            labels=[l.to(device) for l in labels]

            predicted_locs,predicted_scores=model(images)
            loss=criterion(predicted_locs,predicted_scores,boxes,labels)
            losses.update(loss.item(),images.shape[0])

    return losses.avg


if __name__=="__main__":
    data_folder='custom_data'
    n_classes=3
    epoch_since_improvement=0
    best_loss=100
    model=Net()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    model=model.to(device)
    train_dataset=ImageDataset(data_folder, split='train', keep_difficult=False)
    val_dataset=ImageDataset(data_folder, split='test', keep_difficult=False)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32,collate_fn=train_dataset.collate_fn,num_workers=4)
    val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=True,collate_fn=val_dataset.collate_fn,num_workers=4,pin_memory=True)
    criterion=MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    for epoch in range(50):
        train(train_loader,model,criterion=criterion,optimizer=optimizer)
        val_loss=validate(val_loader,model,criterion)
        print("val_loss:{},epoch:{}".format(val_loss,epoch))
        is_best=val_loss<100
        best_loss=min(val_loss,best_loss)
        if not is_best:
            epoch_since_improvement+=1
        else:
            epoch_since_improvement=0

        torch.save(model.state_dict(),'mymodel')

        # print("done",epoch)