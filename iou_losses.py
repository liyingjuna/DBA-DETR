# coding=utf-8

# Copyright 2023 Zhi Cai. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# HungarianMatcher
# Copyright 2022 The IDEA Authors. All rights reserved.
# ------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from detrex.layers import box_cxcywh_to_xyxy, generalized_box_iou,box_iou
from torchvision.ops.boxes import box_area
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes (int): The number of boxes.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
def binary_cross_entropy_loss_with_logits(inputs, pos_weights, neg_weights, avg_factor):
    p = inputs.sigmoid()
    loss = -pos_weights * p.log() - neg_weights * (1-p).log() 
    return loss.sum()/avg_factor



def get_local_rank( quality, indices):
    #quality: one-dimension tensor 
    #indices: matching result
    bs = len(indices)
    device = quality.device
    tgt_size = [len(tgt_ind) for _,tgt_ind in indices]
    ind_start = 0
    rank_list = []
    for i in range(bs):
        if  tgt_size[i] == 0:
            rank_list.append(torch.zeros(0,dtype=torch.long,device=device))
            continue     
        num_tgt = max(indices[i][1]) + 1
        # split quality of one item
        quality_per_img = quality[ind_start:ind_start+tgt_size[i]]
        ind_start += tgt_size[i]
        #suppose candidate bag sizes are equal        
        k = torch.div(tgt_size[i], num_tgt,rounding_mode='floor')
        #sort quality in each candidate bag
        quality_per_img = quality_per_img.reshape(num_tgt, k)
        ind = quality_per_img.sort(dim=-1,descending=True)[1]
        #scatter ranks, eg:[0.3,0.6,0.5] -> [2,0,1]
        rank_per_img = torch.zeros_like(quality_per_img, dtype=torch.long, device = device)
        rank_per_img.scatter_(-1, ind, torch.arange(k,device=device, dtype=torch.long).repeat(num_tgt,1))
        rank_list.append(rank_per_img.flatten())

    return torch.cat(rank_list, 0)

def IOU_IA_BCE_loss(src_logits,pos_idx_c, src_boxes, src_ious,target_boxes, indices, avg_factor, alpha,gamma, w_prime=1,):
    prob = src_logits.sigmoid()
    #init positive weights and negative weights
    pos_weights = torch.zeros_like(src_logits)
    neg_weights =  prob ** gamma
    #ious_scores between matched boxes and GT boxes
    # b_iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy( target_boxes))[0]
    # src_ious_logit = src_ious.sigmoid().flatten()
    # iou_scores = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy( target_boxes))[0])
    iou_scores = src_ious.sigmoid().flatten()

    #t is the quality metric
    t = prob[pos_idx_c]**alpha * iou_scores ** (1-alpha)
    t = torch.clamp(t, 0.01).detach()
    rank = get_local_rank(t, indices)
    #define positive weights for SoftBceLoss  
    if type(w_prime) != int:
        rank_weight = w_prime[rank]
    else:
        rank_weight = w_prime
    
    t = t * rank_weight
    pos_weights[pos_idx_c] = t 
    neg_weights[pos_idx_c] = (1 -t)    
    
    loss = -pos_weights * prob.log() - neg_weights * (1-prob).log()
    # print(loss)
    return loss.sum()/avg_factor, rank_weight

def loss_ious(src_ious, src_boxes, target_boxes,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """

    target_iou = box_iou_pairwise(box_cxcywh_to_xyxy(src_boxes).clamp(min=0, max=1),
                                          box_cxcywh_to_xyxy(target_boxes))[0].detach()
    src_logits_iou = src_ious.sigmoid().flatten()

    loss_iou = l2_loss(src_logits_iou, target_iou)

    # losses = {}
    # losses['loss_iou'] = loss_iou

    return loss_iou

def l2_loss(input, target):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond ** 2 / pos_inds.shape[0]
    else:
        loss = input * 0.0
    return loss.sum()

def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union