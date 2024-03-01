import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
                # ignore background
            for c in range(1, out_shape[1]):
                posmask = img_gt[b].astype(np.bool)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        compute boundary loss for binary segmentation
        input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
            gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
        output: boundary_loss; sclar
        """
        outputs_soft = F.softmax(outputs, dim=1)
        gt_sdf = self.compute_sdf1_1(gt, outputs_soft.shape)
        pc = outputs_soft[:,self.idx,...]
        dc = torch.from_numpy(gt_sdf[:,self.idx,...]).cuda()
        multipled = torch.einsum("bkxyz,bkxyz->bkxyz", pc, dc)
        #multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss




class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes
    