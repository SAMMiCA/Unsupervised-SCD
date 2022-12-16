from symbol import star_expr
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim,ssim
from models.our_models.mod import warp
import cv2
import numpy as np

SMOOTH = 1e-6


# Expect outputs and labels to have same shape (ie: torch.Size([batch:1, 224, 224])), and type long
def iou_segmentation(outputs: torch.Tensor, labels: torch.Tensor):
    # Will be zero if Truth=0 or Prediction=0
    intersection = (outputs & labels).float().sum((1, 2))
    # Will be zzero if both are 0
    union = (outputs | labels).float().sum((1, 2))

    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch

def EPE(input_flow, target_flow, sparse=False, mean=True, sum=False,valid_flow=None):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if valid_flow is not None:
        EPE_map = valid_flow*EPE_map
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/batch_size


def L1_loss(input_flow, target_flow):
    L1 = torch.abs(input_flow-target_flow)
    L1 = torch.sum(L1, 1)
    return L1


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=[1.0,1.0,1.0,1.0,1.0], size_average=True,device='cuda'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device=device
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha]).to(device)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha).to(device)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def L1_charbonnier_loss(input_flow, target_flow, sparse=False, mean=True, sum=False):

    batch_size = input_flow.size(0)
    epsilon = 0.01
    alpha = 0.4
    L1 = L1_loss(input_flow, target_flow)
    norm = torch.pow(L1 + epsilon, alpha)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        norm = norm[~mask]
    if mean:
        return norm.mean()
    elif sum:
        return norm.sum()
    else:
        return norm.sum()/batch_size


def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_flow, robust_L1_loss=False, mask=None, weights=None,
                  sparse=False, mean=False, use_flow=None):
    '''
    here the ground truth flow is given at the higest resolution and it is just interpolated
    at the different sized (without rescaling it)
    '''

    def one_scale(output, target, sparse, robust_L1_loss=False, mask=None, mean=False, use_flow=None):
        b, _, h, w = output.size()
        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))

            if mask is not None:
                mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w))
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
        else:
            target_scaled = F.interpolate(target, (h, w), mode='bilinear', align_corners=False)

            if mask is not None:
                # mask can be byte or float or uint8 or int
                # resize first in float, and then convert to byte/int to remove the borders
                # which are values between 0 and 1
                mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear', align_corners=False).byte()
                mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

        if robust_L1_loss:
            if mask is not None:
                return L1_charbonnier_loss(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
            else:
                return L1_charbonnier_loss(output, target_scaled, sparse, mean=mean, sum=False)
        else:
            if mask is not None:
                return EPE(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False,valid_flow=use_flow)
            else:
                return EPE(output, target_scaled, sparse, mean=mean, sum=False,valid_flow=use_flow)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(output, target_flow, sparse, robust_L1_loss=robust_L1_loss, mask=mask, mean=mean, use_flow=use_flow)
    return loss


def realEPE(output, target, mask_gt, ratio_x=None, ratio_y=None, sparse=False, mean=True, sum=False):
    '''
    in this case real EPE, the network output is upsampled to the size of
    the target (without scaling) because it was trained without the scaling, it should be equal to target flow
    mask_gt can be uint8 tensor or byte or int
    :param output:
    :param target: flow in range [0, w-1]
    :param sparse:
    :return:
    '''
    # mask_gt in shape bxhxw, can be torch.byte or torch.uint8 or torch.int
    b, _, h, w = target.size()
    if ratio_x is not None and ratio_y is not None:
        upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
        upsampled_output[:,0,:,:] *= ratio_x
        upsampled_output[:,1,:,:] *= ratio_y
    else:
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    # output interpolated to original size (supposed to be in the right range then)

    flow_target_x = target.permute(0, 2, 3, 1)[:, :, :, 0]
    flow_target_y = target.permute(0, 2, 3, 1)[:, :, :, 1]
    flow_est_x = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 0]  # BxH_xW_
    flow_est_y = upsampled_output.permute(0, 2, 3, 1)[:, :, :, 1]

    flow_target = \
        torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                   flow_target_y[mask_gt].unsqueeze(1)), dim=1)
    flow_est = \
        torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                   flow_est_y[mask_gt].unsqueeze(1)), dim=1)
    return EPE(flow_est, flow_target, sparse, mean=mean, sum=sum)


def multiscaleCE(network_output, target_change, mask=None, weights=None, criterion = FocalLoss()):
    '''
    here the ground truth flow is given at the higest resolution and it is just interpolated
    at the different sized (without rescaling it)
    :param network_output:
    :param target_flow:
    :param weights:
    :param sparse:
    :return:
    '''

    def one_scale(output, target, mask=None):
        b, _, h, w = output.size()

        target_scaled = F.interpolate(target.float(), (h, w), mode='nearest')

        if mask is not None:
            # mask can be byte or float or uint8 or int
            # resize first in float, and then convert to byte/int to remove the borders
            # which are values between 0 and 1
            mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='nearest', align_corners=False).byte()
            mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

        if mask is not None:
            return criterion(output * mask.float(), (target_scaled * mask.float()).long())
        else:
            return criterion(output, target_scaled.long())

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(output, target_change,mask=mask)
    return loss


def ms_ssim_loss(flows, src_img, tgt_img, cng_masks, inv_mask=False,
                 weights=None):
   
    _, _, h_orig, w_orig = src_img.size()

    def one_scale(flow, src_img, tgt_img, mask=None):
        b, _, h, w = flow.size()
        if mask is not None:
            mask = 1 - mask if inv_mask else mask
            # mask = F.softmax(mask, dim=1)[:, 0, :, :].unsqueeze(1)

        src_img = F.interpolate(src_img.float().cuda(), (h, w), mode='bilinear', align_corners=False)
        tgt_img = F.interpolate(tgt_img.float().cuda(), (h, w), mode='bilinear', align_corners=False)

        div_factor = h / h_orig
        warped_src_img, vmask = warp(src_img, flow*div_factor, disable_flow=None, get_vmask=True)

        # pad 0 to keep the shape of ssim_map same as the img
        warped_src_img = F.pad(warped_src_img, (5, 5, 5, 5), 'constant', 0)
        tgt_img = F.pad(tgt_img, (5, 5, 5, 5), 'constant', 0)
        ssim_map = ssim(warped_src_img, tgt_img, size_average=False)

        ssim_map = torch.clamp(ssim_map, min=0., max=1.)  # non-negative ssim map
        ssim_map = ssim_map.mean(dim=1, keepdim=True)  # B x 1 x h x w, channel-wise average
            
        if mask is not None:
            mask = vmask.unsqueeze(1).detach() * mask
            masked_ssim_map = mask*ssim_map
            numerator = masked_ssim_map.flatten(start_dim=1).sum(dim=-1, keepdim=True)
            if inv_mask:
                denominator = mask.flatten(start_dim=1).sum(dim=-1, keepdim=True) # weighted average 
            else:
                denominator = mask.flatten(start_dim=1).sum(dim=-1, keepdim=True) # average

            score = numerator / (denominator)  # average
            return score.mean()
        else:
            ssim_map = vmask.detach() * ssim_map
            score = ssim_map.flatten(start_dim=1).mean(dim=-1, keepdim=True)
            return score.mean()

    if type(flows) not in [tuple, list]:
        flows = [flows]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(flows))

    loss = 0
    for flow, change_mask, weight in zip(flows, cng_masks, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        # flow = flow.detach()
        ssim_score = one_scale(flow, src_img, tgt_img, mask=change_mask)
        ssim_score = ssim_score if inv_mask else 1-ssim_score
        loss += weight * ssim_score
    return loss


def ms_smooth_reg(img, outs, edge_aware=False, weights=None, edge_weight=1.0):
    def gradient(x):
        d_dy = x[:, :, 1:] - x[:, :, :-1]
        d_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return d_dx, d_dy

    def one_scale(img, out, edge_aware):
        b, _, h, w = out.size()
        out_dx, out_dy = gradient(out)
        
        if edge_aware:
            img = F.interpolate(img.float().cuda(), (h, w), mode='bilinear', align_corners=False)
            img_dx, img_dy = gradient(img)
            w_x = torch.exp(-torch.mean(edge_weight*torch.abs(img_dx), 1, keepdim=True))
            w_y = torch.exp(-torch.mean(edge_weight*torch.abs(img_dy), 1, keepdim=True))
        else:
            w_x = 1.0
            w_y = 1.0
        
        smooth_x = torch.abs(out_dx) * w_x
        smooth_y = torch.abs(out_dy) * w_y    
        smooth_loss = torch.mean(smooth_x) + torch.mean(smooth_y)
        return smooth_loss

    if type(outs) not in [tuple, list]:
        outs = [outs]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(outs))

    loss = 0
    for out, weight in zip(outs, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(img, out, edge_aware)
    return loss


def ms_rgb_loss(flows, src_img, tgt_img, change_masks, 
                  inv_mask=False, loss_type='robust', weights=None):
   
    _, _, h_orig, w_orig = src_img.size()

    def one_scale(flow, src_img, tgt_img, mask=None):
        b, _, h, w = flow.size()

        src_img = F.interpolate(src_img.float().cuda(), (h, w), mode='bilinear', align_corners=False)
        tgt_img = F.interpolate(tgt_img.float().cuda(), (h, w), mode='bilinear', align_corners=False)

        div_factor = h / h_orig
        warped_src_img, vmask = warp(src_img, flow*div_factor, disable_flow=None, get_vmask=True)

        diff = warped_src_img - tgt_img
        if loss_type == 'robust':
            diff = (diff.abs() + 0.01).pow(0.4)
        elif loss_type == 'charbonnier':
            diff = (diff**2 + 1e-6).pow(0.4)
        
        vmask = vmask.unsqueeze(1).repeat(1, diff.size(1), 1, 1).detach()

        if mask is not None:
            mask = 1 - mask if inv_mask else mask
            # mask = F.softmax(mask, dim=1)[:, 0, :, :].unsqueeze(1)
            mask = mask * vmask
            
            numerator = (mask*diff).flatten(1).sum(dim=-1, keepdim=True)
            denominator = mask.flatten(1).sum(dim=-1, keepdim=True) # weighted average 
            score = numerator / denominator  # average
            
            return score.mean()
        else:
            return diff.mean()
            

    if type(flows) not in [tuple, list]:
        flows = [flows]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(flows))

    loss = 0
    for flow, change_mask, weight in zip(flows, change_masks, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        photo_loss = one_scale(flow, src_img, tgt_img, mask=change_mask)
        photo_loss = -photo_loss if inv_mask else photo_loss
        loss += weight * photo_loss
    return loss


def ms_photo_loss(flows, src_img, tgt_img, cng_masks, 
                   inv_mask=False, weights=None, 
                   photometric='robust', wavg=False):
    _, _, h_orig, w_orig = src_img.size()

    def one_scale(flow, src_img, tgt_img, mask=None):
        
        flow_orig = F.interpolate(flow, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
        warped_src_img, vmask = warp(src_img, flow_orig, disable_flow=None, get_vmask=True)
        
        if photometric == 'robust':
            diff = ((warped_src_img - tgt_img).abs() + 0.01).pow(0.4)
        elif photometric == 'charbonnier':
            diff = ((warped_src_img - tgt_img)**2 + 1e-6).pow(0.4)
        elif photometric == 'ssim':
            warped_src_img = F.pad(warped_src_img, (5, 5, 5, 5), 'constant', 0)
            tgt_img = F.pad(tgt_img, (5, 5, 5, 5), 'constant', 0)
            ssim_map = ssim(warped_src_img, tgt_img, size_average=False)  # B x 3 x h x w
            ssim_map = torch.clamp(ssim_map, min=0., max=1.)  # non-negative ssim map
            diff = 1 - ssim_map.mean(dim=1, keepdim=True)  # B x 1 x h x w, channel-wise average    
        elif photometric == 'census':
            warped_src_trf, tgt_trf = _ternary_transform(warped_src_img), _ternary_transform(tgt_img)  
            diff = _hamming_distance(warped_src_trf, tgt_trf)
            diff = (diff.abs() + 0.01).pow(0.4) 
        
        vmask = vmask.unsqueeze(1)

        if mask is not None:
            mask = (1-mask)*vmask if inv_mask else mask*vmask
            # mask = F.softmax(mask, dim=1)[:, 0, :, :].unsqueeze(1)
            return mask_average(diff, mask, wavg=wavg)  
        else:
            return mask_average(diff, vmask, wavg=wavg)
            
    if type(flows) not in [tuple, list]:
        flows = [flows]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(flows))

    loss = 0
    for flow, cng_mask, weight in zip(flows, cng_masks, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        photo_loss = one_scale(flow, src_img, tgt_img, mask=cng_mask)
        photo_loss = -photo_loss if inv_mask else photo_loss
        loss += weight * photo_loss
    return loss

def _ternary_transform(image):
    max_distance = 3
    patch_size = 2*max_distance + 1
    R, G, B = torch.split(image, 1, 1)
    intensities_torch = (0.2989 * R + 0.5870 * G + 0.1140 * B)  # * 255  # convert to gray
    out_channels = patch_size * patch_size
    w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
    w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
    weight = torch.from_numpy(w_).float()
    if image.is_cuda:
        weight = weight.cuda()
    patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1], 
                                    padding=[max_distance, max_distance])
    transf = patches_torch - intensities_torch
    transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
    return transf_norm

def _hamming_distance(t1, t2):
    dist = (t1 - t2) ** 2
    dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
    return dist

def mask_average(x, m, wavg=False):
    # x: b x c x h x w, m: b x 1 x h x w
    mx = m*x
    if wavg:
        wavg_mx = mx.flatten(1).sum(-1, keepdim=True) / (m.flatten(1).sum(-1, keepdim=True) + 1e-6)
        return wavg_mx.mean()
    else:
        return mx.mean()
    
def spatial_centering(x, vmask=None, bias=0.0):
    
    B, _, H, W = x.size()
    x_c = x.flatten(1).mean(-1) if vmask is None else (x*vmask).flatten(1).sum(-1) / vmask.flatten(1).sum(-1)
    x_std = x.flatten(1).std(-1) if vmask is None else (x*vmask).flatten(1).std(-1)
    x_c = x_c.view(-1, 1, 1, 1).repeat(1, 1, H, W)
    return x - x_c + bias

def calc_diff_map(flow, src_img, tgt_img, photometric='ssim'):
    _, _, h_orig, w_orig = src_img.size()
    _, _, h_f, w_f = flow.size()
    flow_orig = F.interpolate(flow, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
    # src_img = F.interpolate(src_img, size=(h_f, w_f), mode='bilinear')
    # tgt_img = F.interpolate(tgt_img, size=(h_f, w_f), mode='bilinear')
    warped_src_img, vmask = warp(src_img, flow_orig, disable_flow=None, get_vmask=True)
    
    if photometric == 'robust':
        diff = ((warped_src_img - tgt_img).abs() + 0.01).pow(0.4).mean(dim=1, keepdim=True)
    elif photometric == 'charbonnier':
        diff = ((warped_src_img - tgt_img)**2 + 1e-6).pow(0.4).mean(dim=1, keepdim=True)
    elif photometric == 'ssim':
        warped_src_img = F.pad(warped_src_img, (5, 5, 5, 5), 'constant', 0)
        tgt_img = F.pad(tgt_img, (5, 5, 5, 5), 'constant', 0)
        ssim_map = ssim(warped_src_img, tgt_img, size_average=False)  # B x 3 x h x w
        ssim_map = torch.clamp(ssim_map, min=0., max=1.)  # non-negative ssim map
        diff = 1 - ssim_map.mean(dim=1, keepdim=True)  # B x 1 x h x w, channel-wise average    
    elif photometric == 'census':
        warped_src_trf, tgt_trf = _ternary_transform(warped_src_img), _ternary_transform(tgt_img)  
        diff = _hamming_distance(warped_src_trf, tgt_trf)
        diff = (diff.abs() + 0.01).pow(0.4).mean(dim=1, keepdim=True)
        
    return diff, vmask.unsqueeze(1)