import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch.nn import CrossEntropyLoss
from scipy.ndimage import binary_erosion, binary_dilation
import SimpleITK as sitk
from pytorch3d.ops import add_points_features_to_volume_densities_features
from torch import nn
# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, target, predictive, ep=1e-8):
        intersection = 2 * torch.sum(predictive * target) + ep
        union = torch.sum(predictive) + torch.sum(target) + ep
        loss = 1 - intersection / union
        return loss

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_vertices: Tensor, gt_vertices: Tensor) -> Tensor:
        # assert simplex(probs)
        # assert not one_hot(dist_maps)

        pred_contour = add_points_features_to_volume_densities_features(points_3d=pred_vertices,
                                                               points_features=pred_vertices,
                                                               volume_densities=torch.zeros(2, 1, 128, 128, 128).cuda(),
                                                               volume_features=torch.zeros(2, 3, 128, 128, 128).cuda())[1].transpose(1,0)
        gt_contour = add_points_features_to_volume_densities_features(points_3d=gt_vertices.cuda(),
                                                               points_features=gt_vertices.cuda(),
                                                               volume_densities=torch.zeros(2, 1, 128, 128, 128).cuda(),
                                                               volume_features=torch.zeros(2, 3, 128, 128, 128).cuda())[1].transpose(1,0)
        pred = pred_contour.sum(dim=1).unsqueeze(0).type(torch.float32)
        gt = gt_contour.sum(dim=1).unsqueeze(0).type(torch.float32)

        multipled = einsum("bkwhd,bkwhd->bkwhd", pred, gt)

        loss = multipled.mean()

        return loss

class SurfaceLoss_dice_entropy(nn.Module):
    def __init__(self):
        super(SurfaceLoss_dice_entropy, self).__init__()
        self.crossentroy = CrossEntropyLoss()
        self.dice_loss = dice_loss()
    # probs: bcwh, dist_maps: bcwh
    def forward(self, pred_vertices, gt_vertices):
        # assert simplex(probs)
        # assert not one_hot(dist_maps)

        # pc = probs[:, self.idc, ...].type(torch.float32)
        # dc = dist_maps[:, self.idc, ...].type(torch.float32)
        # pred = pred_vertices.unsqueeze(0)
        #将分割标签提取边界函数
        # gt = gt_vertices.squeeze(0).cpu().numpy()
        # gt = sitk.GetImageFromArray(gt)
        # gt_image = sitk.LabelContour(gt)
        # gt_image = sitk.GetArrayFromImage(gt_image)
        # gt= torch.from_numpy(gt_image).to('cuda:1').unsqueeze(0).unsqueeze(0).type(torch.long)
        pred_contour = add_points_features_to_volume_densities_features(points_3d=pred_vertices,
                                                               points_features=pred_vertices,
                                                               volume_densities=torch.zeros(4, 1, 128, 144, 128).cuda(),
                                                               volume_features=torch.zeros(4, 3, 128, 144, 128).cuda(),
                                                                # grid_sizes=torch.zeros(4,3).cuda()
                                                                        )[1].transpose(1,0)
        gt_contour = add_points_features_to_volume_densities_features(points_3d=gt_vertices.cuda(),
                                                               points_features=gt_vertices.cuda(),
                                                               volume_densities=torch.zeros(4, 1, 128, 144, 128).cuda(),
                                                               volume_features=torch.zeros(4, 3, 128, 144, 128).cuda())[1].transpose(1,0)
        pred = pred_contour.sum(dim=1).unsqueeze(0).type(torch.float32)
        gt = gt_contour.sum(dim=1).unsqueeze(0).type(torch.long)
        dice = self.dice_loss(pred, gt)
        pred_background = (~(pred.bool())).type(torch.float32)
        pred_res = torch.cat([pred_background, pred], dim=1)
        gt_background = (~(gt.bool())).type(torch.long)
        gt_res = torch.cat([gt_background, gt], dim=1)
        gt_res = torch.argmax(gt_res, dim=1)
        # x = torch.argmax(pred_res, dim=1)
        cross = self.crossentroy(pred_res, gt_res)
        loss = dice + cross
        # kernel_size = 3
        # structuring_element = torch.ones((1, 1, kernel_size, kernel_size, kernel_size)).to('cuda:4')
        #
        # eroded_image = F.conv3d(dc, structuring_element, padding=1)
        # eroded_image = binary_erosion(dc.numpy, mask=True, border_value=1)
        # dc = dc-eroded_image
        # print('pc', pc)
        # print('dc', dc)
        # multipled = 0
        # for i in range(pc.shape[2]):
        #     pc_ = pc[:,:,i,:,:]
        #     dc_ = dc[:,:,i,:,:]
        #     multipled = einsum("bcwh,bcwh->bcwh", pc_, dc_).mean() + multipled
        #
        # loss = multipled

        return loss


if __name__ == "__main__":
    # data = torch.tensor([[[0, 0, 0, 0, 0, 0, 0],
    #                       [0, 1, 1, 0, 0, 0, 0],
    #                       [0, 1, 1, 0, 0, 0, 0],
    #                       [0, 0, 0, 0, 0, 0, 0]]])
    #
    # data2 = class2one_hot(data, 2)
    # # print(data2)
    # data2 = data2[0].numpy()
    # data3 = one_hot2dist(data2)  # bcwh
    #
    # # print(data3)
    # print("data3.shape:", data3.shape)
    #
    # logits = torch.tensor([[[0, 0, 0, 0, 0, 0, 0],
    #                         [0, 1, 1, 1, 1, 1, 0],
    #                         [0, 1, 1, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0]]])
    #
    # logits = class2one_hot(logits, 2)
    data3 = torch.rand(1,1,128,144,128)
    logits = torch.rand(1,1,128,144,128)
    Loss = dice_loss(data3, logits)
    # data3 = torch.tensor(data3).unsqueeze(0)

    res = Loss(logits, data3, None)
    print('loss:', res)
