import torch


def iou_component(preds, targets, num_class, ignore_idx=None):
    """
    Calcualate intersection and union(IoU) value for segmentation.

        Args:
                preds (torch.Tensor): model prediction tensor(B, H, W). Each pixels in [0, num_class-1]
                targets (torch.Tensor): Ground truth mask tensor(B, H, W). Each pixels in [0, num_class-1]
                num_class (int): Number of valid classes (Except background)
                ignore_class: Ignore class when calculating IoU
    """
    if ignore_idx is not None:
        valid_mask = targets != ignore_idx
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    intersection_sum = torch.zeros(num_class, dtype=torch.long, device=preds.device)
    union_sum = torch.zeros(num_class, dtype=torch.long, device=preds.device)

    preds, targets = preds.flatten(), targets.flatten()

    for class_id in range(num_class):
        is_pred = preds == class_id
        is_target = targets == class_id

        intersection = is_pred & is_target
        union = is_pred | is_target

        intersection_sum[class_id] = intersection.sum()
        union_sum[class_id] = union.sum()

    return intersection_sum, union_sum


def iou_calculation(intersection_sum, union_sum, eps=1e-6):
    """
    Calculate IoU for each class and get mIoU.

    Args:
            intersection_sum(torch.Tensor): Intersection value for each class. Type torch.long
            union_sum(torch.Tensor): Union vlaue for each class.Type torch.long
    """
    intersection_f = intersection_sum.float()
    union_f = union_sum.float()

    iou_per_class = torch.zeros_like(intersection_f)
    valid_mask = union_f > 0
    iou_per_class = intersection_f[valid_mask] / (union_f[valid_mask] + eps)

    if valid_mask == 0:
        miou = torch.tensor(0.0, dtype=torch.long, device=intersection_f.device)
    else:
        miou = iou_per_class.mean()

    return miou.item(), iou_per_class.item()
