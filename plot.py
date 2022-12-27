import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_bbox(img, target):
    """
    Plots an image and its bounding boxes.

    Args:
        img (PIL.Image or torch.Tensor): Image to plot.
        target (dict): Dictionary containing the bounding boxes.
    """
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(3, 3)

    # Convert tensor to PIL image
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img).convert('RGB')

    # Plot image
    a.imshow(img)

    # Get bounding boxes
    boxes = target['boxes']

    # Convert tensor to list
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()

    # Plot bounding boxes
    for box in (boxes):
        x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')
        a.add_patch(rect)

    return fig
