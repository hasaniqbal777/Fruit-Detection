import os

import cv2
import torch
import numpy as np
from xml.etree import ElementTree as et


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading
    images and annotations from a directory.
    Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    # Arguments
        files_dir: Directory containing images and annotations.
        width: Width of the images.
        height: Height of the images.
        transforms: Image transformations.

    # Returns
        img_res: Resized image.
        target: Dictionary containing bounding boxes, labels, areas, iscrowd, and image_id.
    """

    def __init__(self, files_dir: str, width: int, height: int, transforms=None):
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # Image transformations
        self.transforms = transforms

        # Load all image filenames
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                     if image[-4:] == '.jpg']

        # Load all classes
        self.classes = ['_', 'apple', 'banana', 'orange']

    def __getitem__(self, idx):
        """
        Returns a single image and its corresponding annotations.

        # Arguments
            idx: Index of the image.

        # Returns
            img: a PIL Image of size (H, W).
            target: a dict containing the following fields:
                boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
                labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
                image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
                area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
                iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
                (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
                (optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. 
        """

        # Create image path
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # Read image and resize using OpenCV
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img,
                               cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb,
                             (self.width, self.height),
                             cv2.INTER_AREA)

        # Normalize image
        img_res /= 255.0

        # Create annotation path
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)

        # Read annotation file
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        wt = img.shape[1]
        ht = img.shape[0]

        # Loop through all objects in the annotation file
        for member in root.findall('object'):
            # Add label
            labels.append(self.classes.index(member.find('name').text))

            # Add bounding box coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            # Normalize bounding box coordinates
            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height

            # Add bounding box coordinates to list
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # Boxes tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Compute areas
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # iscrowd tensor (default value is 0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # Labels tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        # Apply transformations
        if self.transforms:
            sample = self.transforms(image=img_res,
                                     bboxes=target['boxes'],
                                     labels=labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        # Return image and target
        return img_res, target

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.imgs)


def detect_extension(filename: str) -> str:
    """Detects the extension of a file."""
    try:
        extension = os.path.splitext(filename)[-1]
        if extension == '':
            raise ValueError
        return extension

    except ValueError as err:
        raise ValueError("No extension found for file %s", filename) from err
