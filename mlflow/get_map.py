import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from utils.general import box_iou, xywh2xyxy
from utils.metrics import ap_per_class
import mlflow
from mlflow import log_metric, log_artifacts


def get_tensor_from_txt(
        file_path: str, 
        image_height: int, 
        image_width: int, 
        inference: bool):
    """
    Reads bounding box data from a text file and returns a PyTorch tensor.

    Args:
        file_path (str): The path to the text file containing the bounding box data.
        image_height (int): The height of the image.
        image_width (int): The width of the image.
        inference (bool): A flag indicating whether inference is being performed. 
            If True, the text file is expected to contain an additional score value for each bounding box.

    Returns:
        torch.Tensor: A PyTorch tensor containing the bounding box data. 
            Each row in the tensor represents a bounding box and contains the following values:
            - For inference: class_id, bbox_x, bbox_y, bbox_width, bbox_height, score
            - For training: class_id, bbox_x, bbox_y, bbox_width, bbox_height
            The tensor has a dtype of torch.float32.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        current_data = []
        for line in lines:
            if inference:
                class_id, x, y, width, height, score = line.strip().split()
            else:
                class_id, x, y, width, height = line.strip().split()

            bbox_width = int(float(width) * image_width)
            bbox_height = int(float(height) * image_height)
            bbox_x = int((float(x) * image_width) - (bbox_width / 2))
            bbox_y = int((float(y) * image_height) - (bbox_height / 2))

            if inference:
                current_data.append([int(class_id), bbox_x, bbox_y, bbox_width, bbox_height, float(score)])
            else:
                current_data.append([int(class_id), bbox_x, bbox_y, bbox_width, bbox_height])
        
        return torch.tensor(current_data, dtype=torch.float32)

def test(
    classes_name: str,
    images_mapping: Dict,
    ground_truth_labels: str,
    predictions_labels: str,
    image_height: int,
    image_width: int,
    device: str = "cpu"
):
    """
    Perform testing and evaluation on predicted bounding box labels and load the metrics to mlflow

    Args:
        classes_name (str): A string containing the names of the classes.
        images_mapping (Dict): A dictionary mapping ground truth file names to predicted file names.
        ground_truth_labels (str): The path to the directory containing ground truth labels.
        predictions_labels (str): The path to the directory containing predicted labels.
        image_height (int): The height of the input images.
        image_width (int): The width of the input images.
        device (str, optional): The device on which to perform the testing. Defaults to "cpu".

    Returns:
        None

    Prints the evaluation results including Precision, Recall, and mAP scores for each class,
    as well as overall statistics for all classes.
    """

    ground_truth_labels = Path(ground_truth_labels)
    predictions_labels = Path(predictions_labels)

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    names = {k: v for k, v in enumerate(classes_name)}
    nc = len(names)
    seen = 0
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map= 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []

    for ground_truth_file, pred_file in images_mapping.items():

        gt_file_path = list(ground_truth_labels.glob(ground_truth_file[:-4] + ".txt"))[0]
        labels = get_tensor_from_txt(gt_file_path, image_height, image_width, inference=False)

        pred_file_path = list(predictions_labels.glob(pred_file[:-4] + ".txt"))[0]
        pred = get_tensor_from_txt(pred_file_path, image_height, image_width, inference=True)

        nl = pred.shape[0]
        seen += 1
        tcls = labels[:, 0].tolist() if nl else []  # target class

        predn = pred.clone()

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]
            pcls_tensor = pred[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            predn = xywh2xyxy(predn[:, 1:5])

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        # stats.append((correct.cpu(), torch.ones(pred.shape[0], device=device), pred[:, 0].cpu(), tcls))
        stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 0].cpu(), tcls))
    
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, v5_metric=False, save_dir="plot", names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    log_metric("all-Precision", mp)
    log_metric("all-Recall", mr)
    log_metric("all-mAP_.5", map50)
    log_metric("all-mAP_.5_.95", map)
    log_artifacts("plot", artifact_path="plots")

    # Print results per class
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        current_class = names[c]
        if current_class == "m&m":
            current_class = "mm"
        log_metric(f"{current_class}-Precision", p[i])
        log_metric(f"{current_class}-Recall", r[i],)
        log_metric(f"{current_class}-mAP_.5", ap50[i])
        log_metric(f"{current_class}-mAP_.5_.95", ap[i])

    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get metrics for object detection')
    parser.add_argument('--images_mapping', type=str, default='images_mapping.json', nargs='?', help='Path to the folder containing the original images')
    parser.add_argument('--ground_truth_labels', type=str, default='test/labels', nargs='?', help='Folder of to the ground truth labels')
    parser.add_argument('--predictions_labels', type=str, default='labels_yolov7tiny_fp32', nargs='?', help='Folder of the predictions labels')
    parser.add_argument('--image_height', type=int, default=1080, nargs='?', help='image height')
    parser.add_argument('--image_width', type=int, default=1920, nargs='?', help='image width')

    args = parser.parse_args()

    with open(args.images_mapping, 'r') as file:
        images_mapping = json.load(file)

    expr_name = "object_detection"  

    s3_bucket = f"s3://elvis-s3-mlflow/mlruns/{expr_name}"  

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.get_experiment_by_name(expr_name)
    if not experiment:
        mlflow.create_experiment(expr_name, s3_bucket)

    mlflow.set_experiment(expr_name)
    mlflow.start_run('39d3c926757c423d8fc1eb55405bd021')

    print(mlflow.get_artifact_uri())
    # mlflow.set_tag("mlflow.runName", args.predictions_labels[7:])
    test(
        classes_name=['m&m', 'doritos', 'coca-cola', 'pringles', 'person', 'hand'],
        images_mapping=images_mapping,
        ground_truth_labels=args.ground_truth_labels,
        predictions_labels=args.predictions_labels,
        image_height=args.image_height,
        image_width=args.image_width
    )
    mlflow.end_run()
