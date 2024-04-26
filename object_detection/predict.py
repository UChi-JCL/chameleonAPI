import numpy as np
import cv2
import torch
from dataset.coco_utils import get_coco_label_names


def select_top_predictions(predictions, threshold):
    idx = torch.where(predictions['scores'] > threshold)[0]
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    '''
    Simple function that adds fixed colors depending on the class
    '''
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype('uint8')
    return colors


def overlay_boxes(image, predictions, mapping_dict):
    '''
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    '''
    labels = predictions['labels']
    boxes = predictions['boxes']
    colors = compute_colors_for_labels(labels).tolist()
    i = -1
    for box, color in zip(boxes, colors):
        i += 1
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        if labels[i].item() not in mapping_dict['wl1']: continue
        
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 3,
        )

    return image


def overlay_class_names(image, predictions, categories, mapping_dict):
    '''
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    '''
    scores = predictions['scores'].tolist()
    labels = predictions['labels'].tolist()
    raw_labels = predictions['labels'].tolist()
    labels = [categories[i] if i in categories else "Other" for i in labels ]
    boxes = predictions['boxes']

    template = '{}: {:.2f}'
    i = -1
    for box, score, label in zip(boxes, scores, labels):
        i += 1
        x, y = box[:2]
        s = str(label)
        # if raw_labels[i] not in mapping_dict['wl1']: continue
        cv2.putText(image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,)
        
    return image


def predict(img, model, device, mapping_dict):
    model = model.eval()

    result = img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3 x 416 x 416
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img)

    with torch.no_grad():
        output = model([img.to(device)])
    top_predictions = select_top_predictions(output[0], 0.3)
    top_predictions = {k: v.cpu() for k, v in top_predictions.items()}

    result = overlay_boxes(result, top_predictions, mapping_dict)
    coco_label_names, coco_class_ids, coco_cls_colors, coco_index_to_name_mapping = get_coco_label_names()

    result = overlay_class_names(result, top_predictions, coco_index_to_name_mapping, mapping_dict)
    return result, output, top_predictions
