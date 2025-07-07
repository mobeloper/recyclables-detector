import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

# === Config ===
IMAGE_DIR = '../rc40cocodataset_splitted/test/images'
ANNOTATION_FILE = '../rc40cocodataset_splitted/test/coco_test.json'
MODEL_DIR = './'
INPUT_SIZE = (1024, 1024)
VISUALIZE = True
MAX_IMAGES = 5

# === Load model ===
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures['serving_default']

# === Load COCO annotations ===
coco = COCO(ANNOTATION_FILE)
img_id_map = {img['file_name']: img['id'] for img in coco.dataset['images']}
cat_map = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

# === Select up to 5 images ===
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))])[:MAX_IMAGES]

for img_file in tqdm(image_files):
    # Load and resize image
    img_path = os.path.join(IMAGE_DIR, img_file)
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(original, INPUT_SIZE)
    h, w = INPUT_SIZE
    input_tensor = tf.convert_to_tensor([resized], dtype=tf.uint8)

    # Run inference
    outputs = infer(input_tensor)
    boxes = outputs['detection_boxes'].numpy()[0]
    scores = outputs['detection_scores'].numpy()[0]
    classes = outputs['detection_classes'].numpy()[0].astype(int)
    masks = outputs.get('detection_masks', None)
    num_detections = int(outputs['num_detections'].numpy()[0])

    vis_img = resized.copy()

    # === Overlay predictions ===
    for i in range(num_detections):
        if scores[i] < 0.5:
            continue
        y1, x1, y2, x2 = boxes[i]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        label = cat_map.get(classes[i], f"id:{classes[i]}")
        cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Pred: {label} ({scores[i]:.2f})", pt1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if masks is not None:
            mask = cv2.resize(masks[i].numpy(), (w, h))
            binary_mask = (mask > 0.5).astype(np.uint8)
            vis_img = overlay_mask(vis_img, binary_mask, color=(0, 255, 0), alpha=0.4)

    # === Overlay ground truth boxes/masks ===
    img_id = img_id_map.get(img_file)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        x, y, w_, h_ = [int(coord * INPUT_SIZE[0] / original.shape[1]) if i % 2 == 0 else int(coord * INPUT_SIZE[1] / original.shape[0])
                        for i, coord in enumerate(bbox)]
        pt1 = (x, y)
        pt2 = (x + w_, y + h_)
        label = cat_map.get(ann['category_id'], f"id:{ann['category_id']}")
        cv2.rectangle(vis_img, pt1, pt2, (255, 0, 0), 2)
        cv2.putText(vis_img, f"GT: {label}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if 'segmentation' in ann and ann['segmentation']:
            rle = mask_utils.frPyObjects(ann['segmentation'], original.shape[0], original.shape[1])
            m = mask_utils.decode(rle).sum(axis=2)
            m = (m > 0).astype(np.uint8)
            m = cv2.resize(m, INPUT_SIZE)
            vis_img = overlay_mask(vis_img, m, color=(255, 0, 0), alpha=0.3)

    # === Display image ===
    if VISUALIZE:
        plt.figure(figsize=(12, 12))
        plt.imshow(vis_img)
        plt.title(f"Predictions (Green) vs Ground Truth (Red) — {img_file}")
        plt.axis('off')
        plt.show()
