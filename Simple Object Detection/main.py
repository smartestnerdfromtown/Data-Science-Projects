import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pathlib
import torch

MODEL = YOLO(model="yolov8n.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL.to(device=DEVICE)

def read_image(image_path: str | pathlib.Path):
    image = cv2.imread(filename=image_path)
    image_rgb = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    return image_rgb

def predict(image_rgb, model: YOLO, confidence_threshold: float):
    results = model(image_rgb, conf=confidence_threshold, device=DEVICE)[0]

    return results

def visualize(img_rgb, to_be_annotated_img_rgb,  results, colors):
    boxes = results.boxes
    class_names = results.names

    class_labels = {}
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        color = colors[class_id % len(colors)].tolist()
        confidence = float(box.conf[0])

        cv2.putText(
            to_be_annotated_img_rgb,
            f"{class_name} {confidence:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        cv2.rectangle(
            img=to_be_annotated_img_rgb, 
            pt1=(x1, y1), 
            pt2=(x2, y2), 
            color=color, 
            thickness=2
        )

        class_labels[class_name] = color
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Detected Objects')
    plt.imshow(to_be_annotated_img_rgb)
    plt.axis('off')

    legend_handles = []
    for class_name, color in class_labels.items():
        normalized_color = np.array(color) / 255.0 
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                           markerfacecolor=normalized_color, markersize=10))

    plt.legend(handles=legend_handles, loc='upper right', title='Classes')

    plt.tight_layout()
    plt.show()
    
    return to_be_annotated_img_rgb


img_rgb = read_image(image_path="guy_and_cats.jpg")
to_be_annotated_img_rgb = img_rgb.copy()
results = predict(image_rgb=img_rgb, model=MODEL, confidence_threshold=0.35)

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(results.names), 3))

visualize(
    img_rgb=img_rgb,
    to_be_annotated_img_rgb=to_be_annotated_img_rgb,
    results=results,
    colors=colors
)
