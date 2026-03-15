
import cv2
import torch
from PIL import Image

def detect_objects(
    frame_bgr,
    processor,
    model,
    device,
    labels,
    box_threshold=0.30,
    text_threshold=0.20
):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    inputs = processor(images=image, text=[labels], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    detections = []

    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append({
            "label": str(label),
            "score": float(score),
            "box": [x1, y1, x2, y2]
        })

    return detections
