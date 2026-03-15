
import cv2

def check_helmet_per_person(detections, annotated):
    persons = []
    helmets = []

    for det in detections:
        label = det["label"].lower()
        x1, y1, x2, y2 = det["box"]

        if "person" in label or "worker" in label:
            persons.append((x1, y1, x2, y2))

        if "helmet" in label or "hard hat" in label:
            helmets.append((x1, y1, x2, y2))

    alerts = 0

    for (px1, py1, px2, py2) in persons:
        helmet_found = False

        head_region_y2 = py1 + int((py2 - py1) * 0.35)

        for (hx1, hy1, hx2, hy2) in helmets:
            helmet_center_x = int((hx1 + hx2) / 2)
            helmet_center_y = int((hy1 + hy2) / 2)

            if (
                px1 <= helmet_center_x <= px2 and
                py1 <= helmet_center_y <= head_region_y2
            ):
                helmet_found = True
                break

        if not helmet_found:
            alerts += 1
            cv2.putText(
                annotated,
                "ALERTA: PERSONA SIN CASCO",
                (px1, max(py1 - 20, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

    return annotated, alerts
