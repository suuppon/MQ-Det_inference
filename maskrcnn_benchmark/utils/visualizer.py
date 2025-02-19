import cv2
import numpy as np

class Visualizer:
    def __init__(self, entities=None):
        self.entities = entities if entities is not None else []

    def visualize_with_predictions(
        self,
        image,
        predictions,
        threshold=0.5,
        alpha=0.5,
        box_pixel=2,
        text_size=0.5,
        text_pixel=1,
        text_offset=10,
        text_offset_original=10,
        color="green",
    ):
        # OpenCV 색상 설정
        color_map = {"green": (0, 255, 0), "blue": (255, 0, 0), "red": (0, 0, 255)}
        box_color = color_map.get(color, (0, 255, 0))

        output = image.copy()

        # 원본 이미지 크기와 현재 이미지 크기 가져오기
        size = predictions.size
        orig_w, orig_h = size[0], size[1]
        new_h, new_w = image.shape[:2]

        # 박스 좌표 변환
        boxes = predictions.bbox.cpu().numpy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] / orig_w) * new_w  # x_min, x_max 변환
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] / orig_h) * new_h  # y_min, y_max 변환

        scores = predictions.get_field("scores").cpu().numpy()
        labels = predictions.get_field("labels").cpu().numpy() if "labels" in predictions.fields() else None

        info = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < threshold:
                continue

            x_min, y_min, x_max, y_max = map(int, box)

            # 좌표 클램핑
            x_min = max(0, min(x_min, new_w - 1))
            y_min = max(0, min(y_min, new_h - 1))
            x_max = max(0, min(x_max, new_w - 1))
            y_max = max(0, min(y_max, new_h - 1))

            # 바운딩 박스 그리기
            cv2.rectangle(output, (x_min, y_min), (x_max, y_max), box_color, box_pixel)

            # 클래스명 추가
            label_text = ""
            if labels is not None and self.entities and labels[i] < len(self.entities):
                label_text = self.entities[labels[i]]

            # 텍스트 위치 보정
            text = f"{label_text} {score:.2f}" if label_text else f"{score:.2f}"
            text_y = max(10, y_min - text_offset)

            # 텍스트 그리기
            cv2.putText(output, text, (x_min, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, box_color, text_pixel)

            info.append({"box": (x_min, y_min, x_max, y_max), "score": score, "label": label_text})

        # 원본과 블렌딩
        output = cv2.addWeighted(image, 1 - alpha, output, alpha, 0)

        return output, info