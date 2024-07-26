import time

import cv2
import numpy as np
from utils import create_rknn_session


class RK3588:
    def __init__(self, model_path: str, net_size: int = 640):
        self.rknn_lite = create_rknn_session(model_path)
        self.net_size = net_size
        self.classes = (
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        )

    @staticmethod
    def letterbox(
        im: np.ndarray,
        new_shape: tuple = (640, 640),
        color: tuple = (114, 114, 114),
        auto: bool = True,
        scaleup: bool = True,
        stride: int = 32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border

        return im, r, (dw, dh)

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (self.net_size, self.net_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        return img

    def inference(self, img: np.ndarray) -> list[np.ndarray] | None:
        return self.rknn_lite.inference(inputs=[img])

    def post_process(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        return np.array([]), np.array([]), np.array([])

    def draw(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = map(int, box)
            cv2.rectangle(
                img=img,
                pt1=(top, left),
                pt2=(right, bottom),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.putText(
                img=img,
                text=f"{self.classes[cl]} {score:.2f}",
                org=(top, left - 6),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 255),
                thickness=2,
            )

        return img

    def run(self, img: np.ndarray) -> np.ndarray:
        pre_img, ratio, dwdh = self.pre_process(img)
        outputs = self.inference(pre_img)
        if outputs is None:
            return img

        boxes, classes, scores = self.post_process(outputs)

        if all(x is not None for x in (boxes, classes, scores)):
            boxes -= np.array(dwdh * 2)
            boxes /= ratio
            boxes = boxes.round().astype(np.int32)

            inf_img = self.draw(img, boxes, classes, scores) # type: ignore

            return inf_img

        return img

    def main(self, source):
        cap = cv2.VideoCapture(source)

        try:
            while cap.isOpened():
                t1 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                result_img = self.run(frame)

                cv2.imshow("Inf_results", result_img)
                if cv2.waitKey(1) == ord("q"):
                    break
                print(f"FPS: {1 / (time.time() - t1):.2f}", end="\r")
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
