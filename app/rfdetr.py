"""
rf_detr_onnx.py

Класс-обёртка для RF-DETR в формате ONNX.
Поддерживает:
- загрузку модели;
- препроцессинг изображения;
- инференс;
- постпроцессинг детекций;
- отрисовку боксов.

Пример использования:

    from rf_detr_onnx import RFDetrONNX

    detector = RFDetrONNX(
        model_path=r"D:\...\weights.onnx",
        input_size=(640, 640),
        conf_threshold=0.5,
    )

    detections, vis = detector.predict_from_path(
        r"D:\...\05_КМ_page_0001.png"
    )
"""

from __future__ import annotations

import logging
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort

Detection = Dict[str, Any]


class RFDetrONNX:
    """
    Обёртка для RF-DETR ONNX-модели.

    Ожидаемый формат:
      - вход:  input, float32[1, 3, H, W] (BGR->RGB, нормализация ImageNet);
      - выход: dets:   float32[1, N, 4]   (cx, cy, w, h), нормированные [0,1];
                labels: float32[1, N, C]   (логиты по классам).

    Параметры
    ---------
    model_path:
        Путь к ONNX-файлу модели.
    input_size:
        Размер входного тензора (width, height).
    conf_threshold:
        Порог уверенности для фильтрации детекций.
    providers:
        Список execution providers onnxruntime
        (по умолчанию ["CPUExecutionProvider"]).
    logger:
        Экземпляр logging.Logger. Если не передан, создаётся новый
        с именем класса.
    """

    def __init__(
            self,
            model_path: str | Path,
            input_size: Tuple[int, int] = (640, 640),
            conf_threshold: float = 0.5,
            providers: Optional[List[str]] = None,
            logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.providers = providers or ["CPUExecutionProvider"]

        self.logger = logger or logging.getLogger(self.__class__.__name__)
        # Если логгер без хендлеров — добавим базовый, чтобы что-то было видно
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.session = self._create_session()

    # ------------------------------------------------------------------ #
    # Внутренние утилиты
    # ------------------------------------------------------------------ #

    def _create_session(self) -> ort.InferenceSession:
        """Создаёт onnxruntime.InferenceSession и логирует базовую информацию."""
        self.logger.info("Loading RF-DETR ONNX model from: %s", self.model_path)
        t0 = time.time()
        session = ort.InferenceSession(
            self.model_path,
            providers=self.providers,
        )
        dt = time.time() - t0
        self.logger.info("Model loaded in %.3f s", dt)

        input_names = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        self.logger.debug("Model inputs:  %s", input_names)
        self.logger.debug("Model outputs: %s", output_names)

        return session

    @staticmethod
    def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
        """
        cx,cy,w,h (нормированные) -> x1,y1,x2,y2 (та же шкала).

        Параметры
        ---------
        x:
            np.ndarray[..., 4]

        Возвращает
        ----------
        np.ndarray[..., 4]
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2.0
        y[..., 1] = x[..., 1] - x[..., 3] / 2.0
        y[..., 2] = x[..., 0] + x[..., 2] / 2.0
        y[..., 3] = x[..., 1] + x[..., 3] / 2.0
        return y

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Сигмоида."""
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------ #
    # Препроцесс / постпроцесс
    # ------------------------------------------------------------------ #

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """
        Препроцесс изображения под RF-DETR ONNX.

        Шаги:
        - resize до self.input_size;
        - BGR -> RGB;
        - нормализация в [0,1];
        - нормализация по статистике ImageNet;
        - HWC -> NCHW.

        Параметры
        ---------
        bgr:
            Исходное изображение в формате BGR (cv2.imread).

        Возвращает
        ----------
        np.ndarray:
            Тензор формы (1, 3, H, W), dtype float32.
        """
        w, h = self.input_size
        self.logger.debug("Preprocess: resize to (%d, %d)", w, h)

        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        chw = np.transpose(rgb, (2, 0, 1))
        nchw = np.expand_dims(chw, axis=0).astype(np.float32)
        return nchw

    def _postprocess(
            self,
            boxes: np.ndarray,
            logits: np.ndarray,
            image_shape: Tuple[int, int, int],
    ) -> List[Detection]:
        """
        Постпроцесс сырых выходов модели.

        Параметры
        ---------
        boxes:
            np.ndarray (N, 4) — нормированные cx,cy,w,h в [0,1].
        logits:
            np.ndarray (N, C) — логиты по классам.
        image_shape:
            (H, W, C) исходного изображения.

        Возвращает
        ----------
        List[Detection]:
            Список словарей вида:
                {
                    "class_id": int,
                    "confidence": float,
                    "bbox": (x1, y1, x2, y2),
                }
        """
        h, w = image_shape[:2]
        results: List[Detection] = []

        for b, logit_vec in zip(boxes, logits):
            scores = self._sigmoid(logit_vec)
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf < self.conf_threshold:
                continue

            box_xyxy = self._xywh2xyxy(b)
            x1 = int(box_xyxy[0] * w)
            y1 = int(box_xyxy[1] * h)
            x2 = int(box_xyxy[2] * w)
            y2 = int(box_xyxy[3] * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                {
                    "class_id": class_id,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return results

    # ------------------------------------------------------------------ #
    # Визуализация
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_color_for_class(class_id: int, color_map: Dict[int, Tuple[int, int, int]]):
        if class_id not in color_map:
            color_map[class_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        return color_map[class_id]

    def draw_detections(
            self,
            image: np.ndarray,
            detections: List[Detection],
            thickness: int = 2,
    ) -> np.ndarray:
        """
        Рисует прямоугольники и подписи поверх изображения.

        Параметры
        ---------
        image:
            Исходное изображение (BGR).
        detections:
            Список детекций из _postprocess.
        thickness:
            Толщина линий.

        Возвращает
        ----------
        np.ndarray:
            Копия изображения с нарисованными боксами.
        """
        out = image.copy()
        color_map: Dict[int, Tuple[int, int, int]] = {}

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class_id"]
            conf = det["confidence"]

            color = (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            label = f"{class_id}:{conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                out,
                (x1, y1 - th - baseline),
                (x1 + tw, y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out

    # ------------------------------------------------------------------ #
    # Публичные методы инференса
    # ------------------------------------------------------------------ #

    def predict(self, image_bgr: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Запускает инференс на уже загруженном изображении (BGR).

        Параметры
        ---------
        image_bgr:
            Изображение в формате BGR (как возвращает cv2.imread).

        Возвращает
        ----------
        detections:
            Список детекций.
        vis_image:
            Изображение с нарисованными боксами.
        """
        inp = self.preprocess(image_bgr)
        self.logger.debug("Input blob shape: %s", inp.shape)

        t0 = time.time()
        dets, labels = self.session.run(["dets", "labels"], {"input": inp})
        dt = time.time() - t0
        self.logger.info("Inference time: %.1f ms", dt * 1000)

        dets = dets[0]  # (N, 4)
        labels = labels[0]  # (N, C)
        self.logger.debug("Raw dets shape: %s, labels shape: %s", dets.shape, labels.shape)

        detections = self._postprocess(dets, labels, image_bgr.shape)
        self.logger.info(
            "Kept %d detections with conf >= %.2f",
            len(detections),
            self.conf_threshold,
        )

        vis = self.draw_detections(image_bgr, detections)
        return detections, vis

    def predict_from_path(self, image_path: str | Path) -> Tuple[List[Detection], np.ndarray]:
        """
        Запускает инференс по пути к изображению.

        Параметры
        ---------
        image_path:
            Путь к файлу изображения.

        Возвращает
        ----------
        detections:
            Список детекций.
        vis_image:
            Изображение с нарисованными боксами.
        """
        image_path = str(image_path)
        self.logger.info("Reading image from: %s", image_path)

        img = cv2.imread(image_path)
        if img is None:
            self.logger.error("Cannot read image: %s", image_path)
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        return self.predict(img)

    def predict_from_image(self, image: Image.Image):
        """Предсказание из PIL Image"""
        # Конвертируем PIL Image в формат, подходящий для модели
        import numpy as np

        # Преобразуем изображение в numpy array
        img_array = np.array(image)

        # Вызываем основной метод предсказания
        return self.predict(img_array)


__all__ = ["RFDetrONNX"]

if __name__ == "__main__":
    # Простой пример автономного запуска модуля
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    MODEL_PATH = r"/ai_version/detr_640\my-first-project-ybk4b\7\weights.onnx"
    IMAGE_PATH = r"/ai_version/data\05_КМ\05_КМ_page_0001.png"

    detector = RFDetrONNX(
        model_path=MODEL_PATH,
        input_size=(640, 640),
        conf_threshold=0.5,
    )

    dets, vis = detector.predict_from_path(IMAGE_PATH)
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d["bbox"]
        print(
            f"{i:02d}: cls={d['class_id']} "
            f"conf={d['confidence']:.3f} "
            f"box=({x1}, {y1}, {x2}, {y2})"
        )

    out_path = "rf_detr_result.png"
    cv2.imwrite(out_path, vis)
    print(f"Result saved to {out_path}")
