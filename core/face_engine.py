"""
Face detection and neural feature encoding engine using OpenCV YuNet and SFace.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from core.models import FaceDetectionResult, BoundingBox

# Suppress OpenCV non-critical DNN engine diagnostic messages
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


class FaceEngine:
    """
    State-of-the-art Deep Neural Network Face Engine:
    - Face Detection: YuNet (CNN based, ultra-fast and accurate)
    - Face Recognition: SFace (128-dimensional deep metric learning embeddings)
    """

    DEFAULT_YUNET_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    DEFAULT_SFACE_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")

    def __init__(
        self,
        yunet_path: Optional[str] = None,
        sface_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.yunet_path = str(base_dir / (yunet_path or self.DEFAULT_YUNET_PATH))
        self.sface_path = str(base_dir / (sface_path or self.DEFAULT_SFACE_PATH))
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(f"YuNet model not found at {self.yunet_path}")
        if not os.path.exists(self.sface_path):
            raise FileNotFoundError(f"SFace model not found at {self.sface_path}")

        # Initialize detector with default placeholder size (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            self.yunet_path,
            "",
            (320, 320),
            self.conf_threshold,
            self.nms_threshold,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")

    def process_image(
        self,
        image_path: str,
        save_crop: bool = True,
        crop_dir: str = "data/crops",
    ) -> FaceDetectionResult:
        """
        Detects the primary face in an image, extracts 128D embedding vector,
        and computes the SHA-256 fingerprint.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")

        h, w, _ = img.shape

        # YuNet requires input size matching image dimensions
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)

        if faces is None or len(faces) == 0:
            # Fallback: try with resized image if image is excessively large
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                rh, rw, _ = resized.shape
                self.detector.setInputSize((rw, rh))
                _, faces = self.detector.detect(resized)
                if faces is not None and len(faces) > 0:
                    # Rescale face coordinates back to original
                    faces[:, 0:4] /= scale
                    faces[:, 4:14] /= scale

        if faces is None or len(faces) == 0:
            raise ValueError(f"No face detected in {image_path}. Please supply an image with a visible face.")

        # Pick the face with highest confidence (faces is sorted or column 14 is confidence)
        primary_face = max(faces, key=lambda f: f[-1])
        conf = float(primary_face[-1])
        x, y, fw, fh = (
            int(primary_face[0]),
            int(primary_face[1]),
            int(primary_face[2]),
            int(primary_face[3]),
        )

        # Landmark points: 5 landmarks (x, y) = 10 values at indices 4..13
        landmarks = []
        for i in range(4, 14, 2):
            landmarks.append([float(primary_face[i]), float(primary_face[i + 1])])

        # Align & Crop face using SFace recognizer
        aligned_face = self.recognizer.alignCrop(img, primary_face)

        # Extract 128D normalized feature vector
        feature = self.recognizer.feature(aligned_face)
        embedding = feature.flatten().tolist()

        # Compute deterministic SHA-256 fingerprint
        embedding_hash = FaceDetectionResult.compute_embedding_hash(embedding)

        # Save crop thumbnail if requested
        crop_path = None
        if save_crop:
            base_dir = Path(__file__).resolve().parent.parent
            out_dir = base_dir / crop_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            crop_filename = f"{stem}_crop_{embedding_hash[:8]}.jpg"
            crop_path = str(out_dir / crop_filename)
            cv2.imwrite(crop_path, aligned_face)

        return FaceDetectionResult(
            bounding_box=BoundingBox(x=x, y=y, width=fw, height=fh),
            confidence=round(conf, 4),
            landmarks=landmarks,
            embedding=embedding,
            embedding_hash=embedding_hash,
            source_image=image_path,
            crop_path=crop_path,
        )

    def compare_embeddings(self, embedding1: list, embedding2: list) -> float:
        """
        Computes cosine similarity between two 128D embeddings.
        Returns a score in [-1.0, 1.0] where > 0.363 typically indicates the same person in SFace.
        """
        vec1 = np.array(embedding1, dtype=np.float32).reshape(1, -1)
        vec2 = np.array(embedding2, dtype=np.float32).reshape(1, -1)
        score = self.recognizer.match(vec1, vec2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)
