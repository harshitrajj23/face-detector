"""
Tests for FaceEngine (YuNet Face Detection & SFace Recognition).
"""

import os
import pytest
from core.face_engine import FaceEngine
from core.models import FaceDetectionResult


@pytest.fixture
def face_engine():
    return FaceEngine()


def test_face_detection_and_embedding(face_engine):
    img_path = "data/samples/sample_face_1.jpg"
    assert os.path.exists(img_path)

    res = face_engine.process_image(img_path)
    assert isinstance(res, FaceDetectionResult)
    assert res.confidence > 0.85
    assert len(res.landmarks) == 5
    assert len(res.embedding) == 128
    assert len(res.embedding_hash) == 64  # SHA-256 hex string


def test_deterministic_embedding_hash(face_engine):
    img_path = "data/samples/sample_face_1.jpg"
    res1 = face_engine.process_image(img_path)
    res2 = face_engine.process_image(img_path)
    assert res1.embedding_hash == res2.embedding_hash


def test_face_comparison(face_engine):
    res1 = face_engine.process_image("data/samples/sample_face_1.jpg")
    res2 = face_engine.process_image("data/samples/sample_face_2.jpg")

    self_sim = face_engine.compare_embeddings(res1.embedding, res1.embedding)
    diff_sim = face_engine.compare_embeddings(res1.embedding, res2.embedding)

    assert self_sim > 0.99
    assert diff_sim < 0.50
