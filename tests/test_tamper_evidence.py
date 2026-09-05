"""
Tests proving tamper-evidence and immediate cryptographic rejection of altered data.
"""

from core.models import SocialPost
from core.blockchain import NativeMerkleLedger, EVMVerificationLedger


def test_tamper_evidence_post_modification(tmp_path):
    test_db = str(tmp_path / "tamper_ledger.json")
    ledger = NativeMerkleLedger(ledger_path=test_db)

    original_post = SocialPost(
        platform="x",
        author="Alex Chen",
        post_url="https://x.com/alexchen/status/100",
        content_text="Authentic original post anchored to blockchain.",
    )
    face_hash = "5e6fee70b9ba8cb5bebe86b5edf03e3cbdfc5c414515c446a330b0e1b36eb9d4"

    rec = ledger.record_verification(face_hash, original_post)

    # 1. Verify authentic record succeeds
    valid_audit = ledger.verify_record(rec.record_id, face_hash, original_post.post_hash)
    assert valid_audit.is_valid is True

    # 2. Simulate forged/tampered post with altered text
    tampered_post = SocialPost(
        platform="x",
        author="Alex Chen",
        post_url="https://x.com/alexchen/status/100",
        content_text="Tampered fraudulent text trying to pass verification.",
    )
    tampered_audit = ledger.verify_record(rec.record_id, face_hash, tampered_post.post_hash)

    assert tampered_audit.is_valid is False
    assert tampered_audit.face_hash_matches is True
    assert tampered_audit.post_hash_matches is False


def test_tamper_evidence_face_modification(tmp_path):
    test_db = str(tmp_path / "tamper_ledger_face.json")
    ledger = NativeMerkleLedger(ledger_path=test_db)

    post = SocialPost(
        platform="linkedin",
        author="Dr. Sarah Lin",
        post_url="https://linkedin.com/posts/1",
        content_text="Genuine verification.",
    )
    face_hash = "b41ce7aff8c8c302a8ba78b3a8d3ebcfd60452827bdb5ba94c4715a18f366df6"
    tampered_face_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    rec = ledger.record_verification(face_hash, post)

    audit = ledger.verify_record(rec.record_id, tampered_face_hash, post.post_hash)
    assert audit.is_valid is False
    assert audit.face_hash_matches is False
    assert audit.post_hash_matches is True
