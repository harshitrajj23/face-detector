"""
Tests for Dual-Chain verification (Native Merkle Ledger and EVM Smart Contract).
"""

import os
import pytest
from core.models import SocialPost
from core.blockchain import NativeMerkleLedger, EVMVerificationLedger, MerkleTree


@pytest.fixture
def sample_post():
    return SocialPost(
        platform="linkedin",
        author="Dr. Sarah Lin",
        author_handle="sarah-lin",
        post_url="https://linkedin.com/posts/sarah-lin-123",
        content_text="Decentralized biometric provenance",
    )


@pytest.fixture
def face_hash():
    return "b41ce7aff8c8c302a8ba78b3a8d3ebcfd60452827bdb5ba94c4715a18f366df6"


def test_merkle_tree():
    h1 = "a" * 64
    h2 = "b" * 64
    root = MerkleTree.compute_root([h1, h2])
    assert len(root) == 64
    assert root == MerkleTree.compute_root([h1, h2])


def test_native_merkle_ledger_record_and_verify(tmp_path, sample_post, face_hash):
    test_db = str(tmp_path / "ledger.json")
    ledger = NativeMerkleLedger(ledger_path=test_db)

    rec = ledger.record_verification(face_hash, sample_post)
    assert rec.record_id is not None
    assert rec.block_number == 1
    assert rec.block_hash.startswith("0")

    audit = ledger.verify_record(rec.record_id, face_hash, sample_post.post_hash)
    assert audit.is_valid is True
    assert audit.face_hash_matches is True
    assert audit.post_hash_matches is True

    # Chain audit check
    is_valid, _ = ledger.verify_chain_integrity()
    assert is_valid is True


def test_evm_ledger_record_and_verify(sample_post, face_hash):
    evm = EVMVerificationLedger()
    if not evm.is_available():
        pytest.skip("EVM tester not available")

    rec = evm.record_verification(face_hash, sample_post)
    assert rec.record_id is not None
    assert rec.block_number >= 0

    audit = evm.verify_record(rec.record_id, face_hash, sample_post.post_hash)
    assert audit.is_valid is True
    assert audit.face_hash_matches is True
    assert audit.post_hash_matches is True
