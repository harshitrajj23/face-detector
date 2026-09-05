"""
End-to-End Face Identification and Blockchain Verification Pipeline.
Orchestrates:
1. Face Scan & 128D Neural Feature Extraction
2. Web / Social Media Reverse Search
3. Blockchain Ledger Anchoring (Native Merkle / EVM Smart Contract)
4. Cryptographic On-Chain Re-Verification
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel
from core.models import FaceDetectionResult, SocialPost, BlockchainRecord, VerificationAuditResult
from core.face_engine import FaceEngine
from core.search_engine import SocialMediaSearchEngine
from core.blockchain import NativeMerkleLedger, EVMVerificationLedger


class PipelineOutput(BaseModel):
    face_result: FaceDetectionResult
    post: SocialPost
    blockchain_record: BlockchainRecord
    audit_result: VerificationAuditResult
    execution_summary: Dict[str, Any]


class VerificationPipeline:
    """
    Complete, end-to-end pipeline linking biometric facial identity with
    discovered social media posts through an immutable blockchain ledger.
    """

    def __init__(
        self,
        blockchain_type: str = "native",  # 'native' or 'evm'
        serpapi_key: Optional[str] = None,
        evm_rpc_url: Optional[str] = None,
        evm_private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
        allow_offline: bool = False,
    ):
        self.blockchain_type = blockchain_type.lower()
        self.allow_offline = allow_offline
        self.face_engine = FaceEngine()
        self.search_engine = SocialMediaSearchEngine(
            serpapi_key=serpapi_key,
            face_engine=self.face_engine,
            allow_offline=allow_offline,
        )

        if self.blockchain_type == "evm":
            self.ledger = EVMVerificationLedger(
                rpc_url=evm_rpc_url,
                private_key=evm_private_key,
                contract_address=contract_address,
            )
            if not self.ledger.is_available():
                # Fallback to native if EVM is unavailable
                self.ledger = NativeMerkleLedger()
                self.blockchain_type = "native"
        else:
            self.ledger = NativeMerkleLedger()

    def run(
        self,
        image_path: str,
        query_hint: Optional[str] = None,
        preferred_platform: Optional[str] = None,
        save_crop: bool = True,
        allow_offline: Optional[bool] = None,
    ) -> PipelineOutput:
        """
        Executes the 4-stage pipeline:
        Stage 1: Face detection & 128D embedding extraction
        Stage 2: Web / Social Media search for matching post (genuine live search)
        Stage 3: Blockchain upload & block mining
        Stage 4: On-chain re-verification against immutable record
        """
        if allow_offline is not None:
            self.search_engine.allow_offline = allow_offline
            if allow_offline and not any(isinstance(p, VerifiedSampleCorpusProvider) for p in self.search_engine.providers):
                from core.search_engine import VerifiedSampleCorpusProvider
                self.search_engine.providers.append(VerifiedSampleCorpusProvider())

        # --- Stage 1: Face Identification ---
        face_result = self.face_engine.process_image(image_path, save_crop=save_crop)

        # --- Stage 2: Social Media / Web Search ---
        discovered_post = self.search_engine.find_matching_post(
            face_result,
            query_hint=query_hint,
            preferred_platform=preferred_platform,
        )

        # --- Stage 3: Blockchain Upload & Anchoring ---
        metadata = {
            "source_image": image_path,
            "confidence": face_result.confidence,
            "crop_path": face_result.crop_path,
            "match_source": discovered_post.match_source,
        }
        blockchain_rec = self.ledger.record_verification(
            face_hash=face_result.embedding_hash,
            post=discovered_post,
            metadata=metadata,
        )

        # --- Stage 4: On-Chain Re-Verification ---
        audit_result = self.ledger.verify_record(
            record_id=blockchain_rec.record_id,
            provided_face_hash=face_result.embedding_hash,
            provided_post_hash=discovered_post.post_hash,
        )

        summary = {
            "record_id": blockchain_rec.record_id,
            "blockchain": blockchain_rec.blockchain_type,
            "block_number": blockchain_rec.block_number,
            "block_hash": blockchain_rec.block_hash,
            "transaction_hash": blockchain_rec.transaction_hash,
            "face_hash": face_result.embedding_hash,
            "post_hash": discovered_post.post_hash,
            "verified": audit_result.is_valid,
        }

        return PipelineOutput(
            face_result=face_result,
            post=discovered_post,
            blockchain_record=blockchain_rec,
            audit_result=audit_result,
            execution_summary=summary,
        )

    def re_verify(
        self,
        record_id: str,
        face_hash: str,
        post_hash: str,
    ) -> VerificationAuditResult:
        """Re-verifies an existing record from the blockchain ledger."""
        return self.ledger.verify_record(record_id, face_hash, post_hash)
