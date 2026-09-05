"""
Core data models for Face Identification and Blockchain Verification pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import hashlib
import json
from datetime import datetime


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceDetectionResult(BaseModel):
    bounding_box: BoundingBox
    confidence: float
    landmarks: List[List[float]] = Field(default_factory=list, description="5 facial landmarks [[x, y], ...]")
    embedding: List[float] = Field(default_factory=list, description="128-dimensional facial embedding vector")
    embedding_hash: str = Field(description="SHA-256 hash of the normalized face embedding vector")
    source_image: str = Field(description="Path or URL of the source image")
    crop_path: Optional[str] = Field(default=None, description="Path to extracted face crop image")

    @classmethod
    def compute_embedding_hash(cls, embedding: List[float]) -> str:
        """Computes a deterministic SHA-256 hash for a normalized 128D embedding vector."""
        # Format floats to 6 decimal places for deterministic precision
        formatted = [f"{val:.6f}" for val in embedding]
        raw_bytes = ",".join(formatted).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()


class SocialPost(BaseModel):
    platform: str = Field(description="Platform name e.g. twitter, x, reddit, instagram, linkedin, web")
    platform_type: str = Field(default="SOCIAL", description="SOCIAL (social media post) or WEB (generic news/article)")
    author: str = Field(description="Author display name or username")
    author_handle: Optional[str] = Field(default="", description="Social handle e.g. @elonmusk")
    post_url: str = Field(description="Direct URL to the post or thread")
    content_text: str = Field(description="Extracted caption, tweet text, or post body")
    published_date: Optional[str] = Field(default=None, description="Original publication timestamp if available")
    retrieved_at: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"), description="Retrieval UTC timestamp")
    image_url: Optional[str] = Field(default=None, description="Image or thumbnail URL associated with post")
    match_source: str = Field(default="live_search_sface_ranked", description="Engine used (google_lens, ddgs_live, offline_corpus)")
    search_status: str = Field(default="LIVE_SEARCH", description="LIVE_SEARCH or OFFLINE_FALLBACK")
    confidence_score: float = Field(default=0.95, description="Visual/entity match confidence")
    similarity_score: Optional[float] = Field(default=None, description="SFace facial embedding cosine similarity against input scan")
    candidate_count: int = Field(default=0, description="Total live search candidates retrieved and ranked")

    @property
    def post_hash(self) -> str:
        """Deterministic cryptographic SHA-256 fingerprint of the post content and metadata."""
        canonical_dict = {
            "platform": self.platform.strip().lower(),
            "platform_type": self.platform_type.strip().upper(),
            "author": self.author.strip(),
            "author_handle": (self.author_handle or "").strip(),
            "post_url": self.post_url.strip(),
            "content_text": self.content_text.strip(),
            "image_url": (self.image_url or "").strip(),
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class BlockchainRecord(BaseModel):
    record_id: str = Field(description="Unique deterministic ID (SHA-256 of face_hash + post_hash)")
    face_hash: str = Field(description="Cryptographic SHA-256 hash of the face embedding")
    post_hash: str = Field(description="Cryptographic SHA-256 hash of the discovered post")
    platform: str
    post_url: str
    author: str
    timestamp: int = Field(description="Unix timestamp of when verification was recorded")
    block_number: int = Field(default=0, description="Block index in which transaction was included")
    block_hash: str = Field(default="", description="Hash of the mined block")
    transaction_hash: str = Field(default="", description="Transaction ID or hash")
    blockchain_type: str = Field(default="native_merkle", description="native_merkle or evm")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationAuditResult(BaseModel):
    is_valid: bool
    record_id: str
    on_chain_face_hash: str
    provided_face_hash: str
    face_hash_matches: bool
    on_chain_post_hash: str
    provided_post_hash: str
    post_hash_matches: bool
    block_number: int
    block_hash: str
    timestamp: int
    message: str
    blockchain_type: str
