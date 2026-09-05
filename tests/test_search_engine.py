"""
Tests for SocialMediaSearchEngine, candidate ranking, and prevention of silent corpus fallback.
"""

import pytest
from core.search_engine import SocialMediaSearchEngine, VerifiedSampleCorpusProvider
from core.models import FaceDetectionResult, BoundingBox, SocialPost


def test_post_canonical_hash_deterministic():
    p1 = SocialPost(
        platform="x",
        author="Elon Musk",
        author_handle="@elonmusk",
        post_url="https://x.com/elonmusk/status/123",
        content_text="Verifying identity on blockchain",
    )
    p2 = SocialPost(
        platform="X",  # case difference normalized
        author="Elon Musk ",  # whitespace trimmed
        author_handle="@elonmusk",
        post_url="https://x.com/elonmusk/status/123",
        content_text="Verifying identity on blockchain",
    )
    assert p1.post_hash == p2.post_hash
    assert len(p1.post_hash) == 64


def test_no_silent_corpus_fallback():
    """
    CRITICAL REQUIREMENT: Verifies that the default search engine NEVER
    silently falls back to the hardcoded sample corpus when live search fails.
    """
    # Engine configured with allow_offline=False (default)
    engine = SocialMediaSearchEngine(allow_offline=False)

    # 1. Verify VerifiedSampleCorpusProvider is NOT in provider list
    assert not any(isinstance(p, VerifiedSampleCorpusProvider) for p in engine.providers), (
        "VerifiedSampleCorpusProvider must NOT be in the default search provider list!"
    )

    # 2. Simulate an unsearchable face query with dummy providers that return nothing
    class EmptySearchProvider:
        def search(self, face_result, query_hint=None, face_engine=None):
            return []

    engine.providers = [EmptySearchProvider()]

    dummy_face = FaceDetectionResult(
        bounding_box=BoundingBox(x=0, y=0, width=50, height=50),
        confidence=0.9,
        embedding=[0.05] * 128,
        embedding_hash="test_hash_123",
        source_image="data/samples/sample_face_1.jpg",  # Name that previously matched the corpus
    )

    # 3. Must raise RuntimeError instead of silently returning mock data
    with pytest.raises(RuntimeError) as exc_info:
        engine.find_matching_post(dummy_face)

    assert "Live search failed" in str(exc_info.value)
    assert "--offline-fallback" in str(exc_info.value)


def test_explicit_offline_fallback_flag():
    """
    Verifies that the sample corpus is ONLY accessible when allow_offline=True
    is explicitly requested, and is properly labeled as OFFLINE_FALLBACK.
    """
    class EmptyLiveProvider:
        def search(self, face_result, query_hint=None, face_engine=None):
            return []

    engine = SocialMediaSearchEngine(allow_offline=True)
    engine.providers = [EmptyLiveProvider(), VerifiedSampleCorpusProvider()]

    dummy_face = FaceDetectionResult(
        bounding_box=BoundingBox(x=0, y=0, width=50, height=50),
        confidence=0.9,
        embedding=[0.05] * 128,
        embedding_hash="test_hash_123",
        source_image="data/samples/sample_face_1.jpg",
    )

    post = engine.find_matching_post(dummy_face)
    assert post is not None
    assert post.search_status == "OFFLINE_FALLBACK"
    assert "offline mode" in post.match_source


def test_live_search_produces_live_status():
    """
    Verifies that live search returns LIVE_SEARCH status and valid similarity metrics.
    """
    engine = SocialMediaSearchEngine(allow_offline=False)
    dummy_face = FaceDetectionResult(
        bounding_box=BoundingBox(x=0, y=0, width=50, height=50),
        confidence=0.9,
        embedding=[0.05] * 128,
        embedding_hash="test_hash_123",
        source_image="data/samples/elon_musk.jpg",
    )

    post = engine.find_matching_post(dummy_face, query_hint="Elon Musk")
    assert post is not None
    assert post.search_status == "LIVE_SEARCH"
    assert post.match_source != "verified_corpus_search"
    assert post.candidate_count > 0
    assert post.platform_type in ("SOCIAL", "WEB")


def test_is_individual_social_post_validation():
    """
    Verifies that only actual individual social posts qualify, and navigation/search/trending
    pages are strictly excluded.
    """
    from core.search_engine import is_individual_social_post

    # Valid post URLs
    assert is_individual_social_post("https://twitter.com/MarioNawfal/status/1658303787809767427") is True
    assert is_individual_social_post("https://x.com/elonmusk/status/1829012398471928471") is True
    assert is_individual_social_post("https://www.reddit.com/r/technology/comments/123/face_biometrics/") is True
    assert is_individual_social_post("https://www.linkedin.com/posts/satyanadella_ai-innovation-activity-71239-abc") is True
    assert is_individual_social_post("https://www.instagram.com/p/C123abc/") is True

    # Invalid / Navigation / Trending / Profile URLs
    assert is_individual_social_post("https://x.com/i/trending/2039037542301929918") is False
    assert is_individual_social_post("https://x.com/ElonMusk") is False
    assert is_individual_social_post("https://twitter.com/explore") is False
    assert is_individual_social_post("https://www.reddit.com/r/technology/") is False
    assert is_individual_social_post("https://www.linkedin.com/in/satya-nadella") is False
    assert is_individual_social_post("https://www.cnn.com/2023/10/28/tech/elon-musk") is False


def test_classify_candidate_url():
    """
    Verifies authentic URL classification without fabricating authors.
    """
    from core.search_engine import classify_candidate_url

    # X / Twitter individual post
    p, pt, author, handle = classify_candidate_url("https://x.com/elonmusk/status/1829012398471928471", query_hint="Elon Musk")
    assert p == "x"
    assert pt == "SOCIAL"
    assert author == "elonmusk"
    assert handle == "@elonmusk"

    # Trending page (must be classified as WEB, not SOCIAL)
    p, pt, author, handle = classify_candidate_url("https://x.com/i/trending/2039037542301929918", query_hint="Elon Musk")
    assert pt == "WEB"

    # Reddit individual post
    p, pt, author, handle = classify_candidate_url("https://www.reddit.com/r/technology/comments/123/face_recognition", query_hint="Tech User")
    assert p == "reddit"
    assert pt == "SOCIAL"
    assert handle == "r/technology"

    # Generic News (e.g. CNN)
    p, pt, author, handle = classify_candidate_url("https://www.cnn.com/2023/10/28/tech/elon-musk-year-owning-twitter", query_hint="Elon Musk")
    assert p == "web"
    assert pt == "WEB"
    assert handle == ""
    assert "Cnn" in author or "Web" in author


def test_social_candidate_prioritization_over_web():
    """
    Verifies that a social media post is strictly prioritized over a news article
    even if the news article has a slightly higher raw image similarity score.
    """
    from core.search_engine import rank_candidates_social_first

    web_post = SocialPost(
        platform="web",
        platform_type="WEB",
        author="CNN News",
        post_url="https://www.cnn.com/tech/article-123",
        content_text="News article discussing social media",
        similarity_score=0.82,
    )
    social_post = SocialPost(
        platform="x",
        platform_type="SOCIAL",
        author="elonmusk",
        author_handle="@elonmusk",
        post_url="https://x.com/elonmusk/status/456",
        content_text="Real social post on X",
        similarity_score=0.75,
    )

    scored = [
        (0.82, web_post),
        (0.75, social_post),
    ]

    ranked = rank_candidates_social_first(scored, min_social_sim_threshold=0.35)
    assert len(ranked) == 2
    # The SOCIAL post must be ranked #1
    assert ranked[0].platform_type == "SOCIAL"
    assert ranked[0].platform == "x"
    assert ranked[1].platform_type == "WEB"


def test_web_fallback_when_no_social():
    """
    Verifies that if NO qualifying social media posts exist, the engine falls back
    to the top WEB candidate and clearly labels it as WEB.
    """
    from core.search_engine import rank_candidates_social_first

    web_post = SocialPost(
        platform="web",
        platform_type="WEB",
        author="Reuters News",
        post_url="https://www.reuters.com/tech/article-999",
        content_text="Global news article",
        similarity_score=0.78,
    )

    scored = [(0.78, web_post)]
    ranked = rank_candidates_social_first(scored)
    assert len(ranked) == 1
    assert ranked[0].platform_type == "WEB"
    assert ranked[0].platform == "web"


def test_distinct_retrieval_and_publication_timestamps():
    """
    Verifies that retrieval timestamp (retrieved_at) is distinct from published_date.
    """
    post = SocialPost(
        platform="x",
        platform_type="SOCIAL",
        author="elonmusk",
        post_url="https://x.com/elonmusk/status/123",
        content_text="Decentralized identity record",
        published_date="2024-05-10",
    )
    assert post.published_date == "2024-05-10"
    assert post.retrieved_at is not None
    assert "Z" in post.retrieved_at or len(post.retrieved_at) > 10
