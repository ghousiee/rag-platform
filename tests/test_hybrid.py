from rag_platform.retrieval.hybrid import fuse_dense_and_bm25
from rag_platform.stores.base import SearchHit


def test_fuse_prefers_agreement():
    dense = [SearchHit("1", "a", 0.9, {}), SearchHit("2", "b", 0.5, {})]
    bm25 = [SearchHit("2", "b", 0.8, {}), SearchHit("3", "c", 0.7, {})]
    fused = fuse_dense_and_bm25(dense, bm25, bm25_weight=0.4, k=2)
    assert len(fused) == 2
    assert {h.id for h in fused} <= {"1", "2", "3"}


def test_fuse_shared_id_ranks_high():
    dense = [SearchHit("1", "a", 0.9, {}), SearchHit("2", "b", 0.5, {})]
    bm25 = [SearchHit("2", "b", 0.8, {}), SearchHit("3", "c", 0.7, {})]
    fused = fuse_dense_and_bm25(dense, bm25, bm25_weight=0.4, k=3)
    assert fused[0].id == "1"
    assert fused[1].id == "2"


def test_fuse_dense_only_when_bm25_empty():
    dense = [SearchHit("1", "a", 0.9, {})]
    fused = fuse_dense_and_bm25(dense, [], bm25_weight=0.4, k=1)
    assert len(fused) == 1
    assert fused[0].id == "1"