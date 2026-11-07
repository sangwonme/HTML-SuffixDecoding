"""TF-IDF retrieval over train_pool query_rich (used by baseline B2 and few-shot example selection)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class Retriever:
    def __init__(self, pool: list[dict], field: str = "query_rich"):
        self.pool = pool
        self.field = field
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.X = self.vec.fit_transform([r[field] for r in pool])

    def topk(self, query: str, k: int = 3, max_html_chars: int | None = None) -> list[dict]:
        sims = linear_kernel(self.vec.transform([query]), self.X).ravel()
        out = []
        for i in sims.argsort()[::-1]:  # deterministic: argsort is stable for ties by index order reversed
            r = self.pool[int(i)]
            if max_html_chars and len(r["gt_html_flow"]) > max_html_chars:
                continue
            out.append(r)
            if len(out) == k:
                break
        return out
