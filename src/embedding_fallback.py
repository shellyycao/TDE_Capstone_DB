"""Sentence-embedding fallback for labels the taxonomy rules miss.

Experimental alternative to the TF-IDF fallback in jobs/charge_type.py, selected
with `--fallback embedding`. It differs from TF-IDF in two deliberate ways:

  * meaning instead of spelling -- a multilingual sentence-embedding model, so
    "Weekend Charge" can land near "Saturday Delivery" and Dutch labels
    ("brandstoftoeslag") near their English equivalents, where character n-grams
    only see shared substrings like "charge";
  * real labels as references -- each unmatched label is compared (k nearest
    neighbours) with the labels the rules already classified, plus one short
    name document per subcategory so a subcategory with no rule-matched examples
    can still be chosen. TF-IDF compares against the regex fragments joined into
    one string per subcategory, which reads nothing like a real charge label.

Runs locally through fastembed (ONNX runtime, no torch -- torch has no current
build for the x86_64 Python this repo's venv uses). The model is downloaded to
the fastembed cache on first use; no label text leaves the machine.
"""

from collections import defaultdict

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class EmbeddingFallback:
    def __init__(self, references, threshold, k=5, model_name=MODEL_NAME):
        """references: list of (text, (major, sub)) pairs to vote with."""
        from fastembed import TextEmbedding  # lazy: only this fallback needs it

        self.model = TextEmbedding(model_name)
        self.threshold = threshold
        self.k = k
        self.ref_labels = [label for _, label in references]
        self.ref_vectors = self._embed([t for t, _ in references])

    def _embed(self, texts):
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def classify(self, texts):
        """Return ((major, sub), similarity) or None for each text.

        The k most similar references vote, weighted by similarity. The reported
        similarity -- and what the threshold is checked against -- is the best
        single match within the winning subcategory, so it stays comparable
        with the TF-IDF fallback's "best match" score.
        """
        results = []
        for sims in self._embed(texts) @ self.ref_vectors.T:
            top = np.argsort(sims)[::-1][: self.k]
            votes = defaultdict(float)
            for i in top:
                votes[self.ref_labels[i]] += sims[i]
            best = max(votes, key=votes.get)
            best_sim = float(max(sims[i] for i in top if self.ref_labels[i] == best))
            results.append((best, best_sim) if best_sim >= self.threshold else None)
        return results
