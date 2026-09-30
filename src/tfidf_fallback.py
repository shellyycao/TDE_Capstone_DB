"""TF-IDF fallback for labels the taxonomy rules miss.

Character n-gram (char_wb, 3-5) TF-IDF: each subcategory's regex patterns, with the
regex syntax stripped, are joined into one reference document, and a label goes to
the subcategory whose document it is most cosine-similar to -- if that similarity
clears the threshold. Used by jobs/charge_type.py (the default --fallback) and by
eval/evaluate_fallback.py, so both always score the same model.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import TAXONOMY, TFIDF_EXCLUDE

# Raised from 0.25 after scoring the fallback on a hand-labelled set of labels the
# rules miss (eval/fallback_eval_set.csv): at 0.25 66% of its matches were correct,
# at 0.35 80%. Labels below the bar stay uncategorized and show up in charge_review
# instead of landing in a wrong category.
SIMILARITY_THRESHOLD = 0.35


class TfidfFallback:
    def __init__(self, threshold=SIMILARITY_THRESHOLD):
        self.threshold = threshold
        docs, self.ref_labels = [], []
        for major, subcats in TAXONOMY.items():
            for sub, patterns in subcats.items():
                if (major, sub) in TFIDF_EXCLUDE:
                    continue
                docs.append(" ".join(p.replace(r"\b", "").replace(".*", " ").replace("'?", "").replace("?", "")
                                     for p in patterns))
                self.ref_labels.append((major, sub))
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.ref_vectors = self.vectorizer.fit_transform(docs)

    def classify(self, texts):
        """Return ((major, sub), similarity) or None for each normalized label text."""
        if not texts:
            return []
        sims = cosine_similarity(self.vectorizer.transform(texts), self.ref_vectors)
        results = []
        for row in sims:
            best = row.argmax()
            results.append((self.ref_labels[best], float(row[best])) if row[best] >= self.threshold else None)
        return results
