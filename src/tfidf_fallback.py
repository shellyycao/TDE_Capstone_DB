"""TF-IDF fallback for labels the taxonomy rules miss.

Character n-gram (char_wb, 3-5) TF-IDF: each subcategory's regex patterns, with the
regex syntax stripped, are joined into one reference document, and a label goes to
the subcategory whose document it is most cosine-similar to -- if that similarity
clears the threshold. Used by jobs/charge_type.py and by
eval/evaluate_fallback.py, so both always score the same model.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import TAXONOMY, TFIDF_EXCLUDE

# Raised from 0.25 after scoring the fallback on a hand-labelled set of labels the
# rules miss (eval/fallback_eval_set.csv): at 0.25 66% of its matches were correct,
# at 0.35 80%. Labels below the bar stay uncategorized and show up in charge_review
# instead of landing in a wrong category.
SIMILARITY_THRESHOLD = 0.35

# Words nearly every charge label contains. They say nothing about the category, but
# left in they make short reference documents win: "fuel surcharge fuel fsc" pulled
# "Early Surcharge", "JFK Surcharge", "DG Surcharge" into Fuel Surcharge on the shared
# word alone. Stripped from both the reference documents and the labels; on the eval
# set this categorized more labels at the same precision.
GENERIC_WORDS = re.compile(r"\b(sur)?charges?\b|\bfees?\b|\bcosts?\b")


def _strip_generic(text):
    return GENERIC_WORDS.sub(" ", text)


class TfidfFallback:
    def __init__(self, threshold=SIMILARITY_THRESHOLD):
        self.threshold = threshold
        docs, self.ref_labels = [], []
        for major, subcats in TAXONOMY.items():
            for sub, patterns in subcats.items():
                if (major, sub) in TFIDF_EXCLUDE:
                    continue
                doc = " ".join(p.replace(r"\b", "").replace(".*", " ").replace("'?", "").replace("?", "")
                               for p in patterns)
                docs.append(_strip_generic(doc))
                self.ref_labels.append((major, sub))
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.ref_vectors = self.vectorizer.fit_transform(docs)

    def classify(self, texts):
        """Return ((major, sub), similarity) or None for each normalized label text."""
        if not texts:
            return []
        sims = cosine_similarity(self.vectorizer.transform([_strip_generic(t) for t in texts]), self.ref_vectors)
        results = []
        for row in sims:
            best = row.argmax()
            results.append((self.ref_labels[best], float(row[best])) if row[best] >= self.threshold else None)
        return results
