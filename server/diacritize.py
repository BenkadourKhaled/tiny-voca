"""Server-side tashkeel with CAMeL Tools (MIT), loaded once and cached.

The browser demo cannot diacritise reliably; on a server we can. CAMeL Tools
1.6 uses the CALIMA MSA morphology database and an MLE disambiguator, which
must be built once at startup (``camel_data -i defaults`` at image build time).
"""

from __future__ import annotations

from functools import lru_cache

_disambiguator = None


def _engine():
    global _disambiguator
    if _disambiguator is None:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.disambig.mle import MLEDisambiguator

        _disambiguator = MLEDisambiguator(Analyzer(MorphologyDB.builtin_db("calima-msa-r13")))
    return _disambiguator


@lru_cache(maxsize=2000)
def vocalize(text: str) -> str:
    """Unvocalised Arabic -> fully vocalised Arabic (best effort)."""
    from camel_tools.tokenizers.word import simple_word_tokenize

    tokens = simple_word_tokenize(text)
    analysed = _engine().disambiguate(tokens)
    return " ".join(item.analyses[0].analysis.get("diac", item.word) for item in analysed)
