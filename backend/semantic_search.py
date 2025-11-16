from collections import Counter
from query_expansion import generateQueryVariations

def retrieveRelevantDocs(question: str, vectorStore):
    variations = generateQueryVariations(question)
    print("🔍 Query Variations Generated:\n")
    for v in variations:
        print(f" - {v}")

    allDocs = []
    for query in variations:
        results = vectorStore.similarity_search(query, k=3)
        allDocs.extend(results)

    freqCounter = Counter(
        (doc.metadata["verse_no"], doc.metadata["translation_index"]) for doc in allDocs
    )

    bestTranslationPerVerse = {}
    for (verse_no, translation_index), count in freqCounter.items():
        if verse_no not in bestTranslationPerVerse or count > bestTranslationPerVerse[verse_no]["count"]:
            bestTranslationPerVerse[verse_no] = {
                "translation_index": translation_index,
                "count": count
            }

    topN = 2
    topVerses = sorted(
        bestTranslationPerVerse.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:topN]

    selected = {}
    for doc in allDocs:
        verse_no = doc.metadata["verse_no"]
        translation_index = doc.metadata["translation_index"]

        for topVerse, info in topVerses:
            if verse_no == topVerse and translation_index == info["translation_index"]:
                selected[(verse_no, translation_index)] = doc
    print("✅ Top Matching Verses Identified.")
    print()
    return selected