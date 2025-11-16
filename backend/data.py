import re
import os
import pandas as pd

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "structured_gita.csv")

def loadAndProcessGita():
    from datasets import load_dataset  # Local import to avoid extra dependency at top

    dataset = load_dataset("utkarshpophli/bhagwat_gita")
    gitaDf = dataset["train"].to_pandas()

    structured = []
    currentVerse, spokenBy, sanskritText, translations = None, None, None, []

    for _, row in gitaDf.iterrows():
        text = row["text"].strip()

        if text.startswith("<s>[INST]"):
            if currentVerse:
                structured.append([currentVerse, spokenBy, sanskritText] + translations)
            translations = []

            verseMatch = re.search(r"verse (\d+\.\d+)", text)
            currentVerse = verseMatch.group(1) if verseMatch else None

            speakerMatch = re.search(r"spoken by ([^.\[\]/]+)", text)
            spokenBy = speakerMatch.group(1).strip() if speakerMatch else None

        elif text.startswith("Sanskrit:"):
            sanskrit = text.replace("Sanskrit:", "").strip()
            sanskritText = re.sub(r"\d+$", "", sanskrit).strip()

        elif text.startswith("Translations:") or (text and text[0].isdigit()):
            translations.append(text.split(" ", 1)[-1] if " " in text else text)

    if currentVerse:
        structured.append([currentVerse, spokenBy, sanskritText] + translations)

    maxTranslations = max(len(row) - 3 for row in structured)
    for row in structured:
        while len(row) < 3 + maxTranslations:
            row.append("")

    columns = ["verse_no", "spoken_by", "sanskrit_text"] + [f"translation_{i+1}" for i in range(maxTranslations)]
    df = pd.DataFrame(structured, columns=columns)

    # Drop translation_1 if it's empty and rename others
    if "translation_1" in df.columns:
        df = df.drop(columns=["translation_1"])

    renameMap = {
        f"translation_{i}": f"translation_{i-1}"
        for i in range(2, 7)
        if f"translation_{i}" in df.columns
    }
    df = df.rename(columns=renameMap)

    speakerMap = {
        "अर्जुन": "Arjun",
        "सञ्जय": "Sanjay",
        "संजय": "Sanjay",
        "धृतराष्ट्र": "Dhritrashtra",
        "भगवान": "Krishna"
    }
    df["spoken_by"] = df["spoken_by"].replace(speakerMap)

    for col in [col for col in df.columns if col.startswith("translation_")]:
        df[col] = df[col].str.lower()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    print("✅ Gita structured data saved.")
    return df