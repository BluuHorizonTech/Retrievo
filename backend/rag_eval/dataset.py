import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("golden dataset must be a JSON list")
    return data


def relevant_id_set(item: Dict, hits: List[Dict], level: str) -> Set[int]:
    """
    Returns the set of chunk_ids that should count as 'relevant'
    based on the label level chosen in config.
    `hits` is the retrieved list from engine.retrieve() — used only
    to translate filename/page labels into the ids actually present.
    """
    labels = item.get("relevant_chunks", [])
    if level == "chunk_id":
        return {int(l["chunk_id"]) for l in labels if l.get("chunk_id") is not None}

    if level == "page":
        want = {(l["filename"], l["page"]) for l in labels if l.get("filename")}
        return {
            h["chunk_id"]
            for h in hits
            if (h["filename"], h["page"]) in want
        }

    if level == "filename":
        want = {l["filename"] for l in labels if l.get("filename")}
        return {h["chunk_id"] for h in hits if h["filename"] in want}

    raise ValueError(f"unknown relevance level: {level}")