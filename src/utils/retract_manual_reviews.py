import json

def normalize(text):
    """Normalize text for safer matching."""
    if text is None:
        return ""
    return " ".join(text.lower().split())

def build_manual_review_lookup(json1):
    """
    Build lookup:
    {
      filename: [
        {
          type, reasoning, location, manual_review
        }
      ]
    }
    """
    lookup = {}

    for img in json1.get("image_scores", []):
        filename = img["filename"]
        for match in img.get("matched_artifacts", []):
            if "manual_review" not in match:
                continue

            result_artifact = match.get("result_artifact")
            if not result_artifact:
                continue

            entry = {
                "type": normalize(result_artifact.get("type")),
                "reasoning": normalize(result_artifact.get("reasoning")),
                "location": normalize(result_artifact.get("location")),
                "manual_review": match["manual_review"]
            }

            lookup.setdefault(filename, []).append(entry)

    return lookup


def copy_manual_reviews(json1, json2):
    lookup = build_manual_review_lookup(json1)

    for img in json2.get("results", []):
        filename = img["filename"]
        if filename not in lookup:
            continue

        for artifact in img.get("artifacts", []):
            for candidate in lookup[filename]:
                if (
                    normalize(artifact.get("type")) == candidate["type"]
                    and normalize(artifact.get("reasoning")) == candidate["reasoning"]
                    and normalize(artifact.get("location")) == candidate["location"]
                ):
                    artifact["manual_review"] = candidate["manual_review"]
                    break

    return json2

json1_path = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/semantic_evaluation.json"
json2_path = "/home/usluesyr/ai_image_detector/data/fake/test/results/gpt-5.2/prompt3/2026-01-20_18-46-16/results.json"


# ---------- usage ----------
with open(json1_path, "r", encoding="utf-8") as f:
    json1 = json.load(f)

with open(json2_path, "r", encoding="utf-8") as f:
    json2 = json.load(f)

updated_json2 = copy_manual_reviews(json1, json2)

with open("json2_with_manual_reviews.json", "w", encoding="utf-8") as f:
    json.dump(updated_json2, f, indent=2, ensure_ascii=False)
