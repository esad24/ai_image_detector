import json

json1 = "/home/usluesyr/ai_image_detector/data/ground_truth/gt_1/fake_test_labeled_1.json"
json2 = "/home/usluesyr/ai_image_detector/data/ground_truth/gt_1/new_artifacts.json"

with open(json1, "r", encoding="utf-8") as f:
    json1 = json.load(f)

with open(json2, "r", encoding="utf-8") as f:
    json2 = json.load(f)

# Build filename lookup
json1_lookup = {
    item["filename"]: item
    for item in json1["results"]
}

# Merge artifacts
for item in json2:
    filename = item["filename"]

    if filename in json1_lookup:
        json1_lookup[filename]["artifacts"].extend(
            item.get("artifacts", [])
        )
    else:
        json1["results"].append(item)



with open("json1_updated.json", "w", encoding="utf-8") as f:
    json.dump(
        json1,
        f,
        indent=4,
        ensure_ascii=True   # ← forces ASCII escaping
    )
