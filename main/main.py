# import json

# def analyze(model, prompt):
#     return

# def classification():
#     with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         # writer.writerow(["filename", "classification"])
#         writer.writerows(results)


# def artifacts():
#     with open(CLASSIFICATION_ARTIFACTS_CSV, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=['filename', 'classification', 'artifacts'])
        
#         for filename, result in results:
#             writer.writerow({
#                 'filename': filename,
#                 'classification': result_JSON.get('classification', ''),
#                 'artifacts': ", ".join(result_JSON.get('artifacts', []))
#             })

#     print(f"\nSaved results to {CLASSIFICATION_ARTIFACTS_CSV}")


# def artifacts_explained():
#     OUTPUT_JSON = "results/classification_artifacts_explanation.json"
#     # Convert results to serializable format
#     serializable_results = []
#     for filename, result in results:
#         serializable_results.append({
#             "filename": filename,
#             "classification": result.classification,
#             "artifacts": [a.__dict__ for a in result.artifacts]
#         })
#     # Save to JSON file
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(serializable_results, f, ensure_ascii=False, indent=4)

#     print(f"\nSaved all results to {OUTPUT_JSON}")





# def main():
#     # Load image file paths
#     images = []
#     for root, dirs, files in os.walk(IMAGE_FOLDER):
#         for f in files:
#             if f.lower().endswith(("jpg", "jpeg", "png")):
#                 images.append(os.path.join(root, f))

#     if not images:
#         print("No images found in folder:", IMAGE_FOLDER)
#         return

#     print(f"Found {len(images)} images. Starting analysis...\n")

#     results = []

#     #i = 0
#     for img_path in images:
#         print(f"Uploading: {img_path}")
#         base64_image  = encode_image(img_path)

#         print("Analyzing...")
#         result = analyze_image(base64_image)

#         print(f"Result for {os.path.basename(img_path)} → {result}\n")

#         results.append([os.path.basename(img_path), result])

#         #i += 1
#         #if i == 2: break



    


# if __name__ == "__main__":
#     main()
