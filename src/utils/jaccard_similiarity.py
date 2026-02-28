# List 1 and List 2 (example filenames)
list1 = [
      "1baf3f78a5ea367863f70e867819f795ca6bcf73a331f741ba00f484aed98b61.jpeg",
      "b9836f50be0bc4098534bf668a8d68209c3708e6574595f492faeff454fcba7f.jpeg",
      "pexels-maryiaplashchynskaya-3400764.jpeg",
      "c38189c54230e478e366311946620a3a.jpeg",
      "pexels-cottonbro-10485417.jpeg",
      "b472b7624e736851155da6ece8d096ed1d6ffb74cd3710667a4e956d0812cfc9.jpeg",
      "6d8fba24ae67e57a85da5aa46b49dce4.jpeg",
      "bb940e5b5eb4667fa106086a3ddd34ce.jpeg",
      "03a001b85768dd4c7bcb8e20486bf0f20973314b1ec269ef1057f6565de34016.jpeg",
      "caabe61d5e181b8da352b0e7b73dfeb4.jpeg",
      "c362188ff928cf10b6cb734ba674c8f91af2ab48af337ebc873cd06769b429a1.jpeg",
      "0b96dd5fd38f6c18236dcec8df90c549.jpeg"
    ]

list2 = [
      "1baf3f78a5ea367863f70e867819f795ca6bcf73a331f741ba00f484aed98b61.jpeg",
      "b12ad54e54c8cece38e862c6b11edf66f1f15459d504b5107c3af796aec8c334.jpeg",
      "e4fd0b7c8c7d3dca42aab01618224fabc57cb8690ec4f3566b3335f83c8ab0ff.jpeg",
      "pexels-maryiaplashchynskaya-3400764.jpeg",
      "c65987f9418f465555948957aa66dced2c94386ef70da3809c09ba7a7fd8dd8b.jpeg",
      "pexels-cottonbro-10485417.jpeg",
      "d07a9e73914dad147383eb40cb21a296.jpeg",
      "b472b7624e736851155da6ece8d096ed1d6ffb74cd3710667a4e956d0812cfc9.jpeg",
      "765078d524b481fbb30b8acc1a7adf6926c79df9a914c4bd94fc97418e090acc.jpeg",
      "pexels-orestsv-1821699.jpeg",
      "99c5f61aa990fcbcaff73a63fdaef4cd.jpeg",
      "c362188ff928cf10b6cb734ba674c8f91af2ab48af337ebc873cd06769b429a1.jpeg"
    ]

# Convert to sets (important for intersection & union)
set1 = set(list1)
set2 = set(list2)

# Count shared files
shared_files = set1.intersection(set2)
num_shared = len(shared_files)

# Calculate Jaccard similarity
union_files = set1.union(set2)
similarity = num_shared / len(union_files) if len(union_files) > 0 else 0

# Print results
print("Shared files:", num_shared)
print("Shared filenames:", shared_files)
print("Dataset similarity (Jaccard):", round(similarity, 4))