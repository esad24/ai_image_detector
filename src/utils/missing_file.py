import os

def compare_folders(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    only_in_folder1 = files1 - files2
    only_in_folder2 = files2 - files1

    print(f"\nFiles only in {folder1}:")
    for file in sorted(only_in_folder1):
        print(file)

    print(f"\nFiles only in {folder2}:")
    for file in sorted(only_in_folder2):
        print(file)


if __name__ == "__main__":
    folder1 = "/home/usluesyr/ai_image_detector/data/fake/test/images"
    folder2 = "/home/usluesyr/ai_image_detector/data/ground_truth/gt_2/labled_test_second_round"

    compare_folders(folder1, folder2)
