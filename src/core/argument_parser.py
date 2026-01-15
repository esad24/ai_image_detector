import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--prompt_id", required=True, type=int)
    parser.add_argument("--model", required=True, type=str)

    return parser.parse_args()
