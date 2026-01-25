import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", dest="image_folder", type=str, required=True, help="options: real_train, real_test, fake_train, fake_test, or Path to the folder containing images")
    parser.add_argument("-m", dest="model", type=str, required=True, help="Model name (e.g., gpt-5.2)")
    parser.add_argument("-p", dest="prompt_id", type=int, required=True, help="ID of the prompt to use")
    parser.add_argument("-r", dest="resume", action="store_true", help="Resume from the latest previous run if it exists")
    return parser.parse_args()
