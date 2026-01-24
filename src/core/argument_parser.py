import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-k", dest="api_key", type=str, required=True, help="OpenAI API Key")  # neu
    parser.add_argument("-i", dest="image_folder", type=str, required=True, help="options: train, test or Path to the folder containing images")
    parser.add_argument("-m", dest="model", type=str, required=True, help="Model name (e.g., gpt-5.2)")
    parser.add_argument("-p", dest="prompt_id", type=int, required=True, help="ID of the prompt to use")

    return parser.parse_args()
