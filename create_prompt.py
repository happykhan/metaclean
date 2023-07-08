import os
import json
import logging

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def generate_prompt_text(biosample_path, prompt_dir, prompt_text):
    """
    Generates prompt text files for each biosample file in the specified directory.

    Args:
        biosample_path (str): The path to the directory containing the biosample files.
        prompt_dir (str): The path to the directory where the prompt text files will be written.
        prompt_text (str): The path to the file containing the prompt text.
    """   
    for sub_dir_path in [os.path.join(biosample_path,subdir) for subdir in os.listdir(biosample_path) if os.path.isdir(os.path.join(biosample_path,subdir))]:
        for biosample_path, accession in [(os.path.join(sub_dir_path, f), f.split('.')[0]) for f in os.listdir(sub_dir_path) if f.endswith(".txt")]:
            subdir = accession[0:8]
            out_dir =  os.path.join(prompt_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            out_prompt = os.path.join(prompt_dir, subdir,  f"prompt_{accession}.txt")
            with open(out_prompt, 'w') as outfile:
                outfile.write(open(biosample_path).read())
                outfile.write('\n\n')        
                outfile.write(open(prompt_text).read())


def main(prompt_text: str="prompt_text.txt", prompt_dir: str='prompts/', biosample_path :str="biosamples/"):
    """
    The main function that orchestrates the fetching of biosamples, fields, and prompt text.
    """
    generate_prompt_text(biosample_path, prompt_dir, prompt_text)


if __name__ == "__main__":
    typer.run(main)