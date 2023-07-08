"""
fetch_biosample.py

This script fetches biosample IDs from an enterobase table, fetches missing biosamples, and generates
prompt text files for each biosample file in the specified directory.

Usage:
    python fetch_biosample.py  [--prompt_text=<prompt_text>]
    [--prompt_dir=<prompt_dir>] [--biosample_path=<biosample_path>] [--entero_file=<entero_file>]
"""
from Bio import Entrez
import xmltodict
import os 
import urllib
import json 
import logging
import typer

# Set email address for NCBI Entrez
Entrez.email = "nabil@happyykhan.com"
# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def fetch_biosamples(entero_file="entero_all_7.7.23.tsv"):
    """
    Fetches the biosample IDs from an enterobase table.

    Args:
        entero_file (str): The path to the enterobase table file. Defaults to "entero_all_7.7.23.tsv".

    Returns:
        list: A list of valid biosample IDs.
    """
    # Open the entero_all_7.7.23.tsv file and read its contents
    with open(entero_file, "r") as f:
        lines = f.readlines()
        sample_index = lines[0].strip().split("\t").index("Sample ID")
        count = 0
        biosample_ids = []
        for line in lines[1:]:
            # Get index of Sample ID in line.split("\t")
            biosample = line.strip().split("\t")[sample_index]
            # is it a valid biosample id?
            if biosample.startswith("SAMN"):
                biosample_ids.append(biosample)
        return biosample_ids

def fetch_missing_biosamples(biosample_ids, biosample_path="biosamples/"):
    """
    Fetches missing biosamples and writes them to files in the specified directory.

    Args:
        biosample_ids (list): A list of biosample IDs to fetch.
        biosample_path (str): The path to the directory where the biosample files will be written.
                            Defaults to "biosamples/".

    Raises:
        HTTPError: If there is an error fetching the biosamples.

    Notes:
        This function splits the biosample IDs into chunks of 1000 and fetches them using the Entrez API.
        The fetched biosamples are written to files in the specified directory.
        The biosamples are written as pretty JSON.
    """
    # split biosample_ids into chunks of 1000
    logging.info(f'Fetching missing biosamples {len(biosample_ids)}')
    biosample_ids_chunks = [biosample_ids[i:i + 1000] for i in range(0, len(biosample_ids), 1000)]
    for biosample_id_chunk in biosample_ids_chunks:
        try:
            handle = Entrez.esearch(db="biosample", term=' OR '.join(biosample_id_chunk), retmax=1000)
            record = Entrez.read(handle)
            id_list_string = ",".join(record['IdList'])
            epost_xml = Entrez.epost(db="biosample", id=id_list_string, retmax=1000).read()
            epost_dict = xmltodict.parse(epost_xml)
            webenv = epost_dict['ePostResult']['WebEnv']
            logging.info(f"Fetching {len(biosample_id_chunk)} biosamples")
            record_chunks = xmltodict.parse(Entrez.efetch(db="biosample", id=id_list_string, webenv=webenv, rettype="xml", retmax=1000).read())
            for dict_data in record_chunks['BioSampleSet']['BioSample']:
                accession = dict_data['@accession']
                # write dict_data as pretty JSON to file 
                # make sub dir if it doesnt exist
                subdir = accession[0:8]
                if not os.path.exists(os.path.join(biosample_path, subdir)):
                    os.mkdir(os.path.join(biosample_path, subdir))
                with open(os.path.join(biosample_path, subdir, f"{accession}.txt"), 'w') as outfile:
                    outfile.write(json.dumps(dict_data, indent=4))
        except urllib.error.HTTPError as e:
            logging.error(f"Could not fetch biosamples: {e}")
            continue

def check_biosample_file_exists(biosample_id, biosample_path="biosamples/"):
    """
    Checks if a biosample file exists in the specified directory.

    Args:
        biosample_id (str): The ID of the biosample to check.
        biosample_path (str): The path to the directory containing the biosample files.
                            Defaults to "biosamples/".

    Returns:
        bool: True if the biosample file exists, False otherwise.
    """
    subdir = biosample_id[0:8]
    return os.path.exists(f"{biosample_path}/{subdir}/{biosample_id}.txt")


def check_biosample(biosample_path_ori :str="biosamples/"):
    """
    Cleans biosample files by removing unnecessary information and writing the cleaned files to a new directory.

    Args:
        biosample_path_ori (str): The path to the directory containing the biosample files. Defaults to "biosamples/".

    Notes:
        This function searches for biosample files in the specified directory and its subdirectories.
        It only considers files with the ".txt" extension.
        The cleaned files are written to a new directory with the same structure as the original directory.
        The new directory is located in the same parent directory as the original directory and is named "cleaned_biosamples".

    Examples:
    >>> check_biosample("path/to/biosamples/")
    """    
    for sub_dir_path, subdir in [(os.path.join(biosample_path_ori,subdir),subdir) for subdir in os.listdir(biosample_path_ori) if os.path.isdir(os.path.join(biosample_path_ori,subdir))]:
        for biosample_path, accession in [(os.path.join(sub_dir_path, f), f.split('.')[0]) for f in os.listdir(sub_dir_path) if f.endswith(".txt")]:
            keep_for_now = '' 
            with open(biosample_path) as f:
                try:
                    dict_data = json.load(f)
                    if dict_data.get('BioSampleSet'):
                        dict_data = dict_data['BioSampleSet']['BioSample']
                        keep_for_now = dict_data
                        logging.warn(f'Cleaning {accession}')
                except json.decoder.JSONDecodeError as e:
                    logging.error(f"Could not parse {biosample_path}: {e}") 
                    continue
            if keep_for_now != '':
                with open(os.path.join(biosample_path_ori, subdir, f"{accession}.txt"), 'w') as outfile:
                    outfile.write(json.dumps(dict_data, indent=4))
                keep_for_now = ''   

def main(prompt_text: str="prompt_text.txt", prompt_dir: str='prompts/', biosample_path :str="biosamples/", entero_file :str ="entero_all_7.7.23.tsv", validate_files=False):
    """
    The main function that orchestrates the fetching of biosamples, fields, and prompt text.

    Args:
        fields_path (str): The path to the CSV file where the biosample fields will be written.
                        Defaults to "biosamples/fields.csv".
        prompt_text (str): The path to the file containing the prompt text.
                        Defaults to "prompt_text.txt".
        prompt_dir (str): The path to the directory where the prompt text files will be written.
                        Defaults to "prompts/".
        biosample_path (str): The path to the directory containing the biosample files.
                            Defaults to "biosamples/".
        entero_file (str): The path to the enterobase table file.
                        Defaults to "entero_all_7.7.23.tsv".
    """
    biosamples_ids = fetch_biosamples(entero_file=entero_file) 
    missing_biosamples_ids = [biosample_id for biosample_id in biosamples_ids if not check_biosample_file_exists(biosample_id)]
    if len(missing_biosamples_ids) > 0:
        fetch_missing_biosamples(missing_biosamples_ids, biosample_path=biosample_path)
    if validate_files:
        check_biosample(biosample_path_ori=biosample_path)

if __name__ == "__main__":
    typer.run(main)