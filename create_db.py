import os
import json
import logging
import csv 
import typer
import pandas as pd

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def biosample_fields(biosample_path="biosamples/"):
    """
    Returns a dictionary of biosample fields and their first record where seen.

    Args:
        biosample_path (str): The path to the directory containing the biosample files. Defaults to "biosamples/".

    Returns:
        dict: A dictionary where the keys are the biosample field names and the values are the accession numbers
        of the first record where the field was seen.

    Raises:
        FileNotFoundError: If the specified biosample_path does not exist.
        JSONDecodeError: If a biosample file cannot be parsed as JSON.

    Notes:
        This function searches for biosample files in the specified directory and its subdirectories.
        It only considers files with the ".txt" extension.

    Examples:
    >>> biosample_fields("path/to/biosamples/")
    {'organism': 'SAMN00000001', 'sex': 'SAMN00000002', 'age': 'SAMN00000003'}
    """
    fields = {} # key: field, value: first record where seen
    for sub_dir_path in [os.path.join(biosample_path,subdir) for subdir in os.listdir(biosample_path) if os.path.isdir(os.path.join(biosample_path,subdir))]:
        for biosample_path in [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if f.endswith(".txt")]:
            with open(biosample_path) as f:
                try:
                    dict_data = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    logging.error(f"Could not parse {biosample_path}: {e}") 
                    continue
                accession = dict_data.get('@accession')
                if not accession:
                    logging.error(f"Could not find accession in {biosample_path}")
                    continue
                if dict_data.get("Attributes"):
                    for attribute in dict_data.get("Attributes", {}).get("Attribute", []):
                        if isinstance(attribute, str):
                            attribute = dict_data.get("Attributes", {}).get("Attribute", [])
                        field_name = attribute.get("@harmonized_name", attribute.get("@attribute_name")) 
                        if not field_name:
                            logging.error(f"Could not find attribute name in {biosample_path}")
                            continue
                        if not fields.get(field_name):
                            fields[field_name] = accession
    return fields

def output_fields(fields, fields_path="biosamples/fields.csv"):
    """
    Writes the biosample fields and their first record where seen to a CSV file.

    Args:
        fields (dict): A dictionary where the keys are the biosample field names and the values are the accession numbers
                    of the first record where the field was seen.
        fields_path (str): The path to the CSV file where the fields will be written.
                        Defaults to "biosamples/fields.csv".
    """
    os.makedirs(os.path.dirname(fields_path), exist_ok=True)
    with open(fields_path, "w") as f:
        # write header
        f.write('field,first_seen\n')
        # write field and accession
        for field, acc in fields.items():
            f.write(f'{field},{acc}\n')

def create_attribute_csv(fields_path, biosample_path, output_table='all_attributes.csv'):
    """
    Creates a CSV file containing the biosample attributes.

    Args:
        fields_path (str): The path to the file containing the fields to extract.
        biosample_path (str): The path to the directory containing the biosample files.
        output_table (str): The name of the output CSV file. Defaults to 'all_attributes.csv'.

    Notes:
        This function extracts the attributes from the biosample files and writes them to a CSV file.
        The CSV file contains one row for each biosample file.
        The columns of the CSV file correspond to the fields specified in the fields file.
        If a field is not present in a biosample file, the corresponding value in the CSV file is set to None.

    Examples:
        >>> create_attribute_csv('fields.txt', 'biosamples/', 'attributes.csv')
    """
    all_attributes = [] 
    fields = [line.split(',')[0].strip() for line in open(fields_path).readlines()[1:]]
    for sub_dir_path in [os.path.join(biosample_path, subdir) for subdir in os.listdir(biosample_path) if os.path.isdir(os.path.join(biosample_path,subdir))]:
        for biosample_file_path in [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if f.endswith(".txt")]:
            with open(biosample_file_path) as f:
                try:
                    dict_data = json.load(f)
                    # make a new dictionary with all the keys from fields using the data from dict_data, if a field is not present, use None
                    if dict_data.get("Attributes", {}):
                        attributes = dict_data.get("Attributes", {}).get("Attribute", {})
                        new_record = {field: None for field in fields}
                        for attribute in attributes:
                            if isinstance(attribute, str):
                                attribute = dict_data.get("Attributes", {}).get("Attribute", [])                        
                            field_name = attribute.get("@harmonized_name", attribute.get("@attribute_name"))
                            if field_name in fields:
                                new_record[field_name] = attribute.get("#text")
                        all_attributes.append(new_record)
                except json.decoder.JSONDecodeError as e:
                    logging.error(f"Could not parse {biosample_path}: {e}") 
                    continue
    with open(output_table, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_attributes)

def get_data(output_table):
    """
    Returns a pandas dataframe with the biosample data.

    Returns:
        pandas.DataFrame: A dataframe with the biosample data.
    """
    df =  pd.read_csv(output_table)
    # Fetch all values for geo_loc_name
    geo_loc_name = []
    lat_lon = []
    collection_date = []
    host = []
    for i in range(len(df)):
        geo_loc_name.append(df['geo_loc_name'][i])
        lat_lon.append(df['lat_lon'][i])
        collection_date.append(df['collection_date'][i])
        host.append(df['host'][i])
    return df




def main(biosample_path:str="biosamples/", fields_path:str="biosamples/fields.csv", output_table:str='all_attributes.csv'):
    # fields = biosample_fields(biosample_path=biosample_path)
    # output_fields(fields, fields_path=fields_path)
    # create_attribute_csv(fields_path, biosample_path, output_table=output_table)
    get_data(output_table)

if __name__ == "__main__":
    typer.run(main)