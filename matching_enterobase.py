import typer
import pandas as pd
import os 
import textwrap

eb_host_categories = list(set([
    'Fish', 'Marine Mammal', 'Shellfish', 'Canine', 'Feline', 'Air', 'Plant', 'Soil/Dust', 
    'Water', 'Animal Feed', 'Meat', 'Composite Food', 'Dairy', 'Fish', 'Meat', 'Shellfish', 
    'Human', 'Laboratory', 'Bovine', 'Camelid', 'Equine', 'Ovine', 'Swine', 'Avian', 
    'Amphibian', 'Avian', 'Bat', 'Bovine', 'Camelid', 'Canine', 'Deer', 'Equine', 'Feline', 
    'Invertebrates', 'Marsupial', 'Other Mammal', 'Ovine', 'Primate', 'Reptile', 'Rodent', 
    'Swine', 'Not determined'
]))

eb_host_niche = list(set([
    'Aquatic',
    'Companion Animal',
    'Environment',
    'Feed',
    'Food',
    'Human',
    'Laboratory',
    'Livestock',
    'Poultry',
    'Wild Animal',
    'Not determined'
]))



def fetch_eb_records(entero_file="entero_all_7.7.23.tsv.gz", records=1000):
    """
    Fetches 1000 records from enterobase_all.7.7.23.tsv.gz
    """
    df = pd.read_csv(entero_file, compression='gzip', sep='\t')
    # Filter where Sample ID is not null
    df = df[df['Sample ID'].notnull() & df['Source Type'].notnull()]
    eb_records = df.sample(n=records)
    eb_records = eb_records[['Sample ID', 'Source Type', 'Source Niche']]
    biosample_ids = eb_records['Sample ID'].tolist()
    return biosample_ids, eb_records



def fetch_field_list(fields="fields.csv"):
    """
    Fetches list of fields from fields.csv
    """
    # desired fields 
    wanted_fields = [
    'collected_by', 
    'isolation_source',
    'attribute_package',
    'ontological term',
    'IFSAC+ Category',
    'source_type',
    'sample_type',
    'culture_collection',
    'package',
    'attribute package',
    'IFSAC+ category',
    'food_origin',
    'disease'
    ]
    # open fields.txt
    with open(fields, 'r') as f:
        all_field_list = f.readlines()[1:]
        # remove new line characters and remove empty strings
        clean_field_list = [field.split(",")[0].strip() for field in all_field_list  if field]
        field_list = [field.split(",")[0].strip() for field in clean_field_list  if field in wanted_fields or field.startswith("host")]
    return field_list
    

def fetch_host_info(biosample_ids, data_table, field_list):
    """
    Fetches host information from all_attributes.csv.gz
    """
    biosample_details = []
    null_values = ['missing', 'not available', 'not collected', 'not applicable', 'not provided', 'not reported', 'unknown']
    # open all_attributes.csv.gz
    df = pd.read_csv(data_table, compression='gzip')
    # Filter where Sample ID is in biosample_ids
    df = df[df['Sample ID'].isin(biosample_ids)]
    df = df.reset_index()
    # fetch dictionary of biosample details using field_list
    for i in range(len(df)):
        new_biosample = { 'Sample ID' : df['Sample ID'][i]}
        for field in field_list:
            if not pd.isna(df[field][i]):
                if not df[field][i].lower() in null_values:   
                    new_biosample[field] = df[field][i]        
        biosample_details.append(new_biosample)
    return biosample_details

def create_prompt(biosample_details, eb_prompt_dir):
    """
    Creates a prompt text file for a biosample.
    """
    if not os.path.exists(eb_prompt_dir):
        os.makedirs(eb_prompt_dir)
    for biosample in biosample_details:
        biosample_string = ','.join([str(key) + ':' + str(value) for key, value in biosample.items()])
        prompt = "Given the ncbi biosample information. {}. Assign the record to one of the following host categories: {}. Only reply with the category".format(biosample_string, ','.join(eb_host_categories))
        with open("{}/{}.txt".format(eb_prompt_dir, biosample['Sample ID']), 'w') as f:
            # Word wrap prompt to 80 characters
            prompt = textwrap.fill(prompt, width=80)
            f.write(prompt)    
    return prompt
                                                                                                                             

def main(fields:str="fields.csv", data_table:str="all_attributes.csv.gz", entero_file:str ="entero_all_7.7.23.tsv.gz", num_records:int=100, eb_prompt_dir:str="eb_prompts"):
    # Pick 1000 enterobase records from enterobase_all.7.7.23.tsv
    biosample_ids, eb_records = fetch_eb_records(entero_file=entero_file, records=num_records)
    # Fetch list of fields and extract valid fields to use for host 
    field_list = fetch_field_list(fields)
    # Fetch corresponging host information from all_attributes.csv.gz
    input_data = fetch_host_info(biosample_ids, data_table, field_list)
    # Generate prompt text file for each biosample record
    create_prompt(input_data, eb_prompt_dir)

if __name__ == "__main__":
    typer.run(main)

