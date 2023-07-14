import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def create_dataframe(entero_file, data_table, training_num_records=100, weight_field='Source Type', equal_sampling=True, random=False):
    """
    Fetches biosample data from two input files and returns a dictionary of training and testing data.

    Args:
        entero_file (str): Path to the enterobase file.
        data_table (str): Path to the all_attributes file.
        training_num_records (int): Number of training records to fetch.
        equal_sampling (bool, optional): Whether to sample each type equally. Defaults to True.
        weight_field (str, optional): Field to use for weighting. Defaults to 'Source Type'.

    Returns:
        tuple: A tuple of two dictionaries containing training and testing data.
    """    
    df = pd.read_csv(entero_file, compression='gzip', sep='\t')
    logging.info(f"Number of records in enterobase file: {len(df)}")
    # Filter where Sample ID is not null
    df = df[df['Sample ID'].notnull() & df['Source Type'].notnull()]
    if random:
        eb_records = df[['Sample ID', 'Source Type', 'Source Niche']].sample(n=(training_num_records), random_state=7)
    else:
        if equal_sampling:
            df['freq'] = 1./df.groupby(weight_field)[weight_field].transform('count') # Sample each type equally 
        else: 
            # Subsample in the proportion of full df
            df['freq'] = df.groupby('Source Type')['Source Type'].transform('count')
        eb_records = df[['Sample ID', 'Source Type', 'Source Niche']].sample(n=(training_num_records), random_state=7, weights=df.freq)
    eb_records_dict = {x['Sample ID']: x for x in  eb_records.to_dict('records') } 
    null_values = ['missing', 'not available', 'not collected', 'not applicable', 'not provided', 'not reported', 'unknown']
    # open all_attributes.csv.gz
    df = pd.read_csv(data_table, compression='gzip')
    logging.info(f"Number of records in all_attributes file: {len(df)}")
    # fetch dictionary of biosample details using field_list
    training_data_table = {} 
    for i in range(len(df)):
        accession = df['Sample ID'][i]
        if eb_records_dict.get(accession):
            new_biosample = { 'Sample ID' : accession}
            new_biosample['EB Source Type'] = eb_records_dict[accession]['Source Type']
            new_biosample['EB Source Niche'] = eb_records_dict[accession]['Source Niche']
            field_list = df.columns.tolist()
            for field in field_list:
                if not pd.isna(df[field][i]):
                    if not str(df[field][i]).lower() in null_values:   
                        new_biosample[field] = str(df[field][i])
            new_biosample.pop('index', None)                        
            training_data_table[accession] = new_biosample
    return training_data_table 
