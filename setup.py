import os
import requests
import pandas as pd


'''
get_pdb:
    about:
        download pdb_id PDB from RCSB and save to output_dir

    params:
        pdb_id = 4-character PDB ID
        output_dir = output directory

    returns:
        path = path to pdb_id
'''
def get_pdb(pdb_id, output_dir, verbose=False):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data = requests.get(f"http://files.rcsb.org/download/{pdb_id.lower()}.pdb")

    path = os.path.join(output_dir, f"{pdb_id}.pdb")
    with open(path, 'wb') as pdb:
        pdb.write(data.content)

    if verbose:
        print(f"[{pdb_id}]: downloaded PDB from RCSB to [{path}]")

    return path


'''
preprocess:
    about:
        downloads PDBs in pdb_id_list from RCSB to output_dir
    
    params:
        pro_id_list = list of 4-character PDB IDs with chain identifiers for data set
        pdb_dir = directory to .pdb records
        
    returns: none
'''
def setup(pro_id_list, pdb_dir):
    
    pro_id_list_unique = [] # remove duplicate pdb_ids
    for pdb_id in pro_id_list:
        if pdb_id not in pro_id_list_unique:
            pro_id_list_unique.append(pdb_id)

    for count, pdb_id in enumerate(pro_id_list_unique):
        try:
            print(f"\n[{pdb_id.upper()}]: {count+1} of {len(pro_id_list_unique)} proteins")
            get_pdb(pdb_id=pdb_id, output_dir=pdb_dir, verbose=False) # download PDB
            
        except Exception as e: 
            print(f"ERROR for [{pdb_id}]: {e}")
            continue


# download PDB files
if __name__ == '__main__':

    pro_data_path = "data/raw/pro/all_structures_cluster_goodfit_surfnet_features_mod.csv"
    pdb_dir = "data/pdb"

    # get pro_id_list = list of 4-character PDB IDs with chain identifiers for data set
    pro_df = pd.read_csv(pro_data_path, sep=',', header=0, dtype=str)
    pro_id_list = list(map(str, pro_df['pdbid_chain'].unique()))
    
    pdb_id_list = [x.split('_')[0].lower() for x in pro_id_list] # remove chain identifiers from pro_id_list

    setup(pro_id_list, pdb_dir)
