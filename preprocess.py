import os
import time
import numpy as np
import pandas as pd
import multiprocessing

import pymol
from rdkit import Chem

from log.logger import Logger

np.set_printoptions(threshold=np.inf)


'''
label_chain_around_ligand:
    about:
        label druggable protein cavity regions around drug-like ligands for a single protein

    params:
        pdb_path = path to input protein as .pdb
        pro_id = protein ID (PDB ID and chain letter)
        pro_dir = directory to save labeled protein and atom labels
        lig_id_list = list of drug-like ligands
        het_id_list = list of other non-protein molecules to include in labeled protein structures
        logger = preprocess logger

        radius_pocket = cutoff radius for labeling a cavity region around a ligand (Angstroms)
        expand_pockets_by_res = whether entire parent residue is included in a cavity region or not
        include_ligand = whether ligand is included or removed in labeled protein structure

    returns:
        pro_path = path to labeled protein as .pdb (using b-factor column)
        lab_path = path to atom labels as .npy
        exit_code = exit code if labeling is completed successfully
'''
def label_chain_around_ligand(pdb_path, pro_id, pro_dir, lig_id_list, het_id_list, logger,
                              radius_pocket,
                              expand_pockets_by_res,
                              include_ligand):

    pdb_id = pro_id.split('_')[0].upper()
    chn_id = pro_id.split('_')[1].upper()
    
    pymol.cmd.feedback('disable', 'all', 'actions')
    pymol.cmd.feedback('disable', 'all', 'results')
    
    pymol.cmd.load(os.path.join(os.getcwd(), pdb_path))

    pymol.cmd.remove(f'not chain {chn_id.upper()}') # remove all other chains
    pymol.cmd.remove('hydrogen')
    pymol.cmd.remove("not alt ''+A") # remove all alternate atom records
    pymol.cmd.alter('all', 'b=0.00') # set all b-factors to 0.00 initially

    # identify which lig_ids are present in the pro_id .pdb
    lig_id_present = []
    for lig_id in lig_id_list:

        lig_id = lig_id.upper()

        if pymol.cmd.select(f'resn {lig_id}'):
            lig_id_present.append(lig_id)

    if not lig_id_present:
        logger.warning(f"[{pro_id}]: no druggable regions labeled")
        pymol.cmd.delete('all')
        exit_code = 1
        return None, None, exit_code

    # concatenate lig_ids and het_ids, and remove all non-polymer, non-lig_id_present, and non-het_id_list
    keep_str = 'polymer + '
    lig_id_present_str = 'resn ' + ' + resn '.join(lig_id_present)
    keep_str += lig_id_present_str

    if het_id_list:
        keep_str += ' + resn '
        keep_str += ' + resn '.join(het_id_list)

    pymol.cmd.remove(f'not ({keep_str})')

    # label regions around lig_ids within radius_pocket
    if lig_id_present:
        if expand_pockets_by_res == True:
            pymol.cmd.select('druggable', f'byres {lig_id_present_str} around {radius_pocket}')
        else:
            pymol.cmd.select('druggable', f'{lig_id_present_str} around {radius_pocket}')

        pymol.cmd.alter('druggable', 'b=1.00')
        logger.info(f"[{pro_id}]: atoms around ligand(s) {lig_id_present} labeled [1, DRUGGABLE]")

    pymol.cmd.set('pdb_conect_all', 'on') # include CONECT records for all atoms
    pymol.cmd.set('pdb_conect_nodup', '0') # include duplicate CONECT records for multiple bonds
        
    # remove lig_ids according to include_ligand
    if include_ligand == False:
        pymol.cmd.remove(f'{lig_id_present_str}')

    # export cleaned and labeled protein structure and labels
    pro_path = os.path.join(pro_dir, f"protein.pdb")
    pymol.cmd.save(pro_path, 'all')

    pymolspace = {'bfactors': []}
    pymol.cmd.iterate('all', 'bfactors.append(b)', space=pymolspace) # obtain b-factors from structure
    b_factors = np.array(pymolspace['bfactors'], dtype='int')

    lab_path = os.path.join(pro_dir, f"labels.npy")
    np.save(lab_path, b_factors)

    pymol.cmd.delete('all')

    # check that RDKit can load labeled protein without error
    try:
        protein_chain = Chem.rdmolfiles.MolFromPDBFile(pro_path, sanitize=True, removeHs=True, proximityBonding=False)
        protein_chain.GetAtoms()

    except Exception as e:
        logger.error(f"[{pro_id}]: RDKit error reading labeled protein as .pdb [{pro_path}]; error message [{e}]")
        exit_code = 1
        return None, None, exit_code
    
    exit_code = 0
    return pro_path, lab_path, exit_code


'''
preprocess:
    about:
        label druggable protein cavity regions around drug-like ligands for a subset of pro_data_path proteins

    params:
        pro_data_path = path to list of proteins
        lig_data_path = path to list of drug-like ligands
        het_data_path = path to list of other non-protein molecules to include in labeled protein structures

        pdb_dir = directory to .pdb records
        output_dir = output directory

        sample_size = preprocess sample size
        radius_pocket = cutoff radius for labeling a cavity region around a ligand (Angstroms)
        expand_pockets_by_res = whether entire parent residue is included in a cavity region or not
        include_ligand = whether ligand is included or removed in labeled protein structure
        
    returns:
        output_df = dataframe of all processed paths and parameters
        output_path = path to dataframe of all processed paths and parameters as .csv
'''
def preprocess(pro_data_path, lig_data_path, het_data_path,
                  pdb_dir, output_dir,
                  sample_size,
                  radius_pocket=5,
                  expand_pockets_by_res=False,
                  include_ligand=True):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # log parameters
    logger = Logger(os.path.join(output_dir, 'preprocess.log'))

    logger.critical("PREPROCESS PARAMETERS")
    logger.info(f"pro_data_path = {pro_data_path}")
    logger.info(f"lig_data_path = {lig_data_path}")
    logger.info(f"het_data_path = {het_data_path}")
    logger.info(f"pdb_dir = {pdb_dir}")
    logger.info(f"output_dir = {output_dir}")
    logger.info(f"sample_size = {sample_size}")
    logger.info(f"radius_pocket = {radius_pocket}")
    logger.info(f"expand_pockets_by_res = {expand_pockets_by_res}")
    logger.info(f"include_ligand = {include_ligand}")

    logger.critical("PREPROCESS PROPERTIES")
    # PRO_ID
    # get pro_id_list = list of 4-character PDB IDs with chain identifiers for data set
    if os.path.exists(pro_data_path):
        pro_df = pd.read_csv(pro_data_path, sep=',', header=0, dtype=str)
        pro_df_unique = pro_df['pdbid_chain'].unique() # MAY NEED TO MODIFY
        pro_id_list = list(np.random.choice(pro_df_unique, size=sample_size, replace=False))
    else:
        pro_id_list = []

     # check that all pro_id identifiers are valid ([4]_[1] alphanumeric characters)
    for pro_id in pro_id_list:
        if not pro_id.isalnum() and len(pro_id) not in [6] and '_' not in pro_id[4]:
            raise Exception(f"[{pro_id}]: improperly formatted pro_id [{pro_id}]; must be [4]_[1] alphanumeric characters")
        
    logger.info(f"LEN pro_id_list = {len(pro_id_list)}")

    # LIG_ID
    # get lig_id_list = list of ligand het_ids
    if os.path.exists(lig_data_path):
        lig_df = pd.read_csv(lig_data_path, sep='\t', header=0)
        lig_id_list = list(map(str, list(lig_df['HET_ID']))) # MAY NEED TO MODIFY
    else:
        lig_id_list = []

    # check that all lig_id identifiers are valid (1-3 alphanumeric characters)
    for lig_id in lig_id_list:
        if not lig_id.isalnum() and len(lig_id) not in [1,2,3]:
            raise Exception(f"[{lig_id}]: improperly formatted lig_id [{lig_id}]; must be 1-3 alphanumeric characters")

    logger.info(f"LEN lig_id_list = {len(lig_id_list)}")

    # HET_ID
    # get het_id_list = list of het_ids
    if os.path.exists(het_data_path):
        het_df = pd.read_csv(het_data_path, sep='\t', header=0)
        het_id_list = list(map(str, list(het_df['HET_ID']))) # MAY NEED TO MODIFY
    else:
        het_id_list = []
    
    # check that all het_id identifiers are valid (1-3 alphanumeric characters)
    for het_id in het_id_list:
        if not het_id.isalnum() and len(het_id) not in [1,2,3]:
            raise Exception(f"[{het_id}]: improperly formatted het_id [{het_id}]; must be 1-3 alphanumeric characters")
    
    logger.info(f"LEN het_id_list = {len(het_id_list)} ")

    # start preprocess
    logger.critical("PREPROCESS START")

    preprocess_data = []
    count = 1

    for pro_id in pro_id_list:
        start = time.time()
        
        pro_id = pro_id.upper()
        print(f"\n[{pro_id}]: {count} of {len(pro_id_list)} protein chains")
        
        pro_dir = os.path.join(output_dir, pro_id)
        if not os.path.exists(pro_dir):
            os.mkdir(pro_dir)

        pdb_id = pro_id.split('_')[0].upper()
        pdb_path = os.path.join(pdb_dir, f"{pdb_id.lower()}.pdb")
        
        pro_path, lab_path, exit_code = label_chain_around_ligand(pdb_path, pro_id, pro_dir, lig_id_list, het_id_list, logger,
                                                                  radius_pocket,
                                                                  expand_pockets_by_res,
                                                                  include_ligand)

        if exit_code == 0: # if labeling is completed correctly
            preprocess_data.append([pro_id, pro_dir,
                                       pro_path, lab_path, pdb_path,
                                       radius_pocket,
                                       expand_pockets_by_res,
                                       include_ligand])
            
        elif exit_code == 1: # if error occurs in labeling
            logger.error(f"[{pro_id}]: error in label_chain_around_ligand; ignoring this .pdb")
        
        print(f"[{pro_id}]: completed in {np.round(time.time()-start,4)} seconds")
        count += 1

    logger.critical("PREPROCESS END")
    
    # export data
    output_df = pd.DataFrame(preprocess_data, columns=['pro_id','pro_dir',
                                                          'pro_path','lab_path','pdb_path',
                                                          'radius_pocket',
                                                          'expand_pockets_by_res',
                                                          'include_ligand'])
    
    output_path = os.path.join(output_dir, f"preprocessed.csv")
    output_df.to_csv(output_path)
    logger.info(f"output_path = {output_path}")
    logger.info(f"LEN output_df = {len(output_df)}")

    return output_df, output_path


if __name__ == '__main__':
    
    # preprocess parameters
    pro_data_path = "data/pro/all_structures_cluster_goodfit_surfnet_features_mod.csv"
    lig_data_path = "data/lig/druglike_ligands.tsv"
    het_data_path = ""

    pdb_dir = "data/pdb"
    output_dir = "data/dataset"

    # run preprocess
    preprocess(pro_data_path, lig_data_path, het_data_path,
               pdb_dir, output_dir,
               sample_size=2243,
               radius_pocket=4,
               expand_pockets_by_res=False,
               include_ligand=False)
