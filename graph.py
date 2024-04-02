import os
import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx
import multiprocessing
import torch
import torch_geometric as pyg

from rdkit import Chem
from rdkit.Chem import rdFreeSASA

from log.logger import Logger

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


'''
one_hot:
    about:
        function for one-hot encoding

    params:
        val = value
        set = set of possible values

    returns:
        one-hot vector with index of val set to 1
'''
def one_hot(val, set):

    if val not in set:
        val = set[0]
    
    return list(map(lambda s: val == s, set))


'''
get_nodes:
    about:
        gets node and node feature tensor for a labeled protein using RDKit to determine relevant physiochemical features

    params:
        protein_chain = RDKit mol data file for protein chain

    returns:
        x = torch tensor of node and node features
'''
def get_nodes(protein_chain):

    G = nx.Graph()

    atoms = protein_chain.GetAtoms() # iterate over all atoms in protein chain

    ptable = Chem.GetPeriodicTable()
    vdw_radii = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in protein_chain.GetAtoms()] # get Van der Waals radii for all atoms in protein chain
    rdFreeSASA.CalcSASA(protein_chain, vdw_radii) # compute solvent-accessible surface area (SASA) for all atoms in protein

    sasa = []
    surface_accessible = []

    for i in range(len(atoms)):

        atom = atoms[i]
        atom_index = atom.GetIdx()

        # ELEMENT SYMBOL [other, C, N, O, P, S, F, CL, Br, I]
        element_symbol = one_hot(atom.GetSymbol(), ['Other','C','N','O','P','S'])
        
        # DEGREE [0, 1, 2, 3, 4, 5, 6, 7]
        degree = one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7])

        # FORMAL CHARGE [0, 1, 2, -1, -2]
        formal_charge = one_hot(atom.GetFormalCharge(), [0, 1, 2, -1, -2])
        
        # RADICAL ELECTRONS [0, >=1]
        radical_electrons = one_hot(atom.GetNumRadicalElectrons(), [0, 1])

        # IMPLICIT VALENCE [0, 1, 2, 3, 4, 5, 6]
        implicit_valence = one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])

        # IMPLICIT HYDROGENS [0, 1, 2, 3, 4]
        implicit_hydrogens = one_hot(atom.GetNumImplicitHs(), [0, 1, 2, 3, 4])

        # HYBRIDIZATION [SP, SP2, SP3, SP3D, SP3D2]
        hybridization = one_hot(atom.GetHybridization(), [Chem.HybridizationType.SP,
                                                          Chem.HybridizationType.SP2,
                                                          Chem.HybridizationType.SP3,
                                                          Chem.HybridizationType.SP3D,
                                                          Chem.HybridizationType.SP3D2])
        
        # AROMATIC [0/1]
        aromatic = atom.GetIsAromatic()

        # SASA [FLOAT] (NON-FEATURE TENSOR)
        a = float(atom.GetProp('SASA'))
        sasa.append(a)

        # SURFACE ACCESSIBLE [0/1] (NON-FEATURE TENSOR)
        surface_accessible.append(a > 0)

        features = np.hstack((element_symbol,
                              degree,
                              formal_charge,
                              radical_electrons,
                              implicit_valence,
                              implicit_hydrogens,
                              hybridization,
                              aromatic)) # combine all 39 features into one vector and add to graph

        G.add_node(atom_index, feats=torch.from_numpy(features))

    x = torch.stack([feats['feats'] for n, feats in G.nodes(data=True)]).float() # convert to tensor
    sasa = torch.tensor(sasa) # conver to tensor
    surface_accessible = torch.tensor(surface_accessible) # convert to tensor

    return x, sasa, surface_accessible


'''
get_edges:
    about:
        gets edge and edge feature tensor for a protein using RDKit to determine relevant physiochemical features

    params:
        protein_chain = RDKit mol data file for protein chain
        radius_ncov = cutoff radius for determining non-covalent edges around an atom in Angstroms

    returns:
        edge_index = torch tensor of edge indices
        edge_attr = torch tensor of edge features
'''
def get_edges(protein_chain, radius_ncov, ncov_within_residue):

    G = nx.Graph()

    pos = protein_chain.GetConformers()[0].GetPositions() # get atomic coordinates
    dist_matrix = sp.spatial.distance_matrix(pos, pos) # calculate distances between all atoms

    # GET COVALENT INTERACTIONS
    for bond in protein_chain.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        G.add_edge(i, j, type=1, dist=dist_matrix[i, j])

    # GET NON-COVALENT INTERACTIONS
    ncov_idx = np.where((dist_matrix <= radius_ncov))

    for i, j in zip(ncov_idx[0], ncov_idx[1]):
        i = int(i)
        j = int(j)

        if ncov_within_residue == False:
            atom_i = protein_chain.GetAtomWithIdx(i)
            atom_j = protein_chain.GetAtomWithIdx(j)
            if atom_i.GetPDBResidueInfo().GetResidueNumber() != atom_j.GetPDBResidueInfo().GetResidueNumber(): # omit noncovalent edges within a residue
                if (not protein_chain.GetBondBetweenAtoms(i, j)) and (i != j): # omit covalent edges
                    G.add_edge(i, j, type=0, dist=dist_matrix[i, j])
        else:
            if (not protein_chain.GetBondBetweenAtoms(i, j)) and (i != j): # omit covalent edges
                G.add_edge(i, j, type=0, dist=dist_matrix[i, j])
    
    G = G.to_directed()
    
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in G.edges(data=False)]).T # convert to tensor
    edge_attr = torch.stack([torch.FloatTensor((a['type'], a['dist'])) for _, _, a in G.edges(data=True)]) # convert to tensor

    return edge_index, edge_attr


'''
convert_to_graph:
    about:
        creates .pyg torch graph Data object for a single protein_path;
        atoms are nodes with relevant node features;
        covalent and non-covalent interactions are edges with relevant edge features

    params:
        protein_path = path to labeled protein as .pdb (using b-factor column)
        labels_path = path to atom labels as .npy
        graph_path = path to torch graph as .pyg
        radius_ncov = cutoff radius for non-covalent edges (Angstroms)

    returns: none
'''
def convert_to_graph(protein_path, labels_path, graph_path, radius_ncov, ncov_within_residue, output_dir):

    protein_chain = Chem.rdmolfiles.MolFromPDBFile(protein_path, sanitize=True, removeHs=True, proximityBonding=False)
    labels = np.load(labels_path)
    
    # get nodes and node features, edges and edge features, node labels, and node coordinates
    x, sasa, surface_accessible = get_nodes(protein_chain)
    edge_index, edge_attr = get_edges(protein_chain, radius_ncov, ncov_within_residue)
    y = torch.FloatTensor(labels)
    pos = torch.FloatTensor(protein_chain.GetConformers()[0].GetPositions())

    # cast graph components
    x = x.type(torch.FloatTensor)
    edge_index = edge_index.type(torch.LongTensor)
    edge_attr = edge_attr.type(torch.FloatTensor)
    y = y.type(torch.LongTensor)
    pos = pos.type(torch.FloatTensor)
    sasa = sasa.type(torch.FloatTensor)
    surface_accessible = surface_accessible.type(torch.BoolTensor)

    # calculate graph shape
    num_nodes = int(x.size()[0])
    num_node_features = int(x.size()[1])
    num_edges = int(edge_attr.size()[0])
    num_edge_features = int(edge_attr.size()[1])
    num_dimensions = int(pos.size()[1])

    # create and save graph
    graph = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos,
                          sasa=sasa, surface_accessible=surface_accessible,
                          num_nodes=num_nodes, num_node_features=num_node_features,
                          num_edges=num_edges, num_edge_features=num_edge_features,
                          num_dimensions=num_dimensions,
                          graph_path=graph_path, protein_path=protein_path, labels_path=labels_path, radius_ncov=radius_ncov)

    torch.save(graph, graph_path)
    
    logger = Logger(os.path.join(output_dir, 'graph.log'))
    logger.info(f"[{graph_path}]: graph = {graph}")


'''
graph:
    about:
        creates .pyg torch graph Data object for all protein_paths in data_path using multiprocessing

    params:
        data_path = path to dataframe of all processed paths and parameters as .csv
        output_dir = output directory
        radius_ncov = cutoff radius for non-covalent edges (Angstroms)
        num_process = num_workers for multiprocessing

    returns: none
'''
def graph(data_path, output_dir, radius_ncov=10, ncov_within_residue=True, num_process=4):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # log parameters
    logger = Logger(os.path.join(output_dir, 'graph.log'))

    logger.critical("GRAPH PARAMETERS")
    logger.info(f"data_path = {data_path}")
    logger.info(f"output_dir = {output_dir}")
    logger.info(f"radius_ncov = {radius_ncov}")
    logger.info(f"ncov_within_residue = {ncov_within_residue}")
    logger.info(f"num_process = {num_process}")

    # get data from data_path
    data_df = pd.read_csv(data_path, sep=',', header=0, dtype=str)
    
    pro_id_list = data_df['pro_id'].tolist()
    pro_dir_list = data_df['pro_dir'].tolist()
    pro_path_list = data_df['pro_path'].tolist()
    lab_path_list = data_df['lab_path'].tolist()
    pdb_path_list = data_df['pdb_path'].tolist()
    radius_pocket_list = data_df['radius_pocket'].tolist()
    expand_pockets_by_res_list = data_df['expand_pockets_by_res'].tolist()
    include_ligand_list = data_df['include_ligand'].tolist()
    graph_path_list = []
    radius_ncov_list = []
    ncov_within_residue_list = []
    output_dir_list = []
    
    logger.critical("GRAPH PROPERTIES")
    logger.info(f"LEN pro_id_list = {len(pro_id_list)}")

    for pro_dir in pro_dir_list:
        graph_path_list.append(os.path.join(pro_dir, "graph.pyg"))
        radius_ncov_list.append(radius_ncov)
        ncov_within_residue_list.append(ncov_within_residue)
        output_dir_list.append(output_dir)

    # start graph
    logger.critical("GRAPH START")
    pool = multiprocessing.Pool(num_process)
    pool.starmap(convert_to_graph, zip(pro_path_list, lab_path_list, graph_path_list, radius_ncov_list, ncov_within_residue_list, output_dir_list))
    pool.close()
    pool.join()

    logger.critical("GRAPH END")

    # export data
    output_df = pd.DataFrame({'pro_id': pro_id_list,
                              'pro_dir': pro_dir_list,
                              'pro_path': pro_path_list,
                              'lab_path': lab_path_list,
                              'pdb_path': pdb_path_list,
                              'radius_pocket': radius_pocket_list,
                              'expand_pocket_by_res': expand_pockets_by_res_list,
                              'include_ligand': include_ligand_list,
                              'graph_path': graph_path_list,
                              'radius_ncov': radius_ncov_list,
                              'ncov_within_residue': ncov_within_residue_list})
    
    output_path = os.path.join(output_dir, f"graphed.csv")
    output_df.to_csv(output_path)
    logger.info(f"output_path = {output_path}")
    logger.info(f"LEN output_df = {len(output_df)}")


if __name__ == '__main__':

    # graph parameters
    data_path = "data/dataset/preprocessed.csv"
    output_dir = "data/dataset"

    # run graph
    graph(data_path, output_dir, radius_ncov=10, ncov_within_residue=False, num_process=16)
