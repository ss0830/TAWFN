import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import load_GO_annot
import numpy as np
import os
from utils import aa2idx
import sys

def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()

class GoTermDataset(Dataset):

    def __init__(self, set_type, task):
        self.task = task
        if set_type != 'AF2test':
            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
        else:
            prot2annot, goterms, gonames, counts = load_GO_annot('data/nrSwiss-Model-GO_annot.tsv')

        self.processed_dir = 'data/processed'

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt")) 
        if set_type == 'AF2test':
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))["test_pdbch"]
        else:
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        self.y_true = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list])
        self.y_true = torch.tensor(self.y_true)

        prot2annot1, goterms1, gonames1, counts1 = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")

        graph_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_graph.pt"))
        self.graph_list += graph_list_af
        self.pdbch_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
        y_true_af = np.stack([prot2annot1[pdb_c][self.task] for pdb_c in self.pdbch_list_af])

        self.y_true = np.concatenate([self.y_true, y_true_af],0)
        self.y_true = torch.tensor(self.y_true)


    def __getitem__(self, idx):

        return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)
