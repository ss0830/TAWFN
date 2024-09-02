import torch
from torch_geometric.data import Data, Dataset, Batch
import csv
import numpy as np
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser

RES2ID = {
    'A':0,
    'R':1,
    'N':2,
    'D':3,
    'C':4,
    'Q':5,
    'E':6,
    'G':7,
    'H':8,
    'I':9,
    'L':10,
    'K':11,
    'M':12,
    'F':13,
    'P':14,
    'S':15,
    'T':16,
    'W':17,
    'Y':18,
    'V':19,
    '-':20
}
def aa2idx(seq):
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx

# def load_GO_annot(filename):
#     # Load GO annotations
#     onts = ['mf', 'bp', 'cc']
#     prot2annot = {}
#     goterms = {ont: [] for ont in onts}
#     gonames = {ont: [] for ont in onts}
#     with open(filename, mode='r') as tsvfile:
#         reader = csv.reader(tsvfile, delimiter='\t')
#
#         # molecular function
#         next(reader, None)  # skip the headers
#         goterms[onts[0]] = next(reader)
#         next(reader, None)  # skip the headers
#         gonames[onts[0]] = next(reader)
#
#         # biological process
#         next(reader, None)  # skip the headers
#         goterms[onts[1]] = next(reader)
#         next(reader, None)  # skip the headers
#         gonames[onts[1]] = next(reader)
#
#         # cellular component
#         next(reader, None)  # skip the headers
#         goterms[onts[2]] = next(reader)
#         next(reader, None)  # skip the headers
#         gonames[onts[2]] = next(reader)
#
#         next(reader, None)  # skip the headers
#         counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
#         for row in reader:
#             prot, prot_goterms = row[0], row[1:]
#             # if len(row) < 1:
#             #     continue
#             # prot  = row[0]
#             prot2annot[prot] = {ont: [] for ont in onts}
#
#             for i in range(3):
#                 # prot_goterms = row[i + 1] if len(row) > (i + 1) else ''
#                 goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
#                 prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
#                 prot2annot[prot][onts[i]][goterm_indices] = 1.0
#                 counts[onts[i]][goterm_indices] += 1.0
#
#                 # # print(f"prot_goterms[{i}]:", prot_goterms)  # 输出 prot_goterms 的值
#
#                 # for goterm in prot_goterms.split(','):
#                 #
#                 #     if goterm != '':
#                 #         if goterm in goterms[onts[i]]:
#                 #             goterm_indices = goterms[onts[i]].index(goterm)
#                 #             prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
#                 #             prot2annot[prot][onts[i]][goterm_indices] = 1.0
#                 #             counts[onts[i]][goterm_indices] += 1.0
#                 #         else:
#                 #             print(f"Warning: GO term '{goterm}' not found in list.")
#                 #     else:
#                 #         prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
#
#     return prot2annot, goterms, gonames, counts
def load_GO_annot(filename):
    # Load GO annotations
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts

def log(*args):
    print(f'[{datetime.now()}]', *args)
