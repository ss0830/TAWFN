import csv
import pickle
import obonet
import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score as aupr
import utils
from matplotlib import pyplot as plt

plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('font', family='arial')

# go.obo
go_graph = obonet.read_obo(open("data/go-basic.obo", 'r'))

''' helper functions follow '''
def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi=False):
    '''
    evaluate a set of protein predictions using normalized semantic distance
    value of 0 means perfect predictions, larger values denote worse predictions,
    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, predicted binary label ndarray (not compressed). Must have hard predictions (0 or 1, not posterior probabilities)
        termIC: output of ic function above
    OUTPUT:
        depending on returnRuMi and avg. To get the average sd over all proteins in a batch/dataset
        use avg = True and returnRuMi = False
        To get result per protein, use avg = False
    '''

    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)

    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru ** 2 + mi ** 2)

    if not returnRuMi:
        return sd

    return [ru, mi, sd]

def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom

    if avg:
        nru = np.mean(nru)
    return nru

def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num = np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom = np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom

    if avg:
        nmi = np.mean(nmi)

    return nmi

def _cafa_go_aupr(labels, preds,task):
    # propagate goterms (take into account goterm specificity)

    # number of test proteins
    n = labels.shape[0]
    _, goterms, _, _ = utils.load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
    goterms = goterms[task]
    goterms = np.asarray(goterms)
    ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}

    prot2goterms = {}
    for i in range(0, n):
        all_gos = set()
        for goterm in goterms[np.where(labels[i] == 1)[0]]:
            all_gos = all_gos.union(nx.descendants(go_graph, goterm))
            all_gos.add(goterm)
        all_gos.discard(ont2root[task])
        prot2goterms[i] = all_gos

    # CAFA-like F-max predictions
    F_list = []
    AvgPr_list = []
    AvgRc_list = []
    thresh_list = []

    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int)

        m = 0
        precision = 0.0
        recall = 0.0
        for i in range(0, n):
            pred_gos = set()
            for goterm in goterms[np.where(predictions[i] == 1)[0]]:
                pred_gos = pred_gos.union(nx.descendants(go_graph,
                                                         goterm))
                pred_gos.add(goterm)
            pred_gos.discard(ont2root[task])

            num_pred = len(pred_gos)
            num_true = len(prot2goterms[i])
            num_overlap = len(prot2goterms[i].intersection(pred_gos))
            if num_pred > 0 and num_true > 0:
                m += 1
                precision += float(num_overlap) / num_pred
                recall += float(num_overlap) / num_true

        if m > 0:
            AvgPr = precision / m
            AvgRc = recall / n

            if AvgPr + AvgRc > 0:
                F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                # record in list
                F_list.append(F_score)
                AvgPr_list.append(AvgPr)
                AvgRc_list.append(AvgRc)
                thresh_list.append(threshold)

    F_list = np.asarray(F_list)
    AvgPr_list = np.asarray(AvgPr_list)
    AvgRc_list = np.asarray(AvgRc_list)
    thresh_list = np.asarray(thresh_list)

    return AvgRc_list, AvgPr_list, F_list, thresh_list

def _function_centric_aupr(Y_true,Y_pred):
    """ Compute functon-centric AUPR """

    # micro average
    micro_aupr = aupr(Y_true, Y_pred, average='micro')
    # macro average
    macro_aupr = aupr(Y_true, Y_pred, average='macro')

    # each function
    aupr_goterms = aupr(Y_true, Y_pred, average=None)

    return micro_aupr, macro_aupr, aupr_goterms
def _function_centric_aupr_test(Y_true,Y_pred):
    """ Compute functon-centric AUPR """
    keep_goidx = np.where(Y_true.sum(axis=0) > 0)[0]

    print("### Number of functions =%d" % (len(keep_goidx)))

    Y_true = Y_true[:, keep_goidx]
    Y_pred = Y_pred[:, keep_goidx]

    # if self.method_name.find('FFPred') >= 0:
    #    goidx = np.where(Y_pred.sum(axis=0) > 0)[0]
    #    Y_true = Y_true[:, goidx]
    #    Y_pred = Y_pred[:, goidx]

    # micro average
    micro_aupr = aupr(Y_true, Y_pred, average='micro')
    # macro average
    macro_aupr = aupr(Y_true, Y_pred, average='macro')

    # each function
    aupr_goterms = aupr(Y_true, Y_pred, average=None)

    return micro_aupr, macro_aupr, aupr_goterms

def _protein_centric_fmax(Y_true,Y_pred,task):
    """ Compute protein-centric AUPR """
    # compute recall/precision
    Recall, Precision, Fscore, thresholds = _cafa_go_aupr(Y_true,
                                                        Y_pred,task)
    return Fscore, Recall, Precision, thresholds

def fmax(Y_true,Y_pred,task):
    fscore, _, _, _ = _protein_centric_fmax(Y_true,Y_pred,task)
    return max(fscore)

def macro_aupr(Y_true,Y_pred):
    _, macro_aupr, _ = _function_centric_aupr(Y_true,Y_pred)
    return macro_aupr
def macro_aupr_test(Y_true,Y_pred):
    _, macro_aupr, _ = _function_centric_aupr_test(Y_true,Y_pred)
    return macro_aupr

def smin(termIC, Y_true,Y_pred):
    '''
    get the minimum normalized semantic distance
    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrThresholds: the number of thresholds to check.
    OUTPUT:
        the minimum nsd that was achieved at the evaluated thresholds
    '''
    nrThresholds = 100
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ss = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        ss[i] = normalizedSemanticDistance(Y_true, (Y_pred >= t).astype(int), termIC, avg=True,
                                           returnRuMi=False)

    return np.min(ss)