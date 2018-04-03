import read_text as rt
import numpy as np

# get ranking coefficent between these rankings
def compareRankings(rank1, rank2):
    score = 0
    return score

def evaluateCellState(cluster_rankings, cell_state, ppmi_values):
    rank_ppmi = compareRankings(cluster_rankings, ppmi_values)
    print("Rank on PPMI", rank_ppmi)
    cell_ppmi = compareRankings(cell_state, ppmi_values)
    print("Cell on PPMI", cell_ppmi)
    rank_cell = compareRankings(cell_state, cluster_rankings)
    print("Cell on Rank", rank_cell)
    return [rank_ppmi, cell_ppmi, rank_cell]

def evaluateAndSave(cluster_rankings_fn, cell_state_fn, ppmi_values_fn, file_name):
    cluster_rankings = np.load(cluster_rankings_fn)
    cell_state = np.load(cell_state_fn)
    ppmi_values = np.load(ppmi_values_fn)
    scores = evaluateCellState(cluster_rankings, cell_state, ppmi_values)
    rt.writeArray(scores, file_name)

cluster_rankings_fn = ""
cell_state_fn = ""
ppmi_values_fn = ""
file_name = ""
evaluateAndSave(cluster_rankings_fn, cell_state_fn, ppmi_values_fn, file_name)