####################################
# Author: Shiva Sharooh  #
# Date	: 2022/05/04               #
####################################
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy import sparse
from math import ceil
from collections import OrderedDict
from matplotlib import pyplot as plt, dates as mdates
from joblib import Parallel, delayed, parallel_backend
import copy
from joblib import Parallel, delayed
import argparse
import csv
import matplotlib.pyplot as plt

plt.rcdefaults()
global edgelist
edgelist = []

global edgeDict
edgeDict = {}


def get_key(val):
    for key, value in node_dict.items():
        if val == value:
            return key


def log_star(x):
    """
    Compute universal code length used for encode real number

    INPUTS
    Real number

    OUTPUTS
    Approximate universal code length of the real number
    """
    return 2 * np.log2(x) + 1


def edgelist_to_matrix(edgelist):
    """
    Transform edgelist to adjacency matrix

    INPUTS
    Edge List

    OUTPUTS
    Adjacency matrix and dictionary of nodes
    """
    global node_dict
    node_dict = {e: idx for idx, e in enumerate(np.unique(edgelist))}

    ajm = np.zeros((len(node_dict), len(node_dict)), dtype="uint8")

    c = 0
    for e in edgelist:
        ajm[node_dict[e[0]]][node_dict[e[1]]] = 1
        c += 1
    return ajm, node_dict


def compute_mdl(ajm, order, start, count):
    """
    Encode matrix by given order and the information of smurf patterns

    INPUTS
    ajm: Binary adjacency matrix
    order: Order used to reorder the matrix
    start: Start positions of each smurf pattern
    count: number of detected patterns and intermediaries

    OUTPUTS
    [Encoding description length, Purity]
    """
    purity, mdl, n = [], 0, len(ajm)
    order.extend([i for i in range(n) if i not in order])
    ajm = np.array(ajm)[np.ix_(order, order)]

    ### Encode sub-matrix A, B and C
    for idx in range(1, len(start)):
        s, e = start[idx - 1], start[idx] - 1
        k = e - s + 1
        e1 = np.sum(ajm[s + 1 : e, s : e - 1]) * (2 * ceil(np.log2(k - 1)))
        e2 = np.sum(ajm[e + 1 : -1, s:e]) * (ceil(np.log2(n)) + ceil(np.log2(n - k)))
        e3 = np.sum(ajm[s:e, e + 1 : -1]) * (ceil(np.log2(n)) + ceil(np.log2(n - k)))
        et = e1 + e2 + e3
        mdl += et
        sum_abc = (
            np.sum(ajm[s:e, s:e])
            + np.sum(ajm[e + 1 : -1, s:e])
            + np.sum(ajm[s:e, e + 1 : -1])
        )
        purity.append((k - 2) * 2 / sum_abc)

    ### Encode sub-matrix D
    ajm = 1 - ajm
    mdl += np.sum(ajm[start[-1] : -1, start[-1] : -1]) * (2 * ceil(np.log2(n)))
    ### Encode the real number of found patterns and intermediaries
    mdl += ceil(log_star(count[0])) + ceil(log_star(count[1]))
    ### Encode the indexes of senders, receivers and intermediaries
    mdl += np.sum(count) * ceil(np.log2(n))
    ### Encode for the start point of each pattern
    mdl += ceil(log_star(len(start) - 1))

    return mdl, np.mean(purity)


def AA_Smurf(ajm, max_iter, visualize, ThresholdSmurf):
    """
    Identify the best order for spotting smurf pattern

    INPUTS
    ajm: Binary adjacency matrix
    max_iter: Maximum iteration to run the algorithm
    visualize: Path of visualization result

    OUTPUTS
    [Reordered matrix, Best order for reordering]
    """
    ### In case using 'cfd_injected.pkl' file
    # ajm, node_dict = edgelist_to_matrix(edgelist)

    ### Get edge-pairs which have the number of intermediaries hgiher than the threshold c
    print("Get Edge-Pairs...")
    row, col = ajm, ajm.T
    # exit()
    edis = OrderedDict()
    dis_mtr = (sparse.csr_matrix(ajm) * sparse.csr_matrix(ajm)).todense()

    for idx1, idx2 in zip(*dis_mtr.nonzero()):
        val = dis_mtr[idx1, idx2]
        if val >= ThresholdSmurf:
            edis[(idx1, idx2)] = [
                val,
                np.arange(len(row))[(row[idx1] + col[idx2]) == 2],
            ]

    edis = OrderedDict(sorted(edis.items(), key=lambda t: t[1][0])[::-1])
    print("Done!\n")

    ### Heuristically identify the best order by MDL and purity
    print("Identify Best Order...")

    def func(ajm, key, value, order, count, start, prev_mdl):
        if key[0] not in order and key[1] not in order:
            order.append(key[0])
            tmp_mid = [a for a in value[1] if a not in order]
            if len(tmp_mid) == 0:
                return -1, -1, -1, -1, -1
            order.extend(tmp_mid)
            order.append(key[1])
            start.append(len(order))

            mdl, purity = compute_mdl(
                ajm,
                copy.copy(order),
                copy.copy(start),
                [count[0] + 1, count[1] + len(tmp_mid), count[2] + 1],
            )
            score = ((prev_mdl - mdl) / prev_mdl) * purity

            if mdl < prev_mdl:
                count = [count[0] + 1, count[1] + len(tmp_mid), count[2] + 1]

                return mdl, score, order, start, count
        return -1, -1, -1, -1, -1

    old_mdl = np.ceil(np.sum(1 - ajm)) * (2 * ceil(np.log2(len(ajm))))
    count_arr, order_arr, start_arr, mdl_arr = [[0, 0, 0]], [[]], [[0]], [old_mdl]
    iter = 0
    while True:
        prev_mdl = mdl_arr[-1]
        results = Parallel(backend="loky", n_jobs=-1)(
            [
                delayed(func)(
                    ajm,
                    key,
                    value,
                    copy.copy(order_arr[-1]),
                    copy.copy(count_arr[-1]),
                    copy.copy(start_arr[-1]),
                    prev_mdl,
                )
                for idx, (key, value) in enumerate(edis.items())
            ]
        )
        tmp_mdl = [r[0] for r in results]
        # print('tmp mdl> ' + str(tmp_mdl))
        tmp_score = [r[1] for r in results]
        # print('tmp tmp_score> ' + str(tmp_score))
        tmp_order = [r[2] for r in results]
        # print('tmp tmp_order> ' + str(tmp_order))
        tmp_start = [r[3] for r in results]
        # print('tmp tmp_start> ' + str(tmp_start))
        tmp_count = [r[4] for r in results]
        # print('tmp tmp_count> ' + str(tmp_count))
        break
    print("Done!\n")

    return tmp_order


########################################################################
def tupleFinder(tuples, tmpList):
    # Convert to sets just once, rather than repeatedly
    # within the nested for-loops.
    subsets = set(tuples)
    mainsets = [set(xs) for xs in tmpList]
    countr = 0
    # Same as your algorithm, but written differently.
    for items in mainsets:
        if items == subsets:
            countr += 1
    if countr == 0:
        return 0
    else:
        return 1


# Time order function
def Temporal_order(Smurfs):
    # Smurfs=['A','B','C','D','E']
    pathDict = {}
    count = 1
    outgoingEdge = []
    incomingEdge = []
    MultiGraph = {}
    if Smurfs[0] == Smurfs[-1]:
        return 0, {}
    for i in Smurfs[1:-1]:
        try:
            # print(str(Smurfs[0]) + str(i))
            # print(str(i) + str(Smurfs[-1]))
            MultiGraph[str(Smurfs[0]) + str(i)].append(
                edgeDict[str(Smurfs[0]) + str(i)]
            )
            MultiGraph[str(i) + str(Smurfs[-1])].append(
                edgeDict[str(i) + str(Smurfs[-1])]
            )

        except:
            # print(edgeDict[i+Smurfs[-1]])
            MultiGraph[str(Smurfs[0]) + str(i)] = edgeDict[str(Smurfs[0]) + str(i)]
            MultiGraph[str(i) + str(Smurfs[-1])] = edgeDict[str(i) + str(Smurfs[-1])]

    # the first item in the Smurfs list is the source and the last item is the sink node
    for item in Smurfs[1:-1]:
        # pathDict['path_%s' % count] = {Smurfs[0]+item: MultiGraph[Smurfs[0] + item], item+Smurfs[-1]:MultiGraph[ item+Smurfs[-1]]}
        pathDict["path_%s" % count] = {Smurfs[0] + item: [], item + Smurfs[-1]: []}
        outgoingEdge.append(item + Smurfs[-1])
        incomingEdge.append(Smurfs[0] + item)
        count += 1

    # to search for each edge in the edgelist
    tempFanIn = []
    tempFanOut = []
    # print(MultiGraph)
    for k, v in pathDict.items():
        count = 0
        # retreive fan in and fan out of each path in multigraph
        orderedList = []

        for key, val in v.items():
            orderedList.append(key)
            if count == 0:
                tempFanIn = MultiGraph[key]
            else:
                tempFanOut = MultiGraph[key]

            count += 1

        for i in tempFanIn:
            for j in tempFanOut:
                # check if first item is smaller
                if i[0] > j[0]:
                    tempFanOut.remove(j)
                    continue
                if i[0] <= j[0]:
                    # add the attributes to the path as true order
                    if tupleFinder(i, v[orderedList[0]]) == 0:
                        v[orderedList[0]].append(i)
                    if tupleFinder(j, v[orderedList[1]]) == 1:
                        continue
                    v[orderedList[1]].append(j)
                    tempFanOut.remove(j)
                    continue
            v[orderedList[0]].append(i)

    smurfNum = 0

    new_pathDict = {}
    # 'FanIn' 'FanOut
    counter = 0
    for k, v in pathDict.items():
        c = 0
        tmpDict = {}
        bit = 0  # if a value is empty don't count it repeatedly
        for key, val in v.items():
            if c == 0:
                tmpDict["FanIn"] = val
                c += 1
            else:
                tmpDict["FanOut"] = val
            if len(val) == 0 and bit == 0:
                bit += 1
                counter += 1

        new_pathDict[k] = tmpDict
    # print(new_pathDict)
    smurfNum = (len(Smurfs) - 2) - counter
    # new_pathDict is equal to MG_graph
    # print(smurfNum)
    return smurfNum, new_pathDict


def FlowCompute(PathList):
    MFlowDict = {}
    tmpFaninflow = (
        0  # this variable will hold the amount of moneyremains in fan in edge
    )
    # each edge is [time, quantity, Fan_bit]
    for key, val in PathList.items():
        count = 0
        for edges in val:
            # first edge is for sure a fanin edge so we ignore it and just add the quantity to tmpFaninflow
            # if it see the fan in edge, it will just add up to the  tmpFaninflow
            if edges[2] == 0 or count == 0:
                tmpFaninflow += edges[1]
                count += 1
                continue
            # else if it is a fan out node compute maxflow
            maxflow = min(tmpFaninflow, edges[1])
            MFlowDict[edges[0]] = maxflow
            tmpFaninflow -= maxflow
    # print(MFlowDict)
    return MFlowDict


###########################################################

# maxflow compute
def max_flowComputer(MG_Smurf):
    Maxflow = 0
    EdgeorderList = {}
    alledges = []
    for k, v in MG_Smurf.items():
        FanInSum = 0
        FanOutSum = 0
        Templist = []
        # print(v['FanIn'])
        for item in v["FanIn"]:
            alledges.append(item)
            # add 0 as a fan in and 1 to fan out bit indicator
            Templist.append(item + [0])
            FanInSum += item[1]

        for i in v["FanOut"]:
            alledges.append(i)
            FanOutSum += i[1]
            Templist.append(i + [1])
            # order the edges based on their timestamp for each path
            EdgeorderList[k] = sorted(Templist, key=itemgetter(0))
        Maxflow += min(FanOutSum, FanInSum)
    return Maxflow, alledges, EdgeorderList


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser(description="Parameters for AA-Smurf of AutoAudit")
    # parser.add_argument('--f', default='data2/smurfGraphSmall.txt', type=str, help='Input Path')
    # parser.add_argument('--f', default='data2/sample1Adjc.txt', type=str, help='Input Path')
    parser.add_argument(
        "--o", default="results/AA-Smurf_result.png", type=str, help="Output Path"
    )
    parser.add_argument("--i", default=None, type=int, help="Maximum Iteration")
    args = parser.parse_args()
    EnronData = 100000
    ThresholdSmurf = 10
    ThresholdFlow = 150

    # ajm = np.loadtxt(args.f)
    import csv

    countx = 0
    edgelist = []
with open("../analysis/data/5_10_SmurfCfd_trans.csv", newline="") as f:
    reader = csv.reader(f)
    # print(list(reader))
    for item in list(reader):
        if countx == 0:
            # print(item)
            countx = countx + 1
            continue
        if countx == 200000:
            break
        else:
            edgelist.append([int(item[0]), int(item[1])])
            countx += 1

    # print(edgelist)
    ajm, nodDic = edgelist_to_matrix(edgelist)

    print("smurf list created")
    print("done")
    smurf_val = []
    # print(ajm)
    smurf_order = AA_Smurf(ajm, args.i, args.o, ThresholdSmurf)
    for items in smurf_order:
        tmp = []
        for i in items:
            tmp.append(get_key(i))
        smurf_val.append(tmp)
    end = datetime.now()
    tmpend = end - start
    print("Time: ", tmpend)

    print("All smurf detected: " + str(len(smurf_val)))
    alert = 0
    xcount = 0
    fishyAcc = []
    for items in smurf_val:
        # print('length: '+ str(len(items)))
        smurfNumber, pathDictionary = Temporal_order(items)
        if smurfNumber < ThresholdSmurf:
            continue
        xcount += 1

        max_Flow, all_edge, edge_orderList = max_flowComputer(pathDictionary)
        if max_Flow < ThresholdFlow:
            continue
        alert += 1
        print(all_edge)
        print(edge_orderList)
        exit()
        # ploting(MAxflowTime, alledges, timSet)
        # ploting(MAxflowTime, alledges)

    stop = datetime.now()
    print("All temporal smurf detected: " + str(xcount))
    print("All temporalflow smurf detected: " + str(alert))
    tmpStop = stop - start
    print("Difference Time: ", tmpStop)
