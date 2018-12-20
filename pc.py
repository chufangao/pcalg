import itertools
from itertools import combinations, chain
from gsq.ci_tests import ci_test_bin, ci_test_dis
from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import math


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


# Purpose:  For any unshielded triple A-B-C, consider all subsets D of
# the neighbors of A and of the neighbors of C, and record the sets
# D for which A and C are conditionally independent given D. If B
# is in none of these sets, do nothing (it is a
# v-structure) and also delete B from sepset(A,C) if present (so we are
# sure that a v-structure will be created). If B is in all sets, do nothing
# (it is not a v-structure) and also add B to sepset(A,C) if not present
# (so we are sure that a v-structure will not be created). If maj.rule=FALSE
# the normal conservative version is applied, hence if B is in
# some but not all sets, mark the triple as "ambiguous". If maj.rule=TRUE
# we mark the triple as "ambiguous" if B is in exactly 50% of the cases,
# if less than 50% define it as a v-structure, and if in more than 50%
# no v-structure.
# ----------------------------------------------------------------------
# Arguments: - sk: output returned by function "skeleton"
#            - suffStat: Sufficient statistics for independent tests
#            - indepTest: Function for independence test
#            - alpha: Significance level of test
#            - version.unf[1]: 1 it checks if b is in some sepsets,
#                              2 it also checks if there exists a sepset
#                              which is a subset of the neighbours.
#            - version.unf[2]: 1 same as in Tetrad (do not consider
#                              the initial sepset), 2 it also considers
#                              the initial sepset
#            - maj.rule: FALSE/TRUE if the majority rule idea is applied
# ----------------------------------------------------------------------
# Value: - unfTripl: Triple that were marked as unfaithful
#        - vers: vector containing the version (1 or 2) of the
#                corresponding triple saved in unfTripl (1=normal
#                unfaithful triple that is B is in some sepsets;
#                2=triple coming from version.unf[1]==2
#                that is a and c are indep given the initial sepset
#                but there doesn't exist a subset of the neighbours
#                that d-separates them)
#        - sk: updated skelet object, sepsets might have been updated
# ----------------------------------------------------------------------
# Author: Markus Kalisch, Date: 12 Feb 2010, 10:43
# Modifications: Diego Colombo
def checkTriple(a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, suffStat, alpha, indepTest):
    nr_indep = 0

    temp = []
    # temp.append(b in sepsetC or b in sepsetA)

    if len(nbrsA) > 0:
        for s in powerset(nbrsA):
            pval = indepTest(suffStat, a, c, list(s))
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    if len(nbrsC) > 0:
        for s in powerset(nbrsC):
            pval = indepTest(suffStat, a, c, list(s))
            if pval >= alpha:
                nr_indep += 1
                temp.append(b in s)

    newsepsetA = set(list(sepsetA))
    newsepsetC = set(list(sepsetC))

    res = 3
    # print(sum(temp) / len(temp))
    if sum(temp) / len(temp) < .5:
        res = 1
        try:
            newsepsetA.remove(b)
            newsepsetC.remove(b)
        except:
            pass
    elif sum(temp) / len(temp) > .5:
        res = 2
        newsepsetA.add(b)
        newsepsetC.add(b)
    else:
        # print(a,b,c)
        # unfaithful
        pass
    # print(res)
    # print(res, temp)
    return res, {'sepsetA': newsepsetA, 'sepsetC': newsepsetC}


def pc_cons_intern(graphDict, suffstat, alpha, indepTest, version_unf=(None, None), maj_rule=False,
                   verbose=False):
    sk = graphDict['sk']
    p = len(sk)
    unfTripl = [None for i in range(min(p * p, 100000))]
    counter = -1

    if sk.any():
        ind = []
        for i in range(len(sk)):
            for j in range(len(sk)):
                if sk[i][j] == True:
                    ind.append((i, j))
        tripleMatrix = []
        # go thru all edges
        for a, b in ind:
            for c in range(len(sk)):
                if a > c and sk[a][c] == False and sk[b][c] == True:
                    tripleMatrix.append((a, b, c))
            # print(tripleMatrix)
            for a, b, c in tripleMatrix:
                nbrsA = [i for i in range(len(sk)) if sk[i][a] == True]
                nbrsC = [i for i in range(len(sk)) if sk[i][c] == True]

                # print(nbrsA, nbrsC)
                # print(graphDict['sepset'][a][c], graphDict['sepset'][c][a])
                res, r_abc = checkTriple(a, b, c, nbrsA, nbrsC, graphDict['sepset'][a][c],
                                         graphDict['sepset'][c][a],
                                         suffstat, alpha, indepTest)

                if res == 3:
                    if 'unfTriples' in graphDict.keys():
                        graphDict['unfTriples'].add((a, b, c))
                    else:
                        graphDict['unfTriples'] = {(a, b, c)}

                graphDict['sepset'][a][c] = r_abc['sepsetA']
                graphDict['sepset'][c][a] = r_abc['sepsetC']

    # print(np.array(graphDict['sepset']))
    return graphDict


# Purpose: Generate the next set in a list of all possible sets of size
#          k out of 1:n;
#  Also returns a boolean whether this set was the last in the list.
# ----------------------------------------------------------------------
# Arguments:
# - n,k: Choose a set of size k out of numbers 1:n
# - set: previous set in list
# ----------------------------------------------------------------------
# Author: Markus Kalisch, Date: 26 Jan 2006, 17:37
def skeleton(suffStat, indepTest, alpha, labels, method,
             fixedGaps, fixedEdges,
             NAdelete, m_max, numCores, verbose):
    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

    # form complete undirected graph, true if edge i--j needs to be investigated
    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    # save maximal p val
    pMax = [[float('-inf') for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)):
        # pvalue with itsself is 1, don't need to investigate i--i
        pMax[i][i] = 1
        G[i][i] = False

    # done flag
    done = False

    ord = 0
    n_edgetests = {0: 0}

    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0

        done = True

        ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    ind.append((i, j))

        G1 = G.copy()

        for x, y in ind:
            if G[x][y] == True:
                nbrsBool = [row[x] for row in G1]
                nbrsBool[y] = False
                nbrs = [i for i in range(len(nbrsBool)) if nbrsBool[i] == True]

                # print('len nbrs', len(nbrs), ord)
                if len(nbrs) >= ord:
                    if len(nbrs) > ord:
                        done = False

                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        # print(ord)
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        # get pvalue, if dependent, pval should be small
                        pval = indepTest(suffStat, x, y, list(nbrs_S))
                        if pMax[x][y] < pval:
                            pMax[x][y] = pval
                        if pval >= alpha:
                            # then independent
                            # print(x, y, 'independent given', list(nbrs_S))
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = list(nbrs_S)
                            break
        ord += 1
    # fix p values
    for i in range(0, len(labels) - 1):
        for j in range(1, len(labels)):
            pMax[i][j] = pMax[j][i] = max(pMax[i][j], pMax[j][i])

    # print(np.array(sepset))
    return {'sk': np.array(G), 'pMax': np.array(pMax), 'sepset': sepset}


# Purpose: Perform PC-Algorithm, i.e., estimate skeleton of DAG given data
# ----------------------------------------------------------------------
# Arguments:
# - dm: Data matrix (rows: samples, cols: nodes)
# - C: correlation matrix (only for continuous)
# - n: sample size
# - alpha: Significance level of individual partial correlation tests
# - corMethod: "standard" or "Qn" for standard or robust correlation
#              estimation
# - G: the adjacency matrix of the graph from which the algorithm
#      should start (logical)
# - datatype: distinguish between discrete and continuous data
# - NAdelete: delete edge if pval=NA (for discrete data)
# - m.max: maximal size of conditioning set
# - u2pd: Function for converting udag to pdag
#   "rand": udag2pdag
#   "relaxed": udag2pdagRelaxed
#   "retry": udag2pdagSpecial
# - gTrue: Graph suffStatect of true DAG
# - conservative: If TRUE, conservative PC is done
# - numCores: handed to skeleton(), used for parallelization
# ----------------------------------------------------------------------
# Author: Markus Kalisch, Date: 26 Jan 2006; Martin Maechler
# Modifications: Sarah Gerster, Diego Colombo, Markus Kalisch
def udag2pdagRelaxed(graph):
    def orientConflictCollider(pdag, x, y, z):
        # x -> y <- z
        # pdag: 2d list, pdag[x,y] = 1 and pdag[y,x] = 0 means x -> y
        # returns updated pdag
        print(np.array(pdag))
        if pdag[x][y] == 1:
            pdag[y][x] = 0
        else:
            pdag[x][y] = pdag[y][x] = 2

        if pdag[z][y] == 1:
            pdag[y][z] = 0
        else:
            pdag[z][y] = pdag[y][z] = 2

        return pdag

    def rule1(pdag, solve_conf=False, unfVect=None):
        # Rule 1: a -> b - c goes to a -> b -> c
        # Interpretation: No new collider is introduced
        # Out: Updated pdag
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))
        # sort to correspond with r
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            # for a, b in ind:
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (
                        search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)
            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and (
                            (a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                        # if unfaithful, skip
                        continue
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    def rule2(pdag, solve_conf=False):
        # Rule 2: a -> c -> b with a - b: a -> b
        # Interpretation: Avoid cycle
        # normal version = conservative version
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))
        # sort to correspond with r
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            # for a,b in ind:
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (
                        search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        # Rule 3: a-b, a-c1, a-c2, c1->b, c2->b but c1 and c2 not connected;
        # then a-b => a -> b
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))
        # sort to correspond with r
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            # for a,b in ind:
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (
                        search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        if 'unfTriples' in graph.keys() and (
                                (c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            # if unfaithful, skip
                            continue
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    # print(graph['sk'])
    pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]
    # print(np.array(pdag))
    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))
    # need to sort to correspond with R version
    for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
        # for x,y in ind:
        # print(x, y)
        allZ = []
        for z in range(len(pdag)):
            if graph['sk'][y][z] == True and z != x:
                allZ.append(z)
        # print(allZ)
        # print(x, y, allZ)
        for z in allZ:
            if graph['sk'][x][z] == False and graph['sepset'][x][z] != None and graph['sepset'][z][x] != None and not (
                    y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0
                # pdag = orientConflictCollider(pdag,x,y,z)

    # do while
    old_dag = pdag.copy()
    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    while old_dag != pdag:
        pdag = rule1(pdag)
        pdag = rule2(pdag)
        pdag = rule3(pdag)

    # print(np.array(graph['sepset']))
    return np.array(pdag)


def pc(suffStat, alpha, labels, indepTest=ci_test_dis, p='Use labels',
       fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float("inf"),
       u2pd=("relaxed", "rand", "retry"),
       skel_method=("stable", "original", "stable.fast"),
       conservative=False, maj_rule=False, solve_confl=False,
       numCores=1, verbose=False):
    # get skeleton
    graphDict = skeleton(suffStat, indepTest, alpha, labels=labels, method=skel_method,
                         fixedGaps=fixedGaps, fixedEdges=fixedEdges,
                         NAdelete=NAdelete, m_max=m_max, numCores=numCores, verbose=verbose)
    # print(graphDict['sepset'])
    # orient edges
    graph = pc_cons_intern(graphDict, suffStat, alpha, indepTest)
    # print(np.array(graph['sk']))
    return udag2pdagRelaxed(graph)


def fci(suffstat, indepTest, alpha, levels, p, type='adaptive', fixedGaps=None, fixedEdges=None, NAdelete=True,
        m_max=float('inf'), pdsep_max=float('inf')):
    pass


def gaussCItest(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]

    # def pcorOrder(i, j, k, C, cut_at=0.9999999):
    cut_at = 0.9999999
    if len(S) == 0:
        r = C[x, y]
    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))
    else:
        m = C[np.ix_([x] + [y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(m)
        # print(PM)
        r = -1 * PM[0, 1] / math.sqrt(PM[0, 0] * PM[1, 1])
    r = min(cut_at, max(-1 * cut_at, r))
    # return r
    # print(r)
    # def zstat(x, y, S, C, n):
    #     r = pcorOrder(x, y, S, C)
    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    # return res
    # z = zstat(x, y, S, C=suffstat["C"], n=suffstat["n"])
    return 2 * (1 - norm.cdf(abs(res)))


if __name__ == '__main__':
    file = 'C:/Users/gaoan/OneDrive - purdue.edu/2018 Fall/CS590AI/pcalg/datasets/ID1 Disc (1).csv'
    # file = 'datasets/mdata.txt'
    # file = 'datasets/BD Cont.csv'
    # file = 'datasets/BD5 Cluster X2 Disc X1 Outcome (1).csv'

    # file = 'gmD.csv'
    data = pd.read_csv(file)
    # print(data)
    p = pc(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05, labels=[str(i) for i in range(7)],
           indepTest=gaussCItest)
    # p = pc(suffStat=data.values, alpha=.05, labels=[str(i) for i in range(5)], indepTest=ci_test_dis)
    print(p)
