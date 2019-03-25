import copy
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
def checkTriple(a, b, c, nbrsA, nbrsC, sepsetA, sepsetC, suffStat, alpha, indepTest, maj_rule=True):
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

    if len(temp) == 0:
        temp.append(False)

    res = 3
    if maj_rule:
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
            # unfaithful
            pass
    else:
        if sum(temp) / len(temp) == 0:
            res = 1
            try:
                newsepsetA.remove(b)
                newsepsetC.remove(b)
            except:
                pass
        elif sum(temp) / len(temp) == 1:
            res = 2
            newsepsetA.add(b)
            newsepsetC.add(b)
        else:
            # unfaithful
            pass

    return res, {'sepsetA': newsepsetA, 'sepsetC': newsepsetC}


def pc_cons_intern(graphDict, suffstat, alpha, indepTest, version_unf=(None, None), maj_rule=True,
                   verbose=False):
    sk = graphDict['sk']

    if np.any(sk):
        # ind = []
        # for i in range(len(sk)):
        #     for j in range(len(sk)):
        #         if sk[i][j] == True:
        #             ind.append((i, j))
        # ind = sorted(ind, key=lambda x: (x[1], x[0]))
        ind = np.transpose(np.where(sk == 1))
        ind = ind[ind[:, 1].argsort()]

        tripleMatrix = []
        # go thru all edges
        for a, b in ind:
            if verbose: print(a,b)

            for c in range(len(sk)):
                if a < c and sk[a, c] == 0 and sk[b, c] == 1:
                    tripleMatrix.append((a, b, c))
            # print(tripleMatrix)
        for a, b, c in tripleMatrix:
            # nbrsA = [i for i in range(len(sk)) if sk[i][a] == True]
            nbrsA = np.transpose(np.where(sk[:, a] == 1))
            # nbrsC = [i for i in range(len(sk)) if sk[i][c] == True]
            nbrsC = np.transpose(np.where(sk[:, c] == 1))

            res, r_abc = checkTriple(a, b, c, nbrsA, nbrsC, graphDict['sepset'][(a, c)],
                                     graphDict['sepset'][(c, a)],
                                     suffstat, alpha, indepTest, maj_rule=maj_rule)
            if res == 3:
                if 'unfTriples' in graphDict.keys():
                    graphDict['unfTriples'].add((a, b, c))
                else:
                    graphDict['unfTriples'] = {(a, b, c)}

            graphDict['sepset'][(a, c)] = r_abc['sepsetA']
            graphDict['sepset'][(c, a)] = r_abc['sepsetC']

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
    # sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]
    sepset = {}
    for i in itertools.permutations([i for i in range(len(labels))], 2):
        sepset[i] = set()

    # form complete undirected graph, true if edge i--j needs to be investigated
    # G = [[True for i in range(len(labels))] for i in range(len(labels))]
    G = np.ones((len(labels), len(labels)))

    # save maximal p val
    # pMax = [[float('-inf') for i in range(len(labels))] for i in range(len(labels))]
    pMax = np.full((len(labels), len(labels)), np.NINF)

    for i in range(len(labels)):
        # pvalue with itsself is 1, don't need to investigate i--i
        pMax[i, i] = 1
        G[i, i] = 0

    # done flag
    done = False

    ord = 0
    n_edgetests = {}

    while done != True and np.any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0

        done = True

        # ind = []
        # for i in range(len(G)):
        #     for j in range(len(G[i])):
        #         if G[i][j] == True:
        #             ind.append((i, j))
        ind = np.transpose(np.where(G == 1))

        G1 = np.copy(G)

        for x, y in ind:
            if G[y, x] == 1:
                # nbrsBool = [row[x] for row in G1]
                # nbrsBool[y] = False
                # nbrs = [i for i in range(len(G1)) if G1[x][i] == True and i != y]
                nbrs = np.transpose(np.where(G1[:, x] == 1))
                nbrs = nbrs[nbrs != y]

                # print('len nbrs', len(nbrs), ord)
                if len(nbrs) >= ord:
                    if len(nbrs) > ord:
                        done = False

                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        # print(ord)
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        # get pvalue, if dependent, pval should be small
                        pval = indepTest(suffStat, x, y, list(nbrs_S))
                        if pMax[x, y] < pval:
                            pMax[x, y] = pval
                        if pval >= alpha:
                            # then independent
                            # print(x, y, 'independent given', list(nbrs_S))
                            G[x, y] = G[y, x] = False
                            sepset[(x, y)] = list(nbrs_S)
                            break
        ord += 1
    # fix p values
    for i in range(0, len(labels) - 1):
        for j in range(1, len(labels)):
            pMax[i, j] = pMax[j, i] = max(pMax[i, j], pMax[j, i])

    return {'sk': G, 'pMax': pMax, 'sepset': sepset, "unfTriples": set(), "max_ord": ord - 1}


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
        if pdag[x, y] == 1:
            pdag[y, x] = 0
        else:
            pdag[x, y] = pdag[y, x] = 2
        if pdag[z, y] == 1:
            pdag[y, z] = 0
        else:
            pdag[z, y] = pdag[y, z] = 2
        return pdag

    def rule1(pdag, solve_conf=False, unfVect=None):
        # Rule 1: a -> b - c goes to a -> b -> c
        # Interpretation: No new collider is introduced
        # Out: Updated pdag
        search_pdag = np.copy(pdag)
        # ind = []
        # for i in range(len(pdag)):
        #     for j in range(len(pdag)):
        #         if pdag[i][j] == 1 and pdag[j][i] == 0:
        #             ind.append((i, j))
        ind = np.transpose(np.where(np.logical_and(pdag == 1, pdag.transpose() == 0)))
        ind = ind[ind[:, 1].argsort()]

        # sort to correspond with r
        # for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
        for a, b in ind:
            # isC = []
            # for i in range(len(search_pdag)):
            #     if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (
            #             search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
            #         isC.append(i)
            isC = np.squeeze(np.transpose(np.where(
                np.logical_and(
                    np.logical_and(search_pdag[b, :] == 1, search_pdag[:, b] == 1),
                    np.logical_and(search_pdag[a, :] == 0, search_pdag[:, a] == 0)
                ))), axis=1)

            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and (
                            (a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                        # if unfaithful, skip
                        continue
                    if pdag[b, c] == 1 and pdag[c, b] == 1:
                        pdag[b, c] = 1
                        pdag[c, b] = 0
                    elif pdag[b, c] == 0 and pdag[c, b] == 1:
                        pdag[b, c] = pdag[c, b] = 2

        return pdag

    def rule2(pdag, solve_conf=False):
        # Rule 2: a -> c -> b with a - b: a -> b
        # Interpretation: Avoid cycle
        # normal version = conservative version
        search_pdag = pdag.copy()
        # ind = []
        # for i in range(len(pdag)):
        #     for j in range(len(pdag)):
        #         if pdag[i][j] == 1 and pdag[j][i] == 1:
        #             ind.append((i, j))
        ind = np.transpose(np.where(np.logical_and(pdag == 1, pdag.transpose() == 1)))
        ind = ind[ind[:, 1].argsort()]

        # sort to correspond with r
        # for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
        for a, b in ind:
            # isC = []
            # for i in range(len(search_pdag)):
            #     if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (
            #             search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
            #         isC.append(i)
            isC = np.squeeze(np.transpose(np.where(
                np.logical_and(
                    np.logical_and(search_pdag[a, :] == 1, search_pdag[:, a] == 0),
                    np.logical_and(search_pdag[:, b] == 1, search_pdag[b, :] == 0)
                ))), axis=1)
            if len(isC) > 0:
                if pdag[a, b] == 1 and pdag[b, a] == 1:
                    pdag[a, b] = 1
                    pdag[b, a] = 0
                elif pdag[a, b] == 0 and pdag[b, a] == 1:
                    pdag[a, b] = pdag[b, a] = 2

        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        # Rule 3: a-b, a-c1, a-c2, c1->b, c2->b but c1 and c2 not connected;
        # then a-b => a -> b
        search_pdag = pdag.copy()
        # ind = []
        # for i in range(len(pdag)):
        #     for j in range(len(pdag)):
        #         if pdag[i][j] == 1 and pdag[j][i] == 1:
        #             ind.append((i, j))
        ind = np.transpose(np.where(np.logical_and(pdag == 1, pdag.transpose() == 1)))
        ind = ind[ind[:, 1].argsort()]

        # sort to correspond with r
        # for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
        for a, b in ind:
            # isC = []
            # for i in range(len(search_pdag)):
            #     if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (
            #             search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
            #         isC.append(i)
            isC = np.squeeze(np.transpose(np.where(
                np.logical_and(
                    np.logical_and(search_pdag[a, :] == 1, search_pdag[:, a] == 1),
                    np.logical_and(search_pdag[:, b] == 1, search_pdag[b, :] == 0)
                ))), axis=1)
            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1,c2] == 0 and search_pdag[c2,c1] == 0:
                        if 'unfTriples' in graph.keys() and (
                                (c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            # if unfaithful, skip
                            continue
                        if search_pdag[a, b] == 1 and search_pdag[b, a] == 1:
                            pdag[a, b] = 1
                            pdag[b, a] = 0
                            break
                        elif search_pdag[a, b] == 0 and search_pdag[b, a] == 1:
                            pdag[a, b] = pdag[b, a] = 2
                            break
        return pdag

    # print(graph['sk'])
    # pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]
    pdag = np.copy(graph['sk'])
    # print(np.array(pdag))
    # ind = []
    # for i in range(len(pdag)):
    #     for j in range(len(pdag[i])):
    #         if pdag[i][j] == 1:
    #             ind.append((i, j))
    ind = np.transpose(np.where(pdag == 1))
    ind = ind[ind[:,1].argsort()]

    # need to sort to correspond with R version
    # for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
    for x, y in ind:
        # allZ = []
        # for z in range(len(pdag)):
        #     if graph['sk'][y][z] == True and z != x:
        #         allZ.append(z)
        allZ = np.transpose(np.where(graph['sk'][y, :] == 1))
        allZ = allZ[allZ != x]

        for z in allZ:
            if graph['sk'][x, z] == False and graph['sepset'][(x, z)] != None and graph['sepset'][(z, x)] != None and \
                    not (y in graph['sepset'][(x, z)] or y in graph['sepset'][(z, x)]):
                pdag[x, y] = pdag[z, y] = 1
                pdag[y, x] = pdag[y, z] = 0
                # pdag = orientConflictCollider(pdag,x,y,z)

    # do while
    old_dag = np.copy(pdag)
    print(pdag); quit()
    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    # while old_dag != pdag:
    while not np.array_equal(old_dag, pdag):
        print(pdag)
        pdag = rule1(pdag)
        pdag = rule2(pdag)
        pdag = rule3(pdag)

    # print(np.array(graph['sepset']))
    return np.array(pdag)


def pc(suffStat, alpha, labels, indepTest=ci_test_dis, p='Use labels',
       fixedGaps=None, fixedEdges=None, NAdelete=True, m_max=float("inf"),
       u2pd=("relaxed", "rand", "retry"),
       skel_method=("stable", "original", "stable.fast"),
       conservative=False, maj_rule=True, solve_confl=False,
       numCores=1, verbose=False):
    # get skeleton
    graphDict = skeleton(suffStat, indepTest, alpha, labels=labels, method=skel_method,
                         fixedGaps=fixedGaps, fixedEdges=fixedEdges,
                         NAdelete=NAdelete, m_max=m_max, numCores=numCores, verbose=verbose)
    if verbose:
        print('Got skeleton')
    # orient edges
    graph = pc_cons_intern(graphDict, suffStat, alpha, indepTest, verbose=verbose)
    if verbose:
        print('Got sepsets')
    # apply rules
    return udag2pdagRelaxed(graph)


def legal_path(a, b, c, amat):
    a_b = amat[a][b]
    if a == c or a_b == 0 or amat[b][c] == 0:
        return False
    return amat[a][c] != 0 or (a_b == 2 and amat[c][b] == 2)


def qreach(x, amat):
    # print(amat)
    A = [[1 if amat[i][j] != 0 else 0 for i in range(len(amat))] for j in range(len(amat))]
    PSEP = [i for i in range(len(A[x])) if A[x][i] != 0]
    Q = nb = copy.deepcopy(PSEP)
    P = [x for i in range(len(Q))]
    for i in nb:  # delete edge to nbrs
        A[x][i] = 0
    while (len(Q) > 0):
        # print("Q:",Q,"P:",P)

        a = Q[0]
        Q = Q[1:len(Q)]
        pred = P[0]
        P = P[1:len(P)]
        # print("Select",pred,"towards",a)
        nb = [i for i in range(len(A[a])) if A[a][i] != 0]
        # print("Check nbrs",nb)

        for b in nb:
            lres = legal_path(pred, a, b, amat)
            if lres == True:
                A[a][b] = 0
                Q.append(b)
                P.append(a)
                PSEP.append(b)
            # print(lres,end=" ")
        # print("\n")
    while x in PSEP:
        PSEP.remove(x)
    return sorted(set(PSEP))


def pdsep(skel, suffStat, indepTest, p, sepSet, alpha, pMax, m_max=float('inf'), pdsep_max=float('inf'), unfVect=None):
    G = [[0 if skel['sk'][i][j] == False else 1 for i in range(len(skel['sk']))] for j in range(len(skel['sk']))]
    n_edgetest = [0 for i in range(1000)]
    ord = 0
    allPdsep_tmp = [set() for i in range(p)]

    amat = copy.deepcopy(G)
    ind = []
    # orient colliders
    for i in range(len(G)):
        for j in range(len(G[i])):
            if G[i][j] == 1:
                ind.append((i, j))
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepSet[x][z] or y in sepSet[z][x]):
                if len(unfVect) == 0:  # normal version
                    amat[x][y] = amat[z][y] = 2
                else:  # conservative version, check if x-y-z faithful
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = 2

    allPdsep = [qreach(x, amat) for x in range(p)]
    allPdsep_tmp = [[] for i in range(p)]

    for x in range(p):
        an0 = [True if amat[x][i] != 0 else False for i in range(len(amat))]
        tf1 = [i for i in allPdsep[x] if i != x]

        adj_x = [i for i in range(len(an0)) if an0[i] == True]

        for y in adj_x:
            tf = [i for i in tf1 if i != y]
            diff_set = [i for i in tf if i not in adj_x]
            allPdsep_tmp[x] = tf + [y]

            if len(tf) > pdsep_max:
                pass
            elif len(diff_set) > 0:
                done = False
                ord = 0
                while not done and ord < min(len(tf), m_max):
                    # print(ord)
                    ord += 1
                    if ord == 1:
                        for S in diff_set:
                            pval = indepTest(suffStat, x, y, [S])
                            n_edgetest[ord + 1] += 1
                            if pval > pMax[x][y]:
                                pMax[x][y] = pval
                            if pval >= alpha:
                                amat[x][y] = amat[y][x] = 0
                                sepSet[x][y] = sepSet[y][x] = {S}
                                done = True
                                break
                    else:  # ord > 1
                        tmp_combn = combinations(tf, ord)
                        if ord <= len(adj_x):
                            for S in tmp_combn:
                                if not set(S).issubset(adj_x):
                                    pval = indepTest(suffStat, x, y, list(S))
                                    n_edgetest[ord + 1] += 1
                                    if pval > pMax[x][y]:
                                        pMax[x][y] = pval
                                    if pval > alpha:
                                        amat[x][y] = amat[y][x] = 0
                                        sepSet[x][y] = sepSet[y][x] = set(S)
                                        done = True
                                        break
                        else:
                            for S in tmp_combn:
                                pval = indepTest(suffStat, x, y, list(S))
                                n_edgetest[ord + 1] += 1
                                if pval > pMax[x][y]:
                                    pMax[x][y] = pval
                                if pval > alpha:
                                    amat[x][y] = amat[y][x] = 0
                                    sepSet[x][y] = sepSet[y][x] = set(S)
                                    done = True
                                    break
    for i in range(len(amat)):
        for j in range(len(amat[i])):
            if amat[i][j] == 0:
                G[i][j] = False
            else:
                G[i][j] = True

    return {'G': G, "sepset": sepSet, "pMax": pMax, "allPdsep": allPdsep_tmp, "max_ord": ord}


def updateList(path, set, old_list):  # arguments are all lists
    temp = []
    if len(old_list) > 0:
        temp.append(old_list)
    temp.extend([path + [s] for s in set])
    return temp


def faith_check(cp, unfVect, p, boolean=True):
    # check if every triple in circle path unambiguous
    if (boolean == False):
        res = 0
    n = len(cp)
    i1 = [i for i in range(n)]
    i2 = [(i + 1) % n for i in i1]
    i3 = [(i + 2) % n for i in i1]

    for i in range(len(i1)):
        if (i1[i], i2[i], i3[i]) in unfVect or (i3[i], i2[i], i1[i]) in unfVect:
            if boolean:
                return False
    if boolean:
        return True


def minDiscPath(pag, a, b, c):
    p = len(pag)
    visited = [False for i in range(len(pag))]
    visited[a] = visited[b] = visited[c] = True

    indD = [i for i in range(len(pag)) if pag[a][i] != 0 and pag[i][a] == 2 and visited[i] == False]
    if len(indD) > 0:
        path_list = updateList([a], indD, [])
        while len(path_list) > 0:
            mpath = path_list[0]
            m = len(mpath)
            d = mpath[-1]
            if pag[c][d] == 0 and pag[d][c] == 0:
                print(mpath, b, c)
                return mpath.reverse() + [b, c]
            else:
                pred = mpath[-2]
                path_list = path_list[1:len(path_list)]
                visited[d] = True

                if pag[d][c] == 2 and pag[c][d] == 3 and pag[pred][d] == 2:
                    indR = [i for i in range(len(pag)) if pag[d][i] != 0 and pag[i][d] == 2 and visited[i] == False]
                    if len(indR) > 0:
                        path_list = updateList(mpath, indR, path_list)
    return []


def minUncovCircPath(p, pag, path, unfVect):
    visted = [False for i in range(p)]
    for i in path:
        visted[i] = True
    a = path[0]
    b = path[1]
    c = path[2]
    d = path[3]
    min_ucp_path = []

    indX = [i for i in range(p) if pag[c][i] == 1 and pag[i][c] == 1 and visted[i] == False]
    if len(indX) > 0:
        path_list = updateList([c], indX, [])
        done = False
        while done == False and len(path_list) > 0:
            mpath = path_list[0]
            x = mpath[-1]
            path_list = path_list[1:len(path_list)]
            visted[x] = True
            if pag[x][d] == 1 and pag[d][x] == 1:
                mpath = [a] + mpath + [d, b]
                n = len(mpath)
                uncov = True
                for i in range(len(mpath) - 2):
                    if pag[mpath[i]][mpath[i + 2]] == 0 and pag[mpath[i + 2]][mpath[i]] == 0:
                        uncov = False
                        break
                if uncov == True:
                    if len(unfVect) == 0 or faith_check(mpath, unfVect, p):
                        min_ucp_path = mpath
                        done = True
            else:
                indR = [i for i in range(p) if pag[x][i] == 1 and pag[i][x] == 1 and visted[i] == False]
                if len(indR) > 0:
                    path_list = updateList(mpath, indR, path_list)
    return min_ucp_path


def minUncovPdPath(p, pag, a, b, c, unfVect):
    visited = [False for i in range(p)]
    visited[a] = visited[b] = visited[c] = True
    min_upd_path = []
    indD = [i for i in range(p) if
            (pag[b][i] == 1 or pag[b][i] == 2) and
            (pag[i][b] == 1 or pag[i][b] == 3) and
            pag[i][a] == 0 and visited[i] == False]
    if len(indD) > 0:
        path_list = updateList([b], indD, [])
        done = False
        while len(path_list) > 0 and done == False:
            mpath = path_list[0]
            path_list = path_list[1:len(path_list)]
            m = len(mpath)
            d = mpath[-1]
            visited[d] = True
            if pag[d][c] in [1, 2] and pag[c][d] in [1, 3]:
                # pd path found
                mpath = [a] + mpath + [c]
                uncov = True
                for i in range(len(mpath) - 2):
                    if not (pag[mpath[i]][mpath[i + 2]] == 0 and pag[mpath[i + 2][mpath[i]]] == 0):
                        uncov = False
                        break
                if uncov == True:
                    if len(unfVect) == 0 or faith_check(mpath, unfVect, p):
                        min_upd_path = mpath
                        done = True
            else:
                indR = [i for i in range(p) if
                        (pag[d][i] == 1 or pag[d][i] == 2) and
                        (pag[i][d] == 1 or pag[i][d] == 3) and
                        visited[i] == False]
                if len(indR) > 0:
                    path_list = updateList(mpath, indR, path_list)
    return min_upd_path


def udag2pag(pag, sepset, rules=(True, True, True, True, True, True, True, True, True, True), unfVect=None,
             orientCollider=True):
    pag = [[0 if pag[i][j] == False else 1 for i in range(len(pag))] for j in range(len(pag))]
    p = len(pag)
    indA = []  # store for R10
    if orientCollider == True:
        ind = []
        for i in range(len(pag)):
            for j in range(len(pag[i])):
                if pag[i][j] == 1:
                    ind.append((i, j))
        ind = sorted(ind, key=lambda x: (x[1], x[0]))
        for x, y in ind:
            allZ = [i for i in range(len(pag[y])) if pag[y][i] != 0 and i != x]
            for z in allZ:
                if pag[x][z] == 0 and not (y in sepset[x][z] or y in sepset[z][x]):
                    if len(unfVect) == 0:
                        pag[x][y] = pag[z][y] = 2
                    else:
                        if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                            pag[x][y] = pag[z][y] = 2
    # end orient collider

    # old_pag1 = [[0 for i in range(len(pag))] for j in range(len(pag))]
    old_pag1 = None
    while old_pag1 != pag:
        old_pag1 = copy.deepcopy(pag)
        if rules[0]:  # R1
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, b in ind:
                indC = [i for i in range(len(pag)) if
                        pag[b][i] != 0 and pag[i][b] == 1 and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indC) != 0:
                    if len(unfVect) != 0:
                        for c in indC:
                            pag[b][c] = 2
                            pag[c][b] = 3
                    else:
                        for c in indC:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                pag[b][c] = 2
                                pag[c][b] = 3
        if rules[1]:  # R2
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 1 and pag[j][i] != 0:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == 2 and pag[i][a] == 3 and pag[c][i] != 0 and pag[i][c] == 2) or
                        (pag[a][i] == 2 and pag[i][1] != 0 and pag[c][i] == 3 and pag[i][c] == 2)]
                if len(indB) > 0:
                    pag[a][c] = 2
        if rules[2]:  # R3
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, d in ind:
                indAC = [i for i in range(len(pag)) if
                         pag[b][i] != 0 and pag[i][b] == 2 and pag[i][d] == 1 and pag[d][i] != 0]
                if len(indAC) >= 2:
                    if len(unfVect) == 0:
                        counter = -1
                        while counter < len(indAC) - 1 and pag[d][b] != 2:
                            counter += 1
                            ii = counter
                            while ii < len(indAC) - 1 and pag[d][b] != 2:
                                ii += 1
                                if pag[indAC[counter]][indAC[ii]] == 0 and pag[indAC[ii]][indAC[counter]] == 0:
                                    pag[d][b] = 2
                    else:
                        for a, c in combinations(indAC, 2):
                            if pag[a][c] == 0 and pag[c][a] == 0 and c != a:
                                if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                    pag[d][b] = 2
        if rules[3]:  # R4
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                b, c = ind[0]
                ind = ind[1:len(ind)]
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == 2 and pag[i][b] != 0 and pag[c][i] == 3 and pag[i][c] == 2]

                while len(indA) > 0 and pag[c][b] == 1:
                    a = indA[0]
                    indA = indA[1:len(indA)]
                    done = False
                    while done == False and pag[a][b] != 0 and pag[a][c] != 0 and pag[b][c] != 0:
                        md_path = minDiscPath(pag, a, b, c)
                        if len(md_path) == 1:
                            done = True
                        else:
                            # a path exists
                            if b in sepset[md_path[0]][md_path[-1]] or b in sepset[md_path[-1]][md_path[0]]:
                                pag[b][c] = 2
                                pag[c][b] = 3
                            else:
                                pag[a][b] = pag[b][c] = pag[c][b] = 2
                            done = True

        if rules[4]:  # R5
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 1 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind[0]
                ind = ind[1:len(ind)]
                indC = [i for i in range(len(pag)) if
                        pag[a][i] == 1 and pag[i][a] == 1 and pag[b][i] == 0 and pag[i][b] == 0 and i != b]
                indD = [i for i in range(len(pag)) if
                        pag[b][i] == 1 and pag[i][b] == 1 and pag[a][i] == 0 and pag[i][a] == 0 and i != a]
                if len(indD) > 0 and len(indC) > 0:
                    counterC = -1
                    while counterC < len(indC) - 1 and pag[a][b] == 1:
                        counterC += 1
                        c = indC[counterC]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[a][b] == 1:
                            counterD += 1
                            d = indD[counterD]
                            if pag[c][d] == 1 and pag[d][c] == 1:
                                if len(unfVect) == 0:
                                    pag[a][b] = pag[b][a] = 3
                                    pag[a][c] = pag[c][a] = 3
                                    pag[c][d] = pag[c][d] = 3
                                    pag[d][b] = pag[b][d] = 3
                                else:
                                    path2check = [a, c, d, b]
                                    if faith_check(path2check, unfVect, p):
                                        pag[a][b] = pag[b][a] = 3
                                        pag[a][c] = pag[c][a] = 3
                                        pag[c][d] = pag[c][d] = 3
                                        pag[d][b] = pag[b][d] = 3
                            else:
                                ucp = minUncovCircPath(p, pag=pag, path=(a, c, d, b), unfVect=unfVect)
                                if len(ucp) > 1:
                                    pag[ucp[0]][ucp[-1]] = pag[ucp[-1]][ucp[0]] = 3
                                    for j in range(len(ucp) - 1):
                                        pag[ucp[j]][ucp[j + 1]] = pag[ucp[j + 1]][ucp[j]] = 3
        if rules[5]:  # R6
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            for b, c in ind:
                if len([i for i in range(len(pag)) if pag[b][i] == 3 and pag[i][b] == 3]) > 0:
                    pag[c][b] = 3
        if rules[6]:  # R7
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] != 0 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, c in ind:
                indA = [i for i in range(len(pag)) if
                        pag[b][i] == 3 and pag[i][b] == 1 and pag[c][i] == 0 and pag[i][c] == 0 and i != c]
                if len(indA) > 0:
                    if len(unfVect) == 0:
                        pag[c][b] = 3
                    else:
                        for a in indA:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                pag[c][b] = 3
        if rules[7]:  # R8
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag)) if
                        pag[i][a] == 3 and (pag[a][i] == 2 or pag[a][i] == 1) and pag[c][i] == 3 and pag[i][c] == 2]
                if len(indB) > 0:
                    pag[c][a] = 3
        if rules[8]:  # R8
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                a, c = ind[0]
                ind = ind[1:len(ind)]
                indB = [i for i in range(len(pag)) if
                        (pag[a][i] == 2 or pag[a][i] == 1) and
                        (pag[i][a] == 1 or pag[i][a] == 3) and
                        (pag[c][i] == 0 and pag[i][c] == 0) and
                        i != c]
                while len(indB) > 0 and pag[c][a] == 1:
                    b = indB[0]
                    indB = indB[1:len(indB)]
                    upd = minUncovPdPath(p, pag, a, b, c, unfVect=unfVect)
                    if len(upd) > 1:
                        pag[c][a] = 3
        if rules[9]:  # R10
            ind = []
            for i in range(len(pag)):
                for j in range(len(pag[i])):
                    if pag[i][j] == 2 and pag[j][i] == 1:
                        ind.append((i, j))
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind[0]
                ind = ind[1:len(ind)]
                indB = [i for i in range(p) if pag[c][i] == 3 and pag[i][c] == 2]
                if len(indB) >= 2:
                    counterB = -1
                    while counterB < len(indB) - 1 and pag[c][a] == 1:
                        counterB += 1
                        b = indB[counterB]
                        indD = [i for i in indB if i != b]
                        counterD = -1
                        while counterD < len(indD) - 1 and pag[c][a] == 1:
                            counterD += 1
                            d = indD[counterD]
                            if (
                                    (pag[a][b] == 1 or pag[a][b] == 2) and
                                    (pag[b][a] == 1 or pag[b][a] == 3) and
                                    (pag[a][d] == 1 or pag[a][d] == 2) and
                                    (pag[d][a] == 1 or pag[d][a] == 3) and
                                    pag[d][b] == 0 and pag[b][d] == 0
                            ):
                                if len(unfVect) == 0:
                                    pag[c][a] = 3
                                else:
                                    if (b, a, d) not in unfVect and (d, a, b) not in unfVect:
                                        pag[c][a] = 3
                            else:
                                indX = [i for i in range(p) if
                                        (pag[a][i] == 1 or pag[a][i] == 2) and
                                        (pag[i][a] == 1 or pag[i][a] == 3) and
                                        i != c]
                                if len(indX) >= 2:
                                    counterX1 = -1
                                    while counterX1 < len(indX) - 1 and pag[c][a] == 1:
                                        counterX1 += 1
                                        first_pos = indA[counterX1]
                                        indX2 = [i for i in indX if i != first_pos]
                                        counterX2 = -1
                                        while counterX2 < len(indX2) - 1 and pag[c][a] == 1:
                                            counterX2 += 1
                                            sec_pos = indX2[counterX2]
                                            t1 = minUncovPdPath(p, pag, a, first_pos, b, unfVect=unfVect)
                                            if len(t1) > 1:
                                                t2 = minUncovPdPath(p, pag, a, sec_pos, d, unfVect=unfVect)
                                                if len(t2) > 1 and first_pos != sec_pos and pag[first_pos][
                                                    sec_pos] == 0:
                                                    # we found 2 uncovered pd paths
                                                    if len(unfVect) == 0:
                                                        pag[c][a] = 3
                                                    elif (first_pos, a, sec_pos) not in unfVect and (
                                                            sec_pos, a, first_pos) not in unfVect:
                                                        pag[c][a] = 3
        return pag


def fci(suffStat, indepTest, alpha, labels, skel_method=("stable", "original", "stable.fast"), type='adaptive',
        fixedGaps=None, fixedEdges=None, NAdelete=True,
        m_max=float('inf'), pdsep_max=float('inf'), numCores=1, verbose=False, maj_rule=True):
    graphDict = skeleton(suffStat, indepTest, alpha, labels=labels, method=skel_method,
                         fixedGaps=fixedGaps, fixedEdges=fixedEdges,
                         NAdelete=NAdelete, m_max=m_max, numCores=numCores, verbose=verbose)
    # get sepset
    pc_ci = pc_cons_intern(graphDict, suffStat, alpha, indepTest, maj_rule=False)

    # recalculate sepsets and G, orient v structures
    pdSepRes = pdsep(graphDict, suffStat, indepTest=indepTest, p=len(labels), alpha=alpha, pMax=graphDict["pMax"],
                     m_max=graphDict["max_ord"], sepSet=pc_ci["sepset"], unfVect=pc_ci["unfTriples"])

    res = udag2pag(pdSepRes["G"], sepset=pdSepRes["sepset"], unfVect=set())
    print(np.array(res))
    return res


def gaussCItest(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]

    # def pcorOrder(i, j, k, C, cut_at=0.9999999):
    cut_at = 0.9999999
    if len(S) == 0:
        r = C[x, y]
    elif len(S) == 1:
        r = (C[x, y] - C[x, S[0]] * C[y, S[0]]) / math.sqrt(
            (1 - math.pow(C[y, S[0]], 2)) * (1 - math.pow(C[x, S[0]], 2)))
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
    # file = 'datasets/gmD.csv'
    # file = 'datasets/BD Cont.csv'
    # file = 'datasets/BD Disc.csv'
    # file = 'datasets/BD5 Cluster X Disc Y Outcome (2).csv'
    # file = 'datasets/BD5 Cluster X2 Cont X1 Outcome (1).csv'
    # file = 'datasets/BD5 Cluster X2 Disc X1 Outcome (1).csv'
    # file = 'datasets/ID1 Disc (1).csv'
    # file = 'datasets/ID1 Disc (2).csv'
    file = 'datasets/mdata2.csv'
    data = pd.read_csv(file)

    print(data.columns)
    # p = fci(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05,labels=[str(i) for i in data.columns], indepTest=gaussCItest)
    # p = fci(suffStat=data.values, alpha=.05, labels=[str(i) for i in data.columns], indepTest=ci_test_dis)
    p = pc(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05,labels=[str(i) for i in data.columns], indepTest=gaussCItest, verbose=False)
    # p = pc(suffStat=data.values, alpha=.05, labels=[str(i) for i in data.columns], indepTest=ci_test_dis)
    print(p)
