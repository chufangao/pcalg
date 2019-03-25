import copy
import itertools
from itertools import combinations, chain
from gsq.ci_tests import ci_test_dis
from scipy.stats import norm
import pandas as pd
import numpy as np
import math
from ges import ges


def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def gaussCItest(suffstat, x, y, S):
    C = suffstat["C"]
    n = suffstat["n"]
    cut_at = 0.9999999
    if len(S) == 0:
        r = C[x, y]
    elif len(S) == 1:
        r = (C[x, y] - C[x, S[0]] * C[y, S[0]]) / math.sqrt(
            (1 - math.pow(C[y, S[0]], 2)) * (1 - math.pow(C[x, S[0]], 2)))
    else:
        M = C[np.ix_([x] + [y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(M)
        r = -1 * PM[0, 1] / math.sqrt(PM[0, 0] * PM[1, 1])
    r = min(cut_at, max(-1 * cut_at , r))

    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    return 2 * (1 - norm.cdf(abs(res)))


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

    if sepsetA == None:
        sepsetA = set()
    if sepsetC == None:
        sepsetC = set()

    if len(temp) == 0:
        temp.append(False)

    res = 3
    if maj_rule:
        if sum(temp) / len(temp) < .5:
            res = 1
            if b in sepsetA:
                sepsetA.remove(b)
            if b in sepsetC:
                sepsetC.remove(b)
        elif sum(temp) / len(temp) > .5:
            res = 2
            sepsetA.add(b)
            sepsetC.add(b)
        else:
            pass  # unfaithful
    else:
        if sum(temp) / len(temp) == 0:
            res = 1
            if b in sepsetA:
                sepsetA.remove(b)
            if b in sepsetC:
                sepsetC.remove(b)
        elif sum(temp) / len(temp) == 1:
            res = 2
            sepsetA.add(b)
            sepsetC.add(b)
        else:
            pass  # unfaithful

    return res, {'sepsetA': sepsetA, 'sepsetC': sepsetC}


def pc_cons_intern(graphDict, suffstat, alpha, indepTest, version_unf=(None, None), maj_rule=True,
                   verbose=False):
    sk = graphDict['sk']

    if sk.any():
        ind = [(i, j)
               for i in range(len(sk))
               for j in range(len(sk))
               if sk[i][j] == True
               ]
        ind = sorted(ind, key=lambda x: (x[1], x[0]))

        tripleMatrix = [(a, b, c)
                        for a, b in ind
                        for c in range(len(sk))
                        if a < c and sk[a][c] == False and sk[b][c] == True]
        # go thru all edges
        for a, b, c in tripleMatrix:
            nbrsA = [i for i in range(len(sk)) if sk[i][a] == True]
            nbrsC = [i for i in range(len(sk)) if sk[i][c] == True]

            res, r_abc = checkTriple(a, b, c, nbrsA, nbrsC, graphDict['sepset'][a][c],
                                     graphDict['sepset'][c][a],
                                     suffstat, alpha, indepTest, maj_rule=maj_rule)
            if res == 3:
                if 'unfTriples' in graphDict.keys():
                    graphDict['unfTriples'].add((a, b, c))
                else:
                    graphDict['unfTriples'] = {(a, b, c)}
            graphDict['sepset'][a][c] = r_abc['sepsetA']
            graphDict['sepset'][c][a] = r_abc['sepsetC']

    return graphDict


def skeleton(suffStat, indepTest, alpha, labels, method,
             fixedGaps, fixedEdges,
             NAdelete, m_max, numCores, verbose):
    sepset = [[None for i in range(len(labels))] for i in range(len(labels))]

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
    n_edgetests = {}

    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True
        G1 = G.copy()

        ind = [(i, j)
               for i in range(len(G))
               for j in range(len(G[i]))
               if G[i][j] == True
               ]
        for x, y in ind:
            if G[y][x] == True:
                nbrs = [i for i in range(len(G1)) if G1[x][i] == True and i != y]
                if len(nbrs) >= ord:
                    if len(nbrs) > ord:
                        done = False

                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        # get pvalue, if dependent, pval should be small
                        pval = indepTest(suffStat, x, y, list(nbrs_S))
                        if pMax[x][y] < pval:
                            pMax[x][y] = pval
                        if pval >= alpha:
                            # then independent
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = set(nbrs_S)
                            break
        ord += 1
    # fix p values
    for i in range(0, len(labels) - 1):
        for j in range(1, len(labels)):
            pMax[i][j] = pMax[j][i] = max(pMax[i][j], pMax[j][i])

    return {'sk': np.array(G), 'pMax': np.array(pMax), 'sepset': sepset, "unfTriples": set(), "max_ord": ord - 1}


def udag2pdagRelaxed(graph):
    def orientConflictCollider(pdag, x, y, z):
        # x -> y <- z
        # pdag: 2d list, pdag[x,y] = 1 and pdag[y,x] = 0 means x -> y
        # returns updated pdag
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
        ind = [(i, j)
               for i in range(len(pdag)) for j in range(len(pdag))
               if pdag[i][j] == 1 and pdag[j][i] == 0]
        # sort to correspond with r
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            # for a, b in ind:
            isC = [i for i in range(len(search_pdag))
                   if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1)
                   and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0)]
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
        ind = [(i, j)
               for i in range(len(pdag)) for j in range(len(pdag))
               if pdag[i][j] == 1 and pdag[j][i] == 1]
        # sort to correspond with r
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            # for a,b in ind:
            isC = [i for i in range(len(search_pdag))
                   if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and
                   (search_pdag[i][b] == 1 and search_pdag[b][i] == 0)]
            # for i in range(len(search_pdag)):
            #     if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (
            #             search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
            #         isC.append(i)
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
        ind = [(i, j)
               for i in range(len(pdag)) for j in range(len(pdag))
               if pdag[i][j] == 1 and pdag[j][i] == 1]
        # sort to correspond with r
        for a, b in sorted(ind, key=lambda x: (x[1], x[0])):
            # for a,b in ind:
            isC = [i for i in range(len(search_pdag))
                   if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and
                   (search_pdag[i][b] == 1 and search_pdag[b][i] == 0)
                   ]
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

    pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]
    ind = [(i, j)
           for i in range(len(pdag)) for j in range(len(pdag))
           if pdag[i][j] == 1]
    # need to sort to correspond with R version
    for x, y in sorted(ind, key=lambda x: (x[1], x[0])):
        allZ = [z for z in range(len(pdag))
                if graph['sk'][y][z] == True and z != x]
        for z in allZ:
            if graph['sk'][x][z] == False and graph['sepset'][x][z] != None and graph['sepset'][z][x] != None and \
                    not (y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    # do while
    old_dag = pdag.copy()
    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    while old_dag != pdag:
        pdag = rule1(pdag)
        pdag = rule2(pdag)
        pdag = rule3(pdag)

    return np.array(pdag)


def qreach(x, amat):
    # Purpose: Compute possible-d-sep(x) ("psep")
    def legal_path(a, b, c, amat):
        # Purpose: Is path a-b-c legal (either collider in b or a,b,c is triangle)
        if a == c or amat[a][b] == 0 or amat[b][c] == 0:
            return False
        return amat[a][c] != 0 or (amat[a][b] == 2 and amat[c][b] == 2)

    A = [[1 if amat[i][j] != 0 else 0 for i in range(len(amat))] for j in range(len(amat))]
    PSEP = [i for i in range(len(A[x])) if A[x][i] != 0]
    Q = copy.deepcopy(PSEP)
    nb = copy.deepcopy(PSEP)
    P = [x for i in range(len(Q))]
    for i in nb:  # delete edge to nbrs
        A[x][i] = 0

    while len(Q) > 0:
        a = Q.pop(0)
        pred = P.pop(0)
        nb = [i for i in range(len(A[a])) if A[a][i] != 0]

        for b in nb:
            lres = legal_path(pred, a, b, amat)
            if lres == True:
                # if amat[pred][b] != 0 or (amat[pred][a] == 2 and amat[b][a] == 2):
                A[a][b] = 0
                Q.append(b)
                P.append(a)
                PSEP.append(b)
    while x in PSEP:
        PSEP.remove(x)
    return sorted(set(PSEP))


def pdsep(skel, suffStat, indepTest, p, sepset, alpha, pMax, m_max=float('inf'), pdsep_max=float('inf'), unfVect=None):
    G = [[0 if skel['sk'][i][j] == False else 1 for i in range(len(skel['sk']))] for j in range(len(skel['sk']))]
    n_edgetest = [0 for i in range(1000)]
    ord = 0

    amat = copy.deepcopy(G)
    ind = [(i, j)
           for i in range(len(G)) for j in range(len(G))
           if G[i][j] == 1]
    ind = sorted(ind, key=lambda x: (x[1], x[0]))
    # orient colliders
    for x, y in ind:
        allZ = [i for i in range(len(amat[y])) if amat[y][i] != 0 and i != x]
        for z in allZ:
            if amat[x][z] == 0 and not (y in sepset[x][z] or y in sepset[z][x]):
                if len(unfVect) == 0:  # normal version
                    amat[x][y] = amat[z][y] = 2
                else:  # conservative version, check if x-y-z faithful
                    if (x, y, z) not in unfVect and (z, y, x) not in unfVect:
                        amat[x][y] = amat[z][y] = 2

    allPdsep = [qreach(x, amat) for x in range(p)]
    allPdsep_tmp = [[] for i in range(p)]

    for x in range(p):
        an0 = [True if amat[x][i] != 0 else False for i in range(len(amat))]
        if any(an0):
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
                        ord += 1
                        if ord == 1:
                            for S in diff_set:
                                pval = indepTest(suffStat, x, y, [S])
                                n_edgetest[ord + 1] += 1
                                if pval > pMax[x][y]:
                                    pMax[x][y] = pval
                                if pval >= alpha:
                                    amat[x][y] = amat[y][x] = 0
                                    sepset[x][y] = sepset[y][x] = {S}
                                    done = True
                                    break
                        else:  # ord > 1
                            tmp_combn = combinations(tf, ord)
                            if ord <= len(adj_x):
                                for S in tmp_combn:
                                    if not set(S).issubset(adj_x):
                                        n_edgetest[ord + 1] += 1
                                        pval = indepTest(suffStat, x, y, list(S))
                                        if pval > pMax[x][y]:
                                            pMax[x][y] = pval
                                        if pval >= alpha:
                                            amat[x][y] = amat[y][x] = 0
                                            sepset[x][y] = sepset[y][x] = set(S)
                                            done = True
                                            break
                            else:  # ord > len(adj_x)
                                for S in tmp_combn:
                                    n_edgetest[ord + 1] += 1
                                    pval = indepTest(suffStat, x, y, list(S))
                                    if pval > pMax[x][y]:
                                        pMax[x][y] = pval
                                    if pval >= alpha:
                                        amat[x][y] = amat[y][x] = 0
                                        sepset[x][y] = sepset[y][x] = set(S)
                                        done = True
                                        break
    for i in range(len(amat)):
        for j in range(len(amat[i])):
            if amat[i][j] == 0:
                G[i][j] = False
            else:
                G[i][j] = True

    return {'G': G, "sepset": sepset, "pMax": pMax, "allPdsep": allPdsep_tmp, "max_ord": ord}


def updateList(path, set, old_list):  # arguments are all lists
    temp = []
    if len(old_list) > 0:
        temp = old_list
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
    # Purpose: find a minimal discriminating path for a,b,c.
    p = len(pag)
    visited = [False for i in range(p)]
    visited[a] = visited[b] = visited[c] = True
    # find all neighbours of a  not visited yet
    indD = [i for i in range(len(pag))
            if pag[a][i] != 0 and pag[i][a] == 2 and visited[i] == False]
    if len(indD) > 0:
        path_list = updateList([a], indD, [])
        while len(path_list) > 0:
            # next element in the queue
            mpath = path_list[0]
            d = mpath[-1]
            if pag[c][d] == 0 and pag[d][c] == 0:
                # minimal discriminating path found :
                mpath.reverse()
                return mpath + [b, c]
            else:
                pred = mpath[-2]
                path_list.pop(0)
                visited[d] = True
                # d is connected to c -----> search iteratively
                if pag[d][c] == 2 and pag[c][d] == 3 and pag[pred][d] == 2:
                    # find all neighbours of d not visited yet
                    indR = [i for i in range(len(pag))
                            if pag[d][i] != 0 and pag[i][d] == 2 and visited[i] == False]  # r *-> d
                    if len(indR) > 0:
                        # update the queues
                        path_list = updateList(mpath, indR, path_list)
    # nothing found:  return
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
            path_list.pop(0)
            visted[x] = True
            if pag[x][d] == 1 and pag[d][x] == 1:
                mpath = [a] + mpath + [d, b]
                uncov = True
                for i in range(len(mpath) - 3):
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
            mpath = path_list.pop(0)
            d = mpath[-1]
            visited[d] = True
            if pag[d][c] in [1, 2] and pag[c][d] in [1, 3]:
                # pd path found
                mpath = [a] + mpath + [c]
                uncov = True
                for i in range(len(mpath) - 3):
                    if not (pag[mpath[i]][mpath[i + 2]] == 0 and pag[mpath[i + 2]][mpath[i]] == 0):
                        uncov = False
                        break
                if uncov == True:
                    if len(unfVect) == 0 or faith_check(mpath, unfVect, p):
                        min_upd_path = mpath
                        done = True
            else:
                indR = [i for i in range(p)
                        if (pag[d][i] == 1 or pag[d][i] == 2) and
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
        ind = [(i, j)
               for i in range(p) for j in range(p)
               if pag[i][j] == 1]
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

    old_pag1 = None
    while old_pag1 != pag:
        old_pag1 = copy.deepcopy(pag)
        # R1 ----------------------------------------------------------------------
        if rules[0]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] == 2 and pag[j][i] != 0]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, b in ind:
                indC = [i for i in range(len(pag))
                        if pag[b][i] != 0 and pag[i][b] == 1 and
                        pag[a][i] == 0 and pag[i][a] == 0 and i != a]
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
        # R2 ----------------------------------------------------------------------
        if rules[1]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] == 1 and pag[j][i] != 0]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag))
                        if (pag[a][i] == 2 and pag[i][a] == 3 and pag[c][i] != 0 and pag[i][c] == 2) or
                        (pag[a][i] == 2 and pag[i][1] != 0 and pag[c][i] == 3 and pag[i][c] == 2)]
                if len(indB) > 0:
                    pag[a][c] = 2
        # R3 ----------------------------------------------------------------------
        if rules[2]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] != 0 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, d in ind:
                indAC = [i for i in range(len(pag))
                         if pag[b][i] != 0 and pag[i][b] == 2 and pag[i][d] == 1 and pag[d][i] != 0]
                if len(indAC) >= 2:
                    if len(unfVect) == 0:
                        counter = -1
                        while counter < len(indAC) - 2 and pag[d][b] != 2:
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
        # R4 ----------------------------------------------------------------------
        if rules[3]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] != 0 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                b, c = ind.pop(0)
                indA = [i for i in range(len(pag))
                        if pag[b][i] == 2 and pag[i][b] != 0 and pag[c][i] == 3 and pag[i][c] == 2]

                while len(indA) > 0 and pag[c][b] == 1:
                    a = indA.pop(0)
                    # path is the initial triangle
                    # Done is TRUE if either we found a minimal path or no path exists for this triangle
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
        # R5 ----------------------------------------------------------------------
        if rules[4]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] == 1 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind.pop(0)
                indC = [i for i in range(p)
                        if pag[a][i] == 1 and pag[i][a] == 1 and
                        pag[b][i] == 0 and pag[i][b] == 0 and i != b]
                indD = [i for i in range(p)
                        if pag[b][i] == 1 and pag[i][b] == 1 and
                        pag[a][i] == 0 and pag[i][a] == 0 and i != a]
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
                                    for j in range(len(ucp) - 2):
                                        pag[ucp[j]][ucp[j + 1]] = pag[ucp[j + 1]][ucp[j]] = 3
        # R6 ----------------------------------------------------------------------
        if rules[5]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] != 0 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            for b, c in ind:
                if len([i for i in range(len(pag)) if pag[b][i] == 3 and pag[i][b] == 3]) > 0:
                    pag[c][b] = 3
        # R7 ----------------------------------------------------------------------
        if rules[6]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] != 0 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for b, c in ind:
                indA = [i for i in range(len(pag))
                        if pag[b][i] == 3 and pag[i][b] == 1 and
                        pag[c][i] == 0 and pag[i][c] == 0 and i != c]

                if len(indA) > 0:
                    if len(unfVect) == 0:
                        pag[c][b] = 3
                    else:
                        for a in indA:
                            if (a, b, c) not in unfVect and (c, b, a) not in unfVect:
                                pag[c][b] = 3
        # R8 ----------------------------------------------------------------------
        if rules[7]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] == 2 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            for a, c in ind:
                indB = [i for i in range(len(pag))
                        if pag[i][a] == 3 and (pag[a][i] == 2 or pag[a][i] == 1) and
                        pag[c][i] == 3 and pag[i][c] == 2]
                if len(indB) > 0:
                    pag[c][a] = 3
        # R9 ----------------------------------------------------------------------
        if rules[8]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] == 2 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))

            while len(ind) > 0:
                a, c = ind.pop(0)
                indB = [i for i in range(len(pag))
                        if (pag[a][i] == 2 or pag[a][i] == 1) and
                        (pag[i][a] == 1 or pag[i][a] == 3) and
                        (pag[c][i] == 0 and pag[i][c] == 0) and
                        i != c]
                while len(indB) > 0 and pag[c][a] == 1:
                    b = indB.pop(0)
                    upd = minUncovPdPath(p, pag, a, b, c, unfVect=unfVect)
                    if len(upd) > 1:
                        pag[c][a] = 3
        # R10 ----------------------------------------------------------------------
        if rules[9]:
            ind = [(i, j)
                   for i in range(p) for j in range(p)
                   if pag[i][j] == 2 and pag[j][i] == 1]
            ind = sorted(ind, key=lambda x: (x[1], x[0]))
            while len(ind) > 0:
                a, b = ind.pop(0)
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
                                indX = [i for i in range(p)
                                        if (pag[a][i] == 1 or pag[a][i] == 2) and
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
                                                if len(t2) > 1 and first_pos != sec_pos \
                                                        and pag[first_pos][sec_pos] == 0:
                                                    # we found 2 uncovered pd paths
                                                    if len(unfVect) == 0:
                                                        pag[c][a] = 3
                                                    elif (first_pos, a, sec_pos) not in unfVect and (
                                                            sec_pos, a, first_pos) not in unfVect:
                                                        pag[c][a] = 3
    return np.array(pag)


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
    print(graph['sepset'])
    if verbose:
        print('Got sepsets')
    # apply rules
    return udag2pdagRelaxed(graph)


def fci(suffStat, indepTest, alpha, labels, skel_method=("stable", "original", "stable.fast"), type='adaptive',
        fixedGaps=None, fixedEdges=None, NAdelete=True,
        m_max=float('inf'), pdsep_max=float('inf'), numCores=1, verbose=False, maj_rule=False,
        rules=(True, True, True, True, True, True, True, True, True, True)):
    graphDict = skeleton(suffStat, indepTest, alpha, labels=labels, method=skel_method,
                         fixedGaps=fixedGaps, fixedEdges=fixedEdges,
                         NAdelete=NAdelete, m_max=m_max, numCores=numCores, verbose=verbose)
    # get sepset
    pc_ci = pc_cons_intern(graphDict, suffStat, alpha, indepTest, maj_rule=maj_rule)

    # recalculate sepsets and G, orient v structures
    pdSepRes = pdsep(graphDict, suffStat, indepTest=indepTest, p=len(labels), alpha=alpha, pMax=graphDict["pMax"],
                     m_max=graphDict["max_ord"], sepset=pc_ci["sepset"], unfVect=pc_ci["unfTriples"])
    # print(pdSepRes['allPdsep'])
    # print(pdSepRes['sepset'])

    res = udag2pag(pdSepRes["G"], sepset=pdSepRes["sepset"], unfVect=set(), rules=rules)
    return res


if __name__ == '__main__':
    # file = 'datasets/gmD.csv'
    # file = 'datasets/BD Cont.csv'
    # file = 'datasets/BD Disc.csv'
    # file = 'datasets/BD5 Cluster X Disc Y Outcome (2).csv'
    # file = 'datasets/BD5 Cluster X2 Cont X1 Outcome (1).csv'
    # file = 'datasets/BD5 Cluster X2 Disc X1 Outcome (1).csv'
    # file = 'datasets/ID1 Disc (1).csv'
    # file = 'datasets/ID1 Disc (2).csv'
    # file = 'datasets/mdata.csv'
    # file = 'datasets/mdata2.csv'
    # file = 'datasets/dataset1-continuous.csv'
    file = 'C:/Users/gaoan/Downloads/Microsoft.SkypeApp_kzf8qxf38zg5c!App/All/Learn Model Test/datasets/kaggle/admission 1.1.csv'
    data = pd.read_csv(file)
    print(data.columns)
    # print(gaussCItest({"C": data.corr().values, "n": data.values.shape[0]}, 1, 4, [2,0])); quit()

    # rules = [True for i in range(10)]
    p = fci(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05,
            labels=[str(i) for i in data.columns], indepTest=gaussCItest)
    # p = fci(suffStat=data.values, alpha=.05, labels=[str(i) for i in data.columns], indepTest=ci_test_dis)
    # p = pc(suffStat={"C": data.corr().values, "n": data.values.shape[0]}, alpha=.05,labels=[str(i) for i in data.columns], indepTest=gaussCItest)
    # p = pc(suffStat=data.values, alpha=.05, labels=[str(i) for i in data.columns], indepTest=ci_test_dis)
    # p = ges(data)
    # print(len([1 for i in range(len(p)) for j in range(len(p)) if p[i,j]!=0]))
    print(p)
