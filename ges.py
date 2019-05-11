import copy
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from oct2py import octave


class EssentialGraph:
    def __init__(self, data, numVertices=0):
        self.numVertices = numVertices
        self.G = nx.DiGraph()
        for i in range(self.numVertices):
            self.G.add_node(i)
        self.data = data
        self.undirectedEdges = set()
        self.datacount = [len(data) for i in range(self.numVertices)]
        self.maxVertexDegree = [self.numVertices for i in range(self.numVertices)]

    def local(self, v, C_par):
        if len(C_par) == 0:
            return 0
        X = self.data.iloc[:, C_par]
        y = self.data.iloc[:, v]
        n = self.data.shape[0]
        k = len(C_par)
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        y_hat = reg.predict(X)
        resid = y - y_hat
        sse = sum(resid ** 2)
        BIC = n * np.log(sse / n) + k * np.log(n)
        return -1 * BIC

    def local2(self, v, C_par):
        # ScoreGaussL0PenRaw
        y = self.data.iloc[:, v]
        n = self.data.shape[0]
        a = np.sum(np.square(y))

        if len(C_par) > 0:
            Z = self.data.iloc[:,    C_par].values

            # try:
                # Q, R = np.linalg.qr(Z)
            Q, R = octave.feval('qr', Z, 0, nout=2)
            # print(Q, R)
            # except:
                # return 0

            a -= np.square(np.linalg.norm(y.transpose().dot(Q)))

        if np.abs(a) < 1e-08:
            return float('inf')
        return -.5 * (1 + np.log(a / self.datacount[v])) * self.datacount[v] - .5 * np.log(n) * (1 + len(C_par))
    # def local2(self, c, C_par):
        # ScoreGaussL0PenScatter



    def existsPath(self, a, b, C, undirected=False):
        # trivial cases
        if a in C or b in C:
            return False

        # mark forbidden vertices as visited
        visited = [i for i in C]

        # if a-b is an edge, remove it and add it again in the end
        restore = self.G.has_edge(a, b)
        if restore == True:
            self.G.remove_edge(a, b)

        # check using dfs whether b is reacheable from a without using vertices in C
        nbhd = [a]
        if a not in visited:
            visited.append(a)
        while len(nbhd) > 0:
            v = nbhd.pop(0)
            for vi in nx.all_neighbors(self.G, v):
                if not undirected or self.G.has_edge(vi, v):
                    if vi == b:
                        if restore == True:
                            self.G.add_edge(a, b)
                        return True
                    if vi not in visited:
                        nbhd.append(vi)
                        visited.append(vi)
        if restore == True:
            self.G.add_edge(a, b)
        return False

    def getVertexCount(self):
        # return number of vertices in the graph
        return self.numVertices

    def getPosteriorSet(self, A):
        totalsucc = []
        result = []
        for vi in A:
            totalsucc.append(vi)
            result.append(vi)
            while len(totalsucc) > 0:
                a = totalsucc.pop(0)
                for b in self.G.successors(a):
                    if b not in result:
                        totalsucc.append(b)
                        result.append(b)
        return result

    def getOptimalArrowInsertion(self, v):
        result = {}
        result['score'] = 0
        localScore = {}

        if len(list(nx.all_neighbors(self.G, v))) >= self.getVertexCount():
            return result

        # find maximal clique in the neighborhood of v
        # neighbors = nx.all_neighbors(self.G, v)
        neighbors = self.getNeighbors(v)
        maxCliques = self.getMaxCliques(neighbors)
        parents = self.G.predecessors(v)
        # forbidden = self.getPosteriorSet(self.G.successors(v))
        forbidden = list(nx.dfs_successors(self.G, v))

        # get adjacent vertices to v
        tempSet = nx.all_neighbors(self.G, v)
        for i in tempSet:
            if i not in forbidden:
                forbidden.append(i)
        # v itself
        if v not in forbidden:
            forbidden.append(v)
        for i in range(self.numVertices):
            if self.G.degree[i] >= self.maxVertexDegree[i]:
                if i not in forbidden:
                    forbidden.append(i)

        tempSet = [v]
        posterior = self.getPosteriorSet(tempSet)

        for u in range(self.getVertexCount()):
            if u not in forbidden:
                N = list(set(neighbors).intersection(nx.all_neighbors(self.G, u)))
                cliqueStack = [N]
                stopsets = [N]

                for i in maxCliques:
                    if set(N).issubset(i):
                        # if i not in cliqueStack:
                        #     cliqueStack.append(i)
                        append = True
                        for C_temp in stopsets:
                            if set(i).issubset(C_temp):
                                append = False
                        if append == True:
                            cliqueStack.append(i)

                        while len(cliqueStack) > 0:
                            C = cliqueStack.pop()
                            if u not in posterior or not self.existsPath(v, u, C):
                                C_par = list(set(C).union(parents))
                                if len(localScore.keys()) == 0 or tuple(C_par) not in localScore.keys():
                                    diffScore = -1 * self.local2(v, C_par)
                                    # print('before', diffScore, C_par)
                                    localScore[tuple(C_par)] = diffScore
                                else:
                                    diffScore = localScore[tuple(C_par)]
                                    # print('before', diffScore, C_par)
                                C_par.append(u)

                                diffScore += self.local2(v, C_par)
                                # print('after', diffScore, C_par)

                                if diffScore > result['score']:
                                    result["source"] = u
                                    result["clique"] = C
                                    result["score"] = diffScore

                            for si in C:
                                C_sub = copy.copy(C)
                                while si in C_sub:
                                    C_sub.remove(si)
                                # if C_sub not in cliqueStack:
                                #     cliqueStack.append(C_sub)
                                append = True
                                for C_temp in stopsets:
                                    if set(C_sub).issubset(C_temp):
                                        append = False
                                if append == True:
                                    cliqueStack.append(C_sub)

                        stopsets.append(i)

        return result

    def greedyForward(self, verbose=False):
        v_opt = 0
        optInsertion = {}
        optInsertion["score"] = 0

        # calculate score differences for all possible edges
        for v in range(self.getVertexCount()):
            insertion = self.getOptimalArrowInsertion(v)
            print(insertion, v)

            # if insertion["score"] > optInsertion["score"] - 1e-08:
            if insertion["score"] > optInsertion["score"]:
                optInsertion = insertion
                v_opt = v

        if optInsertion["score"] > 0:
            u_opt = optInsertion["source"]
            if verbose: print(u_opt, v_opt, optInsertion['clique'], optInsertion['score'])
            self.insert(u_opt, v_opt, optInsertion["clique"])
            print(nx.to_numpy_matrix(self.G))
            return True

        return False

    def greedyBackward(self):
        v_opt = 0
        optDeletion = {}
        optDeletion["score"] = 0

        for v in range(self.getVertexCount()):
            deletion = self.getOptimalArrowDeletion(v)
            if deletion['score'] > optDeletion['score']:
                optDeletion = deletion
                v_opt = v

        if optDeletion['score'] > 0:
            print(optDeletion['source'], v_opt, optDeletion['clique'], optDeletion['score'])
            self.remove(optDeletion['source'], v_opt, optDeletion['clique'])
            return True
        else:
            return False

    def insert(self, u, v, C):
        chainComp = self.getChainComponent(v)
        startOrder = copy.copy(C)
        startOrder.append(v)
        chainComp.remove(v)
        startOrder.extend(list(set(chainComp).difference(C)))

        self.lexBFS(startOrder, orient=True)
        self.add_edge(u, v)
        self.replaceUnprotected()

    def remove(self, u, v, C):
        chainComp = self.getChainComponent(v)
        startOrder = copy.copy(C)
        if u not in chainComp:
            startOrder.append(u)
        startOrder.append(v)
        chainComp.remove(v)
        startOrder.extend(list(set(chainComp).difference(C)))

        self.lexBFS(startOrder, orient=True)
        self.remove_edge(u, v, True)
        self.replaceUnprotected()

    def add_edge(self, a, b, undirected=False):
        if not self.G.has_edge(a, b):
            self.G.add_edge(a, b)

        if undirected == True and not self.G.has_edge(b, a):
            self.G.add_edge(b, a)

    def remove_edge(self, a, b, bothDirections=False):
        if self.G.has_edge(a, b):
            self.G.remove_edge(a, b)
        if bothDirections == True and self.G.has_edge(b, a):
            self.G.remove_edge(b, a)

    def getChainComponent(self, v):
        nbhd = [v]
        chainComp = set()

        while len(nbhd) > 0:
            a = nbhd.pop()
            chainComp.add(a)
            for vi in nx.all_neighbors(self.G, a):
                if self.G.has_edge(vi, a) and vi not in nbhd and vi not in chainComp:
                    nbhd.append(vi)
        return list(chainComp)

    def lexBFS(self, startOrder, orient=False):
        ordering = []
        length = len(startOrder)

        # Trivial cases: if not more than one start vertex is provided,
        # return an empty set of edges
        if length == 1:
            ordering.append(startOrder[0])
        if length <= 1:
            return ordering

        # Create sequence of sets ("\Sigma") containing the single set
        # of all vertices in the given start order

        sets = [startOrder]

        while len(sets) > 0:
            # Remove the first vertex from the first set, and remove this set
            # if it becomes empty
            a = sets[0].pop(0)
            if len(sets[0]) == 0:
                sets.pop(0)

            # Append a to the ordering
            ordering.append(a)

            # Move all neighbors of a into own sets, and orient visited edges
            # away from a
            si = 0
            while si < len(sets):
                newSet = []
                tempCopy = sets[si].copy()
                for vi in tempCopy:
                    if self.G.has_edge(a, vi):
                        # Orient edge to neighboring vertex, if requested, and
                        # store oriented edge in return set
                        if orient == True:
                            if self.G.has_edge(vi, a):
                                self.G.remove_edge(vi, a)
                        # directed.add((a,vi))

                        # Move neighoring vertex
                        newSet.append(vi)
                        sets[si].remove(vi)
                if len(sets[si]) == 0 and len(newSet) != 0:
                    sets.remove(sets[si])
                    sets.insert(si, newSet)
                    si += 1
                elif len(sets[si]) == 0:
                    sets.remove(sets[si])
                elif len(newSet) != 0:
                    sets.insert(si, newSet)
                    si += 2
                else:
                    si += 1
                
        return ordering

    def isParent(self, a, b):
        return self.G.has_edge(a, b) and not self.G.has_edge(b, a)

    def isNeighbor(self, a, b):
        return self.G.has_edge(a, b) and self.G.has_edge(b, a)

    def isAdjacent(self, a, b):
        return self.G.has_edge(a, b) or self.G.has_edge(b, a)

    def getNeighbors(self, edge):
        return [i for i in nx.all_neighbors(self.G, edge) if self.isNeighbor(i, edge)]

    def replaceUnprotected(self):
        UNDECIDABLE = 0
        PROTECTED = 1
        NOT_PROTECTED = 2
        arrowFlags = {}
        undecidableArrows = set()
        result = set()

        for edge in list(self.G.edges):
            if not self.G.has_edge(edge[1], edge[0]):
                undecidableArrows.add(edge)
                arrowFlags[edge] = UNDECIDABLE

        # Check whether the arrows are part of a v-structure; if yes, mark them as "protected".
        for v in range(self.getVertexCount()):
            for edge1 in [i for i in arrowFlags.keys() if i[1] == v]:
                for edge2 in [i for i in arrowFlags.keys() if i[1] == v and i[0] > edge1[0]]:
                    if not self.isAdjacent(edge1[0], edge2[0]):
                        arrowFlags[edge1] = PROTECTED
                        arrowFlags[edge2] = PROTECTED
                        if edge1 in undecidableArrows:
                            undecidableArrows.remove(edge1)
                        if edge2 in undecidableArrows:
                            undecidableArrows.remove(edge2)

        # Successively check all undecidable arrows, until no one remains
        labeledArrows = 1
        while len(undecidableArrows) > 0 and labeledArrows > 0:
            for edge in undecidableArrows:
                flag = NOT_PROTECTED
                # check if in configuration a                    
                for edge2 in [e for e in arrowFlags if e[1] == edge[0]]:
                    if not self.isAdjacent(edge2[0], edge[1]):
                        if arrowFlags[edge2] == PROTECTED:
                            flag = PROTECTED
                            break
                        else:
                            flag = UNDECIDABLE
                # check if in configuration c
                if flag==PROTECTED:
                    arrowFlags[edge] = flag                    
                    continue
                for edge2 in [e for e in arrowFlags if e[1] == edge[1]]:
                    if self.isParent(edge[0], edge2[0]):
                        if arrowFlags[edge2] == PROTECTED and arrowFlags[(edge[0], edge2[0])] == PROTECTED:
                            flag = PROTECTED
                            break
                        else:
                            flag = UNDECIDABLE
                # check if in configuration d
                if flag==PROTECTED:
                    arrowFlags[edge] = flag
                    continue
                for edge2 in [e for e in arrowFlags if e[1] == edge[1]]:
                    if flag == PROTECTED:
                        break
                    for edge3 in [e for e in arrowFlags if e[1] == edge[1] and e[0] > edge2[0]]:
                        if self.isNeighbor(edge[0], edge2[0]) and \
                                self.isNeighbor(edge[0], edge3[0]) and \
                                not self.isAdjacent(edge2[0], edge3[0]):
                            if arrowFlags[edge2] == PROTECTED and arrowFlags[edge3] == PROTECTED:
                                flag = PROTECTED
                                break
                            else:
                                flag = UNDECIDABLE
                arrowFlags[edge] = flag

            # Replace unprotected arrows by lines; store affected edges in result set
            # labeledArrows = len(undecidableArrows)
            # arrowFlagsKeys = list(arrowFlags.keys())
            # print('arrowFlagsKeys', arrowFlagsKeys)
            # for edge1 in arrowFlagsKeys:
            #     if edge1 in undecidableArrows and arrowFlags[edge1] != UNDECIDABLE:
            #         undecidableArrows.remove(edge1)
            #     if edge1 in arrowFlags.keys() and arrowFlags[edge1] == NOT_PROTECTED:
            #         self.add_edge(edge1[1], edge1[0])
            #         result.add(edge1)
            #         arrowFlags.pop(edge1)
            labeledArrows = len(undecidableArrows)
            arrowFlagsKeys = list(arrowFlags.keys())
            # print('arrowFlagsKeys', arrowFlagsKeys)
            for edge1 in arrowFlagsKeys:
                if arrowFlags[edge1] != UNDECIDABLE:
                    if edge1 in undecidableArrows:
                        undecidableArrows.remove(edge1)
                if arrowFlags[edge1] == NOT_PROTECTED:
                    self.add_edge(edge1[1], edge1[0])
                    result.add(edge1)
                    arrowFlags.pop(edge1)

            labeledArrows -= len(undecidableArrows)

        if labeledArrows == 0 and len(undecidableArrows) > 0:
            print('invalid graph')
            return None  # Invalid graph passed to replaceUnprotected
            
        return result

    def getOptimalArrowDeletion(self, v):
        result = {}
        result['score'] = 0

        localScore = {}

        # find maximal clique in the neighborhood of v
        # neighbors = nx.all_neighbors(self.G, v)
        neighbors = self.getNeighbors(v)
        parents = self.G.predecessors(v)
        candidates = list(set(neighbors).union(parents))

        for ui in candidates:
            N = list(set(neighbors).union(nx.all_neighbors(self.G, ui)))
            maxCliques = self.getMaxCliques(N)
            cliqueStack = []
            stopsets = []

            for i in maxCliques:
                append = True
                for C_temp in stopsets:
                    if set(i).issubset(C_temp):
                        append = False
                if append == True:
                    cliqueStack.append(i)

                while len(cliqueStack) > 0:
                    C = cliqueStack.pop()
                    # Calculate BIC score difference for current clique C
                    # Use "localScore" as (additional) cache
                    C_par = list(set(C).union(parents))
                    if ui not in C_par:
                        C_par.append(ui)
                    if tuple(C_par) not in localScore.keys():
                        diffScore = -1 * self.local2(v, C_par)
                        localScore[tuple(C_par)] = diffScore
                    else:
                        diffScore = localScore[tuple(C_par)]
                    while ui in C_par:
                        C_par.remove(ui)
                    diffScore += self.local2(v, C_par)

                    if diffScore > result['score']:
                        result["source"] = ui
                        result["clique"] = C
                        result["score"] = diffScore

                    for si in C:
                        C_sub = copy.copy(C)
                        while si in C_sub:
                            C_sub.remove(si)
                        # if C_sub not in cliqueStack:
                        #     cliqueStack.append(C_sub)
                        append = True
                        for C_temp in stopsets:
                            if set(C_sub).issubset(C_temp):
                                append = False
                        if append == True:
                            cliqueStack.append(C_sub)

                    stopsets.append(i)

        return result

    def getMaxCliques(self, neighbors):
        maxCliques = []
        # Trivial case: range of vertices contains at most one vertex
        if len(list(neighbors)) <= 1:
            maxCliques.append(list(neighbors))
            return maxCliques
        # For less trivial cases, first generate a LexBFS-ordering on the provided range of vertices
        ordering = self.lexBFS(list(neighbors))

        nbhdSubset = neighbors.copy()
        for i in reversed(range(len(ordering))):
            nbhdSubset.pop(i)
            vertices = self.getNeighbors(ordering[i])
            C = list(set(vertices).intersection(nbhdSubset))
            C.append(ordering[i])
            included = False
            for i in maxCliques:
                included = set(C).issubset(i)
                if included == True:
                    break
            if included == False:
                maxCliques.append(C)

        return maxCliques


def ges(data, verbose=False):
    graph = EssentialGraph(data, numVertices=len(data.columns))
    # do forward step

    tot = 0
    cont = True
    iterate = False
    while cont == True:
        cont = False
        while graph.greedyForward(verbose=verbose) == True:
            if verbose: print('foward', tot)
            cont = True
        while graph.greedyBackward() == True:
            if verbose: print('backwards', tot)
            cont = True
        tot += 1
        cont = cont and iterate

    if verbose: print(nx.to_numpy_matrix(graph.G))
    # return a graph
    return nx.to_numpy_matrix(graph.G)


if __name__ == '__main__':
    file = 'datasets/gmD.csv'
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
    # file = 'C:/Users/gaoan/Desktop/dataset/dataset1 (0-5).csv'
    data = pd.read_csv(file)
    print(data.columns)
    p = ges(data, True)
