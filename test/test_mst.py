# write tests for bfs
import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """ Helper function to check the correctness of the adjacency matrix encoding an MST.
        Note that because the MST of a graph is not guaranteed to be unique, we cannot 
        simply check for equality against a known MST of a graph. 

        Arguments:
            adj_mat: Adjacency matrix of full graph
            mst: Adjacency matrix of proposed minimum spanning tree
            expected_weight: weight of the minimum spanning tree of the full graph
            allowed_error: Allowed difference between proposed MST weight and `expected_weight`

        Tests:
            - MST has correct expected weight
            - MST has correct number of edges
            - MST fully connects all nodes
            - MST is a proper tree

    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    def connected(mat):
        # The nonzero entries in 𝐴^𝑘 where A=mat indicate pairs of nodes that are connected by a path of length 𝑘. 
        # Thus, if n=(number of nodes) and the sum A+A^2+...A^(n-1) contains no nonzero entries, then the graph is connected.
        sum_mat = mat
        for i in range(2, mat.shape[0]):
            sum_mat = np.add(sum_mat, np.linalg.matrix_power(mat, i))
        return not np.any(sum_mat == 0)


    total = 0
    total_edges = 0
    nonzero_edge = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j] # Compute total cost of MST
            if mst[i, j] != 0:
                total_edges += 1 # Keep track of total number of MST edges
                nonzero_edge = (i, j) # Store one of the MST edges
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    assert total_edges == (adj_mat.shape[0]-1), 'Proposed MST has incorrect number of edges'
    assert connected(mst), 'Proposed MST is not connected'
    # Since an MST is a tree, removing any edge will disconnect it.
    broken_mst = mst
    broken_mst[nonzero_edge] = 0
    assert not connected(broken_mst), 'Proposed MST is not a tree'


def test_mst_small():
    """ Unit test for the construction of a minimum spanning tree on a small graph """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """ Unit test for the construction of a minimum spanning tree using 
    single cell data, taken from the Slingshot R package 
    (https://bioconductor.org/packages/release/bioc/html/slingshot.html)
    """
    file_path = './data/slingshot_example.txt'
    # load coordinates of single cells in low-dimensional subspace
    coords = np.loadtxt(file_path)
    # compute pairwise distances for all 140 cells to form an undirected weighted graph
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """ Check that a proper mst is returned, for a graph with cycles and multiple possible MSTs"""
    file_path = './data/student_test.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 3)

