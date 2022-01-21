import numpy as np
import heapq
from typing import Union

class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """ Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def _add_edges_to_pqueue (self, pqueue, start_node):
        """ A helper function that adds all ougoing edges from start_node into a priority queue. 
        """
        row_slice = self.adj_mat[start_node, :]  
        outgoing_edges = []
        for destination, val in enumerate(row_slice):
            if val != 0:
                outgoing_edges.append((val, start_node, destination))
        # Store all outgoing edges from visited_vertices into the priority queue        
        for e in outgoing_edges:
            heapq.heappush(pqueue, e)

    def construct_mst(self):
        """ Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. 
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric. 
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        """
        self.mst = np.zeros_like(self.adj_mat) # Initialize an empty mst of same shape and type as adj mat (to be filled in)
        num_nodes = self.adj_mat.shape[0]
        pqueue = [] # Priority queue (to be interacted with using heapq lib)

        curr_node = 0 # Initialize search from a random node (in this case, we'll start at the first one)
        visited_vertices = [curr_node]

        # Store all outgoing edges (their weight and their destination)
        self._add_edges_to_pqueue (pqueue, curr_node)
        
        # while (not all vertices have been added to visited_vertices)
        while (len(visited_vertices) != num_nodes):
            # Pop the lowest weight edge from the priority queue
            weight, start_node, destination = heapq.heappop(pqueue)
            # If the destination vertex of the edge is not in visited vertices:
            if destination not in visited_vertices:
                # Add this edge to our MST
                self.mst[start_node][destination] = weight
                self.mst[destination][start_node] = weight
                # Add the destination vertex to visited vertices
                visited_vertices.append(destination)
                # Add all outgoing edges from the destination vertex into the priority queue
                self._add_edges_to_pqueue (pqueue, destination)


