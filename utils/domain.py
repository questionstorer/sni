from typing import *
import logging
from scipy.spatial import Delaunay
import numpy as np
from scipy.spatial import ConvexHull
import networkx as nx
import itertools
import metispy as metis
import copy
import os
from scipy.spatial import KDTree
from utils.polygon import get_counterclockwise_trimesh_boundary
from trimesh.graph import vertex_adjacency_graph
from trimesh.base import Trimesh
from typing import *
logger = logging.getLogger(__name__)

class Domain:
    # abstract 
    def __init__(self, geometry, topology, dim, boundary_marker=None):
        # construct 
        self.geometry = geometry
        self.tree = KDTree(self.geometry)
        assert "boundary" in topology, logger.error("boundary should be specified in topology")
        assert "interior" in topology, logger.error("interior should be specified in topology")

        self.topology = topology
        if boundary_marker is None:
            self.boundary_marker = {'dirichlet': topology["boundary"], 'neumann': []}
        else:
            self.boundary_marker = self.mapBoundaryType2Marker(boundary_marker)
        
        self.dim = dim
        self.num_nodes = self.geometry.shape[0]
    def mapBoundaryType2Marker(self, boundary_type):
        raise NotImplementedError
    

    def getBoundary(self):
        return self.topology["boundary"], self.geometry[self.topology["boundary"]]
    
    def getDirichletBoundary(self):
        return self.boundary_marker["dirichlet"], self.geometry[self.boundary_marker["dirichlet"]]

    def getNeumannBoundary(self):
        return self.boundary_marker["neumann"], self.geometry[self.boundary_marker["neumann"]]
    
    def getInterior(self):
        return self.topology["interior"], self.geometry[self.topology["interior"]]

class MeshDomain(Domain):
    def __init__(self, mesh, dim, boundary_marker=None):
        topology = {}
        geometry = mesh.vertices
        boundary_index = get_trimesh_boundary(mesh)
        interior_index = sorted(list(set(range(geometry.shape[0])) - set(boundary_index)))
        topology["boundary"] = boundary_index
        topology["interior"] = interior_index
        self.mesh = mesh
        super(MeshDomain, self).__init__(geometry, topology, dim, boundary_marker)

    def mapBoundaryType2Marker(self, boundary_type: dict[str, Union[np.ndarray, list]]):
        boundary_marker = {'dirichlet': [], 'neumann': []}
        db_index, nb_index = [], [] 

        if 'dirichlet' in boundary_type:
            if isinstance(boundary_type['dirichlet'], np.ndarray):
                for query in boundary_type['dirichlet']:
                    _, i = self.tree.query(query)
                    db_index.append(i)
            elif isinstance(boundary_type['dirichlet'], list):
                db_index = boundary_type['dirichlet']
            else:
                raise TypeError("boundary_type should contain numpy.array or list of integers")
        if 'neumann' in boundary_type:
            if isinstance(boundary_type['neumann'], np.ndarray):
                for query in boundary_type['neumann']:
                    _, i = self.tree.query(query)
                    nb_index.append(i)
            elif isinstance(boundary_type['neumann'], list):
                nb_index = boundary_type['neumann']
            else:
                raise TypeError("boundary_type should contain numpy.array or list of integers")
        # if a boundary node is specified in boundary_type, assign to dirichlet
        for i in self.topology["boundary"]:
            if i not in db_index + nb_index:
                db_index.append(i)

        boundary_marker["dirichlet"] = db_index
        boundary_marker["neumann"] = nb_index
        return boundary_marker

class SimplePolygonMeshDomain(Domain):
    # domain obtained from simple polgon mesh
    def __init__(self, mesh, boundary_marker=None):

        topology = {}
        geometry = mesh.vertices

        # !!! This is the only difference between SimplePolygonMeshDomain and MeshDomain
        boundary_index, _ = get_counterclockwise_trimesh_boundary(mesh)
        interior_index = sorted(list(set(range(geometry.shape[0])) - set(boundary_index)))

        topology["boundary"] = boundary_index
        topology["interior"] = interior_index
        self.mesh = mesh
        super(SimplePolygonMeshDomain, self).__init__(geometry, topology, 2, boundary_marker)

    def mapBoundaryType2Marker(self, boundary_type: dict[str, Union[np.ndarray, list]]):
        boundary_marker = {'dirichlet': [], 'neumann': []}
        db_index, nb_index = [], [] 

        if 'dirichlet' in boundary_type:
            if isinstance(boundary_type['dirichlet'], np.ndarray):
                for query in boundary_type['dirichlet']:
                    _, i = self.tree.query(query)
                    db_index.append(i)
            elif isinstance(boundary_type['dirichlet'], list):
                db_index = boundary_type['dirichlet']
            else:
                raise TypeError("boundary_type should contain numpy.array or list of integers")
        if 'neumann' in boundary_type:
            if isinstance(boundary_type['neumann'], np.ndarray):
                for query in boundary_type['neumann']:
                    _, i = self.tree.query(query)
                    nb_index.append(i)
            elif isinstance(boundary_type['neumann'], list):
                nb_index = boundary_type['neumann']
            else:
                raise TypeError("boundary_type should contain numpy.array or list of integers")
        # if a boundary node is not specified in boundary_type, assign to dirichlet
        for i in self.topology["boundary"]:
            if i not in db_index + nb_index:
                db_index.append(i)
        
        # get start and end of neumann boundary in counterclockwise order
        nstart, nend = 0, 0
        for i, v in enumerate(self.topology["boundary"]):
            if v in nb_index and self.topology["boundary"][i-1] not in nb_index:
                nstart = i
            if v not in nb_index and self.topology["boundary"][i-1] in nb_index:
                nend = i
        if nstart < nend:
            nb_index = self.topology["boundary"][nstart: nend]
            db_index = self.topology["boundary"][nend:] + self.topology["boundary"][:nstart]
            # since neumann and dirichlet are connected, they should share vertices
            db_index = [self.topology["boundary"][nend-1]] + db_index
            db_index.append(self.topology["boundary"][nstart])
            
        elif nstart > nend:
            nb_index = self.topology["boundary"][nstart:] + self.topology["boundary"][:nend]
            db_index = self.topology["boundary"][nend:nstart]
            # since neumann and dirichlet are connected, they should share vertices
            db_index = [self.topology["boundary"][nend-1]] + db_index
            db_index.append(self.topology["boundary"][nstart])
            
        else:
            db_index = self.topology["boundary"]
            nb_index = []

        boundary_marker["dirichlet"] = db_index
        boundary_marker["neumann"] = nb_index
        return boundary_marker


class DecomposedMeshDomain(MeshDomain):
    def __init__(self, mesh, dim, boundary_marker, n_parts, depth):
        super(DecomposedMeshDomain, self).__init__(mesh, dim, boundary_marker)
        self.mesh = mesh
        self.construct_subDomain(n_parts, depth)

    def construct_subDomain(self, n_parts, depth):
        raise NotImplementedError
    


class DecomposedSimplePolygonMeshDomain(DecomposedMeshDomain):

    def __init__(self, mesh, dim, boundary_marker, n_parts, depth):
        super(DecomposedSimplePolygonMeshDomain, self).__init__(mesh, dim, boundary_marker, n_parts, depth)

    def construct_subDomain(self, n_parts, depth):
        self.G = vertex_adjacency_graph(self.mesh)
        if n_parts > 1:
            self.n_parts, partition = partition_graph(self.G, n_parts)
        else:
            self.n_parts = 1
            partition = [set(self.G.nodes)]
        
        self.partition = extend_partition(self.G, partition)
        self.subDomain = []
        self.mapping = []
        for part in self.partition:
            l2g = sorted(list(part))
            g2l = {g:l for l, g in enumerate(l2g)}
            vertices = self.geometry[l2g]
            faces = np.array([[g2l[v] for v in face] for face in self.mesh.faces if all([v in part for v in face])])
            boundary_marker = {k: np.array([self.geometry[i] for i in v if i in part]) for k, v in self.boundary_marker.items()}
            sd = SimplePolygonMeshDomain(Trimesh(vertices, faces), boundary_marker)
            self.subDomain.append(sd)

            # since Trimesh reorder vertices, we have to reconsturct mapping
            l2g = []
            for v in sd.geometry:
                _, i = self.tree.query(v)
                l2g.append(i)
            g2l = {g:l for l, g in enumerate(l2g)}
            self.mapping.append({"l2g": l2g, "g2l": g2l})
    
    def getArtificalBoundary(self, i):
        # get boundary of domain i that is not in global boundary
        indices = set()
        for e in self.subDomain.edges(i):
            overlap = self.subDomain.edges[e]["overlap"]
            indices = indices.union({n for n in overlap if n not in self.boundary[:, self.dim]} - self.partition[i])
        indices = sorted(list(indices))
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])

        return indices, locations
    
    def getInterior(self, i):
        boundary_indices = set(self.getGlobalBoundary(i)[0] + self.getArtificalBoundary(i)[0])
        indices = sorted(list(self.subDomain.nodes[i]["nodes"] - boundary_indices))
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])

        return indices, locations
    
    def getBoundary(self, i):
        gb = self.getGlobalBoundary(i)
        ab = self.getArtificalBoundary(i)

        return gb[0]+ab[0], np.array(gb[1].tolist() + ab[1].tolist())

    def getSubDomain(self, i):
        interior = self.getInterior(i)
        boundary = self.getBoundary(i)
        return interior[0]+boundary[0], np.array(interior[1].tolist() + boundary[1].tolist())

class DecomposedSpaceTimeSimplePolygonMeshDomain(DecomposedSimplePolygonMeshDomain):
    def __init__(self, *args, time_step=1, time_span=10, **kwargs):
        super(DecomposedSpaceTimeSimplePolygonMeshDomain, self).__init__(*args, **kwargs)
        self.time_step = time_step
        self.time_span = time_span
        self.construct_subTimeIntervals()
        self.num_interval = len(self.subTimeInterval)

    def construct_subTimeIntervals(self):
        self.subTimeInterval = []
        s = 0
        while s+2 < self.time_span:
            e = s + self.time_step
            self.subTimeInterval.append([s, min(self.time_span, e)])
            s = e - 2
        
    def getStart(self, i):
        return self.subTimeInterval[i][0]

    def getEnd(self, i):
        return self.subTimeInterval[i][1]
        
    def getSubInterval(self, i):
        return self.subTimeInterval[i]





def get_trimesh_boundary(mesh):
    boundary_elems = np.unique(mesh.facets_boundary[0])
    boundary_index = sorted(boundary_elems.tolist())
    return boundary_index

def partition_graph(G, n_parts):
    
    # partition graph into >= n_parts
    # return partition as a list of parts
    
    objval, parts = metis.part_graph(G, nparts=n_parts, recursive=True)
    # get subDomain as list
    nparts = len(set(parts))
    partition = [[]] * nparts
    for id, p in enumerate(parts):
        partition[p] = partition[p] + [id]
    partition = [set(d) for d in partition]

    # return connected components
    partition_ccs = []
    for p in partition:
        for cc in nx.connected_components(G.subgraph(p)):
            partition_ccs.append(cc)

    n = len(partition_ccs)
    if n != n_parts:
        logger.warning("number of partition is not the same as input 'n_parts'")
    return n, partition_ccs

def extend_partition(G, partition, depth=1):
    # extend each part of partition by its neighbors with distance
    extended_partition = []
    for i, d in enumerate(partition):
        for _ in range(depth):
            for v in d:
                d = d.union(G.neighbors(v))

        extended_partition.append(d)
    return extended_partition



class DecomposedDomain:
    def __init__(self, G, interior, boundary, dim, n_parts):
        self.dim = dim
        self.interior = interior
        self.boundary = boundary # global boundary
        all = np.concatenate((interior, boundary))
        self.all = all[np.argsort(all[:, -1])]
        self.n_parts, self.partition = partition_graph(G, n_parts)

        self.overlapping_partition = extend_partition(G, self.partition)
        self.tree = KDTree(self.all[:, :-1])
        self.construct_subDomain()

    def query_global_indices(self, point):
        # get the global index of a point
        _, i = self.tree.query(point)
        return i

    def construct_subDomain(self):
        # subDomain is a graph
        # node: group of indices in the partition
        # edge: overlapping indices between two groups
        self.subDomain = nx.Graph()
        self.subDomain.add_nodes_from(
            [
                (i, {"nodes": p}) for i, p in enumerate(self.overlapping_partition)
            ]
        )
        self.subDomain.add_edges_from(
            [
                (i, j, {"overlap": p.intersection(q)}) for (i, p), (j, q) in itertools.combinations(enumerate(self.overlapping_partition), r=2) if p.intersection(q)
            ]
        )

    def getAllGlobalBoundary(self):
        indices = sorted([int(v) for v in self.boundary[:, self.dim]])
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])
        return indices, locations
    
    def getAllInterior(self):
        indices = sorted([int(v) for v in self.interior[:, self.dim]])
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])
        return indices, locations

    def getGlobalBoundary(self, i):
        # get real boundary of domain i
        nodes = self.subDomain.nodes[i]["nodes"]
        indices = sorted([v for v in nodes if v in self.boundary[:, self.dim]])
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])

        return indices, locations
    
    def getArtificalBoundary(self, i):
        # get boundary of domain i that is not in global boundary
        indices = set()
        for e in self.subDomain.edges(i):
            overlap = self.subDomain.edges[e]["overlap"]
            indices = indices.union({n for n in overlap if n not in self.boundary[:, self.dim]} - self.partition[i])
        indices = sorted(list(indices))
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])

        return indices, locations
    
    def getInterior(self, i):
        boundary_indices = set(self.getGlobalBoundary(i)[0] + self.getArtificalBoundary(i)[0])
        indices = sorted(list(self.subDomain.nodes[i]["nodes"] - boundary_indices))
        locations = np.array([(self.all[v][0], self.all[v][1]) for v in indices])

        return indices, locations
    
    def getBoundary(self, i):
        gb = self.getGlobalBoundary(i)
        ab = self.getArtificalBoundary(i)

        return gb[0]+ab[0], np.array(gb[1].tolist() + ab[1].tolist())

    def getSubDomain(self, i):
        interior = self.getInterior(i)
        boundary = self.getBoundary(i)
        return interior[0]+boundary[0], np.array(interior[1].tolist() + boundary[1].tolist())


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def build_graph_delaunay(points):

    fine_tri = Delaunay(points[:, 0:2])
    G = nx.Graph()
    G.add_nodes_from([(i, {"x": p[0], "y": p[1]}) for i, p in enumerate(points)])

    tree = KDTree(points[:, :2])
    max_nb_distance = max([tree.query(c, 5)[0][-1] for c in points[:, :2]])
    for tr in fine_tri.simplices:
        for v, w in itertools.combinations(tr, r=2):
            if np.linalg.norm(points[int(v)][0:2] - points[int(w)][0:2]) <= max_nb_distance:
                G.add_edge(v, w)
    return G

def create_subdomains(interior_points:np.ndarray, boundary_points:np.ndarray, dim, n_parts=6):
    # divide domains of 1D into subintervals, 2D into polygons according to eval_centers
    assert dim in [1, 2], logger.error("dimension has to be 1 or 2")

    points = np.concatenate((interior_points, boundary_points))

    if dim == 2:
        G = build_graph_delaunay(points)

        dcps = DecomposedDomain(G, interior_points, boundary_points, dim=dim, n_parts=n_parts)

    elif dim == 1:
        G = nx.Graph()
        G.add_nodes_from([(i, {"x": p[0]}) for i, p in enumerate(points)])
        points = points[points[:, 0].argsort()]
        G.add_edges_from(pairwise([int(i) for i in points[:, 1].astype("uint8").tolist()]))
        #partition = extend_partition(G, partition)

        dcps = DecomposedDomain(G, interior_points, boundary_points, dim=dim, n_parts=n_parts)

    return G, dcps






