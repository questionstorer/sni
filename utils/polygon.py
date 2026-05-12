import matplotlib.pyplot as plt
import gmsh
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import numpy as np
from itertools import pairwise
import logging
import copy
import random
logger = logging.getLogger(__name__)


def plot_polygon(polygon, out_file_name):
    plt.figure()
    plt.gca().set_aspect("equal")

    for i, (x, y) in enumerate(polygon):
        plt.text(x, y, str(i), horizontalalignment="center", verticalalignment="center")

    # just so that it is plotted as closed polygon
    polygon.append(polygon[0])

    xs, ys = zip(*polygon)
    plt.plot(xs, ys, "r-", linewidth=0.4)

    plt.savefig(out_file_name, dpi=300)
    plt.show()
    plt.close()

def generate_polygon(num_points, type):
    if type == "simple":
        polygon = random_polygon(num_points=num_points)
    elif type == "star":
        polygon = random_star_shaped_polygon(num_points=num_points)
    
    p = np.array(polygon)
    argmin_x = np.nonzero(p[:, 0] == p[:, 0].min())[0]
    argmin_xy = np.nonzero(p[argmin_x, 1] == p[argmin_x, 1].min())[0]
    argmin = argmin_x[argmin_xy][0]

    polygon = polygon[argmin:] + polygon[:argmin]

    return polygon

def generate_polygon_mesh(num_points, type, lc=1e-1):
    # generate simple polygon with num_points vertices and mesh
    
    gmsh.model.remove()
    gmsh.model.add("p1")
    #lc = 1e-1
    #lc = 0.025

    polygon = generate_polygon(num_points, type)

    point_tags = []
    line_tags = []
    for i, (x, y) in enumerate(polygon):
        gmsh.model.geo.addPoint(x, y, 0, lc, i+1)
        point_tags.append(i+1)
    
    for i, (p1, p2) in enumerate(pairwise([i+1 for i in range(len(polygon))] + [1])):
        gmsh.model.geo.addLine(p1, p2, i+1)
        line_tags.append(i+1)
    gmsh.model.geo.addCurveLoop(line_tags, 1)
    gmsh.model.geo.addPhysicalGroup(1, line_tags, 1, name="Physical Curve")

    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addPhysicalGroup(2, [1], 1, name="Physical Surface")
    

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    return polygon



# get boundary of meshed domain as a list of points in counterclockwise order
def get_counterclockwise_dolphinx_boundary(domain, boundary_elems):
    boundary_index = []
    boundary_elems = copy.deepcopy(boundary_elems)
    boundary_elems_with_vertices = []
    
    current_element = boundary_elems[0]
    while len(boundary_elems) > 0:
        boundary_elems.remove(current_element)
        elem = [-1, -1, -1]
        elem[0] = current_element
        vs = domain.topology.connectivity(1, 0).links(current_element).tolist()

        if (vs[0] not in boundary_index) and (vs[1] not in boundary_index):
            # begining of loop
            boundary_index.append(vs[0])
            boundary_index.append(vs[1])
            boundary_elems_with_vertices.append((current_element, vs[0], vs[1]))
        elif (vs[0] not in boundary_index) and (vs[1] in boundary_index):
            boundary_index.append(vs[0])
            boundary_elems_with_vertices.append((current_element, vs[1], vs[0]))
        elif (vs[0] in boundary_index) and (vs[1] not in boundary_index):
            boundary_index.append(vs[1])
            boundary_elems_with_vertices.append((current_element, vs[0], vs[1]))
        else:
            # end of loop
            if vs[0] == boundary_index[-1]:
                boundary_elems_with_vertices.append((current_element, vs[0], vs[1]))
            else:
                boundary_elems_with_vertices.append((current_element, vs[1], vs[0]))
            
        for e in boundary_elems:
            if boundary_index[-1] in domain.topology.connectivity(1, 0).links(e):
                current_element = e
                continue
    
    # judge counterclock order
    boundary_points = domain.geometry.x[boundary_index].tolist()
    o = 0
    num = len(boundary_points)
    for i, _ in enumerate(boundary_points):
        x1, y1, _ = boundary_points[i]
        x2, y2, _ = boundary_points[(i+1) % num]
        o += (x2 - x1) * (y1 + y2)
    if o > 0:
        # clockwise
        boundary_index.reverse()
        boundary_elems_with_vertices.reverse()
        for i, (e, v1, v2) in enumerate(boundary_elems_with_vertices):
            boundary_elems_with_vertices[i] = (e, v2, v1)
        
    boundary_points = domain.geometry.x[boundary_index]
    
    return boundary_elems_with_vertices, boundary_index, boundary_points



def get_counterclockwise_trimesh_boundary(mesh):
    boundary_index = []
    boundary_elems = mesh.facets_boundary[0].tolist()
    
    current_element = boundary_elems[0]
    while len(boundary_elems) > 0:
        boundary_elems.remove(current_element)
        vs = current_element
        for v in vs:
            if v not in boundary_index:
                boundary_index.append(v)

        for e in boundary_elems:
            if boundary_index[-1] in e:
                current_element = e
                continue
    # judge counterclock order
    boundary_points = mesh.vertices[boundary_index].tolist()
    o = 0
    num = len(boundary_points)
    for i, _ in enumerate(boundary_points):
        x1, y1, _ = boundary_points[i]
        x2, y2, _ = boundary_points[(i+1) % num]
        o += (x2 - x1) * (y1 + y2)
    if o > 0:
        # clockwise
        boundary_index.reverse()
        
    boundary_points = mesh.vertices[boundary_index]
    
    # reorder points so that the position of the first point is fixed
    argmin_x = np.nonzero(boundary_points[:, 0] == boundary_points[:, 0].min())[0]
    argmin_xy = np.nonzero(boundary_points[argmin_x, 1] == boundary_points[argmin_x, 1].min())[0]
    argmin = argmin_x[argmin_xy][0]

    boundary_index = boundary_index[argmin:] + boundary_index[:argmin]
    boundary_points = np.roll(boundary_points, -argmin, axis=0)
    
    return boundary_index, boundary_points
