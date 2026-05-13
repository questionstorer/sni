import numpy as np
import torch
import random
from typing import Optional, Tuple
import numpy
from typing import *
import copy
import math
import logging
from models.GNOT.utils import MultipleTensors
logger = logging.getLogger(__name__)


def translate(X, shift):
    return X + shift

def scale(X, scale):
    return X * scale

def rotate(X, theta):
    # rotate points in X wrt p counterclockwise by angle theta
    matrix = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return X @ matrix.T

def reflect(X, o):
    if o > 0:
        return -X
    else:
        return X

class BasicTransform:
    def __init__(self, always_apply=False, p=1.0):
        self.p = p
        self.always_apply = always_apply
    def _apply_transform(self, data):
        raise NotImplementedError
    def __call__(self, graph, u_p, inputs_f):

        p = random.random()
        if self.always_apply or (p > self.p):
            transformed = self._apply_transform(graph, u_p, inputs_f)
        else:
            transformed = (graph, u_p, inputs_f)
        return transformed
    
class ComposedTransform(BasicTransform):
    def __init__(self, transforms, *args):
        super(ComposedTransform, self).__init__(*args)
        self.transforms = transforms

    def _apply_transform(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class Darcy2dTransform(BasicTransform):
    def __init__(self,
                 max_space_scale=[0.05, 1.0], 
                 max_value_scale=[0.05, 1.0],
                 always_apply=False,
                 p: float = 1.0):
        super(Darcy2dTransform, self).__init__(always_apply, p)
        self.max_space_scale = max_space_scale
        self.max_value_scale = max_value_scale
    
    def _apply_transform(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        theta = torch.tensor(random.uniform(0, 2*math.pi))

        # shift and space scale
        graph.ndata["x"] = rotate(graph.ndata["x"], theta)

        # qf
        inputs_f.x[0][:, [0, 1]] = rotate(inputs_f.x[0][:, [0, 1]], theta)
        # uD
        inputs_f.x[1][:, [0, 1]] = rotate(inputs_f.x[1][:, [0, 1]], theta)

        return graph, u_p, inputs_f

class Laplace2dMixedTransform(BasicTransform):
    def __init__(self,
                 max_space_scale=[0.05, 1.0], 
                 max_value_scale=[0.05, 1.0],
                 always_apply=False,
                 p: float = 1.0):
        super(Laplace2dMixedTransform, self).__init__(always_apply, p)
        self.max_space_scale = max_space_scale
        self.max_value_scale = max_value_scale
    
    def _apply_transform(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]
        neumann_func = boundary_func[boundary_func[:, -1] == 1.0]

        space_scale = torch.tensor(random.uniform(*self.max_space_scale))
        value_scale = torch.tensor(random.uniform(*self.max_value_scale))
        theta = torch.tensor(random.uniform(0, 2*math.pi))

        # # shift and space scale
        graph.ndata["x"] = scale(rotate(graph.ndata["x"], theta), space_scale)

        # uD and g
        dirichlet_func[:, [0, 1]] = scale(rotate(dirichlet_func[:, [0, 1]], theta), space_scale)
        neumann_func[:, [0, 1]] = scale(rotate(neumann_func[:, [0, 1]], theta), space_scale)
        
        if neumann_func.shape[0] != 0:
            dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], space_scale)
            graph.ndata["y"] = scale(graph.ndata["y"], space_scale)

        # value scale
        graph.ndata["y"] = scale(graph.ndata["y"], value_scale)
        
        #value_scale = 1 / max_abs
        dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], value_scale)
        neumann_func[:, 2] = scale(neumann_func[:, 2], value_scale)
        boundary_func = torch.cat([dirichlet_func, neumann_func])

        inputs_f = MultipleTensors([boundary_func])

        return graph, u_p, inputs_f

class Laplace2dTransform(BasicTransform):
    def __init__(self,
                 max_space_scale=[0.05, 1.0], 
                 max_value_scale=[0.05, 1.0],
                 always_apply=False,
                 p: float = 1.0):
        super(Laplace2dTransform, self).__init__(always_apply, p)
        self.max_space_scale = max_space_scale
        self.max_value_scale = max_value_scale
    
    def _apply_transform(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]

        space_scale = torch.tensor(random.uniform(*self.max_space_scale))
        value_scale = torch.tensor(random.uniform(*self.max_value_scale))
        theta = torch.tensor(random.uniform(0, 2*math.pi))

        # # shift and space scale
        graph.ndata["x"] = scale(rotate(graph.ndata["x"], theta), space_scale)

        # uD and g
        dirichlet_func[:, [0, 1]] = scale(rotate(dirichlet_func[:, [0, 1]], theta), space_scale)

        # value scale
        graph.ndata["y"] = scale(graph.ndata["y"], value_scale)
        dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], value_scale)
        boundary_func = dirichlet_func

        inputs_f = MultipleTensors([boundary_func])

        return graph, u_p, inputs_f

class Heat2dTransform(BasicTransform):
    def __init__(self,
                 max_space_scale=[0.05, 1.0], 
                 max_value_scale=[0.05, 1.0],
                 always_apply=False,
                 p: float = 1.0):
        super(Heat2dTransform, self).__init__(always_apply, p)
        self.max_space_scale = max_space_scale
        self.max_value_scale = max_value_scale
    
    def _apply_transform(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        initial_func = inputs_f.x[0]
        boundary_func = inputs_f.x[1]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]

        space_scale = torch.tensor(random.uniform(*self.max_space_scale))
        value_scale = torch.tensor(random.uniform(*self.max_value_scale))
        theta = torch.tensor(random.uniform(0, 2*math.pi))

        # # shift and space scale
        graph.ndata["x"] = scale(rotate(graph.ndata["x"], theta), space_scale)

        # uD and g
        dirichlet_func[:, [0, 1]] = scale(rotate(dirichlet_func[:, [0, 1]], theta), space_scale)
        initial_func[:, [0, 1]] = scale(rotate(initial_func[:, [0, 1]], theta), space_scale)
        # # g
        u_p = scale(u_p, space_scale * space_scale)

        # value scale
        graph.ndata["y"] = scale(graph.ndata["y"], value_scale)
        dirichlet_func[:, 2:] = scale(dirichlet_func[:, 2:], value_scale)
        initial_func[:, 2:] = scale(initial_func[:, 2:], value_scale)
        boundary_func = dirichlet_func

        inputs_f = MultipleTensors([initial_func, boundary_func])

        return graph, u_p, inputs_f

class NonlinearPoisson2dTransform(BasicTransform):
    def __init__(self,
                 max_space_scale=[0.05, 1.0], 
                 max_value_scale=[0.05, 1.0],
                 always_apply=False,
                 p: float = 1.0):
        super(NonlinearPoisson2dTransform, self).__init__(always_apply, p)
        self.max_space_scale = max_space_scale
        self.max_value_scale = max_value_scale
    
    def _apply_transform(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]

        space_scale = torch.tensor(random.uniform(*self.max_space_scale))
        value_scale = torch.tensor(random.uniform(*self.max_value_scale))
        theta = torch.tensor(random.uniform(0, 2*math.pi))

        # # shift and space scale
        graph.ndata["x"] = scale(rotate(graph.ndata["x"], theta), space_scale)

        # uD and g
        dirichlet_func[:, [0, 1]] = scale(rotate(dirichlet_func[:, [0, 1]], theta), space_scale)

        # value scale
        graph.ndata["y"] = scale(graph.ndata["y"], value_scale)
        dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], value_scale)
        boundary_func = dirichlet_func

        inputs_f = MultipleTensors([boundary_func])

        return graph, u_p, inputs_f