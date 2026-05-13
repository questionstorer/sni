import numpy as np
import torch
import random
from typing import Optional, Tuple
import numpy
from typing import *
import copy
import math
from utils.augmentation import translate, scale
from models.GNOT.utils import MultipleTensors

class Normalizer:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")

    def preprocess(self, graph, u_p, inputs_f):
        raise NotImplementedError
    def postprocess(self, sol, *reverse_args):
        raise NotImplementedError

class IdentityNormalizer(Normalizer):
    def preprocess(self, graph, u_p, inputs_f):
        return graph, u_p, inputs_f, (None, )
    def postprocess(self, sol, *reverse_args):
        return sol

class Laplace2dNormalizer(Normalizer):
    def preprocess(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]

        # shift to (0, 0)
        rx = [graph.ndata["x"][:, 0].min(), graph.ndata["x"][:, 0].max()]
        ry = [graph.ndata["x"][:, 1].min(), graph.ndata["x"][:, 1].max()]
        shift = torch.tensor([-(rx[0] + rx[1]) * 0.5, -(ry[0] + ry[1]) * 0.5]).to(self.device)
        # scale to 1
        space_scale = torch.tensor(1 / max(rx[1] - rx[0], ry[1] - ry[0], 1)).to(self.device)

        # shift and space scale
        graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)
        dirichlet_func[:, [0, 1]] = scale(translate(dirichlet_func[:, [0, 1]], shift), space_scale)
        
        # shift and normalize value scale to [0, 1]
        value_shift = torch.tensor(-dirichlet_func[:, 2].min()).to(self.device)
        dirichlet_func[:, 2] = translate(dirichlet_func[:, 2], value_shift)
        
        rv = [dirichlet_func[:, 2].min(), dirichlet_func[:, 2].max()]
        if rv[1] > 0.0:
            value_scale = torch.tensor(1 / (rv[1] - rv[0])).to(self.device)
        else:
            value_scale = torch.tensor(1.0).to(self.device)
        # normalize dirichlet boundary value scale to [0, 1]
        dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], value_scale)

        inputs_f = MultipleTensors([dirichlet_func])
        
        return graph, u_p, inputs_f, (-value_shift, 1 / value_scale)
    def postprocess(self, sol, value_shift, value_scale):
        return translate(scale(sol, value_scale), value_shift)

class NonlinearPoisson2dNormalizer(Normalizer):
    def preprocess(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]

        # shift to (0, 0)
        rx = [graph.ndata["x"][:, 0].min(), graph.ndata["x"][:, 0].max()]
        ry = [graph.ndata["x"][:, 1].min(), graph.ndata["x"][:, 1].max()]
        shift = torch.tensor([-(rx[0] + rx[1]) * 0.5, -(ry[0] + ry[1]) * 0.5]).to(self.device)
        # scale to 1
        space_scale = torch.tensor(1 / max(rx[1] - rx[0], ry[1] - ry[0], 1)).to(self.device)

        # shift and space scale
        graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)
        dirichlet_func[:, [0, 1]] = scale(translate(dirichlet_func[:, [0, 1]], shift), space_scale)
        inputs_f = MultipleTensors([dirichlet_func])
        
        return graph, u_p, inputs_f, (None,)
    def postprocess(self, sol, *reverse_args):
        return sol


class Laplace2dMixedNormalizer(Normalizer):
    def preprocess(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]
        neumann_func = boundary_func[boundary_func[:, -1] == 1.0]

        # shift to (0, 0)
        rx = [graph.ndata["x"][:, 0].min(), graph.ndata["x"][:, 0].max()]
        ry = [graph.ndata["x"][:, 1].min(), graph.ndata["x"][:, 1].max()]
        shift = torch.tensor([-(rx[0] + rx[1]) * 0.5, -(ry[0] + ry[1]) * 0.5]).to(self.device)
        # scale to 1
        space_scale = torch.tensor(1 / max(rx[1] - rx[0], ry[1] - ry[0], 1)).to(self.device)

        if neumann_func.shape[0] == 0:
            # pure dirichlet
            graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)
            dirichlet_func[:, [0, 1]] = scale(translate(dirichlet_func[:, [0, 1]], shift), space_scale)
            space_scale = 1.0
        else:
            # shift and space scale
            graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)
            dirichlet_func[:, [0, 1]] = scale(translate(dirichlet_func[:, [0, 1]], shift), space_scale)
            neumann_func[:, [0, 1]] = scale(translate(neumann_func[:, [0, 1]], shift), space_scale)
            dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], space_scale)

        
        # shift and normalize value scale to [0, 1]
        value_shift = torch.tensor(-dirichlet_func[:, 2].min()).to(self.device)
        dirichlet_func[:, 2] = translate(dirichlet_func[:, 2], value_shift)
        
        rv = [torch.cat([dirichlet_func[:, 2], neumann_func[:, 2]]).min(), 
              torch.cat([dirichlet_func[:, 2], neumann_func[:, 2]]).max()]
        if rv[1] > 0.0:
            value_scale = torch.tensor(1 / (rv[1] - rv[0])).to(self.device)
        else:
            value_scale = torch.tensor(1.0).to(self.device)

        dirichlet_func[:, 2] = scale(dirichlet_func[:, 2], value_scale)
        neumann_func[:, 2] = scale(neumann_func[:, 2], value_scale)
        boundary_func = torch.cat([dirichlet_func, neumann_func])

        inputs_f = MultipleTensors([boundary_func])
        
        return graph, u_p, inputs_f, (1.0/space_scale, -value_shift, 1.0/value_scale)
    def postprocess(self, sol, space_scale, value_shift, value_scale):
        return scale(translate(scale(sol, value_scale), value_shift), space_scale)


class Darcy2dNormalizer(Normalizer):
    
    def preprocess(self, graph, u_p, inputs_f, 
                shift=None, space_scale=None, value_scale=None):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)
        
        # shift to (0, 0)
        rx = [graph.ndata["x"][:, 0].min(), graph.ndata["x"][:, 0].max()]
        ry = [graph.ndata["x"][:, 1].min(), graph.ndata["x"][:, 1].max()]
        shift = shift if shift else torch.tensor([-(rx[0] + rx[1]) * 0.5, -(ry[0] + ry[1]) * 0.5]).to(self.device)
        # scale to 1
        space_scale = space_scale if space_scale else torch.tensor(1 / max(rx[1] - rx[0], ry[1] - ry[0], 1)).to(self.device)
        space_value_scale = torch.square(space_scale).to(self.device)

        # shift and space scale
        graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)

        inputs_f.x[0][:, [0, 1]] = scale(translate(inputs_f.x[0][:, [0, 1]], shift), space_scale)
        inputs_f.x[1][:, [0, 1]] = scale(translate(inputs_f.x[1][:, [0, 1]], shift), space_scale)
        inputs_f.x[1][:, 2] = scale(inputs_f.x[1][:, 2], space_value_scale)

        return graph, u_p, inputs_f, (1/space_value_scale, )
    
    def postprocess(self, sol, space_value_scale):
        return scale(sol, space_value_scale)

class Heat2dNormalizer(Normalizer):
    def preprocess(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        initial_func = inputs_f.x[0]
        boundary_func = inputs_f.x[1]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]


        # shift to (0, 0)
        rx = [graph.ndata["x"][:, 0].min(), graph.ndata["x"][:, 0].max()]
        ry = [graph.ndata["x"][:, 1].min(), graph.ndata["x"][:, 1].max()]
        shift = torch.tensor([-(rx[0] + rx[1]) * 0.5, -(ry[0] + ry[1]) * 0.5]).to(self.device)
        # scale to 1
        space_scale = torch.tensor(1 / max(rx[1] - rx[0], ry[1] - ry[0], 1)).to(self.device)

        # shift and space scale
        graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)
        initial_func[:, [0, 1]] = scale(translate(initial_func[:, [0, 1]], shift), space_scale)
        dirichlet_func[:, [0, 1]] = scale(translate(dirichlet_func[:, [0, 1]], shift), space_scale)
        u_p = scale(u_p, torch.square(space_scale))

        # shift and normalize value scale to [0, 1]
        value_shift = torch.tensor(-min(dirichlet_func[:, 2:-1].min(), initial_func[:, 2].min())).to(self.device)

        initial_func[:, 2] = translate(initial_func[:, 2], value_shift)
        dirichlet_func[:, 2:-1] = translate(dirichlet_func[:, 2:-1], value_shift)
        
        rv = [min(dirichlet_func[:, 2:-1].min(), initial_func[:, 2].min()), 
              max(dirichlet_func[:, 2:-1].max(), initial_func[:, 2].max())]

        if rv[1] > 0.0:
            value_scale = torch.tensor(1 / rv[1]-rv[0]).to(self.device)
        else:
            value_scale = torch.tensor(1.0).to(self.device)
        dirichlet_func[:, 2:-1] = scale(dirichlet_func[:, 2:-1], value_scale)
        initial_func[:, 2] = scale(initial_func[:, 2], value_scale)

        boundary_func = dirichlet_func


        inputs_f = MultipleTensors([initial_func, boundary_func])

        return graph, u_p, inputs_f, (-value_shift, 1/value_scale)
    
    def postprocess(self, sol, value_shift, value_scale):
        return translate(scale(sol, value_scale), value_shift)

class NonlinearPossion2dNormalizer(Normalizer):
    def preprocess(self, graph, u_p, inputs_f):
        graph = copy.deepcopy(graph)
        u_p = copy.deepcopy(u_p)
        inputs_f = copy.deepcopy(inputs_f)

        boundary_func = inputs_f.x[0]
        dirichlet_func = boundary_func[boundary_func[:, -1] == 0.0]

        # shift to (0, 0)
        rx = [graph.ndata["x"][:, 0].min(), graph.ndata["x"][:, 0].max()]
        ry = [graph.ndata["x"][:, 1].min(), graph.ndata["x"][:, 1].max()]
        shift = torch.tensor([-(rx[0] + rx[1]) * 0.5, -(ry[0] + ry[1]) * 0.5]).to(self.device)
        # scale to 1
        space_scale = torch.tensor(1 / max(rx[1] - rx[0], ry[1] - ry[0], 1)).to(self.device)

        # shift and space scale
        graph.ndata["x"] = scale(translate(graph.ndata["x"], shift), space_scale)
        dirichlet_func[:, [0, 1]] = scale(translate(dirichlet_func[:, [0, 1]], shift), space_scale)


        inputs_f = MultipleTensors([dirichlet_func])
        
        return graph, u_p, inputs_f, (None, )
    def postprocess(self, sol):
        return sol