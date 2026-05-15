
from types import NoneType
import numpy as np
import torch

import logging


from utils.domain import create_subdomains
import dgl
from models.GNOT.utils import MultipleTensors
from torch.nn.utils.rnn import pad_sequence
from models.GNOT.data_utils import MIODataset
from models.GNOT.utils import TorchQuantileTransformer, UnitTransformer, PointWiseUnitTransformer, MultipleTensors

import typing
from utils.augmentation import Darcy2dTransform, Laplace2dTransform, Laplace2dMixedTransform, Heat2dTransform, Helmholtz2dTransform, NonlinearPoisson2dTransform
from utils.normalization import Laplace2dNormalizer, Laplace2dMixedNormalizer, Darcy2dNormalizer, Heat2dNormalizer, NonlinearPoisson2dNormalizer
from trimesh.interfaces.gmsh import load_gmsh
import gmshparser
logger = logging.getLogger(__name__)

def get_train_dolphinx_dataset(args):
    transform = lambda x, y, z: (x, y, z)
    time_dependent = False
    time_step = None

    if args.dataset == "laplace2d_simple":
        train_path = './data/2d/laplace2d_simple_20000_train.pkl'
        test_path = './data/2d/laplace2d_simple_2500_test.pkl'
        transform = Laplace2dTransform(max_space_scale=[0.5, 1.0],
                            max_value_scale=[1.0, 1.0],
                            always_apply=True)
    
    elif args.dataset == "laplace2d_mixed_simple":
        train_path = './data/2d/laplace2d_mixed_simple_40000_train.pkl'
        test_path = './data/2d/laplace2d_mixed_simple_4000_test.pkl'
        transform = Laplace2dMixedTransform(max_space_scale=[1.0, 1.0],
                            max_value_scale=[1.0, 1.0],
                             always_apply=True)
    
    elif args.dataset == "darcy2d_simple":
        train_path = './data/2d/darcy2d_simple_40000_train.pkl'
        test_path = './data/2d/darcy2d_simple_2500_test.pkl'
    
    elif args.dataset == "heat2d_simple":
        train_path = './data/2d/heat2d_simple_80000_train.pkl'
        test_path = './data/2d/heat2d_simple_12500_test.pkl'
        time_dependent = args.time_dependent
        time_step = args.time_step
        transform = Heat2dTransform(
                            max_space_scale=[0.8, 1.0],
                            max_value_scale=[1.0, 1.0],
                             always_apply=True)
    
    elif args.dataset == "nonlinear_poisson2d_simple":
        train_path = './data/2d/nonlinear_poisson2d_simple_20000_train.pkl'
        test_path = './data/2d/nonlinear_poisson2d_simple_2500_test.pkl'
        transform = NonlinearPoisson2dTransform(max_space_scale=[0.8, 1.0],
                            max_value_scale=[1.0, 1.0],
                            always_apply=True)
    

    args.train_num = int(args.train_num) if args.train_num not in ['all', 'none'] else args.train_num
    args.test_num = int(args.test_num) if args.test_num not in ['all', 'none'] else args.test_num


    train_dataset = DolphinxNeuralOperatorDataset(train_path, name=args.dataset, train=True, train_num=args.train_num,
                               sort_data=args.sort_data,
                               normalize_y=args.normalize_y,
                               normalize_x=args.normalize_x,
                               transform=transform,
                               time_dependent=time_dependent,
                               time_step=time_step)
    test_dataset = DolphinxNeuralOperatorDataset(test_path, name=args.dataset, train=False, test_num=args.test_num,
                              sort_data=args.sort_data,
                              normalize_y=args.normalize_y,
                              normalize_x=args.normalize_x, y_normalizer=train_dataset.y_normalizer,
                              x_normalizer=train_dataset.x_normalizer, up_normalizer=train_dataset.up_normalizer,
                              transform=transform,
                              time_dependent=time_dependent,
                              time_step=time_step)
    args.dataset_config = train_dataset.config

    return train_dataset, test_dataset

def get_inference_dolphinx_dataset(args):
    time_dependent = False
    time_span = None
    time_step = None
    if args.dataset == "laplace2d_schwarz":
        test_path = './data/2d/laplace2d_schwarz_100_test.pkl'
    elif args.dataset == "laplace2d_holes":
        test_path = './data/2d/laplace2d_holes_100_test.pkl'
    elif args.dataset == "laplace2d_bosch":
        test_path = './data/2d/laplace2d_bosch_100_test.pkl'  

    elif args.dataset == "laplace2d_dolphin":
        test_path = './data/2d/laplace2d_dolphin_100_test.pkl'
    elif args.dataset == "laplace2d_disk":
        test_path = './data/2d/laplace2d_disk_100_test.pkl'
    elif args.dataset == "laplace2d_mixed_schwarz":
        test_path = './data/2d/laplace2d_mixed_schwarz_100_test.pkl'
    elif args.dataset == 'laplace2d_mixed_holes':
        test_path = './data/2d/laplace2d_mixed_holes_100_test.pkl'
    elif args.dataset == 'laplace2d_mixed_bosch':
        test_path = './data/2d/laplace2d_mixed_bosch_100_test.pkl'
    elif args.dataset == "darcy2d_holes":
        test_path = './data/2d/darcy2d_holes_100_test.pkl'
    elif args.dataset == "darcy2d_schwarz":
        test_path = './data/2d/darcy2d_schwarz_100_test.pkl'
    elif args.dataset == "darcy2d_bosch":
        test_path = './data/2d/darcy2d_bosch_100_test.pkl'
    elif args.dataset == "nonlinear_poisson2d_schwarz":
        test_path = './data/2d/nonlinear_poisson2d_schwarz_100_test.pkl'
    elif args.dataset == "nonlinear_poisson2d_holes":
        test_path = './data/2d/nonlinear_poisson2d_holes_100_test.pkl'
    elif args.dataset == "nonlinear_poisson2d_bosch":
        test_path = './data/2d/nonlinear_poisson2d_bosch_100_test.pkl'
    elif args.dataset == "heat2d_schwarz":
        test_path = './data/2d/heat2d_schwarz_10_test.pkl'
        time_dependent = True 
        time_span = args.time_span
        time_step = args.time_step
    elif args.dataset == "heat2d_holes":
        test_path = './data/2d/heat2d_holes_10_test.pkl'
        time_dependent = True
        time_span = args.time_span
        time_step = args.time_step
    elif args.dataset == "heat2d_bosch":
        test_path = './data/2d/heat2d_bosch_10_test.pkl'
        time_dependent = True
        time_span = args.time_span
        time_step = args.time_step
    elif args.dataset == "nonlinear_poisson2d_schwarz":
        test_path = './data/2d/nonlinear_poisson2d_schwarz_100_test.pkl'
    elif args.dataset == "nonlinear_poisson2d_holes":
        test_path = './data/2d/nonlinear_poisson2d_holes_100_test.pkl'
    elif args.dataset == "nonlinear_poisson2d_bosch":
        test_path = './data/2d/nonlinear_poisson2d_bosch_100_test.pkl'

    args.test_num = int(args.test_num) if args.test_num not in ['all', 'none'] else args.test_num

    test_dataset = DolphinxNeuralOperatorDataset(test_path, name=args.dataset, train=False, test_num=args.test_num,
                              sort_data=args.sort_data,
                              normalize_y=args.normalize_y,
                              normalize_x=args.normalize_x, 
                              time_dependent=time_dependent,
                              time_span=time_span,
                              time_step=time_step)
    args.dataset_config = test_dataset.config

    return test_dataset

def get_inference_mesh(args):
    domain = args.dataset.split("_")[-1]
    if domain == "schwarz":
        mesh_path = './data/mesh/A-schwarz.msh'
    elif domain == "holes":
        mesh_path = './data/mesh/B-holes.msh'
    elif domain == "bosch":
        mesh_path = './data/mesh/C-bosch.msh'
    elif domain == "dolphin":
        mesh_path = './data/mesh/D-dolphin.msh'
    elif domain == "disk":
        mesh_path = './data/mesh/E-disk.msh'
    
    return gmshparser.parse(mesh_path), load_gmsh(mesh_path) 

def get_inference_boundary_marker(args, gmesh):
    pde = "_".join(args.dataset.split("_")[0:-1])
    domain = args.dataset.split("_")[-1]

    if pde in ["laplace2d", "darcy2d", "heat2d", 'nonlinear_poisson2d']:
        # get boundary nodes of element type 1
        boundary_nodes = set()
        for entity in gmesh.get_element_entities():
            if entity.get_element_type() == 1:
                for element in entity.get_elements():
                    for n in element.get_connectivity():
                        boundary_nodes.add(n)
        # get bounday position of boundary nodes
        boundary_points = []
        for entity in gmesh.get_node_entities():
            for node in entity.get_nodes():
                if node.get_tag() in boundary_nodes:
                    boundary_points.append(node.get_coordinates())
        boundary_points = np.array(boundary_points)

        boundary_marker = {"dirichlet": boundary_points, "neumann":[]}
    elif pde == "laplace2d_mixed":
        if domain == "schwarz":
            # schwarz
            # get boundary nodes of element type 1
            db_index, nb_index = set(), set()
            for entity in gmesh.get_element_entities():
                if (entity.get_element_type() == 1) and (entity.get_tag() in [6]):
                    for element in entity.get_elements():
                        for n in element.get_connectivity():
                            db_index.add(n)
                elif (entity.get_element_type() == 1) and (entity.get_tag() in [7,8,9]):
                    for element in entity.get_elements():
                        for n in element.get_connectivity():
                            nb_index.add(n)
            # get bounday position of boundary nodes
            db_points, nb_points = [], []
            for entity in gmesh.get_node_entities():
                for node in entity.get_nodes():
                    if node.get_tag() in db_index:
                        db_points.append(node.get_coordinates())
                    elif node.get_tag() in nb_index:
                        nb_points.append(node.get_coordinates())
            db_points = np.array(db_points)
            nb_points = np.array(nb_points)
            boundary_marker = {"dirichlet": db_points, "neumann":nb_points}

        elif domain == "holes":
            # holes
            # get boundary nodes of element type 1
            db_index, nb_index = set(), set()
            for entity in gmesh.get_element_entities():
                if (entity.get_element_type() == 1) and (entity.get_tag() in [8, 9]):
                    for element in entity.get_elements():
                        for n in element.get_connectivity():
                            db_index.add(n)
                elif (entity.get_element_type() == 1) and (entity.get_tag() in [1,2,3,4]):
                    for element in entity.get_elements():
                        for n in element.get_connectivity():
                            nb_index.add(n)
            # get bounday position of boundary nodes
            db_points, nb_points = [], []
            for entity in gmesh.get_node_entities():
                for node in entity.get_nodes():
                    if node.get_tag() in db_index:
                        db_points.append(node.get_coordinates())
                    elif node.get_tag() in nb_index:
                        nb_points.append(node.get_coordinates())
            db_points = np.array(db_points)
            nb_points = np.array(nb_points)
            boundary_marker = {"dirichlet": db_points, "neumann":nb_points}
        
        elif domain == "bosch":
            # bosch
            # get boundary nodes of element type 1
            db_index, nb_index = set(), set()
            for entity in gmesh.get_element_entities():
                if (entity.get_element_type() == 1) and (entity.get_tag() in [1]):
                    for element in entity.get_elements():
                        for n in element.get_connectivity():
                            db_index.add(n)
                elif (entity.get_element_type() == 1) and (entity.get_tag() in [2, 3, 4, 5, 6, 8, 9, 11]):
                    for element in entity.get_elements():
                        for n in element.get_connectivity():
                            nb_index.add(n)
            # get bounday position of boundary nodes
            db_points, nb_points = [], []
            for entity in gmesh.get_node_entities():
                for node in entity.get_nodes():
                    if node.get_tag() in db_index:
                        db_points.append(node.get_coordinates())
                    elif node.get_tag() in nb_index:
                        nb_points.append(node.get_coordinates())
            db_points = np.array(db_points)
            nb_points = np.array(nb_points)
            boundary_marker = {"dirichlet": db_points, "neumann":nb_points}

    return boundary_marker

def get_inference_normalizer(args):
    pde = "_".join(args.dataset.split("_")[0:-1])
    if pde == "laplace2d":
        normalizer = Laplace2dNormalizer
    elif pde == "laplace2d_mixed":
        normalizer = Laplace2dMixedNormalizer
    elif pde == "darcy2d":
        normalizer = Darcy2dNormalizer
    elif pde == "heat2d":
        normalizer = Heat2dNormalizer
    elif pde == "nonlinear_poisson2d":
        normalizer = NonlinearPoisson2dNormalizer
    return normalizer
    

class DecomposedDomainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,sort_data=True, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        super(DecomposedDomainDataLoader, self).__init__(dataset=dataset, batch_size=batch_size,
                                           shuffle=shuffle, sampler=sampler,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn,
                                           pin_memory=pin_memory,
                                           drop_last=drop_last, timeout=timeout,
                                           worker_init_fn=worker_init_fn)

        self.sort_data = sort_data
        if sort_data:
            self.batch_indices = [list(range(i, min(i+batch_size, len(dataset)))) for i in range(0, len(dataset), batch_size)]
            if drop_last:
                self.batch_indices = self.batch_indices[:-1]
        else:
            self.batch_indices = list(range(0, (len(dataset) // batch_size)*batch_size)) if drop_last else list(range(0, len(dataset)))
        if shuffle:
            np.random.shuffle(self.batch_indices)

    def __iter__(self):
        # 返回一个迭代器，用于遍历数据集中的每个批次
        for indices in self.batch_indices:
            transposed = zip(*[self.dataset[idx][1:] for idx in indices])
            batched = []
            for sample in transposed:
                if isinstance(sample[0][0], dgl.DGLGraph):
                    gs = []
                    for s in sample:
                        gs += s
                    batched.append(dgl.batch(gs))
                elif isinstance(sample[0][0], torch.Tensor):
                    ts = []
                    for s in sample:
                        ts += s
                    batched.append(torch.stack(ts))
                elif isinstance(sample[0][0], MultipleTensors):
                    ms = []
                    for s in sample:
                        ms += s
                    ms = [pad_sequence([ms[i][j] for i in range(len(ms))]).permute(1, 0, 2) for j in range(len(ms[0]))]
                    sample_ = MultipleTensors(ms)
                    batched.append(sample_)
                else:
                    raise NotImplementedError
            yield batched

    def __len__(self):
        # 返回数据集的批次数
        return len(self.batch_indices)


class DolphinxNeuralOperatorDataset(MIODataset):
    # load data generated from dolphinx as dataset
    # loaded pkl file is assumed to be a list of pair 
    # (numpy array of solution, numpy array of boundary condition)

    def __init__(self, *args, time_dependent=False, time_step=None, time_span=None, **kwargs):
        transform = kwargs.pop('transform', None)
        self.time_dependent = time_dependent
        self.time_step = time_step
        self.time_span = time_span
        super(DolphinxNeuralOperatorDataset, self).__init__(*args, **kwargs)
        self.transform = transform if transform else lambda x,y,z: (x,y,z)

    def process(self):
        if not self.time_dependent:
            # space only process
            self.data_len = len(self.data_list)
            self.graphs = []
            self.inputs_f = []
            self.u_p = []
            for i in range(len(self)):
                #x, y, u_p, input_f = self.data_list[i]
                if len(self.data_list[i]) == 2:
                    sol, inputs_f = self.data_list[i][0], self.data_list[i][1]
                    up = torch.zeros((1, )).float()
                elif len(self.data_list[i]) == 3:
                    sol, u_p, inputs_f = self.data_list[i][0], self.data_list[i][1], self.data_list[i][2]
                    up = torch.tensor((u_p, )).float()

                g = dgl.DGLGraph()
                g.add_nodes(sol.shape[0])
                g.ndata['x'] = torch.from_numpy(sol[:, 0:2]).float()
                g.ndata['y'] = torch.from_numpy(sol[:, 2:]).float()

                self.graphs.append(g)
                self.u_p.append(up) # global input parameters
                if inputs_f is not None:
                    
                    inputs_f = MultipleTensors([torch.from_numpy(f).float() for f in inputs_f])
                    self.inputs_f.append(inputs_f)
                    self.num_inputs = len(inputs_f)
        else:
            # space-time process
            self.graphs = []
            self.inputs_f = []
            self.u_p = []

            for i, data in enumerate(self.data_list):
                sol, u_p, inputs_f = data[0], data[1], data[2]
                
                if self.time_span is None:
                    # training
                    chunck_size = sol[:, 2:].shape[1] / self.time_step
                else:
                    chunck_size = 1
                # assert all times steps can be divided into equa time_step
                assert int(chunck_size) == chunck_size
                boundary_points = inputs_f[0][:, [0, 1]]
                for i, (s, f) in enumerate(zip(np.hsplit(sol[:, 2:], chunck_size), np.hsplit(inputs_f[0][:, 2:-1], chunck_size))):
                    g = dgl.DGLGraph()
                    g.add_nodes(sol.shape[0])
                    g.ndata['x'] = torch.from_numpy(sol[:, 0:2]).float()
                    g.ndata['y'] = torch.from_numpy(s)
                    up = torch.tensor((u_p, )).float()
                    self.graphs.append(g)
                    self.u_p.append(up) # global input parameters

                    # boundary condition and initial condition
                    bc = torch.from_numpy(np.concatenate([boundary_points, 
                                                        f, 
                                                        np.zeros((boundary_points.shape[0], 1))], axis=1)).float()
                    ic = torch.from_numpy(np.concatenate([sol[:, 0:2], s[:, [0]]], axis=1)).float()
                    self.inputs_f.append(MultipleTensors([ic, bc]))
                self.num_inputs = 2
            self.data_len = len(self.graphs)
        if len(self.inputs_f) == 0:
            self.inputs_f = torch.zeros([len(self)])  # pad values, tensor of 0, not list

            # logger.info('processing {}'.format(i))d

        #### sort data if necessary
        if self.sort_data:
            self.__sort_dataset()

        self.u_p = torch.stack(self.u_p)


        #### normalize_y
        if self.normalize_y != 'none':
            self.__normalize_y__()
        if self.normalize_x != 'none':
            self.__normalize_x__()

        self.__update_dataset_config__()

        return
    


    def __update_dataset_config__(self):
        if not self.time_dependent:
            self.config = {
                'input_dim': self.graphs[0].ndata['x'].shape[1],
                'theta_dim': self.u_p.shape[1],
                'output_dim': self.graphs[0].ndata['y'].shape[1],
                'branch_sizes': [x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0
            }
        else:
            self.config = {
                'input_dim': 2,
                'theta_dim': self.u_p.shape[1],
                'output_dim': self.time_step if self.time_step else -1, #-1 for infinity time steps
                'branch_sizes': [x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0
            }
        return
    def __getitem__(self, idx):
        return self.transform(self.graphs[idx], self.u_p[idx], self.inputs_f[idx])


def transform_gt(model, graph):

    if not model.time_dependent:
        inputs_f = MultipleTensors([torch.concat([graph.ndata['x'], graph.ndata['y'], torch.zeros((graph.ndata['x'].shape[0], 1))], dim=1)])
    else:
        boundary_condition = torch.concat([graph.ndata['x'], graph.ndata['y'], torch.zeros((graph.ndata['x'].shape[0], 1))], dim=1)
        initial_condition = torch.concat([graph.ndata['x'], graph.ndata['y'][:, 0:1]], dim=1)
        inputs_f = MultipleTensors([initial_condition, boundary_condition])
    gt_sol = model.initialize(inputs_f)

    return gt_sol