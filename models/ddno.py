import torch
from utils.data_utils import create_subdomains
import dgl
from scipy.spatial import KDTree
from models.GNOT.utils import MultipleTensors
from torch.nn import Parameter
import numpy as np
from torch.nn import ParameterList
import copy
from utils.normalization import IdentityNormalizer
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class DDNO(torch.nn.Module):
    __name__ = "DDNO"

    def __init__(self, 
                 local_operator, 
                 domain, 
                 space_dim, 
                 output_dim=1, 
                 normalizer=None, 
                 time_dependent=False,
                 time_span=None,
                 device=None
                 ):
        super(DDNO, self).__init__()
        self.local_operator = local_operator
        for p in self.local_operator.parameters():
            p.requires_grad = False
        self.device = device if device else torch.device("cpu")

        self.domain = domain
        self.space_dim = space_dim
        self.output_dim = output_dim if output_dim else self.local_operator.output_size
        
        self.time_dependent = time_dependent
        if time_dependent:
            self.time_step = self.local_operator.output_size
            assert time_span is not None, logger.error("for time dependent, time span has to be specifiece")
            self.time_span = time_span
            self.trm = ParameterList(self._get_time_restriction_matrix())
            self.time_masks = ParameterList(self._get_time_masks())

        self.rm = ParameterList(self._get_restriction_matrix())
        self.sbm = ParameterList(self._get_subdomain_boundary_matrix())
        self.sbm_dirichlet = ParameterList(self._get_subdomain_dirichlet_matrix())
        self.sbm_neumann = ParameterList(self._get_subdomain_neumann_matrix())
        self.masks = ParameterList(self._get_subdomain_masks())
        self.graphs = self._get_subdomain_graph()
        
        self.geometry = Parameter(torch.tensor(self.domain.geometry[:, :self.space_dim], dtype=torch.float32))
        self.normalizer = normalizer if normalizer else IdentityNormalizer(self.device)

        for p in self.parameters():
            p.requires_grad = False

    def to(self, device):
        self = super(DDNO, self).to(device)
        self.graphs = [g.to(device) for g in self.graphs]
        self.device = device
        return self

    def _get_subdomain_graph(self):
        X = [d.geometry[:, :self.space_dim] for d in self.domain.subDomain]
        graphs = []
        for x in X:
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = torch.from_numpy(x).float()
            graphs.append(g)
        return graphs

    def _get_restriction_matrix(self):
        # restriction matrix maps global index to local index
        # shape of restriction matrix rm[i] is (n_component_i, n_all)
        rm = [torch.zeros((d.num_nodes, self.domain.num_nodes), dtype=torch.float32) for d in self.domain.subDomain]
        for i, m in enumerate(self.domain.mapping):
            for g, l in m["g2l"].items():
                rm[i][l, g] = 1

        return rm
    
    def _get_time_restriction_matrix(self):
        trm = [torch.zeros((self.time_span, self.time_step)) for i in self.domain.subTimeInterval]
        for i, interval in enumerate(self.domain.subTimeInterval):
            s, e = interval
            for j, t in enumerate(range(s, e)):
                trm[i][t, j] = 1
        return trm


    def _get_subdomain_boundary_matrix(self):
        # maps local index to boundary index
        # shape of sbm is (n_boundary_i, n_component_i)
        sbm = []

        for sd in self.domain.subDomain:
            boundary = sd.topology["boundary"]
            m = torch.zeros((len(boundary), sd.num_nodes), dtype=torch.float32)
            #m = nn.Linear(len(local_indices), len(b_index), False)
            #m.apply(lambda s: nn.init.zeros_(s.weight))
            for i, b in enumerate(boundary):
                m[i, b] = 1
            sbm.append(m)
        
        return sbm

    def _get_subdomain_dirichlet_matrix(self):
        sbdm = []
        for sd in self.domain.subDomain:
            db_index, _ = sd.getDirichletBoundary()
            m = torch.zeros((len(db_index), sd.num_nodes), dtype=torch.float32)
            for i, b in enumerate(db_index):
                m[i, b] = 1
            sbdm.append(m)
            #m = torch.zeros((, ))
        return sbdm
    
    def _get_subdomain_neumann_matrix(self):
        sbnm = []
        for sd in self.domain.subDomain:
            nb_index, _ = sd.getNeumannBoundary()
            m = torch.zeros((len(nb_index), sd.num_nodes), dtype=torch.float32)
            for i, b in enumerate(nb_index):
                m[i, b] = 1
            sbnm.append(m)
            #m = torch.zeros((, ))
        return sbnm
    
    def _get_subdomain_masks(self):
        # masks on nodes in subdomain
        masks = []
        #masks = [torch.zeros(self.domain.num_nodes, dtype=torch.float32, requires_grad=False) for i in range(self.domain.n_parts)]
        for m in self.domain.mapping:
            mask = torch.zeros((self.domain.num_nodes, 1), dtype=torch.float32)
            mask[m["l2g"]] = 1
            masks.append(mask)

        return masks
    
    def _get_time_masks(self):
        time_masks = []
        for interval in self.domain.subTimeInterval:
            s, e = interval
            time_mask = torch.zeros((self.time_span, 1), dtype=torch.float32)
            time_mask[s:e] = 1
            time_masks.append(time_mask)
        return time_masks

    def initialize(self, inputs_f):

        if not self.time_dependent:
            boundary_input = inputs_f[-1]
            sol = torch.zeros((self.domain.num_nodes, 1), dtype=torch.float32).to(self.device)
            (db_index, db_func_value), (nb_index, nb_func_value) = self.map_boundary(boundary_input)
            for l, v in zip(db_index, db_func_value):
                # dirichlet boundary
                sol[l] = v
        else:
            boundary_input = inputs_f[1]
            initial_condition_input = inputs_f[0]

            sol = torch.zeros((self.domain.num_nodes, self.time_span), dtype=torch.float32).to(self.device)
            (db_index, db_func_value), (nb_index, nb_func_value) = self.map_boundary(boundary_input)
            for l, v in zip(db_index, db_func_value):
                # dirichlet boundary
                sol[l] = v
            
            (ic_index, ic_func_value) = self.map_input(initial_condition_input)
            for l, v in zip(ic_index, ic_func_value):
                sol[l, 0] = v
            
        
        return sol

    def map_input(self, input_f):
        index, func_value = [], []
        for f in input_f:
            if self.space_dim == 2:
                query = np.concatenate([f[:2].cpu().numpy(), np.zeros((1, ))])
            else:
                query = f[:3].cpu().numpy()

            _, i = self.domain.tree.query(query)
            
            func_value.append(f[self.space_dim:])
            index.append(i)
        func_value = torch.stack(func_value).to(self.device)

        return index, func_value


    def map_boundary(self, input_f):
        # map locations in input to their corresponding index

        db_func_value, nb_func_value = [], []
        db_index, nb_index = [], []
        for f in input_f:
            if self.space_dim == 2:
                query = np.concatenate([f[:2].cpu().numpy(), np.zeros((1, ))])
            else:
                query = f[:3].cpu().numpy()
            _, i = self.domain.tree.query(query)

            if f[-1] == 0.0:

                db_func_value.append(f[self.space_dim:-1])
                db_index.append(i)
            elif f[-1] == 1.0:
                nb_func_value.append(f[self.space_dim:-1])
                nb_index.append(i)
            else:
                raise ValueError("the last column of input_f should be 0 for dirichlet or 1 for neumann")

        if len(db_func_value) > 0:
            db_func_value = torch.stack(db_func_value).to(self.device)
        else:
            db_func_value = torch.empty((0, input_f.shape[1] - 1 - self.space_dim)).to(self.device)
        if len(nb_func_value) > 0:
            nb_func_value = torch.stack(nb_func_value).to(self.device)
        else:
            nb_func_value = torch.empty((0, input_f.shape[1] - 1 - self.space_dim)).to(self.device)

        return (db_index, db_func_value), (nb_index, nb_func_value)



    def patch_local_sols(self, local_sols):
        # patch local solutions to form global solutions
        sol_sum = sum([self.rm[i].T @ local_sols[i] for i in range(self.domain.n_parts)])
        count = sum(self.masks)
        sol = sol_sum / count
        return sol
    
    def patch_time_sols(self, time_sols):
        sol_sum = sum([time_sols[i] @ self.trm[i].T for i in range(self.domain.num_interval)])
        count = sum(self.time_masks)[:, 0]
        sol = sol_sum / count
        return sol

    def forward(self, sol, bic, u_p, input_func, downsample=1):
        bc, ic = bic
        
        if not self.time_dependent:
            (db_index, db_func_value), (nb_index, nb_func_value) = bc
        else:
            (db_index, db_func_value), (nb_index, nb_func_value) = bc
            (ic_index, ic_func_value) = ic

        sol = torch.clone(sol)

        if not self.time_dependent:
            sol[db_index, -self.output_dim:] = db_func_value
            nb_func = torch.zeros_like(sol)
            nb_func[nb_index, -self.output_dim:] = nb_func_value
        else:
            # for time series, boundary value is from time step 1 instead of 0,
            # time step 0 is initial condition
            sol[db_index] = db_func_value
            sol[ic_index, 0:1] = ic_func_value
            nb_func = torch.zeros_like(sol)
            nb_func[nb_index] = nb_func_value

        
        # solve equation in each domain
        

        if not self.time_dependent:
            local_sols = []

            local_input_func = [[torch.cat((self.rm[i] @ self.geometry, self.rm[i] @ f), dim=1) for f in input_func] for i, _ in enumerate(self.domain.subDomain)]
            local_dirichlet_func = [torch.cat((self.sbm_dirichlet[i] @ self.rm[i] @ self.geometry, self.sbm_dirichlet[i] @ self.rm[i] @ sol), dim=-1)[::downsample, :] for i, _ in enumerate(self.domain.subDomain)]
            local_augmented_dirichlet_func = [torch.cat([local_dirichlet_func[i], torch.zeros((local_dirichlet_func[i].shape[0], 1)).to(self.device)], dim=1) for i, _ in enumerate(self.domain.subDomain)]
            local_neumann_func = [torch.cat((self.sbm_neumann[i] @ self.rm[i] @ self.geometry, self.sbm_neumann[i] @ self.rm[i] @ nb_func), dim=-1) for i, _ in enumerate(self.domain.subDomain)]
            local_augmented_neumann_func = [torch.cat([local_neumann_func[i], torch.ones((local_neumann_func[i].shape[0], 1)).to(self.device)], dim=1) for i, _ in enumerate(self.domain.subDomain)]
            local_boundary_func = [torch.cat([local_augmented_dirichlet_func[i], local_augmented_neumann_func[i]]) for i, _ in enumerate(self.domain.subDomain)]
            
            local_input_func = [MultipleTensors([t for t in local_input_func[i]] + [local_boundary_func[i]]) for i, _ in enumerate(self.domain.subDomain)]
            gs, us, fs, reverse_args = zip(*[self.normalizer.preprocess(self.graphs[i], u_p, local_input_func[i]) for i, _ in enumerate(self.domain.subDomain)])
            if self.local_operator.__name__ == "CGPT":
            
                batched = [dgl.batch(list(gs)), torch.stack(us)]
                fs_ = MultipleTensors(
                            [pad_sequence([fs[i][j] for i in range(len(fs))]).permute(1, 0, 2) for j in range(len(fs[0]))])
                batched.append(fs_)
                indices = [0] + torch.cumsum(batched[0].batch_num_nodes(), 0).cpu().numpy().tolist()
                ls = self.local_operator(*batched)
                local_sol = [ls[indices[i]:indices[i+1], :] for i in range(len(indices) - 1)]
            elif self.local_operator.__name__ == "geofno":
                meshes = []
                input_bcs = []
                indices = []
                u_ps = []
                for g, f in zip(gs, fs):
                    mesh = g.ndata['x'].cpu().numpy()
                    if len(f) > 1:
                        bc = f[1][:, [0, 1, 2, 3]].cpu().numpy()
                        u_p = f[0].cpu().numpy()
                    else:
                        bc = f[0][:, [0, 1, 2, 3]].cpu().numpy()
                        u_p = None
                    #print(f[0][0:10])
                    #input_bc = np.concatenate([mesh, np.zeros((mesh.shape[0], 1))], axis=1)
                    
                    #tree = KDTree(mesh)
                    
                    # for p in bc:
                    #     _, i = tree.query(p[[0, 1]])
                    #     input_bc[i, -1] = p[2]
                        
                    mesh = torch.from_numpy(mesh)

                    #print(bc.shape)
                    input_bc = torch.from_numpy(bc)
                    #u = torch.from_numpy(u)
                    index = (mesh.shape[0], bc.shape[0])
                    input_bc = F.pad(input_bc, pad=(0, 0, 0, 120-input_bc.shape[0]))
                    mesh = F.pad(mesh, pad=(0, 0, 0, 120-mesh.shape[0]))

                    if u_p is not None:
                        u_p = torch.from_numpy(u_p)
                        u_p = F.pad(u_p, pad=(0, 0, 0, 120-u_p.shape[0]))
                    #print(mesh.shape)
                    meshes.append(mesh)
                    input_bcs.append(input_bc)
                    indices.append(index)
                    u_ps.append(u_p)
                    
                inputs = torch.stack(input_bcs, dim=0)
                inputs = inputs.to(torch.float32)
                inputs = inputs.cuda()

                if u_ps[0] is not None:
                    input_u_ps = torch.stack(u_ps, dim=0)
                    input_u_ps = input_u_ps.to(torch.float32)
                    input_u_ps = input_u_ps.cuda()
                else:
                    input_u_ps = None

                meshes = torch.stack(meshes, dim=0)
                meshes = meshes.to(torch.float32)
                meshes = meshes.cuda()

                #print(inputs.shape, meshes.shape)
                ls = self.local_operator(inputs, input_u_ps, meshes)
                local_sol = [l[:ind[0]] for l, ind in zip(ls, indices)]
            
            elif self.local_operator.__name__ == "meshgraphnets":
                # Similar to geofno but for meshgraphnets
                meshes = []
                input_bcs = []
                indices = []
                u_ps = []
                for g, f in zip(gs, fs):
                    mesh = g.ndata['x'].cpu().numpy()
                    if len(f) > 1:
                        bc = f[1][:, [0, 1, 2, 3]].cpu().numpy()
                        u_p = f[0].cpu().numpy()
                    else:
                        bc = f[0][:, [0, 1, 2, 3]].cpu().numpy()
                        u_p = None
                        
                    mesh = torch.from_numpy(mesh)
                    input_bc = torch.from_numpy(bc)
                    index = (mesh.shape[0], bc.shape[0])
                    input_bc = F.pad(input_bc, pad=(0, 0, 0, 120-input_bc.shape[0]))
                    mesh = F.pad(mesh, pad=(0, 0, 0, 120-mesh.shape[0]))

                    if u_p is not None:
                        u_p = torch.from_numpy(u_p)
                        u_p = F.pad(u_p, pad=(0, 0, 0, 120-u_p.shape[0]))
                    
                    meshes.append(mesh)
                    input_bcs.append(input_bc)
                    indices.append(index)
                    u_ps.append(u_p)
                    
                inputs = torch.stack(input_bcs, dim=0)
                inputs = inputs.to(torch.float32)
                inputs = inputs.to(self.device)

                if u_ps[0] is not None:
                    input_u_ps = torch.stack(u_ps, dim=0)
                    input_u_ps = input_u_ps.to(torch.float32)
                    input_u_ps = input_u_ps.to(self.device)
                else:
                    input_u_ps = None

                meshes = torch.stack(meshes, dim=0)
                meshes = meshes.to(torch.float32)
                meshes = meshes.to(self.device)

                ls = self.local_operator(inputs, input_u_ps, meshes)
                local_sol = [l[:ind[0]] for l, ind in zip(ls, indices)]

            local_sols = [self.normalizer.postprocess(s, *a) for s, a in zip(local_sol, reverse_args)]

            return local_sols
        else:
            temporal_local_sols = []

            local_input_func = [[torch.cat((self.rm[i] @ self.geometry, self.rm[i] @ f), dim=1) for f in input_func] for i, _ in enumerate(self.domain.subDomain)]
            local_dirichlet_func = [torch.cat((self.sbm_dirichlet[i] @ self.rm[i] @ self.geometry, self.sbm_dirichlet[i] @ self.rm[i] @ sol), dim=-1) for i, _ in enumerate(self.domain.subDomain)]
            local_augmented_dirichlet_func = [torch.cat([local_dirichlet_func[i], torch.zeros((local_dirichlet_func[i].shape[0], 1)).to(self.device)], dim=1) for i, _ in enumerate(self.domain.subDomain)]
            local_neumann_func = [torch.cat((self.sbm_neumann[i] @ self.rm[i] @ self.geometry, self.sbm_neumann[i] @ self.rm[i] @ nb_func), dim=-1) for i, _ in enumerate(self.domain.subDomain)]
            local_augmented_neumann_func = [torch.cat([local_neumann_func[i], torch.ones((local_neumann_func[i].shape[0], 1)).to(self.device)], dim=1) for i, _ in enumerate(self.domain.subDomain)]
            local_boundary_func = [torch.cat([local_augmented_dirichlet_func[i], local_augmented_neumann_func[i]]) for i, _ in enumerate(self.domain.subDomain)]

            gs, us, fs, reverse_args = [], [], [], []

            for interval in self.domain.subTimeInterval:
                s, e = interval[0], interval[1]
                local_initial_func = [torch.cat([self.rm[i] @ self.geometry, self.rm[i] @ sol[:, [s]]], dim=1) for i, _ in enumerate(self.domain.subDomain)]
                local_temporal_boundary_func = [torch.cat([local_boundary_func[i][:, 0:self.space_dim], 
                                                local_boundary_func[i][:, s+2:e+2], 
                                                local_boundary_func[i][:, -1:]], dim=1) for i, _ in enumerate(self.domain.subDomain)]
                current_local_input_func = [MultipleTensors([t for t in local_input_func[i]] + [local_initial_func[i], local_temporal_boundary_func[i]]).to(self.device) for i, _ in enumerate(self.domain.subDomain)]
                g, u, f, a = (zip(*[self.normalizer.preprocess(self.graphs[i], u_p, current_local_input_func[i]) for i, _ in enumerate(self.domain.subDomain)]))
                gs += g
                us += u
                fs += f
                reverse_args += a
            batched = [dgl.batch(list(gs)), torch.stack(us)]
            fs_ = MultipleTensors(
                           [pad_sequence([fs[i][j] for i in range(len(fs))]).permute(1, 0, 2) for j in range(len(fs[0]))])
            batched.append(fs_)
            indices = [0] + torch.cumsum(batched[0].batch_num_nodes(), 0).cpu().numpy().tolist()

            # inference once for all subdomains in this time step
            ls = self.local_operator(*batched)
            current_local_sol = [ls[indices[i]:indices[i+1], :] for i in range(len(indices) - 1)]

            current_local_sol = [self.normalizer.postprocess(s, *a) for s, a in zip(current_local_sol, reverse_args)]

            for i, interval in enumerate(self.domain.subTimeInterval):
                temporal_local_sols.append(current_local_sol[i*self.domain.n_parts:(i+1)*self.domain.n_parts])

            return temporal_local_sols
                

            

        
