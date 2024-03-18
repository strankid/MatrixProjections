import torch
import numpy as np
from projection_methods import approximator

class RobeApproximator(approximator.Approximator):
    def __init__(self, seed=2342823481, P=2038074743, **kwargs):
        self.seed = seed
        self.P = P
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        print("Hashing using seed", seed)

        self.random_numbers = torch.randint(low=1, high=int(self.P/2) - 1, size=(4,), generator=self.gen)
        self.random_numbers = 2*self.random_numbers + 1
    
    def get_name(self):
        return "RobeApproximator"
    
    def hash(self, numbers, target_size=None):
        assert(target_size < self.P)
        self.random_numbers = self.random_numbers.to(numbers.device)
        return ((numbers * self.random_numbers[0] + torch.square(numbers) * self.random_numbers[1] + self.random_numbers[2]) % self.P) % target_size
    
    ## removed offsets from all functions. Should be okay for single layer approximations. Need to confirm
    def get_general_idx(self, w_shape, target_size):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num)
      idx = (self.hash(global_locations, target_size)) % target_size
      idx = idx.reshape(*w_shape)
      return idx
    
    def get_general_g(self, w_shape):
      total_num = np.prod(w_shape)
      global_locations = torch.arange(total_num)
      g = 2*(self.hash(global_locations, 2)) - 1
      g = g.reshape(*w_shape)
      return g
    
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float = 0.5):
        target_size = int(optim_mat.size * nb_params_share)
        original_shape = optim_mat.shape
        layer_scale = optim_mat.std()

        robe_index = self.get_general_idx(original_shape, target_size).flatten()
        robe_g = self.get_general_g(original_shape).flatten()

        optim_mat = torch.from_numpy(optim_mat).flatten()
        robe_weights = torch.zeros(target_size, dtype=optim_mat.dtype)
        robe_count = torch.zeros(target_size, dtype=optim_mat.dtype)     

        robe_weights = torch.zeros_like(robe_weights).scatter_add_(0, robe_index, robe_g*optim_mat/layer_scale)
        robe_count = robe_count.scatter_add_(0, robe_index, torch.ones_like(robe_index, dtype=robe_count.dtype))
        robe_weights = robe_weights /(1e-6 + robe_count)

        W = torch.mul(robe_weights[robe_index], robe_g) * layer_scale
        W = W.view(original_shape)

        res_dict = dict()
        res_dict["type"] = "RobeApproximator"
        res_dict["robe_weights"] = robe_weights.numpy()
        res_dict["robe_g"] = robe_g.numpy()
        res_dict["robe_index"] = robe_index.numpy()
        res_dict["layer_scale"] = layer_scale
        res_dict["approx_mat_dense"] = W.numpy()
        res_dict["nb_parameters"] = target_size
        return res_dict