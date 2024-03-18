import numpy as np
from projection_methods import approximator

class SVDApproximator(approximator.Approximator):
    def get_name(self):
        return "SVDApproximator"

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float = 0.03125):
        assert len(optim_mat.shape) == 2, "Can only handle matrices for LR approximation"

        print("checkpoint 1")
        max_nb_parameters = int(nb_params_share * optim_mat.size)
        rank = int(max_nb_parameters / (optim_mat.shape[0] + optim_mat.shape[1]))

        print("checkpoint 2")
        U, S, Vh = np.linalg.svd(optim_mat, full_matrices=False)
        S_root = np.sqrt(S)
        left_full_component = U @ np.diag(S_root)
        right_full_component = np.diag(S_root) @ Vh

        print("checkpoint 3")
        left_mat = left_full_component[:, :rank]
        right_mat = right_full_component[:rank, :]

        print("checkpoint 4")
        approx_mat_dense = left_mat @ right_mat
        nb_parameters = left_mat.size + right_mat.size

        print("checkpoint 5")
        res_dict = dict()
        res_dict["type"] = "LRApproximator"
        res_dict["left_mat"] = left_mat
        res_dict["right_mat"] = right_mat
        res_dict["approx_mat_dense"] = approx_mat_dense
        res_dict["nb_parameters"] = nb_parameters
        return res_dict

if __name__ == "__main__":
    # Test case
    optim_mat = np.random.uniform(-1, 1, size=(100, 100))
    
    approximator = SVDApproximator()
    res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.4)
    norm_difference = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")
    print("Norm Difference: " + str(norm_difference))

    halt = 1