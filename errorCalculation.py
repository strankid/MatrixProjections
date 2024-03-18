import pickle 
import numpy as np
import argparse
from projection_methods import monarch, robe, svd

def l2Error(mat1, mat2):
    return l2_norm_in_chunks(mat1 - mat2)

def getMethodFunction(methodName):
    methodMap = {"monarch": monarch.MonarchApproximator(),
                 "robe": robe.RobeApproximator(),
                 "svd": svd.SVDApproximator()}
    
    return methodMap[methodName]

def l2_norm_in_chunks(matrix, chunk_size=1024):
    """
    Calculates the L2 norm (Frobenius norm) of a matrix by splitting it into chunks.
    
    Args:
        matrix (numpy.ndarray): The input matrix.
        chunk_size (int, optional): The size of each chunk. Default is 1000.
        
    Returns:
        float: The L2 norm of the matrix.
    """
    # Flatten the matrix into a 1D array
    flattened = matrix.ravel()
    
    # Split the flattened array into chunks
    chunks = [flattened[i:i+chunk_size] for i in range(0, len(flattened), chunk_size)]
    
    # Compute the sum of squared values for each chunk
    squared_chunks = [np.sum(chunk ** 2) for chunk in chunks]
    
    # Sum the squared values across all chunks
    squared_sum = np.sum(squared_chunks)
    
    # Take the square root to get the L2 norm
    l2_norm = np.sqrt(squared_sum)
    
    return l2_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="mistralai/Mistral-7B-v0.1", help="Model weights to project")
    parser.add_argument("-method", type=str, default="lr", help="Algorithm to use for projection", choices=["monarch", "svd", "robe"])
    parser.add_argument('-debug', action="store_true", help="print debug statements")
    args = parser.parse_args() 

    if args.debug: print("================== IMPORTING MODEL ==================")
    with open(f"models/{args.model}.pickle", 'rb') as f:
        model_weights = pickle.load(f)
    
    method = getMethodFunction(args.method)
    method_errors = dict()


    for layer, weights in model_weights.items():
        if args.debug: print(f"================== PROJECTING LAYER {layer} ==================")
        try:
            res_dict = method.approximate(weights)
        except NotImplementedError:
            continue
        
        if args.debug: print(f"================== CALCULATING ERROR {layer} ==================")
        method_errors[layer] = l2Error(weights, res_dict["approx_mat_dense"])

    if args.debug: print(f"================== SAVING RESULT ==================")
    with open(f"results/{args.model}+{args.method}.pickle", 'wb') as handle:
        pickle.dump(method_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

