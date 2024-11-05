import numpy as np
import matplotlib.pyplot as plt

def normalize_2D_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val == min_val:
        return np.zeros_like(array)
    
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

# Code for saving data needed for running sims
#cond0_dict_list = [0]*num_participants
#for idx in range(num_participants):
#    cond0_dict_list[idx] = {'training':emgs_block1[keys[idx]][0,:,:], 'labels':refs_block1[keys[idx]][0,:,:]}
#
#with open(path+cond0_filename, 'wb') as fp:
#    pickle.dump(cond0_dict_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
#    
#init_decoders = [Ws_block1[keys[i]][:, 0, :, :] for i in range(num_participants)]
#with open(path+all_decs_init_filename, 'wb') as fp:
#    pickle.dump(init_decoders, fp, protocol=pickle.HIGHEST_PROTOCOL)