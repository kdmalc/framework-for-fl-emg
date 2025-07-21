import numpy as np
import pandas as pd
import h5py
import copy

from utils import *


def extract_full_cost_logs(file_dict, update_num=None, 
                           alphaE=1e-6, alphaD=1e-4):
    """
    For each key,path in file_dict, loads the window of data with
    RUSA_load_data(..., update_num=update_num), then computes
      D0 = the decoder (must be constant across the window),
      V_int[t], V_dec[t] = intended & decoded velocity at t,
      cost[t]  = alphaE * ||V_dec[t] - V_int[t]||^2
               + alphaD * ||D0||^2
    
    Returns a dict mapping each same key → 1D numpy array of length T
    (the “full cost log” for that trial/user/update).
    """
    cost_logs = {}
    for key, path in file_dict.items():
        # load raw trial slices
        dec_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst = \
            RUSA_load_data(path, update_num=update_num, static=False)

        # if nothing loaded, return empty
        if len(dec_lst) == 0:
            cost_logs[key] = np.empty((0,))
            continue

        # check decoder is constant
        D0 = dec_lst[0]
        for i, Di in enumerate(dec_lst[1:], start=1):
            if not np.allclose(Di, D0):
                raise ValueError(f"[{key}] decoder changed at index {i}")

        # stack velocity arrays: shape (T, 2)
        V_int = np.stack(vint_lst, axis=0)
        V_dec = np.stack(vdec_lst, axis=0)

        # compute term1 per time‐step, then add constant term2
        diff    = V_dec - V_int                      # (T,2)
        term1   = alphaE * np.sum(diff*diff, axis=1) # (T,)
        term2   = alphaD * np.sum(D0*D0)             # scalar
        cost_log = term1 + term2                     # (T,)

        cost_logs[key] = cost_log

    return cost_logs


def RUSA_load_data(full_file_path, update_num=None, static=False):
    #times_idx=0
    filt_emg_idx=2
    dec_idx=3
    p_act_idx=8
    v_dec_idx=9
    v_int_idx=10
    p_ref_idx=7
    
    if update_num=="ALL":
        init_idx = 0
        final_idx = -1
    elif update_num is None:
        init_idx = 15625
        final_idx = 16827
    else:
        raise ValueError(f"update_num {update_num} not supported, please use None or 'ALL'!")
        
    if not static:
        try:
            with h5py.File(full_file_path, 'r') as handle:
                weiner_dataset = handle['weiner']
                weiner_df = pd.DataFrame(weiner_dataset)
            weiner_df = weiner_df.iloc[init_idx:final_idx]
            # Create lists for the remaining data
            dec_lst = [weiner_df.iloc[i, 0][dec_idx] for i in range(weiner_df.shape[0])]
            emg_lst = [weiner_df.iloc[i, 0][filt_emg_idx] for i in range(weiner_df.shape[0])]
            pref_lst = [weiner_df.iloc[i, 0][p_ref_idx] for i in range(weiner_df.shape[0])]
            pact_lst = [weiner_df.iloc[i, 0][p_act_idx] for i in range(weiner_df.shape[0])]
            vint_lst = [weiner_df.iloc[i, 0][v_int_idx] for i in range(weiner_df.shape[0])]
            vdec_lst = [weiner_df.iloc[i, 0][v_dec_idx] for i in range(weiner_df.shape[0])]

            return dec_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst
        except OSError:
            print(f"Unable to open/find file: {full_file_path}\nThus, downstream will be dummy values of -1")
            return [], [], [], [], [], []
    
def extract_raw_data(file_dict):
    results = {}
    for key, file_path in file_dict.items():
        dec_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst = RUSA_load_data(file_path)
        results[key] = {
            'trial_decoders': dec_lst,
            'trial_emg': emg_lst,
            'trial_reference_position': pref_lst,
            'trial_actual_position': pact_lst, 
            'trial_intended_velocity': vint_lst,
            'trial_decoded_velocity': vdec_lst
        }
    return results


def return_final_averaged_cost(full_file_path, update_num=None,
                               alphaE=1e-6, alphaD=1e-4):
    dec_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst = \
        RUSA_load_data(full_file_path,
                       update_num=update_num,
                       static=False)

    N = len(dec_lst)
    if N == 0:
        return np.empty((0,))

    # 1) check that all decoder weights are the same
    D0 = dec_lst[0]
    for i, Di in enumerate(dec_lst[1:], start=1):
        if not np.allclose(Di, D0):
            raise ValueError(f"Decoder weights differ at index {i}")
    # if you reach here, D0 is your only decoder (entire update uses the same dec)

    # 2) stack vint and vdec into shape (N,2)
    V_int = np.stack(vint_lst, axis=0)  
    V_dec = np.stack(vdec_lst, axis=0)  

    term1 = alphaE * np.linalg.norm(V_dec - V_int)**2
    term2 = alphaD * np.linalg.norm(D0)**2
    # 5) cost
    cost = term1 + term2                    

    return cost

# Function to extract cost and time-domain performance
def extract_cost(file_dict):
    results = {}
    for key, file_path in file_dict.items():
        results[key] = {
            'mean_cost': return_final_averaged_cost(file_path),
        }
    return results


def check_unique_decoders(full_file_path):
    """
    Scan the “weiner” table in the given .h5 and extract the decoder
    weight matrix from each record.  Return the list of timestep-indices
    at which a *new* decoder appears.
    """
    dec_idx = 3  # column index within the inner record where D lives

    # 1) load the raw “weiner” dataset
    with h5py.File(full_file_path, 'r') as f:
        weiner_dataset = f['weiner']
        weiner_df = pd.DataFrame(weiner_dataset)

    # 2) extract decoder list: each entry is record[0][dec_idx]
    dec_lst = [weiner_df.iloc[i, 0][dec_idx] for i in range(weiner_df.shape[0])]

    # 3) detect the first‐occurrence indices of each unique decoder
    unique_idxs = []
    if dec_lst:
        last = dec_lst[0]
        unique_idxs.append(0)
        for i, D in enumerate(dec_lst[1:], start=1):
            # compare elementwise; if any entry differs, record a change
            if not np.allclose(D, last):
                unique_idxs.append(i)
                last = D

    return unique_idxs
