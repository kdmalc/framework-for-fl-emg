import os
import h5py
import numpy as np
import pandas as pd
import statistics
import re


def create_linkage_attack_df(extractration_dict, num_clients=14):
    keys = extractration_dict.keys()
    num_updates_lst = []
    for key in keys:
        if "global" in key:
            continue
        num_updates_lst.append(len(extractration_dict[key]))
    mode_update = statistics.mode(num_updates_lst)
    max_update = max(num_updates_lst)  # If there is a global model, this will be number of global rounds, which is much greater than any client's local rounds
    if max_update == mode_update:  # Eg is this the NOFL case
        num_updates = mode_update
    else:
        num_updates = min(max_update, int(statistics.mean(num_updates_lst) + np.sqrt(statistics.stdev(num_updates_lst))))  # int() rounds down
        # ^ np.sqrt(stdev) bc sometimes stdev is massive (like 38), making num_updates just equal to the max...
    print(f"num updates = {num_updates}; max_update {max_update}; avg_num_updates {statistics.mean(num_updates_lst)}")
    
    # Initialize a list of empty DataFrames for each user group
    dec_flattened_list = [pd.DataFrame(columns=['Subject', 'Fold', 'Update Number', 'Flattened Dec']) for _ in range(num_updates)]
    global_dec_flattened_list = [pd.DataFrame(columns=['Subject', 'Fold', 'Update Number', 'Flattened Dec']) for _ in range(max_update)]
    
    # Regular expression pattern to extract subject and fold
    pattern = r"((S\d+)_client_local_model_log_fold(\d+)|global_dec_log_fold(\d+))"
    # Loop through keys and updates to populate the DataFrames
    for key_idx, key in enumerate(keys):
        key_len = len(extractration_dict[key])
        match = re.search(pattern, key)  # Extract the subject and fold using regex
        # Group0: Entire match (the entire local key? Not sure what happens if global is the match...)
        # Group1: Local subject ID (str); Group3: Local fold; Group4: Global fold
        if match:
            if match.group(2):
                long_subject_str = match.group(1)  # e.g., 'S0', 'S1', 'S10'
                subject = re.search(r'S\d+', long_subject_str).group()
                fold = int(match.group(3))  # e.g., '0', '1', '2'
                for update_number in range(num_updates): 
                    if update_number >= key_len:
                        continue
                    else:
                        user_data = np.ravel(extractration_dict[key][update_number])
                        dec_flattened_list[update_number].loc[len(dec_flattened_list[update_number])] = [subject, fold, update_number, user_data]
            elif match.group(4):  # This means it's a 'global_dec_log_fold(\d+)' key
                fold = int(match.group(4))
                for update_number in range(max_update): # Max_update will be the number of global rounds
                    global_data = np.ravel(extractration_dict[key][update_number])
                    # Global should be the same for all clients, for a given round and fold
                    subj_ID_num = 0
                    global_dec_flattened_list[update_number].loc[len(global_dec_flattened_list[update_number])] = ["S"+str(subj_ID_num), fold, update_number, global_data]
    
    # Concat all the dfs into a single training input dataframe
    dec_flattened = pd.concat(dec_flattened_list, ignore_index=True, axis=0)
    flattened_input_df = dec_flattened.join(dec_flattened['Flattened Dec'].apply(pd.Series)).drop('Flattened Dec', axis=1)
    # Same for global
    global_dec_flattened = pd.concat(global_dec_flattened_list, ignore_index=True, axis=0)
    global_flattened_input_df = global_dec_flattened.join(global_dec_flattened['Flattened Dec'].apply(pd.Series)).drop('Flattened Dec', axis=1)
    return flattened_input_df, num_updates, global_flattened_input_df, max_update


def avg_client_results_across_folds(extraction_dict, algorithm, num_clients=14, num_folds=7, verbose=False):
    client_logs = {f'S{i}_client_local_cost_func_comps_log': 0.0 for i in range(num_clients)}  # S0 to S13
    global_logs = {f'S{i}_client_global_cost_func_comps_log': 0.0 for i in range(num_clients)}  # S0 to S13
    for fold in range(num_folds):
        for i in range(num_clients):
            # Access client local test logs for each fold
            client_key = f'S{i}_client_local_cost_func_comps_log_fold{fold}'
            try:
                client_last_n_logs = np.array(extraction_dict[client_key][-10:]) 
                client_data = np.mean([ele[1] for ele in client_last_n_logs])
                if client_logs[f'S{i}_client_local_cost_func_comps_log'] == 0.0:
                    client_logs[f'S{i}_client_local_cost_func_comps_log'] = client_data
                else:
                    client_logs[f'S{i}_client_local_cost_func_comps_log'] += client_data
            except KeyError:
                # It was a testing client and thus not saved, so just skip to the next iter
                pass
    
            if "NOFL" in algorithm:
                pass
            else: 
                # Access global and local test logs for each fold
                global_key = f'S{i}_client_global_cost_func_comps_log_fold{fold}'
                try:
                    client_last_n_global_logs = np.array(extraction_dict[global_key][-10:]) 
                    # Mean across last n values, for one client, within one fold
                    client_global_data = np.mean([ele[1] for ele in client_last_n_global_logs])
                    if global_logs[f'S{i}_client_global_cost_func_comps_log'] == 0.0:
                        global_logs[f'S{i}_client_global_cost_func_comps_log'] = client_global_data
                    else:
                        global_logs[f'S{i}_client_global_cost_func_comps_log'] += client_global_data
                except KeyError:
                    # It was a testing client and thus not saved, so just skip to the next iter
                    pass
    
    return client_logs, global_logs
    

def load_model_logs(cv_results_path, filename, num_clients=14, num_folds=7, verbose=False):
    extraction_dict = dict()
    for i in range(num_folds):
        h5_path = os.path.join(cv_results_path, filename+f"{i}.h5")
        
        # Load data from HDF5 file
        with h5py.File(h5_path, 'r') as f:
            a_group_key = list(f.keys())
            for key in a_group_key:
                if key=="client_local_model_log":
                    client_keys = list(f[key])
                    for ck in client_keys:
                        ed_key = f"{ck}_fold{i}" 
                        if len(list(f[key][ck]))==0:
                            pass
                        else:
                            extraction_dict[ed_key] = list(f[key][ck])
                elif key=="global_dec_log" and "NOFL" not in filename:
                    ed_key = f"{key}_fold{i}"
                    extraction_dict[ed_key] = list(f[key])
                else:
                    pass

    return extraction_dict