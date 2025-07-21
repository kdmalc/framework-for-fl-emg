import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import h5py
import copy
import statistics
import re
from sklearn.svm import SVC

from utils import *

random.seed(a=1)


def plot_inter_user_histograms(dfs, cond_names=None):
    """
    Plots a 2x3 grid of histograms for inter-user distances.
    
    Parameters:
    - dfs: list of 6 DataFrames, each with a 'Distance' column.
    - cond_names: optional list of 6 names for each condition.
    """
    if cond_names is None:
        cond_names = [f"Condition {i+1}" for i in range(len(dfs))]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    
    for i, (df, ax) in enumerate(zip(dfs, axes)):
        ax.hist(df['Distance'], bins=30)
        ax.set_title(cond_names[i])
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.xaxis.grid(False)  # turn off vertical grid lines
    
    plt.suptitle('Inter-User Pairwise Distance Histograms', fontsize=16)
    plt.show()


def plot_within_user_histograms(dfs, cond_names=None):
    """
    Plots a 2x3 grid of histograms for within-user distances.
    
    Parameters:
    - dfs: list of 6 DataFrames, each with a 'Distance' column.
    - cond_names: optional list of 6 names for each condition.
    """
    if cond_names is None:
        cond_names = [f"Condition {i+1}" for i in range(len(dfs))]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    
    for i, (df, ax) in enumerate(zip(dfs, axes)):
        ax.hist(df['Distance'], bins=30)
        ax.set_title(cond_names[i])
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.xaxis.grid(False)  # turn off vertical grid lines
    
    plt.suptitle('Within-User Update Distance Histograms', fontsize=16)
    plt.show()


def process_dict_to_mean_lst(my_dict):
    my_lst = []
    for key in my_dict['client_local_test_log'].keys():
        my_lst.append(np.mean(my_dict['client_local_test_log'][key]))
    return my_lst


def load_error_logs(
    cv_results_path: str,
    filename: str,
    num_folds: int = 7,
    extract_server: bool = True,
    last_n_rounds: int = 10,
    verbose: bool = False
):
    """
    Load / average:
      • server‐level curves (local/global test log), trimmed to last_n_rounds if specified
      • client‐level test logs, also trimmed to last_n_rounds

    Args:
      cv_results_path: directory containing fold files
      filename:       base filename, e.g. 'ExperimentA_KFold'
      num_folds:      number of folds
      extract_server: if True, also average the server‐level test curves
      last_n_rounds:  how many rounds to keep from the end of each curve
                      (None to use full length)
      verbose:        debug prints

    Returns:
      dict with keys:
        'local_test_error_log' (ndarray), 
        'global_test_error_log' (ndarray),
        'client_local_test_log' (dict client→ndarray)
    """
    # buffers
    per_fold_srv = {"local_test_error_log": [], "global_test_error_log": []}
    per_client_raw = {}  # client_key -> list of trimmed arrays

    for i in range(num_folds):
        path = os.path.join(cv_results_path, f"{filename}{i}.h5")
        if not os.path.isfile(path):
            if verbose:
                print(f"Skipping missing fold file: {path}")
            continue

        with h5py.File(path, 'r') as f:
            # --- server‐level ---
            if extract_server:
                for key in per_fold_srv:
                    if key in f and len(f[key])>0:
                        arr = f[key][()]
                        if last_n_rounds is not None:
                            arr = arr[-last_n_rounds:]
                        per_fold_srv[key].append(arr)
                    elif verbose:
                        print(f" Fold {i}: no data for {key}")

            # --- client‐level ---
            grp = "client_local_test_log"
            if grp in f:
                for client_key in f[grp].keys():
                    arr = f[grp][client_key][()]
                    if last_n_rounds is not None:
                        if arr.shape[0] < last_n_rounds:
                            # too short: skip this fold for this client
                            ## This triggers for Cross heldout clients FYI. This is expected behaviour for the Cross case!
                            #print(f"Too few rounds! Skipping client {client_key}, group {grp}")
                            continue
                        arr = arr[-last_n_rounds:]
                    per_client_raw.setdefault(client_key, []).append(arr)
            elif verbose:
                print(f" Fold {i}: missing group '{grp}'")

    out = {}

    # average server‐level
    if extract_server:
        for key, arrs in per_fold_srv.items():
            if not arrs:
                if verbose:
                    print(f"No valid folds for {key}")
                continue
            stacked = np.stack(arrs, axis=0)      # (n_folds, last_n_rounds)
            out[key] = stacked.mean(axis=0)      # (last_n_rounds,)
            if verbose:
                print(f"{key}: stacked {stacked.shape} → mean {out[key].shape}")

    # average client‐level
    client_avg = {}
    if verbose:
        print("\nAveraging client test logs (last", last_n_rounds, "rounds):")
    for client_key, logs in per_client_raw.items():
        if not logs:
            if verbose:
                print(f"  {client_key}: no valid folds (or too short)")
            continue
        stacked = np.stack(logs, axis=0)       # (n_folds_client, last_n_rounds)
        client_avg[client_key] = stacked.mean(axis=0)
        if verbose:
            print(f"  {client_key}: {stacked.shape} → {client_avg[client_key].shape}")

    out["client_local_test_log"] = client_avg
    return out


def x_pos_map(condition_values_lst):
    mapping = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
    return [mapping[val] for val in condition_values_lst]


def algo_map(condition_values_lst):
    mapping = {0: 'Local', 1: 'Per-FedAvg', 2: 'FedAvg', 3: 'Local', 4: 'Per-FedAvg', 5: 'FedAvg'}
    return [mapping[val] for val in condition_values_lst]


def full_LOO_privacy_attack(ordered_conds_lst, augmented_subj_lst, max_updates=6, svc_kernel='linear'):
    """
    Returns two DataFrames:
      - local_privacy_df: per-subject privacy risk from the Local models
      - global_privacy_df: per-subject privacy risk from the Global models
    """
    local_results = []
    global_results = []

    for cond_idx, cond_dict in enumerate(ordered_conds_lst):
        # 1) build flat
        flat_df, num_updates, global_flat_df, max_update = create_linkage_attack_df(cond_dict)

        # --- LOCAL ---
        loc_df = flat_df[flat_df['Fold'] == 0].copy()
        parts = []
        for subj in augmented_subj_lst:
            sbj = loc_df[loc_df['Subject'] == subj].sort_values('Update Number')
            if sbj.empty: continue
            last_u = sbj['Update Number'].unique()[-max_updates:]
            parts.append(sbj[sbj['Update Number'].isin(last_u)])
        sel_loc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if not sel_loc.empty:
            feats = [c for c in sel_loc.columns if c.startswith('w_')]
            X, y = sel_loc[feats].values, sel_loc['Subject'].values
            n = len(y); correct = np.zeros(n, dtype=int)

            for i in range(n):
                mask = np.ones(n, bool); mask[i] = False
                clf = SVC(kernel='linear')
                clf.fit(X[mask], y[mask])
                pred = clf.predict(X[~mask].reshape(1, -1))[0]
                correct[i] = int(pred == y[i])

            for subj in augmented_subj_lst:
                idxs = np.where(y == subj)[0]
                if len(idxs)==0: continue
                acc = correct[idxs].mean()
                local_results.append({
                    'Condition_Number': cond_idx,
                    'Subject':          subj,
                    'Privacy_Risk':     acc
                })

        # --- GLOBAL ---
        if not global_flat_df.empty:
            gdf = global_flat_df[global_flat_df['Fold'] == 0].copy()
            if not gdf.empty:
                last_u_g = np.unique(gdf['Update Number'])[-max_updates:]
                sel_g = gdf[gdf['Update Number'].isin(last_u_g)]

                # duplicate per subject
                pools = []
                for subj in augmented_subj_lst:
                    tmp = sel_g.copy()
                    tmp['Subject'] = subj
                    pools.append(tmp)
                pool = pd.concat(pools, ignore_index=True)

                feats = [c for c in pool.columns if c.startswith('w_')]
                Xg, yg = pool[feats].values, pool['Subject'].values
                m = len(yg); correct_g = np.zeros(m, dtype=int)

                for i in range(m):
                    mask = np.ones(m, bool); mask[i] = False
                    clf = SVC(kernel=svc_kernel)
                    clf.fit(Xg[mask], yg[mask])
                    pred = clf.predict(Xg[~mask].reshape(1, -1))[0]
                    correct_g[i] = int(pred == yg[i])

                for subj in augmented_subj_lst:
                    idxs = np.where(yg == subj)[0]
                    if len(idxs)==0: continue
                    acc = correct_g[idxs].mean()
                    global_results.append({
                        'Condition_Number': cond_idx,
                        'Subject':          subj,
                        'Privacy_Risk':     acc
                    })

    base_local_df  = pd.DataFrame(local_results)
    base_global_df = pd.DataFrame(global_results)

    # Mapping from Condition_Number to the other attributes
    mapping_df = pd.DataFrame({
        'Condition_Number': [0, 1, 2, 3, 4, 5],
        'Algorithm': ['Local', 'Per-FedAvg', 'FedAvg', 'Local', 'Per-FedAvg', 'FedAvg'],
        'x_pos': [0, 1, 2, 0, 1, 2],
        'Scenario': ['Intra-Subject', 'Intra-Subject', 'Intra-Subject',
                    'Cross-Subject', 'Cross-Subject', 'Cross-Subject'],
        'Model': ['Local'] * 6
    })

    local_privacy_df = base_local_df.merge(mapping_df, on='Condition_Number', how='left')
    global_privacy_df = base_global_df.merge(mapping_df, on='Condition_Number', how='left')

    return local_privacy_df, global_privacy_df


def create_linkage_attack_df(extractration_dict):
    """This just sets up the input dataframes, doesn't actually run the attack"""
    keys = extractration_dict.keys()
    num_updates_lst = []
    for key in keys:
        if "global" in key:
            continue
        num_updates_lst.append(len(extractration_dict[key]))
    mode_update = statistics.mode(num_updates_lst)
    max_update = max(num_updates_lst)
    # ^ NOW THAT GLOBAL IS INCLUDED, MAX UPDATE IS WAY OFF FOR LOCAL!
    ## Could find the 2nd largest value, since all global runs will have the same length...
    if max_update == mode_update:  # Eg this is a poor man's proxy for if this is the NOFL case
        num_updates = mode_update
    else:
        num_updates = min(max_update, int(statistics.mean(num_updates_lst) + np.sqrt(statistics.stdev(num_updates_lst))))  # int() rounds down
        # np.sqrt(stdev) bc sometimes stdev is massive (like 38), making num_updates just equal to the max...
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
                #print(f"Group 4: {match.group(4)}")
                fold = int(match.group(4))
                for update_number in range(max_update): # Max_update will be the number of global rounds
                    global_data = np.ravel(extractration_dict[key][update_number])
                    #for subj_ID_num in range(num_clients):  # Global should be the same for all clients, for a given round and fold
                    subj_ID_num = 0
                    global_dec_flattened_list[update_number].loc[len(global_dec_flattened_list[update_number])] = ["S"+str(subj_ID_num), fold, update_number, global_data]
    
    # Concat all the dfs into a single training input dataframe
    # 1) LOCAL models ─ expand “Flattened Dec” into columns
    dec_flattened = pd.concat(dec_flattened_list, ignore_index=True)
    # turn each row’s array into a DataFrame
    dec_expanded = pd.DataFrame(
        dec_flattened['Flattened Dec'].tolist(),
        index=dec_flattened.index,
        columns=[f"w_{i}" for i in range(len(dec_flattened['Flattened Dec'].iloc[0]))]
    )
    flattened_input_df = pd.concat(
        [dec_flattened.drop(columns=['Flattened Dec']), dec_expanded],
        axis=1
    )

    # 2) GLOBAL models (may be empty!)
    global_dec_flattened = pd.concat(global_dec_flattened_list, ignore_index=True)
    if not global_dec_flattened.empty:
        global_expanded = pd.DataFrame(
            global_dec_flattened['Flattened Dec'].tolist(),
            index=global_dec_flattened.index,
            columns=[f"w_{i}" for i in range(len(global_dec_flattened['Flattened Dec'].iloc[0]))]
        )
        global_flattened_input_df = pd.concat(
            [global_dec_flattened.drop(columns=['Flattened Dec']), global_expanded],
            axis=1
        )
    else:
        # no global data in this condition → empty DataFrame
        global_flattened_input_df = pd.DataFrame()

    return flattened_input_df, num_updates, global_flattened_input_df, max_update


def process_trial_dfs_for_plotting(all_dicts_lst, privacy_df, global_privacy_df):
    all_trials_lst = [process_dict_to_mean_lst(my_dict) for my_dict in all_dicts_lst]

    # Data for intra- and cross- subject groups  
    all_trials_intra = [all_trials_lst[0], all_trials_lst[2], all_trials_lst[1]]
    all_trials_cross = [all_trials_lst[3], all_trials_lst[5], all_trials_lst[4]]

    # Build the “exploded” trial‐error DataFrames
    all_trials_intra_df = pd.DataFrame({
        'Condition_Number': [0, 1, 2],
        'x_pos':             [0, 2, 1],
        'Algorithm':         ['Local', 'Per-FedAvg', 'FedAvg'],
        'Scenario':          ['Intra'] * 3,
        'Velocity_Error':    all_trials_intra
    }).explode('Velocity_Error', ignore_index=True)

    all_trials_cross_df = pd.DataFrame({
        'Condition_Number': [4, 5, 6],
        'x_pos':             [0, 2, 1],
        'Algorithm':         ['Local', 'Per-FedAvg', 'FedAvg'],
        'Scenario':          ['Cross'] * 3,
        'Velocity_Error':    all_trials_cross
    }).explode('Velocity_Error', ignore_index=True)

    # Split off privacy‐risk DataFrames
    intra_privacy_df        = privacy_df[privacy_df['Scenario']=='Intra-Subject']
    cross_privacy_df        = privacy_df[privacy_df['Scenario']=='Cross-Subject']
    global_intra_privacy_df = global_privacy_df[global_privacy_df['Scenario']=='Intra-Subject']
    global_cross_privacy_df = global_privacy_df[global_privacy_df['Scenario']=='Cross-Subject']

    # “Full” versions (merge local+global)
    full_intra_privacy_df = pd.concat([intra_privacy_df, global_intra_privacy_df], ignore_index=True)
    full_cross_privacy_df = pd.concat([cross_privacy_df, global_cross_privacy_df], ignore_index=True)

    # Enforce mapping: Local→0, Per-FedAvg→2, FedAvg→1
    pos_map = {'Local': 0, 'Per-FedAvg': 2, 'FedAvg': 1}
    dfs = [
        all_trials_intra_df,
        all_trials_cross_df,
        full_intra_privacy_df,
        full_cross_privacy_df,
        global_intra_privacy_df,
        global_cross_privacy_df
    ]

    for i, df in enumerate(dfs):
        # ensure we have an actual copy, not a view
        df = df.copy()
        # map into x_pos via .loc
        df.loc[:, 'x_pos'] = df['Algorithm'].map(pos_map)
        dfs[i] = df

    (
        all_trials_intra_df,
        all_trials_cross_df,
        full_intra_privacy_df,
        full_cross_privacy_df,
        global_intra_privacy_df,
        global_cross_privacy_df
    ) = dfs

    return (
        all_trials_intra_df,
        all_trials_cross_df,
        full_intra_privacy_df,
        full_cross_privacy_df,
        global_intra_privacy_df,
        global_cross_privacy_df
    )