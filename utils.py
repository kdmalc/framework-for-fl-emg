import os
import h5py
import numpy as np
import pandas as pd


def reorder_list(main_list, data_list, str_2_move="Static"):
    # Find the index of str_2_move in the main list
    static_index = main_list.index(str_2_move)

    # Reorder main list to put str_2_move first
    reordered_main_list = [str_2_move] + main_list[:static_index] + main_list[static_index + 1:]

    # Reorder data list to match the new order of main_list
    reordered_data_list = [data_list[static_index]] + data_list[:static_index] + data_list[static_index + 1:]

    return reordered_main_list, reordered_data_list


def RUSA_load_data(full_file_path, update_num=-2, verbose=False, static=False):
    filt_emg_idx=2
    dec_idx=3
    p_act_idx=8
    p_ref_idx=7
    
    if update_num==-2:
        init_idx = -1600
        final_idx = -800
    elif update_num==-1:
        init_idx = -800
        final_idx = -1
    elif update_num==0:
        init_idx = 0
        final_idx = 800
    else:
        raise ValueError(f"update_num {update_num} not supported, please use 0, -1, or -2!")
        
    if not static:
        try:
            with h5py.File(full_file_path, 'r') as handle:
                weiner_dataset = handle['weiner']
                weiner_df = pd.DataFrame(weiner_dataset)

            # Get the last 1600 points and then only keep points from -1600 to -800
            weiner_df = weiner_df.iloc[-1600:final_idx]

            # Create lists for the remaining data
            dec_lst = [weiner_df.iloc[i, 0][dec_idx] for i in range(weiner_df.shape[0])]
            emg_lst = [weiner_df.iloc[i, 0][filt_emg_idx] for i in range(weiner_df.shape[0])]
            pref_lst = [weiner_df.iloc[i, 0][p_ref_idx] for i in range(weiner_df.shape[0])]
            pact_lst = [weiner_df.iloc[i, 0][p_act_idx] for i in range(weiner_df.shape[0])]
            return dec_lst, emg_lst, pref_lst, pact_lst
        except OSError:
            print(f"Unable to open/find file: {full_file_path}\nThus, downstream will be dummy values of -1")
            return [], [], [], []
    
    
def return_final_averaged_cost_and_tdposerror(full_file_path, update_num=-2):
    dec_lst, emg_lst, pref_lst, pact_lst = RUSA_load_data(full_file_path, update_num=update_num, verbose=False, static=False)
    if len(dec_lst)==0:
        return -1, -1
    else:
        cost_log, perf_log, penalty_log = calc_cost_function(emg_lst, dec_lst, pref_lst, pact_lst)
        td_error = calc_time_domain_error(np.array(pref_lst), np.array(pact_lst))
        return np.mean(cost_log), np.mean(td_error)
    

# Function to extract cost and time-domain performance
def extract_performance(file_dict):
    results = {}
    for key, file_path in file_dict.items():
        mean_cost, mean_tdpe = return_final_averaged_cost_and_tdposerror(file_path)
        results[key] = {
            'mean_cost': mean_cost,
            'mean_tdpe': mean_tdpe
        }
    return results


def avg_client_results_across_folds(extraction_dict, algorithm, num_clients=14, num_folds=7, verbose=False):
    #print(f"ALGO: {algorithm}")
    client_logs = {f'S{i}_client_local_cost_func_comps_log': 0.0 for i in range(num_clients)}  # S0 to S13
    global_logs = {f'S{i}_client_global_cost_func_comps_log': 0.0 for i in range(num_clients)}  # S0 to S13
    for fold in range(num_folds):
        for i in range(num_clients):
            # Access client local test logs for each fold
            client_key = f'S{i}_client_local_cost_func_comps_log_fold{fold}'
            try:
                client_last_n_logs = np.array(extraction_dict[client_key][-10:]) 
                client_data = np.mean([ele[1] for ele in client_last_n_logs])
                #print(f"client data shape: {client_data.shape}")
                if client_logs[f'S{i}_client_local_cost_func_comps_log'] == 0.0:
                    #print("Instantiating empty entry")
                    client_logs[f'S{i}_client_local_cost_func_comps_log'] = client_data
                else:
                    #print("Adding new data")
                    client_logs[f'S{i}_client_local_cost_func_comps_log'] += client_data
                #print(f"Local cli{i} SUCCESS!")
            except KeyError:
                # It was a testing client and thus not saved, so just skip to the next iter
                #print(f"Local cli{i} FAILED!")
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
                    #print(f"Global cli{i} SUCCESS!")
                except KeyError:
                    # It was a testing client and thus not saved, so just skip to the next iter
                    #print(f"Global cli{i} FAILED!")
                    pass
            #print()
    #print()
    
    # Return the results
    return client_logs, global_logs


def load_model_logs(cv_results_path, filename, num_clients=14, num_folds=7, verbose=False):
    extraction_dict = dict()
    for i in range(num_folds):
        h5_path = os.path.join(cv_results_path, filename+f"{i}.h5")
        #print(h5_path)
        
        # Load data from HDF5 file
        with h5py.File(h5_path, 'r') as f:
            a_group_key = list(f.keys())
            #if i==0:
            #    print(a_group_key)
            for key in a_group_key:
                #print(key)
        
                if key=="client_local_model_log":
                    client_keys = list(f[key])
                    #print(client_keys)
                    for ck in client_keys:
                        ed_key = f"{ck}_fold{i}"  # Does this never update from or something...
                        #print(f"Key: {key}, Client: {ck}, Fold: {i}")
    
                        # So this doenst have any knoledge of the fold number???
                        if len(list(f[key][ck]))==0:
                            #print(f"{ed_key} SKIPPED!")
                            pass
                        else:
                            #print(f"{ed_key} SUCCESS!")
                            extraction_dict[ed_key] = list(f[key][ck])
                elif key=="global_dec_log" and "NOFL" not in filename:
                    # Do I need to turn this off for NoFL? Or will it just be empty and append something empty...
                    ed_key = f"{key}_fold{i}"
                    #print(ed_key)
                    extraction_dict[ed_key] = list(f[key])
                else:
                    pass

    return extraction_dict

############################################################
# From analysis_funcs... these funcs are probably elsewhere in this repo...
############################################################


import matplotlib.pyplot as plt
# Only used for boxplots
import matplotlib.patches as mpatches


def calc_time_domain_error(pref, pact):
    assert(pref.shape == pact.shape)
    td_error = np.linalg.norm(pref - pact, axis=1)
    return td_error


def calc_cost_function(F, D, pref, pact, alphaD=1e-4, alphaE=1e-6, Nd=2, Ne=64):
    # F, D, pref, and pact should be lists / npy arrays from the loaded data
    ## Eg vec from one trial
    # Should it be using D or D_{t-1}? ...
    # Likewise should s/F actually be s/F[:,:-1]? ...
    cost_log = [0 for ele in list(F)]
    perf_log = [0 for ele in list(F)]
    penalty_log = [0 for ele in list(F)]
    # Calculate cost at each time point
    for i in range(len(cost_log)):
        w = np.reshape(D[i],(Nd,Ne))
        V = (pref[i] - pact[i])*(1/60)
        Vplus = V[1:]
        term1 = alphaE*(np.linalg.norm((w@F[i] - Vplus))**2)
        term2 = alphaD*(np.linalg.norm(w)**2)
        cost_log[i] = term1 + term2
        perf_log[i] = term1
        penalty_log[i] = term2
    return cost_log, perf_log, penalty_log


# Function to filter out spikes dynamically using IQR
def remove_spikes_via_IQR(data, k=1.5):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    # Calculate Interquartile Range (IQR)
    IQR = Q3 - Q1
    # Define upper and lower bounds for outliers
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    # Filter out values that are outside the bounds (replaces them with np.nan...)
    filtered_data = np.where((data < upper_bound) & (data > lower_bound), data, np.nan)
    return filtered_data


def plot_cost_function(F_lst, D_lst, pref_lst, pact_lst, alphaD=1e-4, alphaE=1e-6, plot_total_cost=True, plot_perf=True, plot_combined=False, remove_spikes=False, plot_penalty=True, my_title="Cost Function", plot_2nd_half_separately=False, plot_separately=False):
    
    cost_log, perf_log, penalty_log = calc_cost_function(F_lst, D_lst, pref_lst, pact_lst, alphaD=alphaD, alphaE=alphaE)

    if remove_spikes:
        # Filtered loss data (removing spikes)
        perf_log = remove_spikes_via_IQR(perf_log)

    if plot_2nd_half_separately==True and plot_separately==False:
        # Create two subplots in a single row
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

        perf_log_half  = len(perf_log)//2
        penalty_log_half  = len(penalty_log)//2

        # First subplot: Velocity Error
        ax1.plot(perf_log[:perf_log_half], label="Velocity Error", color='blue')
        ax1.plot(penalty_log[:penalty_log_half], label="Decoder Penalty", color='red')
        ax1.set_title("Cost Func: First Half")
        ax1.set_xlabel("Training Round")
        ax1.set_ylabel("Loss")
        # Second subplot: Decoder Penalty
        ax2.plot(perf_log[perf_log_half:], label="Velocity Error", color='blue')
        ax2.plot(penalty_log[penalty_log_half:], label="Decoder Penalty", color='red')
        ax2.set_title("Cost Func: Second Half")
        ax2.set_xlabel("Training Round")
        ax2.set_ylabel("Loss")
        # Adjust layout and display
        plt.tight_layout()
        #plt.suptitle(my_title, y=1.02, fontsize=16)  # Optional: Add a super title for both plots
        plt.show()
    elif plot_separately==True and plot_2nd_half_separately==True:
        if plot_combined:
            fig, axes = plt.subplots(3, 2, figsize=(12, 6))  # Adjusted height for better spacing
            # Flatten the axes array to make it a 1D array
            ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

            # First subplot: Velocity Error Half 1
            perf_log_half  = len(perf_log)//2
            ax1.plot(perf_log[:perf_log_half], label="Velocity Error", color='blue')
            ax1.set_title("First Half of Velocity Error")
            ax1.set_xlabel("Training Round")
            ax1.set_ylabel("Loss")
            # Second subplot: Velocity Error Half 2
            ax2.plot(perf_log[perf_log_half:], label="Decoder Penalty", color='blue')
            ax2.set_title("Second Half of Velocity Error")
            ax2.set_xlabel("Training Round")
            ax2.set_ylabel("Loss")

            # Third subplot: Decoder Penalty Half 1
            penalty_log_half  = len(penalty_log)//2
            ax3.plot(penalty_log[:penalty_log_half], label="Decoder Penalty", color='red')
            ax3.set_title("First Half of Decoder Penalty")
            ax3.set_xlabel("Training Round")
            ax3.set_ylabel("Loss")
            # Fourth subplot: Decoder Penalty Half 2
            ax4.plot(penalty_log[penalty_log_half:], label="Decoder Penalty", color='red')
            ax4.set_title("Second Half of Decoder Penalty")
            ax4.set_xlabel("Training Round")
            ax4.set_ylabel("Loss")

            # Fifth subplot: COMBINED Half 1
            ax5.plot(penalty_log[:penalty_log_half], label="Decoder Penalty", color='red')
            ax5.plot(perf_log[:perf_log_half], label="Velocity Error", color='blue')
            ax5.set_title("First Half Combined")
            ax5.set_xlabel("Training Round")
            ax5.set_ylabel("Loss")
            # Sixth subplot: COMBINED Half 2
            ax6.plot(penalty_log[penalty_log_half:], label="Decoder Penalty", color='red')
            ax6.plot(perf_log[perf_log_half:], label="Decoder Penalty", color='blue')
            ax6.set_title("Second Half Combined")
            ax6.set_xlabel("Training Round")
            ax6.set_ylabel("Loss")

            # Adjust layout and display
            plt.tight_layout()
            #plt.suptitle(my_title, y=1.02, fontsize=16)  # Optional: Add a super title for both plots
            plt.show()
        else:
            # Create two subplots in a single row
            #fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(12, 3))
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))  # Adjusted height for better spacing
            # Flatten the axes array to make it a 1D array
            ax1, ax2, ax3, ax4 = axes.ravel()

            # First subplot: Velocity Error Half 1
            perf_log_half  = len(perf_log)//2
            ax1.plot(perf_log[:perf_log_half], label="Velocity Error", color='blue')
            ax1.set_title("First Half of Velocity Error")
            ax1.set_xlabel("Training Round")
            ax1.set_ylabel("Loss")
            # Second subplot: Velocity Error Half 2
            ax2.plot(perf_log[perf_log_half:], label="Decoder Penalty", color='blue')
            ax2.set_title("Second Half of Velocity Error")
            ax2.set_xlabel("Training Round")
            ax2.set_ylabel("Loss")

            # Third subplot: Decoder Penalty Half 1
            penalty_log_half  = len(penalty_log)//2
            ax3.plot(penalty_log[:penalty_log_half], label="Decoder Penalty", color='red')
            ax3.set_title("First Half of Decoder Penalty")
            ax3.set_xlabel("Training Round")
            ax3.set_ylabel("Loss")
            # Fourth subplot: Decoder Penalty Half 2
            ax4.plot(penalty_log[penalty_log_half:], label="Decoder Penalty", color='red')
            ax4.set_title("Second Half of Decoder Penalty")
            ax4.set_xlabel("Training Round")
            ax4.set_ylabel("Loss")

            # Adjust layout and display
            plt.tight_layout()
            #plt.suptitle(my_title, y=1.02, fontsize=16)  # Optional: Add a super title for both plots
            plt.show()
    elif plot_separately==True and plot_2nd_half_separately==False:
        # Create two subplots in a single row
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        # First subplot: Velocity Error
        ax1.plot(perf_log, label="Velocity Error", color='blue')
        ax1.set_title("Velocity Error")
        ax1.set_xlabel("Training Round")
        ax1.set_ylabel("Loss")
        # Second subplot: Decoder Penalty
        ax2.plot(penalty_log, label="Decoder Penalty", color='red')
        ax2.set_title("Decoder Penalty")
        ax2.set_xlabel("Training Round")
        ax2.set_ylabel("Loss")
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
    else:
        if plot_total_cost:
            plt.plot(cost_log, label="Total cost")
        if plot_perf:
            plt.plot(perf_log, label="Velocity Error")
        if plot_penalty:
            plt.plot(penalty_log, label="Decoder Penalty")
        plt.legend()
        plt.title(my_title)
        plt.xlabel("Training Round")
        plt.ylabel("Loss")
        plt.show() 


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_time_domain_error(pref_npy, pact_npy, movavg_window_size=1001):
    if type(pref_npy)==type([]):
        pref_npy = np.array(pref_npy)
    if type(pact_npy)==type([]):
        pact_npy = np.array(pact_npy)
    
    # Calculate the average time-domain error across all subjects
    td_error = calc_time_domain_error(pref_npy, pact_npy)

    ma1000 = moving_average(td_error, window_size=movavg_window_size)

    # Plot Time Domain Error
    fig, ax_big = plt.subplots()
    ax_big.plot(range(len(td_error)), td_error, alpha=0.5, label="Raw")
    ax_big.plot(range(len(ma1000)), ma1000, label="MovAvg 1000")
    ax_big.set_title("TD Pos Error: ||p_ref - p_act||")
    ax_big.set_ylabel("Performance Error Norm")
    ax_big.set_xlabel("Iteration")
    ax_big.legend()
    plt.show()
    
    return td_error, ma1000


def load_data(full_file_path, remove_ramp=True, drop_last_update=True, verbose=False, static=False, ramp_time=5):
    # static --> In static trials, there are no updates, so update_ix cannot be calculated!
    times_idx=0
    filt_emg_idx=2
    dec_idx=3
    p_act_idx=8
    v_dec_idx=9
    v_int_idx=10
    p_ref_idx=7
        
    if static==False:
        unique_dec_lst, update_ix, _, _, _ = set_trial_update_ix_times_and_tscale(full_file_path, verbose=verbose)

        print("Loading Data")
        with h5py.File(full_file_path, 'r') as handle:
            weiner_dataset = handle['weiner']
            weiner_df = pd.DataFrame(weiner_dataset)

        num_dropped_points_start = 0
        # Correct the time scale by subtracting the first time value from all entries
        ## Eg we want the initial time to be 0, not XE9
        initial_time = weiner_df.iloc[0, 0][times_idx][0]
        times_lst = [weiner_df.iloc[i, 0][times_idx][0] for i in range(len(weiner_df))]
        adj_times_lst = np.array(times_lst) - times_lst[0]
        times_lst, dec_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst = [], [], [], [], [], [], []
        for i in range(weiner_df.shape[0]):
            if (remove_ramp==True) and (adj_times_lst[i]  < ramp_time):
                num_dropped_points_start += 1
            elif (drop_last_update==True) and (i > update_ix[-2]):
                continue
            else:
                dec_lst.append(weiner_df.iloc[i, 0][dec_idx])  # Decoder W
                times_lst.append(weiner_df.iloc[i, 0][times_idx])  # Times
                emg_lst.append(weiner_df.iloc[i, 0][filt_emg_idx])  # filt_emg
                pref_lst.append(weiner_df.iloc[i, 0][p_ref_idx])  # reference position (p_ref)
                pact_lst.append(weiner_df.iloc[i, 0][p_act_idx])  # decoded/actual position (p_act)
                vint_lst.append(weiner_df.iloc[i, 0][v_int_idx])  # intended velocity
                vdec_lst.append(weiner_df.iloc[i, 0][v_dec_idx])  # decoded velocity
        print("Successfully loaded in all data and set lists!")
        # Adjust update_ix to account for the dropped values, and pop last update val
        adjusted_update_ix = np.array(update_ix[:-1]) - num_dropped_points_start
        adjusted_update_ix[0] += num_dropped_points_start  # Reset the first value back to zero
        return adjusted_update_ix, unique_dec_lst, dec_lst, times_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst
    elif static==True:
        print("Loading Data")
        with h5py.File(full_file_path, 'r') as handle:
            weiner_dataset = handle['weiner']
            weiner_df = pd.DataFrame(weiner_dataset)

        times_lst, dec_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst = [], [], [], [], [], [], []
        num_dropped_points_start = 0
        for i in range(weiner_df.shape[0]):
            if weiner_df.iloc[i, 0][times_idx] < ramp_time:
                num_dropped_points_start += 1
            else:
                dec_lst.append(weiner_df.iloc[i, 0][dec_idx])  # Decoder W
                times_lst.append(weiner_df.iloc[i, 0][times_idx])  # Times
                emg_lst.append(weiner_df.iloc[i, 0][filt_emg_idx])  # filt_emg
                pref_lst.append(weiner_df.iloc[i, 0][p_ref_idx])  # reference position (p_ref)
                pact_lst.append(weiner_df.iloc[i, 0][p_act_idx])  # decoded/actual position (p_act)
                vint_lst.append(weiner_df.iloc[i, 0][v_int_idx])  # intended velocity
                vdec_lst.append(weiner_df.iloc[i, 0][v_dec_idx])  # decoded velocity
        print("Successfully loaded in all data and set lists!")
        return None, [dec_lst[0]], dec_lst, times_lst, emg_lst, pref_lst, pact_lst, vint_lst, vdec_lst      


def set_trial_update_ix_times_and_tscale(full_file_path, timestamp_idx=0, num_dps=2, dec_idx=3, verbose=True):
    if verbose:
        print(full_file_path)
    with h5py.File(full_file_path, 'r') as handle:
        weiner_dataset = handle['weiner']
        weiner_df = pd.DataFrame(weiner_dataset)

    # These should really be denoted as the full datasets, before applying update_ix
    dec_lst = []
    times_lst = []
    for i in range(weiner_df.shape[0]):
        dec_lst.append(weiner_df.iloc[i, 0][dec_idx])
        times_lst.append(weiner_df.iloc[i, 0][0][timestamp_idx])
        
    assert(len(dec_lst)!=0)
    assert(len(times_lst)!=0)

    trial_len = len(dec_lst)
    times_npy = np.array(times_lst)
    times_npy -= times_npy[0]

    # THIS IS WHAT I CARE ABOUT RIGHT NOW!!!
    update_ix = [0]
    for i in range(len(dec_lst)-1):
        if not np.array_equal(dec_lst[i], dec_lst[i+1]):
            update_ix.append(i) # i+1 is the change, but keeping with CPHS's broken update_ix format I will just use i
    unique_dec_lst = [dec_lst[idx+1] for idx in update_ix]
    update_ix.append(trial_len-1) # They had this in CPHS but it's not a unique dec... idk
    update_times = np.round(times_npy[update_ix], decimals=num_dps)
    update_mins = update_times/60
    tscale = update_ix[-1]/update_times[-1]

    if verbose:
        print("update index in time indices")
        print(update_ix)
        print("Update times in seconds:")
        print(update_times)
        print("Update times in minutes:")
        print(update_mins)
        print("Time scale conversion (index --> seconds): ", tscale)
        print()

    return unique_dec_lst, update_ix, update_times, update_mins, tscale


def plot_single_initial_final_boxplot(update_ix, pref, pact, my_title="TD Pos Error Across Trial", ticks_fontsize=16, title_fontsize=16, labels_fontsize=20):
    
    pref_npy_up1 = np.array(pref[:update_ix[1]])
    pref_npy_finalup = np.array(pref[update_ix[-2]:])
    pact_npy_up1 = np.array(pact[:update_ix[1]])
    pact_npy_finalup = np.array(pact[update_ix[-2]:])
    
    td_error_up1 = calc_time_domain_error(pref_npy_up1, pact_npy_up1)
    td_error_finalup = calc_time_domain_error(pref_npy_finalup, pact_npy_finalup)

    # Create a box plot
    plt.figure(figsize=(6, 4))
    plt.boxplot([td_error_up1, td_error_finalup], patch_artist=True)#, vert=False)
    plt.title(my_title, fontsize=title_fontsize+4)
    plt.ylabel('Norm', fontsize=labels_fontsize)
    plt.xlabel(("Experimental Trial"), fontsize=labels_fontsize)
    plt.xticks([1, 2], ["Trial Initial", "Trial Final"], rotation=45, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.show()


def plot_multiple_dumbbell(update_ix_lst, pref_lst, pact_lst, my_title="Change in TD Pos Error Within Given Trial", xticks_labels_lst=None, ticks_fontsize=16, title_fontsize=16, labels_fontsize=20, legend_fontsize=14):
    num_separate_trials = len(update_ix_lst)
        
    # Create the Dumbbell Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_separate_trials):
        update_ix = update_ix_lst[i]
        pref = pref_lst[i]
        pact = pact_lst[i]
        
        pref_npy_up1 = np.array(pref[:update_ix[1]])
        pref_npy_finalup = np.array(pref[update_ix[-2]:])
        pact_npy_up1 = np.array(pact[:update_ix[1]])
        pact_npy_finalup = np.array(pact[update_ix[-2]:])

        td_error_up1 = calc_time_domain_error(pref_npy_up1, pact_npy_up1)
        td_error_finalup = calc_time_domain_error(pref_npy_finalup, pact_npy_finalup)

        avg_td_error_up1 = np.mean(td_error_up1)
        avg_td_error_finalup = np.mean(td_error_finalup)
        
        ax.plot([i, i], [avg_td_error_up1, avg_td_error_finalup], color='gray', linewidth=2, zorder=1)
        ax.scatter(i, avg_td_error_up1, color='blue', s=100, label='Initial', zorder=2)
        ax.scatter(i, avg_td_error_finalup, color='red', s=100, label='Final', zorder=2)
    ax.set_xticks(range(num_separate_trials))
    if xticks_labels_lst is not None:
        ax.set_xticklabels(xticks_labels_lst)
    ax.set_ylabel('Loss', fontsize=labels_fontsize)
    ax.set_xlabel('Trial', fontsize=labels_fontsize)
    ax.set_title(my_title, fontsize=title_fontsize)
    plt.show()

    
def plot_single_initial_final_dumbbell(update_ix, pref, pact, my_title="Change in TD Pos Error Within Given Trial", ticks_fontsize=16, title_fontsize=16, labels_fontsize=20, legend_fontsize=14):
    
    pref_npy_up1 = np.array(pref[:update_ix[1]])
    pref_npy_finalup = np.array(pref[update_ix[-2]:])
    pact_npy_up1 = np.array(pact[:update_ix[1]])
    pact_npy_finalup = np.array(pact[update_ix[-2]:])
    
    td_error_up1 = calc_time_domain_error(pref_npy_up1, pact_npy_up1)
    td_error_finalup = calc_time_domain_error(pref_npy_finalup, pact_npy_finalup)
    
    avg_td_error_up1 = np.mean(td_error_up1)
    avg_td_error_finalup = np.mean(td_error_finalup)

    # Create the Dumbbell Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 0], [avg_td_error_up1, avg_td_error_finalup], color='gray', linewidth=2, zorder=1)
    ax.scatter(0, avg_td_error_up1, color='blue', s=100, label='Initial', zorder=2)
    ax.scatter(0, avg_td_error_finalup, color='red', s=100, label='Final', zorder=2)
    ax.set_ylabel('Loss', fontsize=labels_fontsize)
    ax.set_xlabel('Trial', fontsize=labels_fontsize)
    ax.set_title(my_title, fontsize=title_fontsize)
    plt.show()
    
    
def plot_multiple_final_initial_cost_func_dumbbell_plots(update_ix_lst, F_lst, D_lst, pref_lst, pact_lst, files_to_load_in=None, trial_names_lst=None, my_title="Change in Cost Func Across Trials", ticks_fontsize=16, title_fontsize=22, labels_fontsize=20, legend_fontsize=14):
    # THIS IS ONLY FASTER IF YOU DONT CALL THIS (or its sibling funcs) A TON
    ## Since then it's faster to load the data in at one place and just pass it in
    if files_to_load_in is not None:
        lists = [[] for _ in range(9)]
        
        for file in files_to_load_in:
            # Call the function and append each value to its corresponding list
            results = load_data(file)
            # ^ RETURNS: update_ix1, unique_dec_lst1, dec_lst1, times_lst1, emg_lst1, pref_lst1, pact_lst1, vint_lst1, vdec_lst1
            for i, value in enumerate(results):
                if i in [1, 3, 7, 8]: # We don't need to log unique_dec_lst1, times_lst1, vint_lst1, or vdec_lst1
                    continue
                else:
                    lists[i].append(value)  # Append each value to the appropriate list
        update_ix_lst = lists[0]
        pref_lst = lists[5]
        pact_lst = lists[6]
    
    num_separate_trials = len(update_ix_lst)
        
    # Create the Dumbbell Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_separate_trials):
        update_ix = update_ix_lst[i]
        pref = pref_lst[i]
        pact = pact_lst[i]
        F = F_lst[i]
        D = D_lst[i]
        
        pref_npy_up1 = np.array(pref[:update_ix[1]])
        pref_npy_finalup = np.array(pref[update_ix[-2]:])
        pact_npy_up1 = np.array(pact[:update_ix[1]])
        pact_npy_finalup = np.array(pact[update_ix[-2]:])
        F_npy_up1 = np.array(F[:update_ix[1]])
        F_npy_finalup = np.array(F[update_ix[-2]:])
        D_npy_up1 = np.array(D[:update_ix[1]])
        D_npy_finalup = np.array(D[update_ix[-2]:])

        cost_log_up1, _, _ = calc_cost_function(F_npy_up1, D_npy_up1, pref_npy_up1, pact_npy_up1)
        cost_log_finalup, _, _ = calc_cost_function(F_npy_finalup, D_npy_finalup, pref_npy_finalup, pact_npy_finalup)
        
        avg_cost_log_up1 = np.mean(cost_log_up1)
        avg_cost_log_finalup = np.mean(cost_log_finalup)
        
        ax.plot([i, i], [avg_cost_log_up1, avg_cost_log_finalup], color='gray', linewidth=2, zorder=1)
        if i==0:
            ax.scatter(i, avg_cost_log_up1, color='blue', s=100, label='Initial', zorder=2)
            ax.scatter(i, avg_cost_log_finalup, color='red', s=100, label='Final', zorder=2)
        else:
            ax.scatter(i, avg_cost_log_up1, color='blue', s=100, zorder=2)
            ax.scatter(i, avg_cost_log_finalup, color='red', s=100, zorder=2)
    ax.set_xticks(range(num_separate_trials))
    if trial_names_lst is not None:
        ax.set_xticklabels(trial_names_lst, rotation=90)
    ax.set_ylabel('Loss', fontsize=labels_fontsize)
    ax.set_xlabel('Trial', fontsize=labels_fontsize)
    ax.set_title(my_title, fontsize=title_fontsize)
    ax.legend(loc='best', fontsize=legend_fontsize)
    plt.show()
    


def plot_multiple_final_initial_position_error_dumbbell_plots(update_ix_lst, pref_lst, pact_lst, files_to_load_in=None, trial_names_lst=None, my_title="Change in Cost Func Across Trials", ticks_fontsize=16, title_fontsize=22, labels_fontsize=20, legend_fontsize=14):
    if files_to_load_in is not None:
        lists = [[] for _ in range(9)]
        
        for file in files_to_load_in:
            # Call the function and append each value to its corresponding list
            results = load_data(file)
            # ^ RETURNS: update_ix1, unique_dec_lst1, dec_lst1, times_lst1, emg_lst1, pref_lst1, pact_lst1, vint_lst1, vdec_lst1
            for i, value in enumerate(results):
                if i in [1, 3, 7, 8]: # We don't need to log unique_dec_lst1, times_lst1, vint_lst1, or vdec_lst1
                    continue
                else:
                    lists[i].append(value)  # Append each value to the appropriate list
        update_ix_lst = lists[0]
        pref_lst = lists[5]
        pact_lst = lists[6]
    
    num_separate_trials = len(update_ix_lst)
        
    # Create the Dumbbell Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_separate_trials):
        update_ix = update_ix_lst[i]
        pref = pref_lst[i]
        pact = pact_lst[i]
        
        pref_npy_up1 = np.array(pref[:update_ix[1]])
        pref_npy_finalup = np.array(pref[update_ix[-2]:])
        pact_npy_up1 = np.array(pact[:update_ix[1]])
        pact_npy_finalup = np.array(pact[update_ix[-2]:])

        td_error_up1 = calc_time_domain_error(pref_npy_up1, pact_npy_up1)
        td_error_finalup = calc_time_domain_error(pref_npy_finalup, pact_npy_finalup)
        
        avg_td_error_up1 = np.mean(td_error_up1)
        avg_td_error_finalup = np.mean(td_error_finalup)
        
        ax.plot([i, i], [avg_td_error_up1, avg_td_error_finalup], color='gray', linewidth=2, zorder=1)
        # Add labels only for the first instance to avoid duplicates
        if i == 0:
            ax.scatter(i, avg_td_error_up1, color='blue', s=100, label='Initial', zorder=2)
            ax.scatter(i, avg_td_error_finalup, color='red', s=100, label='Final', zorder=2)
        else:
            ax.scatter(i, avg_td_error_up1, color='blue', s=100, zorder=2)
            ax.scatter(i, avg_td_error_finalup, color='red', s=100, zorder=2)
    ax.set_xticks(range(num_separate_trials))
    if trial_names_lst is not None:
        ax.set_xticklabels(trial_names_lst, rotation=90)
    ax.set_ylabel('Loss', fontsize=labels_fontsize)
    ax.set_xlabel('Trial', fontsize=labels_fontsize)
    ax.set_title(my_title, fontsize=title_fontsize)
    ax.legend(loc='best', fontsize=legend_fontsize)
    plt.show()


def plot_multiple_final_initial_cost_func_boxplots(update_ix_lst, F_lst, D_lst, pref_lst, pact_lst, files_to_load_in=None, trial_names_lst=None, my_title="Change in Cost Function Across Trials", legend_fontsize=17, ticks_fontsize=16, title_fontsize=22, labels_fontsize=20):
    # THIS IS ONLY FASTER IF YOU DONT CALL THIS (or its sibling funcs) A LOT
    ## Since then it's faster to load the data in at one place and just pass it in
    if files_to_load_in is not None:
        lists = [[] for _ in range(9)]
        
        for file in files_to_load_in:
            # Call the function and append each value to its corresponding list
            results = load_data(file)
            # ^ RETURNS: update_ix1, unique_dec_lst1, dec_lst1, times_lst1, emg_lst1, pref_lst1, pact_lst1, vint_lst1, vdec_lst1
            for i, value in enumerate(results):
                if i in [1, 3, 7, 8]: # We don't need to log unique_dec_lst1, times_lst1, vint_lst1, or vdec_lst1
                    continue
                else:
                    lists[i].append(value)  # Append each value to the appropriate list
        update_ix_lst = lists[0]
        pref_lst = lists[5]
        pact_lst = lists[6]
    
    num_separate_trials = len(update_ix_lst)
    x_positions = []
    colors = ['darkblue', 'lightgreen']
    initial_patch = mpatches.Patch(color=colors[0], label='Initial')
    final_patch = mpatches.Patch(color=colors[1], label='Final')
        
    # Create the Box Plot
    plt.figure(figsize=(18, 6))
    for i in range(num_separate_trials):
        update_ix = update_ix_lst[i]
        pref = pref_lst[i]
        pact = pact_lst[i]
        F = F_lst[i]
        D = D_lst[i]
        
        pref_npy_up1 = np.array(pref[:update_ix[1]])
        pref_npy_finalup = np.array(pref[update_ix[-2]:])
        pact_npy_up1 = np.array(pact[:update_ix[1]])
        pact_npy_finalup = np.array(pact[update_ix[-2]:])
        F_npy_up1 = np.array(F[:update_ix[1]])
        F_npy_finalup = np.array(F[update_ix[-2]:])
        D_npy_up1 = np.array(D[:update_ix[1]])
        D_npy_finalup = np.array(D[update_ix[-2]:])

        cost_log_up1, _, _ = calc_cost_function(F_npy_up1, D_npy_up1, pref_npy_up1, pact_npy_up1)
        cost_log_finalup, _, _ = calc_cost_function(F_npy_finalup, D_npy_finalup, pref_npy_finalup, pact_npy_finalup)
        
        # Calculate the x-tick positions for this set of box plots
        x_pos_up1 = 2 * i  # Initial box for this trial
        x_pos_finalup = 2 * i + 1  # Final box for this trial

        # Append the positions to the list
        x_positions.extend([x_pos_up1, x_pos_finalup])

        # Plot the boxes at the calculated positions
        boxplot_elements = plt.boxplot([cost_log_up1, cost_log_finalup], positions=[x_pos_up1, x_pos_finalup], patch_artist=True, showfliers=False, widths=0.6)
        
        # Set the color for the boxes
        boxplot_elements['boxes'][0].set_facecolor(colors[0])
        boxplot_elements['boxes'][1].set_facecolor(colors[1])
        
        # Add a vertical dividing line after each pair of boxes, except after the last pair
        if i < num_separate_trials - 1:
            plt.axvline(x=x_pos_finalup + 0.5, color='gray', linestyle='--', linewidth=2)
        
    # Add the legend with custom handles
    plt.legend(handles=[initial_patch, final_patch], fontsize=legend_fontsize)
            
    plt.title(my_title, fontsize=title_fontsize)
    plt.ylabel('Norm', fontsize=labels_fontsize)
    plt.xlabel(("Experimental Trial"), fontsize=labels_fontsize)
    if trial_names_lst is None:
        plt.xticks(range(len(update_ix_lst)*2), rotation=90, fontsize=ticks_fontsize)
    else:
        plt.xticks(range(len(update_ix_lst)*2), trial_names_lst, rotation=90, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.show()
    
    
def plot_multiple_final_initial_position_error_boxplots(update_ix_lst, pref_lst, pact_lst, files_to_load_in=None, trial_names_lst=None, my_title="Change in TDError Within Given Trial", xticks_labels_lst=None, legend_fontsize=17, ticks_fontsize=16, title_fontsize=22, labels_fontsize=20):
    if files_to_load_in is not None:
        lists = [[] for _ in range(9)]
        
        for file in files_to_load_in:
            # Call the function and append each value to its corresponding list
            results = load_data(file)
            # ^ RETURNS: update_ix1, unique_dec_lst1, dec_lst1, times_lst1, emg_lst1, pref_lst1, pact_lst1, vint_lst1, vdec_lst1
            for i, value in enumerate(results):
                if i in [1, 3, 7, 8]: # We don't need to log unique_dec_lst1, times_lst1, vint_lst1, or vdec_lst1
                    continue
                else:
                    lists[i].append(value)  # Append each value to the appropriate list
        update_ix_lst = lists[0]
        #dec_lst1 = lists[2]
        #emg_lst1 = lists[4]
        pref_lst = lists[5]
        pact_lst = lists[6]
    
    num_separate_trials = len(update_ix_lst)
    x_positions = []
    colors = ['darkblue', 'lightgreen']
    initial_patch = mpatches.Patch(color=colors[0], label='Initial')
    final_patch = mpatches.Patch(color=colors[1], label='Final')

    # Create the Box Plot
    plt.figure(figsize=(18, 6))
    for i in range(num_separate_trials):
        update_ix = update_ix_lst[i]
        pref = pref_lst[i]
        pact = pact_lst[i]
        
        pref_npy_up1 = np.array(pref[:update_ix[1]])
        pref_npy_finalup = np.array(pref[update_ix[-2]:])
        pact_npy_up1 = np.array(pact[:update_ix[1]])
        pact_npy_finalup = np.array(pact[update_ix[-2]:])

        td_error_up1 = calc_time_domain_error(pref_npy_up1, pact_npy_up1)
        td_error_finalup = calc_time_domain_error(pref_npy_finalup, pact_npy_finalup)
        
        # Calculate the x-tick positions for this set of box plots
        x_pos_up1 = 2 * i  # Initial box for this trial
        x_pos_finalup = 2 * i + 1  # Final box for this trial

        # Append the positions to the list
        x_positions.extend([x_pos_up1, x_pos_finalup])

        # Plot the boxes at the calculated positions
        boxplot_elements = plt.boxplot([td_error_up1, td_error_finalup], positions=[x_pos_up1, x_pos_finalup], patch_artist=True, widths=0.4)
        
        # Set the color for the boxes
        boxplot_elements['boxes'][0].set_facecolor(colors[0])
        boxplot_elements['boxes'][1].set_facecolor(colors[1])
        
        # Add a vertical dividing line after each pair of boxes, except after the last pair
        if i < num_separate_trials - 1:
            plt.axvline(x=x_pos_finalup + 0.5, color='gray', linestyle='--', linewidth=2)
        
    # Add the legend with custom handles
    plt.legend(handles=[initial_patch, final_patch], fontsize=legend_fontsize)
    
    plt.title(my_title, fontsize=title_fontsize)
    plt.ylabel('Norm', fontsize=labels_fontsize)
    plt.xlabel(("Experimental Trial"), fontsize=labels_fontsize)
    if trial_names_lst is None:
        plt.xticks(range(len(update_ix_lst)*2), rotation=90, fontsize=ticks_fontsize)
    else:
        plt.xticks(range(len(update_ix_lst)*2), trial_names_lst, rotation=90, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.show()
