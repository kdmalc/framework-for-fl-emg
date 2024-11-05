import os
import numpy as np
import random
import pickle
import copy
from matplotlib import pyplot as plt

from experiment_params import *
from cost_funcs import *
from fl_sim_client import *
from fl_sim_server import *
from shared_globals import *

np.random.seed(0)
random.seed(0)


# GLOBALS
GLOBAL_METHOD = "PFAFO"  #FedAvg #PFAFO #NOFL
OPT_METHOD = 'FULLSCIPYMIN' if GLOBAL_METHOD=="NOFL" else 'GDLS'
GLOBAL_ROUNDS = 12 if GLOBAL_METHOD=="NOFL" else 250
LOCAL_ROUND_THRESHOLD = 1 if GLOBAL_METHOD=="NOFL" else 20
NUM_STEPS = 1 if GLOBAL_METHOD=="NOFL" else NUM_FL_STEPS  # This is basically just local_epochs, since I don't batch. THIS WRAPS BOTH NOFL AND FL ALGOS!
SCENARIO = "INTRA"  # "CROSS" cant be used in this file!

with open(path+cond0_filename, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)
#with open(path+all_decs_init_filename, 'rb') as fp:
#    init_decoders = pickle.load(fp)    
#cond0_init_decs = [dec[0, :, :] for dec in init_decoders]

cross_val_res_lst = [[0, 0, 0, 0] for _ in range(NUM_KFOLDS)]
for fold_idx in range(NUM_KFOLDS):
    print(f"Fold {fold_idx+1}/{NUM_KFOLDS}")
    
    # Initialize clients for training
    full_client_lst = [Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM, 
                            scenario=SCENARIO, local_round_threshold=LOCAL_ROUND_THRESHOLD, current_fold=fold_idx, global_method=GLOBAL_METHOD, max_iter=MAX_ITER, 
                            num_steps=NUM_STEPS, use_zvel=USE_HITBOUNDS, test_split_type=TEST_SPLIT_TYPE) for i in range(NUM_USERS)]

    server_obj = Server(1, copy.deepcopy(D_0), opt_method=OPT_METHOD, global_method=GLOBAL_METHOD, all_clients=full_client_lst)
    server_obj.set_save_filename(CURRENT_DATETIME)
    # Add these to the init func params...
    server_obj.current_fold = fold_idx
    server_obj.global_rounds = GLOBAL_ROUNDS
    for i in range(GLOBAL_ROUNDS):
        server_obj.execute_FL_loop()

    server_obj.save_results_h5()

    if PLOT_EACH_FOLD:
        plt.figure()  # Create a new figure
        plt.plot(server_obj.local_train_error_log, linestyle='--', color=COLORS_LST[0], alpha=ALPHA, label=f"f{fold_idx} local train")
        plt.plot(server_obj.local_test_error_log, alpha=ALPHA, color=COLORS_LST[0], label=f"f{fold_idx} local test")
        if server_obj.global_method!="NOFL":
            plt.plot(server_obj.global_train_error_log, linestyle='--', color=COLORS_LST[1], alpha=ALPHA, label=f"f{fold_idx} global train")
            plt.plot(server_obj.global_test_error_log, alpha=ALPHA, color=COLORS_LST[1], label=f"f{fold_idx} global test")
        plt.xlabel("Training Round")
        plt.ylabel("Loss")
        plt.title(f"Train/Test Local/Global Error Curves For Fold {fold_idx}")
        plt.legend()
        plt.savefig(server_obj.trial_result_path + f'\\Intra_PLOTEACHFOLD_LossCurvesf{fold_idx}.png', format='png')
        plt.show()

    # Record train and test error logs
    ## Idk if I really need to be using deepcopy here...
    cross_val_res_lst[fold_idx][0] = copy.deepcopy(server_obj.local_train_error_log)
    cross_val_res_lst[fold_idx][1] = copy.deepcopy(server_obj.local_test_error_log)
    cross_val_res_lst[fold_idx][2] = [[0] for _ in range(len(full_client_lst))]
    cross_val_res_lst[fold_idx][3] = [[0] for _ in range(len(full_client_lst))]
    # Also record the client's loss log
    ## Init an empty lst, with a spot for each client
    ## We do all clients instead of just trained clients since otherwise I would also have to save the index...
    cross_val_res_lst[fold_idx][2] = [0 for _ in range(len(server_obj.all_clients))]  
    for idx, cli in enumerate(server_obj.all_clients):
        assert(len(cli.local_test_error_log)>1)
        #print(f"CLI{cli.ID} SUCCESS: LEN = {len(cli.local_test_error_log)}")
        cross_val_res_lst[fold_idx][2][idx] = copy.deepcopy(cli.local_test_error_log)
        cross_val_res_lst[fold_idx][3][cli.ID] = copy.deepcopy(cli.local_gradient_log)

    #server_obj.save_results_h5(save_cost_func_comps=False, save_gradient=False)
    # Save the model for the current fold
    if GLOBAL_METHOD.upper()!="NOFL":
        print("Saving server's final (global) model")
        dir_path = os.path.join(model_saving_dir, server_obj.str_current_datetime+"_"+GLOBAL_METHOD)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, f'servers_final_model_fold{fold_idx}.npy'), server_obj.w)

# Plot all results:
plt.figure()  # Create a new figure
running_train_loss = np.zeros(GLOBAL_ROUNDS+1)  # +1 because we record the initial model loss now
running_test_loss = np.zeros(GLOBAL_ROUNDS+1)
for fold_idx in range(NUM_KFOLDS):
    train_loss = cross_val_res_lst[fold_idx][0]
    test_loss = cross_val_res_lst[fold_idx][1]

    plt.plot(train_loss, alpha=ALPHA, linestyle='--', color=COLORS_LST[fold_idx], label=f"Fold{fold_idx} Train")
    plt.plot(test_loss, alpha=ALPHA, color=COLORS_LST[fold_idx], label=f"Fold{fold_idx} Test")

    running_train_loss += np.array(train_loss)
    running_test_loss += np.array(test_loss)
# Average to get cross val curve:
avg_cv_train_loss = running_train_loss / NUM_KFOLDS
avg_cv_test_loss = running_test_loss / NUM_KFOLDS
plt.plot(avg_cv_train_loss, linewidth=2, color=COLORS_LST[NUM_KFOLDS], linestyle='--', label="Avg CrossVal Train")
plt.plot(avg_cv_test_loss, linewidth=2, color=COLORS_LST[NUM_KFOLDS], label="Avg CrossVal Test")
plt.xlabel("Training Round")
plt.ylabel("Loss")
plt.title("Train/Test Local Error Curves")
plt.legend()
plt.savefig(server_obj.trial_result_path + '\\TrainTestLossCurves.png', format='png')
plt.show()
    
server_obj.save_header()

'''
with h5py.File(server_obj.h5_file_path + "_CrossValResults.h5", 'w') as hf:
    for fold_idx in range(NUM_KFOLDS):
        #if server_obj.global_method!="NOFL":
        #    hf.create_dataset('global_test_error_log', data=self.global_test_error_log)
        #    hf.create_dataset('global_train_error_log', data=self.global_train_error_log)
        hf.create_dataset(f'Fold{fold_idx}_local_test_error_log', data=cross_val_res_lst[fold_idx][1])
        hf.create_dataset(f'Fold{fold_idx}_local_train_error_log', data=cross_val_res_lst[fold_idx][0])
        for idx, cli in enumerate(server_obj.all_clients):
            hf.create_dataset(f'Fold{fold_idx}_client{cli.ID}_local_test_error_log', data=cross_val_res_lst[fold_idx][2][idx])
            hf.create_dataset(f'Fold{fold_idx}_client{cli.ID}_gradient_norm_log', data=cross_val_res_lst[fold_idx][3][cli.ID])
    # Save the averaged cv results
    hf.create_dataset(f'AveragedCV_local_test_error_log', data=avg_cv_test_loss)
    hf.create_dataset(f'AveragedCV_local_train_error_log', data=avg_cv_train_loss)
'''
