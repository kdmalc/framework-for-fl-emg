import os
import numpy as np
import random
import pickle
from sklearn.model_selection import KFold
import copy
from matplotlib import pyplot as plt

from experiment_params import *
from cost_funcs import *
from fl_sim_client import *
from fl_sim_server import *
from shared_globals import *

random.seed(100)
np.random.seed(100)


# GLOBALS
GLOBAL_METHOD = "PFAFO"  #FedAvg #PFAFO #NOFL
OPT_METHOD = 'FULLSCIPYMIN' if GLOBAL_METHOD=="NOFL" else 'GD'
GLOBAL_ROUNDS = NOFL_NUM_ROUNDS if GLOBAL_METHOD=="NOFL" else NUM_KFCV_GLOBAL_ROUNDS
LOCAL_ROUND_THRESHOLD = 1 if GLOBAL_METHOD=="NOFL" else LRT
NUM_STEPS = 1 if GLOBAL_METHOD=="NOFL" else NUM_FL_STEPS 


with open(path + SAVED_DATASET, 'rb') as fp:
    cond0_training_and_labels_lst = pickle.load(fp)

# THIS K FOLD SCHEME IS ONLY FOR CROSS-SUBJECT ANALYSIS!!!
# prepare K-fold on user IDs
kf = KFold(n_splits=NUM_KFOLDS, shuffle=True, random_state=123)
user_ids = list(range(NUM_USERS))
folds    = list(kf.split(user_ids))

# pre-allocate results & success flags
cross_val_res_lst = [None] * NUM_KFOLDS
fold_success      = [False] * NUM_KFOLDS

for fold_idx, (train_ids, test_ids) in enumerate(folds):
    print(f"\n=== Fold {fold_idx+1}/{NUM_KFOLDS} ===")
    print(f"  Train IDs: {train_ids}")
    print(f"   Test IDs: {test_ids}")

    # build train/test clients
    train_clients = [
        Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM, 
               beta=BETA, scenario="CROSS",
               local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, starting_update=STARTING_UPDATE,
               current_fold=fold_idx, num_kfolds=NUM_KFOLDS, smoothbatch_lr=SMOOTHBATCH_LR, 
               global_method=GLOBAL_METHOD, num_steps=NUM_STEPS, test_split_type='KFOLDCV')
        for i in train_ids
    ]
    test_clients = [
        Client(i, copy.deepcopy(D_0), OPT_METHOD, cond0_training_and_labels_lst[i], DATA_STREAM, 
               beta=BETA, scenario="CROSS",
               local_round_threshold=LOCAL_ROUND_THRESHOLD, lr=LR, smoothbatch_lr=SMOOTHBATCH_LR, 
               current_fold=fold_idx, availability=False, val_set=True,
               num_kfolds=NUM_KFOLDS, global_method=GLOBAL_METHOD, starting_update=STARTING_UPDATE,
               num_steps=NUM_STEPS, test_split_type='KFOLDCV')
        for i in test_ids
    ]

    # share the test set from the first test client to all train clients
    test_dataset = test_clients[0].get_testing_dataset()
    for cli in train_clients:
        cli.set_testset(test_dataset)

    full_client_lst = train_clients + test_clients

    # create & configure the server
    server_obj = Server(
        1, copy.deepcopy(D_0), opt_method=OPT_METHOD,
        global_method=GLOBAL_METHOD, 
        all_clients=full_client_lst, # Should this actually be just train_clients? test_clients shouldn't get used at all right?
        C=C, starting_update=STARTING_UPDATE, smoothbatch_lr=SMOOTHBATCH_LR
    )
    server_obj.set_save_filename(CURRENT_DATETIME)
    server_obj.current_fold  = fold_idx
    server_obj.global_rounds = GLOBAL_ROUNDS

    try:
        # run federated‐learning rounds
        for _ in range(GLOBAL_ROUNDS):
            server_obj.execute_FL_loop()

        fold_success[fold_idx] = True

        for cli in server_obj.train_clients:
            print(f"Subject {cli.ID} ran {cli.current_round} rounds and stopped on update {cli.current_update}")
        print()

        # save h5 & optional per‐fold plot
        server_obj.save_results_h5()
        if PLOT_EACH_FOLD:
            plt.figure()
            plt.plot(server_obj.local_train_error_log,
                     '--', color=COLORS_LST[0], alpha=0.5,
                     label=f"F{fold_idx} local train")
            plt.plot(server_obj.local_test_error_log,
                     '-',  color=COLORS_LST[0], alpha=0.5,
                     label=f"F{fold_idx} local test")
            if server_obj.global_method != "NOFL":
                plt.plot(server_obj.global_train_error_log,
                         '--', color=COLORS_LST[1], alpha=0.5,
                         label=f"F{fold_idx} global train")
                plt.plot(server_obj.global_test_error_log,
                         '-',  color=COLORS_LST[1], alpha=0.5,
                         label=f"F{fold_idx} global test")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title(f"Fold {fold_idx} Loss Curves")
            plt.legend()
            plt.savefig(os.path.join(
                server_obj.trial_result_path,
                f'Cross_F{fold_idx}_LossCurves.png'
            ))
            plt.close()

        # stash all per‐fold results
        cross_val_res_lst[fold_idx] = [
            copy.deepcopy(server_obj.local_train_error_log),
            copy.deepcopy(server_obj.local_test_error_log),
            # client-wise test logs
            [copy.deepcopy(cli.local_test_error_log) for cli in server_obj.all_clients],
            # client-wise gradient logs
            [copy.deepcopy(cli.local_gradient_log) for cli in server_obj.all_clients],
        ]

        # optionally save final model
        if GLOBAL_METHOD.upper() != "NOFL":
            outdir = os.path.join(model_saving_dir,
                                  server_obj.str_current_datetime + "_" + GLOBAL_METHOD)
            os.makedirs(outdir, exist_ok=True)
            np.save(
                os.path.join(outdir, f'server_final_model_f{fold_idx}.npy'),
                server_obj.w
            )

    except RuntimeError as e:
        if "Cost blew up" in str(e):
            print(f"→ Fold {fold_idx} aborted: cost explosion.")
        else:
            # rethrow unexpected errors
            raise

# --- final cross‐val average over only the successful folds ---
valid = [i for i, ok in enumerate(fold_success) if ok]
n_valid = len(valid)
if n_valid == 0:
    raise RuntimeError("All folds failed—no valid results to plot.")

plt.figure()
sum_train = np.zeros(GLOBAL_ROUNDS)
sum_test  = np.zeros(GLOBAL_ROUNDS)

for f in valid:
    train_curve = np.array(cross_val_res_lst[f][0])
    test_curve  = np.array(cross_val_res_lst[f][1])
    sum_train  += train_curve
    sum_test   += test_curve

avg_train = sum_train / n_valid
avg_test  = sum_test  / n_valid

for f in valid:
    plt.plot(cross_val_res_lst[f][0],
             color=COLORS_LST[f], alpha=0.3, linestyle='--')
    plt.plot(cross_val_res_lst[f][1],
             color=COLORS_LST[f], alpha=0.3, linestyle='-')

plt.plot(avg_train, '--', color='k', linewidth=2, label="Avg Train")
plt.plot(avg_test,  '-', color='k', linewidth=2, label="Avg Test")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Cross-Subject Avg Train/Test Curves")
plt.legend()
plt.savefig(os.path.join(server_obj.trial_result_path,
                         'CrossAvg_LossCurves.png'))
plt.show()

server_obj.save_header()