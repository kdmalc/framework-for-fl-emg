import numpy as np
import random
import copy
import os
from datetime import datetime
import h5py

from experiment_params import *
from cost_funcs import *
from fl_sim_base import *
        
        
class Server(ModelBase):
    def __init__(self, ID, D0, opt_method, global_method, all_clients, 
                 apply_zero_thresholding=False,
                 smoothbatch_lr=0.75, C=0.35, test_split_type='kfoldcv', starting_update=9, 
                 num_kfolds=7, test_split_frac=0.3, current_round=0, PCA_comps=64, verbose=False, 
                 save_client_loss_logs=True, save_client_trajectory_logs=True, 
                 sequential=False, current_datatime="Overwritten", privacy_attack=False):
        super().__init__(ID, D0, opt_method, smoothbatch_lr=smoothbatch_lr, current_round=current_round, PCA_comps=PCA_comps, 
                         verbose=verbose, num_clients=14, log_init=0, apply_zero_thresholding=apply_zero_thresholding, starting_update=starting_update)
        
        self.type = 'Server'
        self.save_client_loss_logs = save_client_loss_logs
        self.save_client_trajectory_logs = save_client_trajectory_logs
        self.sequential = sequential
        self.privacy_attack = privacy_attack

        # CLIENT LISTS!
        self.num_avail_clients = 0
        self.available_clients_lst = [0]*len(all_clients)
        self.num_chosen_clients = 0
        self.chosen_clients_lst = [0]*len(all_clients)
        self.all_clients = all_clients  # All train/val/static clients
        self.num_total_clients = len(self.all_clients)
        self.train_clients = [cli for cli in self.all_clients if cli.val_set is False]
        self.val_clients = [cli for cli in self.all_clients if cli.val_set is True]
        self.num_train_clients = len(self.train_clients)
        self.num_test_clients = len(self.val_clients)
        self.set_available_clients_list()
        # Init all train clients with simulate_streaming so they have self.s, self.F, and self.p_reference
        for client in self.train_clients:
            client.simulate_data_stream()

        self.global_method = global_method.upper()
        print(f"Running the {self.global_method} algorithm as the global method!")
        self.C = C  # Fraction of clients to use each round

        # TESTING
        self.test_split_type = test_split_type.upper()
        self.num_kfolds = num_kfolds
        self.test_split_frac = test_split_frac
        self.current_fold = 0

        # SAVE FILE RELATED
        # Get the directory of the current script
        self.script_directory = os.path.dirname(os.path.abspath(__file__))  # This returns the path to serverbase... so don't index the end of the path
        # Relative path to results dir
        self.result_path = "\\results\\"
        if current_datatime is None:
            self.set_save_filename()

    
    def report_mean_client_tdpe(self):
        tdpe_lst = []
        for cli in self.train_clients:  # train_clients is all clients who are not in the val_set
            tdpe_lst.append(cli.calc_test_tdpe())
        mean_tdpe = np.mean(np.array(tdpe_lst))
        return mean_tdpe
    

    def report_mean_client_cost(self):
        global_cost_lst = [] 
        local_cost_lst = []
        for cli in self.train_clients:  # train_clients is all clients who are not in the val_set
            global_cost_lst.append(np.mean(cli.global_test_error_log[-8:]))  # Average over the last 8 rounds
            local_cost_lst.append(np.mean(cli.local_test_error_log[-8:]))  # Average over the last 8 rounds
        global_mean_cost = np.mean(np.array(global_cost_lst))
        local_mean_cost = np.mean(np.array(local_cost_lst))
        return global_mean_cost, local_mean_cost


    def set_save_filename(self, current_datetime=None):
        if current_datetime is None:
            current_datetime = datetime.now().strftime("%m-%d_%H-%M")

        # convert datetime obj to string
        self.str_current_datetime = str(current_datetime)
        # Specify the relative path from the script's directory
        self.relative_path = self.result_path + self.str_current_datetime + "_" + self.global_method
        # Combine the script's directory and the relative path to get the full path
        self.trial_result_path = self.script_directory + self.relative_path
        self.h5_file_path = os.path.join(self.trial_result_path, f"{self.opt_method}_{self.global_method}")
        self.paramtxt_file_path = os.path.join(self.trial_result_path, "param_log.txt")
        if not os.path.exists(self.trial_result_path):
            os.makedirs(self.trial_result_path)

                
    # Main Loop
    def execute_FL_loop(self):
        # Update server's global round number
        self.current_round += 1
        #print(f"Server execute_FL_loop! New round: {self.current_round}")
        self.w_prev = copy.deepcopy(self.w)
        
        if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
            # Choose fraction C of available clients
            self.set_available_clients_list()
            self.choose_clients()
            for my_client in self.all_clients:  
                if my_client.val_set==True:
                    # This is a val client, so don't log anything
                    # Could be included in train_metrics but doesnt need to be
                    continue
                elif my_client.availability==False:
                    raise ValueError("Sequential case not implemented yet")
                elif my_client not in self.chosen_clients_lst:
                    # Just not getting trained this round
                    continue
                else:
                    # THESE ARE THE CLIENTS WHICH ACTUALLY GET TRAINED! 
                    # This is latest_global_round and is just used for logging stuff basically
                    my_client.latest_global_round = self.current_round          
                    # Send those clients the current global model
                    my_client.global_w = copy.deepcopy(self.w)
                    my_client.execute_training_loop()
            # AGGREGATION
            self.agg_local_weights()  # This func sets self.w, eg the new decoder
        elif self.global_method=='NOFL':
            for my_client in self.all_clients:  
                if my_client.val_set==True:
                    # This is a val client, so don't log anything
                    continue
                elif my_client.availability==False:
                    raise ValueError("Sequential case not implemented yet")
                else:
                    # We train all (available) clients. Should be all clients every round
                    my_client.latest_global_round = self.current_round          
                    my_client.execute_training_loop()
        else:
            raise('Method not currently supported, please reset method to FedAvg')
        
        # Save the new decoder to the log
        self.global_dec_log.append(copy.deepcopy(self.w))
        # Run train_metrics and test_metrics to log performance on training/testing data
        for client_idx, my_client in enumerate(self.train_clients): # Eg all train-able clients (no witheld val clients from kfoldcv)
            # Reset all clients so no one is chosen for the next round
            my_client.chosen_status = 0
            
            # test_metrics for all TRAIN clients
            local_test_loss = my_client.test_metrics(my_client.w, 'local')
            local_train_loss = my_client.train_metrics(my_client.w, 'local')
            if self.global_method=='FEDAVG' or 'PFA' in self.global_method:
                # Note: this evaluates the server's current global model on ALL trainset clients
                global_test_loss = my_client.test_metrics(self.w, 'global')
                global_train_loss = my_client.train_metrics(self.w, 'global')
            elif self.global_method=='NOFL':
                global_test_loss = 0
                global_train_loss = 0
            
            if client_idx!=0:
                running_global_test_loss += global_test_loss
                running_local_test_loss += local_test_loss

                running_global_train_loss += global_train_loss
                running_local_train_loss += local_train_loss
            else:
                running_global_test_loss = global_test_loss
                running_local_test_loss = local_test_loss

                running_global_train_loss = global_train_loss
                running_local_train_loss = local_train_loss

        # SERVER AVERAGE PERFORMANCE
        self.local_test_error_log.append(running_local_test_loss / len(self.train_clients))
        self.local_train_error_log.append(running_local_train_loss / len(self.train_clients))
        if self.global_method!='NOFL':
            self.global_test_error_log.append(running_global_test_loss / len(self.train_clients))
            self.global_train_error_log.append(running_global_train_loss / len(self.train_clients))
            
            
    def set_available_clients_list(self):
        self.available_clients_full_idx_lst = [0]*len(self.train_clients)
        for idx, my_client in enumerate(self.train_clients):
            if my_client.availability:
                self.available_clients_full_idx_lst[idx] = my_client
        self.available_clients_lst = [cli for cli in self.available_clients_full_idx_lst if cli != 0]
        self.num_avail_clients = len(self.available_clients_lst)
    

    def choose_clients(self):
        # Check what client are available this round
        self.set_available_clients_list()
        # Now choose frac C clients from the resulting available clients
        if self.num_avail_clients > 0:
            self.num_chosen_clients = int(np.ceil(self.num_avail_clients*self.C))
            if self.num_chosen_clients<1:
                raise ValueError(f"ERROR: Chose {self.num_chosen_clients} clients for some reason, must choose more than 1")
            # Right now it chooses 2 at random: 14*.1=1.4 --> 2
            self.chosen_clients_lst = random.sample(self.available_clients_lst, len(self.available_clients_lst))[:self.num_chosen_clients]
            for my_client in self.chosen_clients_lst:
                my_client.chosen_status = 1
        else:
            raise(f"ERROR: Number of available clients must be greater than 0: {self.num_avail_clients}")

    
    def agg_local_weights(self):
        # From McMahan 2017 (vanilla FL)
        summed_num_datapoints = 0
        for my_client in self.chosen_clients_lst:
            summed_num_datapoints += my_client.learning_batch
        # Aggregate local model weights, weighted by normalized local number of datapoint (here, all clients have the same)
        aggr_w = np.zeros((2, self.PCA_comps))
        for my_client in self.chosen_clients_lst:
            aggr_w += (my_client.learning_batch/summed_num_datapoints) * my_client.w
        self.w = aggr_w


    def save_header(self):
        federated_base_str = ("\n\nFEDERATED LEARNING PARAMS\n"
            f"local_round_threshold = {self.train_clients[0].local_round_threshold}\n")

        cphs_base_str = ("\n\nSIMULATION PARAMS\n"
            f"starting_update = {self.starting_update}\n"
            f"total effective clients = {self.num_train_clients}\n"
            f"smoothbatch_lr = {self.smoothbatch_lr}\n")

        param_log_str = (
            "BASE\n"
            f"algorithm = {self.global_method}\n"
            f"scenario = {self.all_clients[0].scenario}\n"
            "\n\nMODEL HYPERPARAMETERS\n"
            f"lambdaF = {self.alphaF}\n"
            f"lambdaD = {self.alphaD}\n"
            f"lambdaE = {self.alphaE}\n"
            f"global_rounds = {self.global_rounds}\n"
            f"optimizer = {self.opt_method}\n"
            f"pca_channels = {self.PCA_comps}\n"
            f"learning rate (only used with GD) = {self.all_clients[0].lr}\n"
            f"max iter (only used with scipy) = {self.all_clients[0].max_iter}\n"
            f"num gradient steps / epochs = {self.all_clients[0].num_steps}\n"
            "\n\nTESTING\n"
            f"test_split_fraction = {self.test_split_frac}\n"
            f"num_kfolds = {self.num_kfolds}\n")

        with open(self.paramtxt_file_path, 'w') as file:
            file.write(param_log_str)
            file.write(federated_base_str)
            file.write(cphs_base_str)
            if "PFA" in self.global_method:
                perfedavg_param_str = ("\n\nPERFEDAVG PARAMS\n"
                    f"beta (PFA outer lr) = {self.all_clients[0].beta}\n")
                file.write(perfedavg_param_str)


    def save_results_h5(self, save_cost_func_comps=True, save_gradient=True, dir_str=None):
        if len(self.local_test_error_log)!=0:
            if dir_str is None:
                extra_dir_str = ""
            else:
                extra_dir_str = "_" + dir_str

            print("File path: " + self.h5_file_path + f"{extra_dir_str}.h5")

            if self.test_split_type=="KFOLDCV":
                if self.privacy_attack:
                    self.save_filename = self.h5_file_path + f"{extra_dir_str}.h5"
                else:
                    self.save_filename = self.h5_file_path + f"_KFold{self.current_fold}.h5"
            else:
                self.save_filename = self.h5_file_path + f"{extra_dir_str}.h5"

            with h5py.File(self.save_filename, 'w') as hf:
                # THESE ARE THE SERVER'S LOGS! Averaged over clients
                if self.global_method!="NOFL":
                    hf.create_dataset('global_test_error_log', data=self.global_test_error_log)
                    hf.create_dataset('global_train_error_log', data=self.global_train_error_log)
                    hf.create_dataset('global_dec_log', data=self.global_dec_log)
                hf.create_dataset('local_test_error_log', data=self.local_test_error_log)
                hf.create_dataset('local_train_error_log', data=self.local_train_error_log)

                group = hf.create_group('client_local_model_log')
                for cli in self.all_clients:
                    group.create_dataset(f"S{cli.ID}_client_local_model_log", data=cli.local_dec_log)

                if save_cost_func_comps:
                    group = hf.create_group('client_local_train_cfc_logs')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_local_train_cfc_logs", data=cli.local_train_cfc_log)
                    group = hf.create_group('client_local_test_cfc_logs')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_local_teset_cfc_logs", data=cli.local_test_cfc_log)

                if self.save_client_loss_logs:
                    group = hf.create_group('client_local_test_log')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_client_local_test_log", data=cli.local_test_error_log)
                    group = hf.create_group('client_local_train_log')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_client_local_train_log", data=cli.local_train_error_log)

                if self.save_client_trajectory_logs:
                    group = hf.create_group('client_pact_test_log')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_client_pact_test_log", data=cli.pact_test_log)
                    group = hf.create_group('client_pref_test_log')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_client_pref_test_log", data=cli.pref_test_log)
                    
                    group = hf.create_group('client_pact_train_log')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_client_pact_train_log", data=cli.pact_train_log)
                    group = hf.create_group('client_pref_train_log')
                    for cli in self.all_clients:
                        group.create_dataset(f"S{cli.ID}_client_pref_train_log", data=cli.pref_train_log)

                if save_gradient:
                    G2 = hf.create_group('gradient_norm_lists_by_client')
                    for idx, cli in enumerate(self.all_clients):
                        G2.create_dataset(f"S{cli.ID}_grad_norm_lst", data=cli.local_gradient_log)
        else:
            print("Saving failed.")

