import numpy as np
import random
import copy
import time
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.optimize import line_search

from fl_sim_base import *
from experiment_params import *
from cost_funcs import *


class Client(ModelBase):
    def __init__(self, ID, w, opt_method, full_client_dataset, data_stream, smoothbatch_lr=0.75, current_round=0, PCA_comps=64, 
                availability=1, final_usable_update_ix=17, global_method='FedAvg', max_iter=1, normalize_EMG=True, starting_update=10, 
                track_cost_components=True, log_decs=True, val_set=False,
                tol=1e-10, lr=1, beta=0.01, track_gradient=True, num_steps=1, use_zvel=False, current_fold=0, scenario="", 
                mix_in_each_steps=False, mix_mixed_SB=False, delay_scaling=0, random_delays=False, download_delay=1, 
                upload_delay=1, validate_memory_IDs=True, local_round_threshold=25, condition_number=3, 
                verbose=False, test_split_type='kfoldcv', num_kfolds=5, test_split_frac=0.3):
        super().__init__(ID, w, opt_method, smoothbatch_lr=smoothbatch_lr, current_round=current_round, PCA_comps=PCA_comps, 
                         verbose=verbose, num_clients=14, log_init=0)
        '''
        Note self.smoothbatch gets overwritten according to the condition number!  
        If you want NO smoothbatch then set it to 0
        '''

        assert(full_client_dataset['training'].shape[1]==64) # --> Shape is (20770, 64)
        # Don't use anything past update 17 since they are different (update 17 is the short one, only like 300 datapoints)
        self.current_update = starting_update
        self.current_train_update = 0
        self.starting_update = starting_update
        self.final_usable_update_ix = final_usable_update_ix
        self.local_dataset = copy.deepcopy(full_client_dataset['training'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])
        self.local_labelset = copy.deepcopy(full_client_dataset['labels'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])

        self.global_method = global_method.upper()
        self.validate_memory_IDs = validate_memory_IDs
        # NOT INPUT
        self.type = 'Client'
        self.chosen_status = 0
        self.latest_global_round = 0
        self.normalize_EMG = normalize_EMG

        # Sentinel Values
        self.F = None
        self.V = None
        self.F2 = None
        self.V2 = None
        self.learning_batch = None

        self.dt = 1.0/60.0
        self.lr = lr  # Learning rate
        self.beta = beta # PFA 2nd step learning rate (not used as of now...)
        # Round minimization output to the nearest int or keep as a float?  Don't need arbitrary precision
        self.round2int = False
        self.max_iter = max_iter
        
        # Maneeshika Code:
        self.use_zvel = use_zvel
        self.hit_bound = 0  # This is just initializing it to 0 (eg no hits yet)
        
        # FL CLASS STUFF
        # Availability for training
        self.availability = availability
        # Toggle streaming aspect of data collection: {Ignore updates and use all the data; 
        #  Stream each update, moving to the next update after local_round_threshold iters have been run; 
        #  After 1 iteration, move to the next update}
        self.data_stream = data_stream  # {'full_data', 'streaming', 'advance_each_iter'} 
        # Number of gradient steps to take when training (eg amount of local computation):
        self.num_steps = num_steps  # This is Tau in PFA!
        # UPDATE STUFF
        self.local_round_threshold = local_round_threshold

        # PRACTICAL / REAL WORLD
        # Not using the delay stuff right now
        # Boolean setting whether or not up/download delays should be random or predefined
        self.random_delays = random_delays
        # Scaling from random [0,1] to number of seconds
        self.delay_scaling = delay_scaling
        # Set the delay times
        if self.random_delays: 
            self.download_delay = random.random()*self.delay_scaling
            self.upload_delay = random.random()*self.delay_scaling
        else:
            self.download_delay = download_delay
            self.upload_delay = upload_delay
        
        # ML Parameters / Conditions
        #cond_dict = {1:(0.25, 1e-3, 1), 2:(0.25, 1e-4, 1), 3:(0.75, 1e-3, 1), 4:(0.75, 1e-4, 1), 5:(0.25, 1e-4, -1), 6:(0.25, 1e-4, -1), 7:(0.75, 1e-3, -1), 8:(0.75, 1e-4, -1)}
        #cond_smoothbatch, self.alphaD, self.init_dec_sign = cond_dict[condition_number]
        if type(smoothbatch_lr)==str and smoothbatch_lr.upper()=='OFF':
            self.smoothbatch_lr = 0  # AKA Use only the new dec, no mixing
        #elif smoothbatch_lr==-1:  
        #    # Let the condition number set smoothbatch
        #    self.smoothbatch_lr = cond_smoothbatch
        else:
            # Set smoothbatch to whatever you manually entered
            self.smoothbatch_lr=smoothbatch_lr
        # LOGS
        self.log_decs = log_decs
        # Overwrite the logs since global and local track in slightly different ways
        self.track_cost_components = track_cost_components
        #self.performance_log = []
        #self.Dnorm_log = []
        #self.Fnorm_log = []
        self.track_gradient = track_gradient
        self.local_gradient_log = []
        self.update_transition_log = []
        self.local_cost_func_comps_log = []
        self.global_cost_func_comps_log = []


        ## Not used AFAIK... --> control where SmoothBatch is used (verify this code still exists if you plan on using)
        self.mix_in_each_steps = mix_in_each_steps
        self.mix_mixed_SB = mix_mixed_SB

        self.global_w = copy.deepcopy(self.w)

        # TRAIN TEST DATA SPLIT
        self.test_split_type = test_split_type.upper() # "KFOLDCV" "UPDATE16" "ENDFRACTION"
        self.test_split_frac = test_split_frac
        self.num_kfolds = num_kfolds
        self.val_set = val_set
        self.scenario = scenario.upper()
        self.current_fold = current_fold
        self.test_update = None
        self.train_update_ix = None

        if self.test_split_type=="KFOLDCV":
            if self.scenario=="INTRA":
                # Include these in the init? ...
                append_leftovers_to_last_fold = False
                shift_leftovers_across_folds = False
                match_num_kfolds_to_num_updates = True

                # Filter update_ix to only include updates 9-16
                valid_updates = [ix for ix in self.update_ix if starting_update <= self.update_ix.index(ix) < final_usable_update_ix]
                # THIS SETS THE TESTING FOLD
                if match_num_kfolds_to_num_updates:
                    # Set number of kfolds to match number of updates in valid_updates
                    self.num_kfolds = len(valid_updates)  # OVERWRITES! Set kfolds to number of updates
                    updates_per_fold = 1  # 1 update per fold since we have 8 updates and 8 folds
                    self.folds = [[update] for update in valid_updates]  # Each fold has exactly one update
                    #if hasattr(self, 'ID') and self.ID == 0:
                    #    print(f"Created {len(self.folds)} folds, each with 1 update:")
                    #    for i, fold in enumerate(self.folds):
                    #        print(f"Fold {i}: {len(fold)} updates: {[self.update_ix.index(i) for i in fold]}")
                    assert len(self.folds) == self.num_kfolds, f"Expected {self.num_kfolds} folds, got {len(self.folds)}"
                elif append_leftovers_to_last_fold:
                    '''Leftover updates are appended to the last fold. Updates are continuous but the last fold will be much bigger...'''
                    # Calculate the number of updates per fold
                    updates_per_fold = len(valid_updates) // self.num_kfolds
                    #print(f"updates_per_fold: {updates_per_fold}")
                    # Ensure we have enough updates for the specified number of folds
                    assert updates_per_fold > 0, "Not enough valid updates for the specified number of folds"
                    # Create folds --> THESE ARE THE TEST UPDATES!
                    self.folds = [valid_updates[i:i+updates_per_fold] for i in range(0, len(valid_updates), updates_per_fold)]
                    # If there are leftover updates, add them to the last fold
                    while len(self.folds) > self.num_kfolds:
                        if self.ID==0:
                            print(f"len(self.folds) {len(self.folds)} > self.num_kfolds {self.num_kfolds} --> Thus, popping and adding one to the last fold!")
                        self.folds[-2].extend(self.folds[-1])
                        self.folds.pop()
                    assert len(self.folds) == self.num_kfolds, f"Expected {self.num_kfolds} folds, got {len(self.folds)}"
                elif shift_leftovers_across_folds:
                    '''Purpose of this code is to shift/push updates to earlier folds, to maintain contintuity, while making test folds more even in size'''
                    # TODO: This has not been validated at all...
                    # Calculate the total number of updates and the ideal number of updates per fold
                    total_updates = len(valid_updates)
                    ideal_updates_per_fold = total_updates / self.num_kfolds
                    self.folds = []
                    start_idx = 0
                    for fold in range(self.num_kfolds):
                        end_idx = int(round((fold + 1) * ideal_updates_per_fold))
                        self.folds.append(valid_updates[start_idx:end_idx])
                        start_idx = end_idx
                    # Ensure all updates are included
                    if start_idx < total_updates:
                        self.folds[-1].extend(valid_updates[start_idx:])
                    if hasattr(self, 'ID') and self.ID == 0:
                        print(f"Created {len(self.folds)} folds:")
                        for i, fold in enumerate(self.folds):
                            print(f"Fold {i}: {len(fold)} updates: {[self.update_ix.index(i) for i in fold]}")
                    assert len(self.folds) == self.num_kfolds, f"Expected {self.num_kfolds} folds, got {len(self.folds)}"
                    assert sum(len(fold) for fold in self.folds) == total_updates, "Not all updates were included in the folds"
                
                # SET THE TEST DATA
                # Get the current fold's update indices
                fold_updates = self.folds[self.current_fold]
                self.test_update = [self.update_ix.index(ele) for ele in fold_updates]
                self.train_update_ix = [ele for ele in self.update_ix if ((self.update_ix.index(ele)>=self.starting_update) and (self.update_ix.index(ele) not in self.test_update) and (self.update_ix.index(ele)<self.final_usable_update_ix))]

                # Find the start and end indices for the TEST dataset
                ## THE BELOW CODE ASSUMES TEST UPDATES IN FOLD ARE CONTIGUOUS
                ## The values in fold_updates represent the START of each update, NOT the BOUNDS!!  Thus the adjustment
                lower_bound_pre_idx = self.update_ix.index(fold_updates[0])
                upper_bound_pre_idx = self.update_ix.index(fold_updates[-1]) + 1  # +1 to include the last update (eg otherwise this update num would be the upperbound, as opposed to included)
                # Upper and lower bounds mapped to the new size (since we do not use anything before starting_update...)
                lower_bound = self.update_ix[lower_bound_pre_idx] - self.update_ix[starting_update]
                upper_bound = self.update_ix[upper_bound_pre_idx] - self.update_ix[starting_update]
                # Set testing data
                self.testing_data = self.local_dataset[lower_bound:upper_bound, :]
                self.testing_labels = self.local_labelset[lower_bound:upper_bound, :]

                # SIMULATE THE FIRST DATA STREAM FOR THE STARTING UPDATE
                ## Eg, set self.V, self.s, etc etc
                # self.train_update_ix starts at starting_update already!
                lower_bound = self.train_update_ix[0] - self.update_ix[self.starting_update]
                upper_bound = lower_bound + 1200
            elif self.scenario=="CROSS":
                self.test_split_idx = -1
                # Setting the testing set to the whole dataset so that it can be extracted
                ## If this is a training cliet this will be overwritten by other client's testing dataset
                ## I guess I could set this to 
                upper_bound = self.local_dataset.shape[0]
                lower_bound = 0
                self.testing_data = self.local_dataset
                self.testing_labels = self.local_labelset

            if "PFA" in self.global_method:
                mid_point = (lower_bound+upper_bound)//2
                self.learning_batch = mid_point - lower_bound

                s_temp = self.local_dataset[lower_bound:mid_point, :]
                s_temp2 = self.local_dataset[mid_point:upper_bound, :]
                p_ref_lim = self.local_labelset[lower_bound:mid_point, :]
                p_ref_lim2 = self.local_labelset[mid_point:upper_bound, :]
                self.p_reference = np.transpose(p_ref_lim)
                self.p_reference2 = np.transpose(p_ref_lim2)

                # First, normalize the entire s matrix
                if self.normalize_EMG:
                    s_normed2 = s_temp2/np.amax(s_temp2)
                else:
                    s_normed2 = s_temp2
                # Now do PCA unless it is set to 64 (AKA the default num channels i.e. no reduction)
                # Also probably ought to find a global transform if possible so I don't recompute it every time...
                if self.PCA_comps!=self.pca_channel_default:  
                    pca = PCA(n_components=self.PCA_comps)
                    s_normed2 = pca.fit_transform(s_normed2)
                self.s2 = np.transpose(s_normed2)
                self.F2 = self.s2[:,:-1] # note: truncate F for estimate_decoder
                v_actual2 = self.w@self.s2
                p_actual2 = np.cumsum(v_actual2, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
                self.V2 = (self.p_reference2 - p_actual2)*self.dt
            else:
                self.learning_batch = upper_bound - lower_bound
                s_temp = self.local_dataset[lower_bound:upper_bound, :]
                # For Maneeshika's code:
                p_ref_lim = self.local_labelset[lower_bound:upper_bound, :]
                # This is the used label
                self.p_reference = np.transpose(p_ref_lim)

            # First, normalize the entire s matrix
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            # Now do PCA unless it is set to 64 (AKA the default num channels i.e. no reduction)
            # Also probably ought to find a global transform if possible so I don't recompute it every time...
            if self.PCA_comps!=self.pca_channel_default:  
                pca = PCA(n_components=self.PCA_comps)
                s_normed = pca.fit_transform(s_normed)
            self.s = np.transpose(s_normed)
            self.F = self.s[:,:-1] # note: truncate F for estimate_decoder
            v_actual = self.w@self.s
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            self.V = (self.p_reference - p_actual)*self.dt
        elif self.test_split_type=="UPDATE16":
            # self.update_ix[17] and further is the short update, self.update_ix[18] is literally the last value
            # This is technically update 17 but it doesnt matter, update 17 is the last usable update (17/19)
            lower_bound = self.update_ix[16]
            #//2  #Use only the second half of each update
            upper_bound = self.update_ix[17]
            self.test_split_idx = lower_bound
            self.testing_data = self.local_dataset[self.test_split_idx:upper_bound, :]
            self.testing_labels = self.local_labelset[self.test_split_idx:upper_bound, :]
            # TODO: There ought to be some assert here to make sure that self.testing_XYZ doesnt have a shape of zero...
        elif self.test_split_type=="ENDFRACTION":
            test_split_product_index = self.local_dataset.shape[0]*self.test_split_frac
            # Convert this value to the closest update_ix value
            train_test_update_number_split = min(self.update_ix, key=lambda x:abs(x-test_split_product_index))
            self.test_split_idx = self.update_ix.index(train_test_update_number_split)
            if self.test_split_idx<17:
                print(f"self.test_split_idx is {self.test_split_idx}, so subtracting 1")
                self.test_split_idx -= 1
            assert(self.test_split_idx > self.starting_update)
            lower_bound = self.test_split_idx
            upper_bound = self.local_dataset.shape[0]
            self.testing_data = self.local_dataset[self.test_split_idx:, :]
            self.testing_labels = self.local_labelset[self.test_split_idx:, :]
        else:
            raise ValueError("test_split_type not working as expected")
        
        if self.scenario=="CROSS":
            self.training_data = self.local_dataset
            self.training_labels = self.local_labelset
            self.test_learning_batch = upper_bound - lower_bound
        s_temp = self.testing_data
        if self.normalize_EMG:
            s_normed = s_temp/np.amax(s_temp)
        else:
            s_normed = s_temp
        if self.PCA_comps!=self.pca_channel_default:  
            pca = PCA(n_components=self.PCA_comps)
            s_normed = pca.fit_transform(s_normed)
        self.s_test = np.transpose(s_normed)
        self.F_test = self.s_test[:,:-1] # note: truncate F for estimate_decoder
        self.p_test_reference = np.transpose(self.testing_labels)
        
            
    # 0: Main Loop
    def execute_training_loop(self):
        self.simulate_data_stream()
        self.train_model()
        
        # LOG EVERYTHING
        #if self.ID==0:
        #    print(f"LOGGING: cli{self.ID}, round {self.current_round}, len(local_dec_log) {len(self.local_dec_log)}")
        ## I guess this is actually after train_model is called...
        self.local_dec_log.append(copy.deepcopy(self.w))
        # TODO: Not sure why this would have higher error than global, given that global is right below...
        self.local_train_error_log.append(self.eval_model(which='local'))
        if self.global_method!="NOFL":
            # TODO: Still doesn't tell me what that global would be better than local...
            self.global_train_error_log.append(self.eval_model(which='global'))

        # Log Cost Comp
        ## I think this is TRAIN only??
        #if self.track_cost_components:
        #    self.performance_log.append(self.alphaE*(np.linalg.norm((self.w@self.F - self.V[:,1:]))**2))
        #    self.Dnorm_log.append(self.alphaD*(np.linalg.norm(self.w)**2))  # This is the scaled norm...
        #    self.Fnorm_log.append(np.linalg.norm(self.F)**2)  # This is FNorm, not the F in the cost func (which is 0 because of self.alphaF)


    def set_testset(self, test_dataset_obj):
        lower_bound = 0

        if type(test_dataset_obj) is type([]):
            running_num_test_samples = 0
            self.s_test = []
            self.F_test = []
            self.p_test_reference = []
            for test_dset in test_dataset_obj:
                self.testing_data = test_dset[0]
                self.testing_labels = test_dset[1]

                # TODO: Verify that shape index is correct
                running_num_test_samples += self.testing_data.shape[0]

                s_temp = self.testing_data
                if self.normalize_EMG:
                    s_normed = s_temp/np.amax(s_temp)
                else:
                    s_normed = s_temp
                if self.PCA_comps!=self.pca_channel_default:  
                    pca = PCA(n_components=self.PCA_comps)
                    s_normed = pca.fit_transform(s_normed)
                self.s_test.append(np.transpose(s_normed))
                self.F_test.append(np.transpose(s_normed)[:,:-1]) # note: truncate F for estimate_decoder
                self.p_test_reference.append(np.transpose(self.testing_labels))

            upper_bound = running_num_test_samples
            # TODO: This might be incorrect... it is calculated differently than the below branch...
            ## Eg maybe this should be a list of each upper-lower? Not sure...
            ## Probably doesn't matter? Not even sure if this is used
            self.test_learning_batch = upper_bound - lower_bound
        else:
            self.testing_data = test_dataset_obj[0]
            self.testing_labels = test_dataset_obj[1]
            # TODO: Verify that shape index is correct
            upper_bound = self.testing_data.shape[0]

            self.test_learning_batch = upper_bound - lower_bound

            s_temp = self.testing_data
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            if self.PCA_comps!=self.pca_channel_default:  
                pca = PCA(n_components=self.PCA_comps)
                s_normed = pca.fit_transform(s_normed)
            self.s_test = np.transpose(s_normed)
            self.F_test = self.s_test[:,:-1] # note: truncate F for estimate_decoder
            self.p_test_reference = np.transpose(self.testing_labels)
    
    
    def get_testing_dataset(self):
        return self.testing_data, self.testing_labels

        
    def simulate_delay(self, incoming):
        if incoming:
            time.sleep(self.download_delay+random.random())
        else:
            time.sleep(self.upload_delay+random.random())
            
            
    def simulate_data_stream(self):
        streaming_method = self.data_stream
        need_to_advance=True
        # TODO: Idk if I want the current_round increment here......
        ## This is sort of fine since it's just the round and doesn't matter that much 
        self.current_round += 1

        if (self.current_update>=self.final_usable_update_ix-1) or (self.train_update_ix is not None and self.current_train_update>=(len(self.train_update_ix)-1)):
            self.logger = "UNNAMED"
            #print("Maxxed out your update (you are on update 18), continuing training on last update only")
            # Probably ought to track that we maxed out --> LOG SYSTEM
            # We are stopping an update early, so use -3/-2 and not -2/-1 (the last update)
            # TODO: This is sus with INTRA, although I doubt it reaches it ever
            if self.scenario=="INTRA":
                # Should this be -2/-1 or -1/? ...?
                lower_bound = self.train_update_ix[-2]
                upper_bound = self.train_update_ix[-1]
            else:
                lower_bound = self.update_ix[16] - self.update_ix[self.starting_update]
                upper_bound = self.update_ix[17] - self.update_ix[self.starting_update]
            self.learning_batch = upper_bound - lower_bound
            need_to_advance=False
        elif streaming_method=='full_data':
            self.logger = "full_data"
            lower_bound = self.update_ix[0]  # Starts at 0 and not update 10, for now
            upper_bound = self.update_ix[-1]
            self.learning_batch = upper_bound - lower_bound
            # TODO: need_to_advance shouldn't ever run, once lower_bound and upper_bound have been set once
            need_to_advance=False
        elif streaming_method=='streaming':
            self.logger = "streaming"
            if self.scenario=="INTRA":
                # Really ought to move this branch back so that the other streaming methods are included too...
                ## Since this version uses the folds to set the streaming index
                ## Eg the existing lower_bounds and upper_bounds would be incorrect (might be incorrect for other versions even wtihout kfcv...)
                ### NAH ignore the old versions. We only doing KFCV now

                # If we pass threshold, move on to the next update
                ## > 2 since starts at 0 but is immediately incremented to 1 in this func (should not update immediately)
                if (self.current_round>2) and (self.current_round%self.local_round_threshold==0):
                    self.current_update += 1
                    self.current_train_update += 1
                    self.update_transition_log.append(self.latest_global_round)
                    if self.verbose==True and self.ID==0:
                        # self.current_update here should really be changed...
                        print(f"Client{self.ID}: New update after lrt passed: (new update, current global round, current local round): {self.current_update, self.latest_global_round, self.current_round}")
                        print()
                    need_to_advance = True
                #elif self.current_round>2:  # This is the base case
                #    need_to_advance = False
                else:  # This is for the init case (current round is 0 or 1)
                    # This should be False actually, I don't need it to advance immediately...
                    need_to_advance = False
            elif self.scenario=="CROSS":
                # If we pass threshold, move on to the next update
                if self.current_round>2 and self.current_round%self.local_round_threshold==0:
                    self.current_update += 1
                    self.update_transition_log.append(self.latest_global_round)
                    if self.verbose==True and self.ID==0:
                        print(f"Client {self.ID}: New update after lrt passed: (new update, current global round, current local round): {self.current_update, self.latest_global_round, self.current_round}")
                        print()
                        
                    need_to_advance = True
                else:
                    # This is for the init case (current round is 0 or 1)
                    # need_to_advance is true, so we overwrite s and such... this is fine 
                    ## Uhh changed it so now this is the default case (removed current_round>2 branch), not sure if this is correct...

                    need_to_advance = False
        elif streaming_method=='advance_each_iter':
            #lower_bound = (self.update_ix[self.current_update] + self.update_ix[self.current_update+1])//2 
            lower_bound = (self.update_ix[self.current_update]) - self.update_ix[self.starting_update]
            upper_bound = self.update_ix[self.current_update+1] - self.update_ix[self.starting_update]
            self.learning_batch = upper_bound - lower_bound
            self.current_update += 1
            self.current_train_update += 1
        else:
            raise ValueError(f'streaming_method ("{streaming_method}") not recognized: this data streaming functionality is not supported')
            
        # This code could be heavily condensed but idc enough to refactor it right now...
        if need_to_advance:
            if self.scenario=="INTRA":
                # Should the 0 be changed to self.current_update... would need to be subtracted by starting_update...
                lower_bound = self.train_update_ix[self.current_train_update] - self.update_ix[self.starting_update]
                # This might be a subpar soln... pretty hardcoded in but it doesnt really matter for this dataset I dont think
                upper_bound = lower_bound + 1200

                if "PFA" in self.global_method:
                    mid_point = (lower_bound+upper_bound)//2
                    self.learning_batch = mid_point - lower_bound

                    s_temp = copy.deepcopy(self.local_dataset[lower_bound:mid_point, :])
                    s_temp2 = copy.deepcopy(self.local_dataset[mid_point:upper_bound, :])
                    p_ref_lim = copy.deepcopy(self.local_labelset[lower_bound:mid_point, :])
                    p_ref_lim2 = copy.deepcopy(self.local_labelset[mid_point:upper_bound, :])
                    self.p_reference = np.transpose(p_ref_lim)
                    self.p_reference2 = np.transpose(p_ref_lim2)

                    # First, normalize the entire s matrix
                    if self.normalize_EMG:
                        s_normed2 = s_temp2/np.amax(s_temp2)
                    else:
                        s_normed2 = s_temp2
                    # Now do PCA unless it is set to 64 (AKA the default num channels i.e. no reduction)
                    # Also probably ought to find a global transform if possible so I don't recompute it every time...
                    if self.PCA_comps!=self.pca_channel_default:  
                        pca = PCA(n_components=self.PCA_comps)
                        s_normed2 = pca.fit_transform(s_normed2)
                    self.s2 = np.transpose(s_normed2)
                    self.F2 = self.s2[:,:-1] # note: truncate F for estimate_decoder
                    v_actual2 = self.w@self.s2
                    p_actual2 = np.cumsum(v_actual2, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
                    self.V2 = (self.p_reference2 - p_actual2)*self.dt
                else:
                    self.learning_batch = upper_bound - lower_bound
                    s_temp = copy.deepcopy(self.local_dataset[lower_bound:upper_bound, :])
                    #print(f"Client{self.ID} TRAINING DATA UPDATE: Mean, var, and norm: {np.mean(s_temp), np.var(s_temp), np.linalg.norm(s_temp)}")
                    # For Maneeshika's code:
                    #p_ref_lim = np.vstack([self.training_labels[self.update_ix.index(update):self.update_ix.index(update)+1, :] 
                    #               for update in train_updates])
                    p_ref_lim = copy.deepcopy(self.local_labelset[lower_bound:upper_bound, :])
                    # This is the used label
                    self.p_reference = np.transpose(p_ref_lim)
            elif self.scenario=="CROSS":
                #lower_bound = (self.update_ix[self.current_update] + self.update_ix[self.current_update+1])//2
                lower_bound = (self.update_ix[self.current_update]) - self.update_ix[self.starting_update]
                upper_bound = self.update_ix[self.current_update+1] - self.update_ix[self.starting_update]
                self.learning_batch = upper_bound - lower_bound

                if "PFA" in self.global_method:
                    mid_point = (lower_bound+upper_bound)//2
                    s_temp = copy.deepcopy(self.training_data[lower_bound:mid_point,:])
                    s_temp2 = copy.deepcopy(self.training_data[mid_point:upper_bound,:])
                    self.p_reference = copy.deepcopy(np.transpose(self.training_labels[lower_bound:mid_point,:]))
                    self.p_reference2 = copy.deepcopy(np.transpose(self.training_labels[mid_point:upper_bound,:]))
                    # For Maneeshika's code, otherwise not used:
                    #p_ref_lim = self.training_labels[lower_bound:mid_point,:]
                    #p_ref_lim2 = self.training_labels[mid_point:upper_bound,:]
                else:  # FEDAVG AND NOFL!
                    s_temp = copy.deepcopy(self.training_data[lower_bound:upper_bound,:])
                    self.p_reference = copy.deepcopy(np.transpose(self.training_labels[lower_bound:upper_bound,:]))
                    # For Maneeshika's code, otherwise not used:
                    #p_ref_lim = self.training_labels[lower_bound:upper_bound,:]
                
            # First, normalize the entire s matrix
            if self.normalize_EMG:
                s_normed = s_temp/np.amax(s_temp)
            else:
                s_normed = s_temp
            # Now do PCA unless it is set to 64 (AKA the default num channels i.e. no reduction)
            # Also probably ought to find a global transform if possible so I don't recompute it every time...
            if self.PCA_comps!=self.pca_channel_default:  
                pca = PCA(n_components=self.PCA_comps)
                s_normed = pca.fit_transform(s_normed)
            self.s = np.transpose(s_normed)
            #print(f"Client{self.ID} TRAINING DATA UPDATE: Mean, var, and norm: {np.mean(self.s), np.var(self.s), np.linalg.norm(self.s)}")
            self.F = self.s[:,:-1] # note: truncate F for estimate_decoder
            v_actual = self.w@self.s
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            if "PFA" in self.global_method:
                if self.normalize_EMG:
                    s_normed2 = s_temp2/np.amax(s_temp2)
                else:
                    s_normed2 = s_temp2
                if self.PCA_comps!=self.pca_channel_default:  
                    pca = PCA(n_components=self.PCA_comps)
                    s_normed2 = pca.fit_transform(s_normed2)
                self.s2 = np.transpose(s_normed2)
                self.F2 = self.s2[:,:-1] # note: truncate F for estimate_decoder
                v_actual2 = self.w@self.s2
                p_actual2 = np.cumsum(v_actual2, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
                
            #####################################################################
            # Add the boundary conditions code here
            if self.use_zvel:
                # Maneeshika code
                if self.current_round<2:
                    self.vel_est = np.zeros_like((p_ref_lim))
                    self.pos_est = np.zeros_like((p_ref_lim))
                    self.int_vel_est = np.zeros_like((p_ref_lim))
                    self.vel_est[0] = self.w@self.s[:,0]  # Translated from: Ds_fixed@emg_tr[0]
                    self.pos_est[0] = [0, 0]
                else:
                    prev_vel_est = self.vel_est[-1]
                    prev_pos_est = self.pos_est[-1]
                    
                    self.vel_est = np.zeros_like((p_ref_lim))
                    self.pos_est = np.zeros_like((p_ref_lim))
                    self.int_vel_est = np.zeros_like((p_ref_lim))
                    
                    self.vel_est[0] = prev_vel_est
                    self.pos_est[0] = prev_pos_est
                for tt in range(1, self.s.shape[1]):
                    # Note this does not keep track of actual updates, only the range of 1 to s.shape[1] (1202ish)
                    vel_plus = self.w@self.s[:,tt]  # Translated from: Ds_fixed@emg_tr[tt]
                    p_plus = self.pos_est[tt-1, :] + (self.vel_est[tt-1, :]*self.dt)
                    # These are just correctives, such that vel_plus can get bounded
                    # x-coordinate
                    if abs(p_plus[0]) > 36:  # 36 hardcoded from earlier works
                        p_plus[0] = self.pos_est[tt-1, 0]
                        vel_plus[0] = 0
                        self.hit_bound += 1 # update hit_bound counter
                    if abs(p_plus[1]) > 24:  # 24 hardcoded from earlier works
                        p_plus[1] = self.pos_est[tt-1, 1]
                        vel_plus[1] = 0
                        self.hit_bound += 1 # update hit_bound counter
                    if self.hit_bound > 200:  # 200 hardcoded from earlier works
                        p_plus[0] = 0
                        vel_plus[0] = 0
                        p_plus[1] = 0
                        vel_plus[1] = 0
                        self.hit_bound = 0
                    # now update velocity and position
                    self.vel_est[tt] = vel_plus
                    self.pos_est[tt] = p_plus
                    # calculate intended velocity
                    self.int_vel_est[tt] = calculate_intended_vels(p_ref_lim[tt], p_plus, 1/self.dt)

                # self.V must be set ONLY WITHIN TRAINING
                self.V = np.transpose(self.int_vel_est[:tt+1])
                #print(f"V.shape: {self.V.shape}")
            else:
                # Original code
                # self.V must be set ONLY WITHIN TRAINING
                self.V = (self.p_reference - p_actual)*self.dt
                if "PFA" in self.global_method:
                    self.V2 = (self.p_reference2 - p_actual2)*self.dt
            

    def train_given_model_1_comm_round(self, model, which):
        '''This can be used for training the in ML pipeline but principally is for local-finetuning (eg training a model after it as completed its global training pipeline).'''
        raise ValueError("train_given_model_1_comm_round is not implemented yet.")
    

    def gd_step_with_line_search(self, w, F, V, w_base=None):
        if w_base is None:
            w_flat = w.flatten()
            gradient = gradient_cost_l2(F, np.reshape(w_flat, (2, 64)), V, alphaE=self.alphaE, alphaD=self.alphaD, flatten=True)
            def objective(w_flat):
                D = np.reshape(w_flat, (2, 64))
                c_e = np.sum((D@F - V[:, 1:]) ** 2)
                c_d = np.sum(D ** 2)
                cost = self.alphaE*c_e + self.alphaD*c_d
                return cost
            def gradient_function(w_flat):
                return gradient_cost_l2(F, np.reshape(w_flat, (2, 64)), V, alphaE=self.alphaE, alphaD=self.alphaD,  flatten=True)
            # Perform line search
            alpha = line_search(objective, gradient_function, w_flat, -gradient)[0]
            if alpha is None:
                print("LINE SEARCH FAILED")
                alpha = self.lr  # Use default learning rate if line search fails
            new_w = w_flat - alpha*gradient
            # Gradient step using optimal lr:
            return np.reshape(new_w, (2, self.PCA_comps))
        else:
            w_flat = w.flatten()
            gradient = gradient_cost_l2(F, np.reshape(w_flat, (2, 64)), V, alphaE=self.alphaE, alphaD=self.alphaD, flatten=True)
            def objective(w_flat):
                # TODO: Use w_base (actual location) or w_flat (AKA w_tilde)
                D = np.reshape(w_flat, (2, 64))
                c_e = np.sum((D@F - V[:, 1:]) ** 2)
                c_d = np.sum(D ** 2)
                cost = self.alphaE*c_e + self.alphaD*c_d
                return cost
            def gradient_function(w_flat):
                # TODO: Use w_base (actual location) or w_flat (AKA w_tilde)
                return gradient_cost_l2(F, np.reshape(w_flat, (2, 64)), V, alphaE=self.alphaE, alphaD=self.alphaD, flatten=True)
            # Perform line search
            alpha = line_search(objective, gradient_function, w_flat, -gradient)[0]
            if alpha is None:
                print("LINE SEARCH FAILED")
                alpha = self.lr  # Use default learning rate if line search fails
            new_w = w_base - alpha*np.reshape(gradient, (2, 64))
            # Gradient step using optimal lr:
            return np.reshape(new_w, (2, self.PCA_comps))
        
    
    def train_model(self):
        ## Set the w_prev equal to the current w:
        self.w_prev = copy.deepcopy(self.w)

        if self.global_method!="NOFL":
            # Overwrite local model with the new global model
            # Is this the equivalent of recieving the global model I'm assuming?
            ## So... this should be D0 and such right... at the start I think...
            self.w_new = copy.deepcopy(self.global_w)
        else:
            # w_new is just the same model it has been
            self.w_new = copy.deepcopy(self.w)
        
        for i in range(self.num_steps):
            D0 = copy.deepcopy(self.w_new)  
            # ^ Was self.w_prev for some reason...
            ## But it should be the current model at the start

            # IF GLOBAL METHOD IS ONE OF THE PFA METHODS, OPT METHOD GETS OVERWRITTEN WITH GDLS
            ## This is a bad loop since it checks both global_method and opt_method...
            if self.global_method=='PFAFO':
                ## THIS IS THE WORKING ONE

                # Step 1: Compute w_tilde
                ## One step of GD on the trained_model weights
                ## VERSION USING SCIPYMIN
                #result = minimize(
                #    lambda D: cost_l2(self.F, D, self.V, self.alphaE, self.alphaD),
                #    D0,
                #    method='BFGS',
                #    jac=lambda D: gradient_cost_l2(self.F, D, self.V, self.alphaE, self.alphaD),
                #    options={'maxiter': 1})
                #w_tilde = np.reshape(result.x, (2, self.PCA_comps))
                ## VERSION USING GDLS
                w_tilde_gd = self.gd_step_with_line_search(self.w_new, self.F, self.V)
                # Just for my purposes, how similar are w_tilde nad w_tilde_gd? Ideally are the same...
                #print(f"\nw_tilde (BFGS) - w_tilde_gd Norm: {np.linalg.norm(w_tilde - w_tilde_gd)}\n")
                # They are 0.2-0.9 in difference...

                ## Step 2: Using w_tilde to inform the update on copy_of_original_weights
                ## FO Only
                self.w_new = self.gd_step_with_line_search(w_tilde_gd, self.F2, self.V2, w_base=D0)
            # This is for LOCAL and FEDAVG:
            elif self.opt_method=='GD':
                ## NOT WORKING?
                grad_cost = np.reshape(gradient_cost_l2(self.F, self.w_new, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                                        (2, self.PCA_comps))
                self.w_new -= self.lr*grad_cost
            elif self.opt_method=='GDLS':
                self.w_new = self.gd_step_with_line_search(self.w_new, self.F, self.V)
            elif self.opt_method=='FULLSCIPYMIN':
                out = minimize(
                    lambda D: cost_l2(self.F, D, self.V, alphaD=self.alphaD, alphaE=self.alphaE, Ne=self.PCA_comps), 
                    D0, method='BFGS', 
                    jac=lambda D: gradient_cost_l2(self.F, D, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps))
                self.w_new = np.reshape(out.x,(2, self.PCA_comps))
            elif self.opt_method=='MAXITERSCIPYMIN':
                out = minimize(
                    lambda D: cost_l2(self.F, D, self.V, alphaD=self.alphaD, alphaE=self.alphaE, Ne=self.PCA_comps), 
                    D0, method='BFGS', 
                    jac=lambda D: gradient_cost_l2(self.F, D, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                    options={'maxiter':self.max_iter})
                self.w_new = np.reshape(out.x,(2, self.PCA_comps))
            else:
                raise ValueError("Unrecognized method")
                
            if self.mix_in_each_steps:
                # TODO: mixed_w appears to not be used // is used as the personalized model?...
                self.mixed_w = self.smoothbatch_lr*self.mixed_w + ((1 - self.smoothbatch_lr)*self.w_new)

            # Log gradient
            self.local_gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, self.w, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)))
            if "PFA" in self.global_method:
                self.local_gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F2, self.w, self.V2, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)))

        # Do SmoothBatch
        #W_new = alpha*D[-1] + ((1 - alpha) * W_hat)
        self.w = self.smoothbatch_lr*self.w_prev + ((1 - self.smoothbatch_lr)*self.w_new)
 
        
    def eval_model(self, which):
        if which=='local':
            my_dec = self.w
            my_V = self.V
        elif which=='global':
            my_dec = self.global_w
            my_V = self.V
        elif which=='pers' and "PFA" in self.global_method:
            my_dec = self.w
            my_V = self.V
        else:
            raise ValueError("Please set <which> to either local or global")
        # Just did this so we wouldn't have the 14 decimals points it always tries to give
        if self.round2int:
            temp = np.ceil(cost_l2(self.F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps))
            # Setting to int is just to catch overflow errors
            # For RT considerations, ints are also generally ints cheaper than floats...
            out = int(temp)
        else:
            temp = cost_l2(self.F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
            out = round(temp, 7)
        return out
        

    def test_metrics(self, model, which, return_cost_func_comps=True):
        ''' No training, just evaluates the given model on the client's testing dataset (which may be the multi-client CROSS testset, if that has been set externally already) '''
        # NOTE: Hit bound code may be incompatible with this!
        
        if which=='local':
            test_log = self.local_test_error_log
            cost_func_comps_log = self.local_cost_func_comps_log
        elif which=='global':
            test_log = self.global_test_error_log
            cost_func_comps_log = self.global_cost_func_comps_log
        else:
            raise ValueError('test_metrics which does not exist. Must be local, global, or pers')
            
        if type(self.F_test) is type([]):
            # LIST VERSION: THIS IS WHAT RUNS BY DEFAULT IN THE CROSS CASE
            ## WHY

            running_test_loss = 0
            running_num_samples = 0
            if return_cost_func_comps:
                running_vel_error = 0
                running_dec_error = 0
            for i in range(len(self.F_test)):
                # TODO: Ensure that this is supposed to be D@s and not D@F...
                # Not totally sure why this iterates... why doesn't it just operate on the entire matrix? Idk

                v_actual = model@self.s_test[i]
                p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
                V_test = (self.p_test_reference[i] - p_actual)*self.dt

                batch_samples = 1 #len(self.F_test[i])  # len is 64, which is NOT the number of samples...
                if return_cost_func_comps:
                    batch_loss, vel_error, dec_error = cost_l2(self.F_test[i], model, V_test, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps, return_cost_func_comps=return_cost_func_comps)
                    running_vel_error += vel_error * batch_samples
                    running_dec_error += dec_error * batch_samples
                else:
                    batch_loss = cost_l2(self.F_test[i], model, V_test, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps, return_cost_func_comps=return_cost_func_comps)
                running_test_loss += batch_loss * batch_samples  # Accumulate weighted loss by number of samples in batch
                running_num_samples += batch_samples
            normalized_test_loss = running_test_loss / running_num_samples  # Normalize by total number of samples
            if return_cost_func_comps:
                normalized_vel_error = running_vel_error / running_num_samples
                normalized_dec_error = running_dec_error / running_num_samples
            test_log.append(normalized_test_loss)
        else:
            # MATRIX VERSION: THIS IS WHAT RUNS BY DEFAULT IN THE INTRA CASE

            # TODO: Ensure that this is supposed to be D@s and not D@F...
            v_actual = model@self.s_test
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            V_test = (self.p_test_reference - p_actual)*self.dt

            batch_samples = self.F_test.shape[1]  # Shape of F is (64, 1201)!
            if return_cost_func_comps:
                test_loss, vel_error, dec_error = cost_l2(self.F_test, model, V_test, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps, return_cost_func_comps=return_cost_func_comps)
                vel_error = vel_error * batch_samples
                dec_error = dec_error * batch_samples
            else:
                test_loss = cost_l2(self.F_test, model, V_test, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps, return_cost_func_comps=return_cost_func_comps)

            total_samples = self.F_test.shape[1]  # Shape is (64, 1201ish) AKA (self.PCA_comps, 1201ish)
            assert(total_samples!=self.PCA_comps)
            normalized_test_loss = test_loss / total_samples  # Normalize by the number of samples
            if return_cost_func_comps:
                normalized_vel_error = vel_error / total_samples
                normalized_dec_error = dec_error / total_samples
            test_log.append(normalized_test_loss)
        
        if return_cost_func_comps:
            cost_func_comps_log.append((normalized_test_loss, normalized_vel_error, normalized_dec_error))
            #return normalized_test_loss, normalized_vel_error, normalized_dec_error
        else:
            pass
        
        # This was returning V_test for some reason
        ## I have None as a placeholder for now, maybe I'll return testing samples? But it already has access to that...
        ## Probably should just remove, just have to remove it everywhere in the code...
        return normalized_test_loss, None
    

    def train_metrics(self, model, which):
        ''' No training, just evaluates the given model on the client's training data '''
        # NOTE: Hit bound code may be incompatible with this!
        
        if which=='local':
            train_log = self.local_train_error_log
        elif which=='global':
            train_log = self.global_train_error_log
        #elif which=='pers':
        #    train_log = self.pers_train_error_log
        else:
            raise ValueError('train_metrics which does not exist. Must be local, global, or pers')
            
        if type(self.F) is type([]):
            running_train_loss = 0
            running_num_samples = 0
            for i in range(len(self.F)):
                # TODO: Ensure that this is supposed to be D@s and not D@F...
                v_actual = model@self.s[i]
                p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
                V = (self.p_reference[i] - p_actual)*self.dt

                batch_loss = cost_l2(self.F[i], model, V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
                batch_samples = self.F[i].shape[0]
                running_train_loss += batch_loss * batch_samples  # Accumulate weighted loss by number of samples in batch
                running_num_samples += batch_samples
            normalized_train_loss = running_train_loss / running_num_samples  # Normalize by total number of samples
            train_log.append(normalized_train_loss)
        else:
            # TODO: Ensure that this is supposed to be D@s and not D@F...
            v_actual = model@self.s
            p_actual = np.cumsum(v_actual, axis=1)*self.dt  # Numerical integration of v_actual to get p_actual
            V = (self.p_reference - p_actual)*self.dt

            train_loss = cost_l2(self.F, model, V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
            total_samples = self.F.shape[1]  # Shape is (64, 1201ish) AKA (self.PCA_comps, 1201ish)
            assert(total_samples!=self.PCA_comps)
            normalized_train_loss = train_loss / total_samples  # Normalize by the number of samples
            train_log.append(normalized_train_loss)
        
        # This was returning V for some reason
        ## I have None as a placeholder for now, maybe I'll return training samples? But it already has access to that...
        return normalized_train_loss, None

