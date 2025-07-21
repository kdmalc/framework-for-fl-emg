import numpy as np
import random
import copy
from sklearn.decomposition import PCA
from scipy.optimize import minimize

from fl_sim_base import *
from experiment_params import *
from cost_funcs import *


class Client(ModelBase):
    def __init__(self, ID, w, opt_method, full_client_dataset, data_stream, 
                 use_new_BOUNDED=True, apply_zero_thresholding=False, 
                 smoothbatch_lr=0.75, current_round=0, starting_update=10,
                 availability=1, final_usable_update_ix=17, global_method='FedAvg', 
                 verbose=False, test_split_type='kfoldcv', num_kfolds=7,
                 track_cost_components=True, log_decs=True, val_set=False,
                 current_fold=0, scenario="", track_gradient=True,
                 use_vint_direct=True,
                 # TUNABLE HYPERPARAMS!
                 local_round_threshold=25, lr=0.0001, beta=0.00001, num_steps=1, 
                 normalize_EMG=False, 
                 alphaE=1e-6, alphaD=1e-4, alphaF=0.0,
                 # IMPLEMENTED BUT NOT USED FOR CURRENT CONFIGURATION:
                 test_split_frac=0.3,  max_iter=None, PCA_comps=64,
                 delay_scaling=0, random_delays=False, download_delay=1, upload_delay=1, 
                 ):
        super().__init__(ID, w, opt_method, smoothbatch_lr=smoothbatch_lr, current_round=current_round, PCA_comps=PCA_comps, 
                         verbose=verbose, num_clients=14, log_init=0, apply_zero_thresholding=apply_zero_thresholding, 
                         alphaE=alphaE, alphaD=alphaD, alphaF=alphaF)
        
        self.use_vint_direct = use_vint_direct

        assert(full_client_dataset['training'].shape[1]==64) # --> Shape is (20770, 64)
        # Don't use anything past update 17 since they are different (update 17 is the short one, only like 300 datapoints)
        self.current_update = starting_update
        self.current_train_update = 0
        self.starting_update = starting_update
        self.final_usable_update_ix = final_usable_update_ix
        self.local_dataset = copy.deepcopy(full_client_dataset['training'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])
        self.local_labelset = copy.deepcopy(full_client_dataset['labels'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])
        if self.use_vint_direct:
            self.local_prefset = copy.deepcopy(full_client_dataset['pref'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])
            self.local_pactset = copy.deepcopy(full_client_dataset['pact'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])
            self.local_vactset = copy.deepcopy(full_client_dataset['vact'][self.update_ix[starting_update]:self.update_ix[final_usable_update_ix], :])

        self.global_method = global_method.upper()
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
        self.pos_est = None
        self.vel_est = None

        self.dt = 1.0/60.0
        self.lr = lr  # Learning rate
        self.beta = beta # PFA 2nd step learning rate
        self.round2int = False
        self.max_iter = max_iter
        
        self.use_new_BOUNDED = use_new_BOUNDED
        # Maneeshika Code:
        self.hit_bound = 0  # This is just initializing it to 0 (eg no hits yet)
        self.hit_bound_prev = 0
        # screen bounds (x, y) --> Hardcoded coming from CPHS
        self.x_bound = 36
        self.y_bound = 24
        self.hit_bound_threshold = 200
        
        # FL CLASS STUFF
        # Availability for training
        self.availability = availability
        # Toggle streaming aspect of data collection: {Ignore updates and use all the data; 
        #  Stream each update, moving to the next update after local_round_threshold iters have been run; 
        #  After 1 iteration, move to the next update}
        self.data_stream = data_stream  # {'full_data', 'streaming', 'advance_each_iter'} 
        # Number of gradient steps to take when training (eg amount of local computation):
        self.num_steps = num_steps  # This is Tau in PFA!
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
        # Do I need to do this here if they get passed in to the base class? Idk
        self.alphaE = alphaE
        self.alphaD = alphaD
        self.alphaF = alphaF
        self.smoothbatch_lr=smoothbatch_lr
        # LOGS
        self.log_decs = log_decs
        # Overwrite the logs since global and local track in slightly different ways
        self.track_cost_components = track_cost_components
        self.track_gradient = track_gradient
        self.local_gradient_log = []
        self.update_transition_log = []
        self.local_train_cfc_log = []
        self.local_test_cfc_log = []
        #self.global_cost_func_comps_log = []
        # 6/18/25 late addition: real tracking error
        # Not needed with self.use_vint_direct!
        self.pref_test_log = []
        self.pact_test_log = []
        self.pref_train_log = []
        self.pact_train_log = []

        self.global_w = copy.deepcopy(self.w)

        # TRAIN TEST DATA SPLIT
        self.test_split_type = test_split_type.upper() # "KFOLDCV" "UPDATE16" "ENDFRAC"
        self.test_split_frac = test_split_frac
        self.num_kfolds = num_kfolds
        self.val_set = val_set
        self.scenario = scenario.upper()
        self.current_fold = current_fold
        self.test_update = None
        self.train_update_ix = None


        if self.use_new_BOUNDED:  # This is now the only option, I removed the depreciated code
            #── TRAIN/TEST SPLIT ───────────────────────────────────────────────────
            if self.test_split_type == "TEST_ON_PRE":
                # Not even going to bothering defining the trainset, this is for eval only
                lower_bound = starting_update-1
                upper_bound = starting_update
                self.testing_data = copy.deepcopy(full_client_dataset['training'][self.update_ix[lower_bound]:self.update_ix[upper_bound], :])
                self.testing_labels = copy.deepcopy(full_client_dataset['labels'][self.update_ix[lower_bound]:self.update_ix[upper_bound], :])
                if self.use_vint_direct:
                    self.testing_pref = copy.deepcopy(full_client_dataset['pref'][self.update_ix[lower_bound]:self.update_ix[upper_bound], :])
                    self.test_v_init = copy.deepcopy(full_client_dataset['pact'][self.update_ix[lower_bound], :])
                    self.test_p_init = copy.deepcopy(full_client_dataset['vact'][self.update_ix[lower_bound], :])

                if self.scenario=="INTRA":
                    # Build folds of update‐indices once (one update per fold)
                    valid_updates = [
                        ix for ix in self.update_ix
                        if self.starting_update <= self.update_ix.index(ix) < self.final_usable_update_ix
                    ]
                    # one‐update‐per‐fold cross‐val:
                    self.folds = [[u] for u in valid_updates]
                    self.num_kfolds = len(self.folds)
                    self.test_update = [self.starting_update - 1]
                    # build train_update_ix (all others in range)
                    self.train_update_ix = [
                        u for u in self.update_ix
                        if (self.starting_update <= self.update_ix.index(u) < self.final_usable_update_ix)
                        and (self.update_ix.index(u) not in self.test_update)
                    ]
            elif self.test_split_type == "KFOLDCV":
                if self.scenario=="INTRA":
                    # Build folds of update‐indices once (one update per fold)
                    valid_updates = [
                        ix for ix in self.update_ix
                        if self.starting_update <= self.update_ix.index(ix) < self.final_usable_update_ix
                    ]
                    # one‐update‐per‐fold cross‐val:
                    self.folds = [[u] for u in valid_updates]
                    self.num_kfolds = len(self.folds)
                    assert len(self.folds) == self.num_kfolds

                    # pick the current fold’s single update as test
                    test_updates = self.folds[self.current_fold]
                    self.test_update = [self.update_ix.index(u) for u in test_updates]

                    # build train_update_ix (all others in range)
                    self.train_update_ix = [
                        u for u in self.update_ix
                        if (self.starting_update <= self.update_ix.index(u) < self.final_usable_update_ix)
                        and (self.update_ix.index(u) not in self.test_update)
                    ]

                    # map those to raw array bounds
                    lo_pre = self.update_ix.index(test_updates[0])
                    hi_pre = self.update_ix.index(test_updates[-1]) + 1
                    lo = self.update_ix[lo_pre] - self.update_ix[self.starting_update]
                    hi = self.update_ix[hi_pre] - self.update_ix[self.starting_update]

                    self.testing_data   = self.local_dataset[lo:hi, :]
                    self.testing_labels = self.local_labelset[lo:hi, :]
                    if self.use_vint_direct:
                        self.testing_pref = self.local_prefset[lo:hi, :]
                        self.test_v_init = self.local_vactset[lo, :]
                        self.test_p_init = self.local_pactset[lo, :]
            elif self.test_split_type == "ENDFRAC":
                if self.scenario == "INTRA":
                    # use the final usable update as the testing update
                    test_upd = self.final_usable_update_ix - 1
                    self.test_update = [test_upd]

                    # train_update_ix = all updates from starting_update (inclusive) up to final_usable_update_ix (exclusive)
                    # except the one we just held out
                    self.train_update_ix = [
                        u for u in self.update_ix
                        if (self.starting_update <= self.update_ix.index(u) < self.final_usable_update_ix)
                        and (self.update_ix.index(u) not in self.test_update)
                    ]

                    # now slice out the exact samples for that held-out update
                    lo = self.update_ix[test_upd]     - self.update_ix[self.starting_update]
                    hi = self.update_ix[test_upd + 1] - self.update_ix[self.starting_update]
                    self.testing_data   = self.local_dataset[lo:hi, :]
                    self.testing_labels = self.local_labelset[lo:hi, :]
                    if self.use_vint_direct:
                        self.testing_pref = self.local_prefset[lo:hi, :]
                        self.test_v_init = self.local_vactset[lo, :]
                        self.test_p_init = self.local_pactset[lo, :]
            else:
                raise ValueError(f"Unknown split type {self.test_split_type}")
            
            # We actually treat cross the same here since it is handled in the main.py
            if self.scenario=="CROSS":
                if self.val_set==True:
                    # ^ If both are true, then this is a testing client, as set in main_cross_subject.py
                    self.test_split_idx = -1

                    # Use the first update
                    lower_bound = 0
                    upper_bound = 1202

                    self.testing_data = self.local_dataset[lower_bound:upper_bound, :]
                    self.testing_labels = self.local_labelset[lower_bound:upper_bound, :]
                    if self.use_vint_direct:
                        self.testing_pref = self.local_prefset[lower_bound:upper_bound, :]
                        self.test_v_init = self.local_vactset[lower_bound, :]
                        self.test_p_init = self.local_pactset[lower_bound, :]
                else:
                    # Just added this for cross, didnt exist before (was breaking the BOUNDED version)
                    # NOTE: self.local_dataset is already starting at starting_update and going till final_usable_update_ix
                    self.training_data = self.local_dataset
                    self.training_labels = self.local_labelset

            #── BUILD TEST FEATURES & POSITION LABELS ────────────────────────────────
            # Test set isn't defined for Cross yet so we can't run this for Cross
            if self.scenario=="INTRA":
                # Normalize & PCA just like in training
                ## Recall that for testing we only need 1 batch for PFA as well
                self.s_test = self.prep_EMG_data(self.testing_data)  # shape (features, T)
                self.F_test = self.s_test[:, :-1]  # next‐step alignment
                if self.use_vint_direct:
                    self.V_test = self.testing_labels.T
                    self.p_test_reference = self.testing_pref.T
                else:
                    self.p_test_reference = self.testing_labels.T  # shape (2, T)
                
                # Velocity labels will be computed inside the below func:
                self.simulate_data_stream()

    
    def apply_bounds_and_compute_V(self, 
        w,           # weight matrix, shape (2, d)
        s,           # feature matrix, shape (d, T)
        p_ref,       # reference positions, shape (T, 2)
        pos0=None,   # initial position, shape (2,)
        vel0=None,   # initial velocity, shape (2,)
        x_bound=36,  # horizontal screen half-width
        y_bound=24,  # vertical screen half-height
        hit_reset=200,  # hits before teleport
        cumulative_hitbound=0
    ):
        """
        Simulate cursor dynamics with boundary constraints and compute intended velocities V.
        Returns:
        V          (2, T)   gap-closing velocities for each time step (self.V, V_optimal)
        pos        (T, 2)   simulated positions after bounding (p_act)
        vel        (T, 2)   simulated velocities after bounding (v_act)
        hit_bound  int      final hit counter
        """
        T = s.shape[1]
        # Initialize arrays
        pos = np.zeros((T, 2))
        vel = np.zeros((T, 2))
        int_vel = np.zeros((T, 2))

        if s.shape[1]==2:
            raise ValueError("T is 2, ought to be the sequence length. Transpose somewhere...")
        if p_ref.shape[0]==2:
            # This is kind of a band-aid solution but should be fine?
            p_ref = p_ref.T

        hit_bound = cumulative_hitbound
        # Seed initial state
        if pos0 is None:
            pos[0] = np.zeros(2)
        else:
            pos[0] = pos0
        if vel0 is None:
            vel[0] = (w @ s[:, 0])
        else:
            vel[0] = vel0

        # Compute intended velocity at t=0 (optional)
        int_vel[0] = self.calculate_intended_vels(p_ref[0], pos[0], self.dt)

        for t in range(1, T):
            # raw velocity from decoder
            vel_plus = w @ s[:, t]
            # integrate previous velocity
            p_plus = pos[t-1] + vel[t-1]*self.dt

            # apply horizontal bound
            if abs(p_plus[0]) > x_bound:
                p_plus[0] = pos[t-1, 0]
                vel_plus[0] = 0
                hit_bound += 1
            # apply vertical bound
            if abs(p_plus[1]) > y_bound:
                p_plus[1] = pos[t-1, 1]
                vel_plus[1] = 0
                hit_bound += 1
            # teleport back to center if too many hits
            if hit_bound > hit_reset:
                p_plus[:] = 0
                vel_plus[:] = 0
                hit_bound = 0

            pos[t] = p_plus
            vel[t] = vel_plus
            # compute the intended (gap-closing) velocity label
            int_vel[t] = self.calculate_intended_vels(p_ref[t], p_plus, self.dt)

        # Rearrange to shape (2, T)
        V = int_vel.T
        return V, pos, vel, hit_bound
    
            
    # 0: Main Loop
    def execute_training_loop(self):
        self.simulate_data_stream()
        self.train_model()
        # Logging
        self.local_dec_log.append(copy.deepcopy(self.w))


    def set_testset(self, test_dataset_obj):
        lower_bound = 0

        # if we have multiple client's test sets
        if type(test_dataset_obj) is type([]):
            raise ValueError("set_testset(): WARNING: multi-dataset version running! This has been discontinued.") 
        else:
            self.testing_data = test_dataset_obj[0]
            self.testing_labels = test_dataset_obj[1]
            self.V_test = self.testing_labels.T
            # Using only the first update since we know pos and vel are zero
            upper_bound = 1202  #self.testing_data.shape[0]
            self.test_learning_batch = upper_bound - lower_bound
            self.s_test = self.prep_EMG_data(self.testing_data)  # shape (features, T)
            self.F_test = self.s_test[:, :-1]  # next‐step alignment
            if self.use_vint_direct:
                self.testing_pref = test_dataset_obj[2]
                self.p_test_reference = self.testing_pref.T 
                self.test_v_init = test_dataset_obj[3]
                self.test_p_init = test_dataset_obj[4]
            else:
                self.p_test_reference = self.testing_labels.T  # shape (2, T)

        self.simulate_data_stream()     


    def get_testing_dataset(self):
        if self.use_vint_direct:
            return self.testing_data, self.testing_labels, self.testing_pref, self.test_v_init, self.test_p_init
        else:
            return self.testing_data, self.testing_labels

    
    def prep_EMG_data(self, s_block):
        X = s_block / np.max(s_block) if self.normalize_EMG else s_block
        if self.PCA_comps != self.pca_channel_default:
            X = PCA(n_components=self.PCA_comps).fit_transform(X)
        return X.T  # -> shape (d, T)

    
    def simulate_data_stream(self):
        streaming_method = self.data_stream  # Seems like this could be fixed and set in init surely
        need_to_advance = True
        self.current_round += 1  # So INTRA and CROSS should both "start" on current_round=1 bc of this

        #print(f"Logging simulate_data_stream! ID {self.ID}, Current round {self.current_round}, Current update {self.current_update}")

        condition1 = (self.current_update>=self.final_usable_update_ix-1)
        condition2 = (self.train_update_ix is not None and self.current_train_update>=(len(self.train_update_ix)-1))
        # If we have reached the final update and need to stop advancing and just start recycling the same update
        if condition1 or condition2:
            self.logger = "UNNAMED"
            #print("Maxxed out your update (you are on update 18), continuing training on last update only")
            if self.scenario=="INTRA":
                lower_bound = self.train_update_ix[-2]
                upper_bound = lower_bound+1202  #self.train_update_ix[-1]
            elif self.scenario=="CROSS":
                # Stay fixed at the final update
                lower_bound = self.update_ix[16] - self.update_ix[self.starting_update]
                upper_bound = self.update_ix[17] - self.update_ix[self.starting_update]
            self.learning_batch = upper_bound - lower_bound
            need_to_advance=False
        elif streaming_method=='streaming':
            self.logger = "streaming"
            # If we pass threshold, move on to the next update
            ## > 2 since starts at 0 but is immediately incremented to 1 in this func (should not update immediately)
            if self.current_round==1:  
                need_to_advance = True
                # "Advance" to/on the current update (ie we don't increment the update on the very first round, we just set everything up)
            elif self.current_round%self.local_round_threshold==0:
                self.current_update += 1
                self.current_train_update += 1
                self.update_transition_log.append(self.latest_global_round)
                if self.current_update == self.final_usable_update_ix:  # (17)
                    print(f"Client{self.ID} reached final update at: (new update, current global round, current local round): {self.current_update, self.latest_global_round, self.current_round}\n")
                if self.verbose==True and self.ID==0:
                    print(f"Client{self.ID}: New update after lrt passed: (new update, current global round, current local round): {self.current_update, self.latest_global_round, self.current_round}\n")
                need_to_advance = True
            else:
                need_to_advance = False
        else:
            raise ValueError(f'streaming_method ("{streaming_method}") not recognized: this data streaming functionality is not supported')
            
        if need_to_advance:
            # ─── Compute lower/upper bounds ─────────────────────────────────────────
            if self.scenario == "INTRA":
                # use the train_update_ix list
                lower_bound = self.train_update_ix[self.current_train_update] - self.update_ix[self.starting_update]
                upper_bound = lower_bound+1202  #self.train_update_ix[self.current_train_update+1] - self.update_ix[self.starting_update]
            elif self.scenario == "CROSS":
                # use the global update_ix directly
                lower_bound =  self.update_ix[self.current_update]   - self.update_ix[self.starting_update]
                upper_bound = (self.update_ix[self.current_update+1] - self.update_ix[self.starting_update])
            else:
                raise ValueError(f"Unknown scenario {self.scenario}")

            # ─── Split PFA vs single‐batch ───────────────────────────────────────────
            # Set s, F, and V for the training data (the new update)
            if "PFA" in self.global_method:
                # two halves
                mid = (lower_bound + upper_bound) // 2
                self.learning_batch = mid - lower_bound

                # raw slices
                s1 = self.local_dataset[lower_bound:mid, :]
                s2 = self.local_dataset[mid:upper_bound, :]
                self.s,  self.s2  = self.prep_EMG_data(s1), self.prep_EMG_data(s2)
                self.F,  self.F2  = self.s[:, :-1], self.s2[:, :-1]

                if self.use_vint_direct:
                    self.V = self.local_labelset[lower_bound:mid, :].T  # shape (2, T1)
                    self.V2 = self.local_labelset[mid:upper_bound, :].T  # shape (2, T2)

                    self.p_reference = self.local_prefset[lower_bound:mid, :].T  # transpose into (2, T)
                    self.p_reference2 = self.local_prefset[mid:upper_bound, :].T  # transpose into (2, T)

                    self.update_v_init = self.local_vactset[lower_bound, :]
                    self.update_p_init = self.local_pactset[lower_bound, :]
                    self.update_v_init2 = self.local_vactset[mid, :]
                    self.update_p_init2 = self.local_pactset[mid, :]

                    # Do a forward pass to get the estimated train pact (note that we are still using the recorded pact)
                    v_actual = self.w @ self.s
                    v_actual[:, 0] = self.update_v_init
                    pact_train_1 = np.cumsum(v_actual, axis=1)*self.dt
                    pact_train_1[:, 0] = self.update_p_init

                    v2_actual = self.w @ self.s2
                    v2_actual[:, 0] = self.update_v_init2
                    pact_train_2 = np.cumsum(v2_actual, axis=1)*self.dt
                    pact_train_2[:, 0] = self.update_p_init2
                    self.pos_est = np.concatenate((pact_train_1, pact_train_2), axis=1)  # p_act (train)
                else:
                    p1 = self.local_labelset[lower_bound:mid, :].T  # shape (2, T1)
                    p2 = self.local_labelset[mid:upper_bound, :].T  # shape (2, T2)
                    self.p_reference = p1  # shape (2, T)
                    self.p_reference2 = p2  # shape (2, T)

                    # batch1: boundary + labels
                    self.V,  pos1, vel1, hb1 = self.apply_bounds_and_compute_V(
                        w=self.w, s=self.s, p_ref=p1, 
                        pos0=self.pos_est[-1] if self.pos_est is not None else None, 
                        vel0=self.vel_est[-1] if self.vel_est is not None else None, 
                        cumulative_hitbound=self.hit_bound
                    )
                    # batch2: boundary + labels, seeded from end of batch1
                    self.V2, pos_est, vel_est, hit_bound = self.apply_bounds_and_compute_V(
                        w=self.w, s=self.s2, p_ref=p2,
                        pos0=pos1[-1], vel0=vel1[-1],  # carry over
                        cumulative_hitbound=hb1               # continue counter
                    )
            else:
                # single‐batch (NOFL / FEDAVG)
                self.learning_batch = upper_bound - lower_bound

                s_block = self.local_dataset[lower_bound:upper_bound, :]
                self.s  = self.prep_EMG_data(s_block)
                self.F  = self.s[:, :-1]

                if self.use_vint_direct:
                    self.update_v_init = self.local_vactset[lower_bound, :]
                    self.update_p_init = self.local_pactset[lower_bound, :]

                    self.V = self.local_labelset[lower_bound:upper_bound, :].T
                    v_actual = self.w @ self.s
                    v_actual[:, 0] = self.update_v_init
                    self.pos_est = np.cumsum(v_actual, axis=1)*self.dt  # p_act (train)
                    self.pos_est[:, 0] = self.update_p_init
                    self.p_reference = self.local_prefset[lower_bound:upper_bound, :].T  # transpose into (2, T)
                else:
                    p_block = self.local_labelset[lower_bound:upper_bound, :].T  # transpose into (2, T)
                    self.p_reference = p_block

                    # boundary + labels
                    self.V, pos_est, vel_est, hit_bound = self.apply_bounds_and_compute_V(
                        w=self.w, s=self.s, p_ref=p_block,
                        pos0=self.pos_est[-1] if self.pos_est is not None else None, 
                        vel0=self.vel_est[-1] if self.vel_est is not None else None, 
                        cumulative_hitbound=self.hit_bound
                    )

            # 1) Generate bounded trajectory & labels, seeded from previous end‐state
            if self.use_vint_direct:
                # I DO need to do a forward pass (set pact)
                v_actual_test = self.w @ self.s_test
                v_actual_test[:, 0] = self.test_v_init
                self.p_actual_test = np.cumsum(v_actual_test, axis=1)*self.dt
                self.p_actual_test[:, 0] = self.test_p_init 
            else:
                self.pos_est, self.vel_est, self.hit_bound = pos_est, vel_est, hit_bound

                if self.scenario=="INTRA":
                    # We need to use the final training pos/vel whenever we get it
                    ## The below is going to be wrong for every single evaluation EXCEPT the very final, post-training one
                    ## I think we just have to accept that this will always be quite wrong for INTRA
                    test_pos0 = self.pos_est[-1]
                    test_vel0 = self.vel_est[-1]
                    test_hitbnd = self.hit_bound
                elif self.scenario=="CROSS":
                    # On the test client, we start from scratch (if testing data starts at update 1)
                    # These inits are only true if the Cross test users are starting from the first update!
                    test_pos0 = None
                    test_vel0 = None
                    test_hitbnd = 0

                self.V_test, pos_sim, vel_sim, hb_final = self.apply_bounds_and_compute_V(
                    w          = self.w,
                    s          = self.s_test,
                    p_ref      = self.p_test_reference,  # shape (T,2)
                    pos0       = test_pos0,
                    vel0       = test_vel0,
                    cumulative_hitbound = test_hitbnd
                )

                self.p_actual_test = pos_sim

            # ——— LOG The raw reference & actual cursor positions —————
            # store the ground‐truth test reference trajectory (T×2)
            self.pref_test_log.append(copy.deepcopy(self.p_test_reference))
            self.pref_train_log.append(copy.deepcopy(self.p_reference))
            # store the decoded trajectory we just simulated (T×2)
            self.pact_test_log.append(copy.deepcopy(self.p_actual_test))
            assert(self.pos_est.shape[1]<1203)
            self.pact_train_log.append(copy.deepcopy(self.pos_est))
        
    
    def train_model(self):
        ## Set the w_prev equal to the current w:
        self.w_prev = copy.deepcopy(self.w)

        if self.global_method!="NOFL":
            # Overwrite local model with the new global model
            self.w_new = copy.deepcopy(self.global_w)
        else:
            # w_new is just the same model
            self.w_new = copy.deepcopy(self.w)
        
        for i in range(self.num_steps):
            # If global method is PFA, then force it to do the GD two step opt
            if self.global_method=='PFAFO':
                # Inner (fast) step
                w_base = copy.deepcopy(self.w_new)
                w_flat = copy.deepcopy(w_base.flatten())
                gradient = gradient_cost_l2(self.F, np.reshape(w_flat, (2, 64)), self.V, alphaE=self.alphaE, alphaD=self.alphaD, flatten=True)
                grad_norm = np.linalg.norm(gradient)
                #print(f"Step1 gradient norm: {grad_norm}")
                if grad_norm>1e10:
                    raise ValueError("Step 1 gradients are blowing up...")
                #assert np.all(np.isfinite(gradient)), "Found non‐finite entries in gradient"
                new_w = w_flat - self.lr*gradient
                w_tilde = np.reshape(new_w, (2, self.PCA_comps))
                
                # Outer (meta) step
                gradient = gradient_cost_l2(self.F2, w_tilde, self.V2, alphaE=self.alphaE, alphaD=self.alphaD, flatten=True)
                grad_norm = np.linalg.norm(gradient)
                #print(f"Step2 gradient norm: {grad_norm}")
                if grad_norm>1e10:
                    raise ValueError("Step 2 gradients are blowing up...")
                #assert np.all(np.isfinite(gradient)), "Found non‐finite entries in gradient2"
                new_w = w_base.flatten() - self.beta*gradient
                self.w_new = np.reshape(new_w, (2, self.PCA_comps))
            # This is for LOCAL and FEDAVG:
            elif self.opt_method=='GD':
                gradient = np.reshape(gradient_cost_l2(self.F, self.w_new, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps), 
                                        (2, self.PCA_comps))
                #print(f"Gradient norm: {np.linalg.norm(gradient)}")
                if np.linalg.norm(gradient)>1e10:
                    raise ValueError("Gradients are blowing up...")
                self.w_new -= self.lr*gradient
            elif self.opt_method=='FULLSCIPYMIN':
                assert(self.F.shape[1]+1 == self.V.shape[1])
                out = minimize(
                    lambda D: cost_l2(self.F, D, self.V, alphaD=self.alphaD, alphaE=self.alphaE, Ne=self.PCA_comps), 
                    self.w_new, method='BFGS', 
                    jac=lambda D: gradient_cost_l2(self.F, D, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps))
                self.w_new = np.reshape(out.x,(2, self.PCA_comps))
            else:
                raise ValueError("Unrecognized method")

            # Log gradient
            self.local_gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F, self.w_new, self.V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)))
            if "PFA" in self.global_method:
                self.local_gradient_log.append(np.linalg.norm(gradient_cost_l2(self.F2, self.w_new, self.V2, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)))

        # SmoothBatch (and also updating self.w) should only happen at the end of an update (NOT each training round).
        # It is fine to do it here for NOFL tho
        # SmoothBatch: only for the user's models (what they were previously using and what they are now using)
        if self.global_method=="NOFL":
            self.w = self.smoothbatch_lr*self.w_prev + ((1 - self.smoothbatch_lr)*self.w_new)
        else:
            self.w = copy.deepcopy(self.w_new)
 
        
    def eval_model(self, which):
        if which=='local':
            my_dec = self.w
        elif which=='global':
            # This evals the CLIENT'S latest global model, not the current global model
            my_dec = self.global_w
        else:
            raise ValueError("Please set <which> to either local or global")
        
        my_F = self.F
        my_V = self.V
        total_num_samples = self.F.shape[1]  # Should be 1200-1202, NOT 64!
        assert(my_F.shape[1]+1 == my_V.shape[1])
        total_cost = cost_l2(my_F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)

        if self.global_method=="PFAFO":
            my_F = self.F2
            my_V = self.V2
            pfa2_samples = self.F2.shape[1]  # Should be 1200-1202, NOT 64!
            assert(my_F.shape[1]+1 == my_V.shape[1])
            pfa2_cost = cost_l2(my_F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps)
            total_cost += pfa2_cost
            total_num_samples += pfa2_samples

        normalized_cost = total_cost / total_num_samples
        return normalized_cost
    

    def calc_test_tdpe(self, model=None):
        if model is None:
            model = self.w

        if self.use_vint_direct:
            v_actual = model @ self.s_test
            v_actual[:, 0] = self.test_v_init
            pact = (np.cumsum(v_actual, axis=1)*self.dt).T
            # Note that it has been transposed here
            pact[0, :] = self.test_p_init
            pref = self.p_test_reference.T
        else:
            # 1) Generate bounded trajectory & labels, seeded from previous end‐state
            if self.scenario=="INTRA":
                # We need to use the final training pos/vel whenever we get it
                ## The below is going to be wrong for every single evaluation EXCEPT the very final, post-training one
                test_pos0 = self.pos_est[-1]
                test_vel0 = self.vel_est[-1]
                test_hitbnd = self.hit_bound
            elif self.scenario=="CROSS":
                # On the test client, we start from scratch
                # This is only true if we are evaluating on the first update! I don't think that is the case!
                test_pos0 = None
                test_vel0 = None
                test_hitbnd = 0
            # self.V_test would only be needed for a cost calculation!
            V_test, pos_sim, vel_sim, hb_final = self.apply_bounds_and_compute_V(
                w          = model,
                s          = self.s_test,
                p_ref      = self.p_test_reference,  # shape (T,2)
                pos0       = test_pos0,
                vel0       = test_vel0,
                cumulative_hitbound = test_hitbnd,
            )

            pref = self.p_test_reference.T  # Also needs to become 1202x2
            pact = pos_sim  # 1202x2

        # Using just the second half, to account for the the fact that the init pos and vel are probably quite far off
        midpoint = 0  #pact.shape[0]//2
        full_test_time_domain_position_error = self.calc_time_domain_error(pact[midpoint:, :], pref[midpoint:, :])  # Should be like 1200x1 I think, can make it 600x1
        mean_tdpe = np.mean(full_test_time_domain_position_error)  # Should be a scalar
        return mean_tdpe
        

    def test_metrics(self, model, which, return_cost_func_comps=False, save_cost_func_comps=True):
        """
        Evaluate `model` on the test split (local or global).
        Computes:
        v_pred = model @ F_test
        p_pred = cumsum(v_pred)*dt
        optionally bounds p_pred
        V_label = (p_test_reference[:,1:] - p_pred[:,1:]) / dt
        loss components via cost_l2
        normalizes by number of timesteps

        return_cost_func_comps means to return them from this function. Note that we should always return them from the cost function. Separate thing despite the same name
        """

        # choose the right log
        if which.upper() == 'LOCAL':
            test_log, comps_log = self.local_test_error_log, self.local_test_cfc_log
        elif which.upper() == 'GLOBAL':
            test_log = self.global_test_error_log
        else:
            raise ValueError("which must be 'local' or 'global'")

        # 2) cost & (optionally) components
        assert self.F_test.shape[1] + 1 == self.V_test.shape[1], \
            f"F_test has {self.F_test.shape[1]} cols but V has {self.V_test.shape[1]}"
        out = cost_l2(
            self.F_test, model, self.V_test,
            alphaE=self.alphaE, alphaD=self.alphaD,
            Ne=self.PCA_comps, return_cost_func_comps=True
        )
        total_cost, vel_err, dec_err = out

        # 3) normalize
        n_samples     = self.V_test.shape[1]
        assert(n_samples!=2)
        loss        = total_cost / n_samples
        vel_err    /= n_samples
        dec_err    /= n_samples

        if save_cost_func_comps==True and which.upper()=='LOCAL':
            comps_log.append((vel_err, dec_err))

        # 4) log
        test_log.append(loss)
        if which.upper()=="GLOBAL" or return_cost_func_comps==False:
            return loss
        elif return_cost_func_comps:
            return loss, (vel_err, dec_err)
        else:
            raise ValueError("This should not run!")
        
    
    def score_half(self, s, F, p_ref, pos0, vel0, hb0, model=None):
        if model is None:
            model = self.w

        V_full, pos_out, vel_out, hb_out = self.apply_bounds_and_compute_V(
        w                  = model,
        s                  = s,         # (d, T)
        p_ref              = p_ref.T,   # (T,2)
        pos0               = pos0,      # e.g. self.pos_est[0]
        vel0               = vel0,      # e.g. self.vel_est[0]
        cumulative_hitbound= hb0        # e.g. self.hit_bound_prev
        )
        assert F.shape[1]+1 == V_full.shape[1]
        # cost_l2(F, model, V_full, …)
        cost = cost_l2(F, model, V_full,
            alphaE=self.alphaE, alphaD=self.alphaD,
            Ne=self.PCA_comps, return_cost_func_comps=False)
        return cost, V_full.shape[1], pos_out, vel_out, hb_out


    def train_metrics(self, model, which='local', save_cost_func_comps=True):
        """
        Exactly the same pipeline as test_metrics, but on the training split.
        This is also called by the server at the end of each training round!
        """

        if which == 'local':
            train_log, comps_log = self.local_train_error_log, self.local_train_cfc_log
        elif which == 'global':
            train_log = self.global_train_error_log
        else:
            raise ValueError("which must be 'local' or 'global'")

        my_F = self.F
        my_V = self.V
        my_dec = model
        total_num_samples = self.F.shape[1]  # Should be 1200-1202, NOT 64!
        assert(total_num_samples!=64)
        assert(my_F.shape[1]+1 == my_V.shape[1])
        out = cost_l2(my_F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps, return_cost_func_comps=True)
        total_cost, total_vel_error, total_dec_err = out

        if self.global_method=="PFAFO":
            my_F = self.F2
            my_V = self.V2
            pfa2_samples = self.F2.shape[1]  # Should be 1200-1202, NOT 64!
            assert(pfa2_samples!=64)
            assert(my_F.shape[1]+1 == my_V.shape[1])
            out2 = cost_l2(my_F, my_dec, my_V, alphaE=self.alphaE, alphaD=self.alphaD, Ne=self.PCA_comps, return_cost_func_comps=True)
            cost2, vel_error2, dec_err2 = out2
            total_cost += cost2
            total_vel_error += vel_error2
            total_dec_err += dec_err2
            total_num_samples += pfa2_samples

        ## 3) normalize
        normalized_cost = total_cost / total_num_samples
        normalized_vel_err = total_vel_error / total_num_samples
        normalized_dec_err = total_dec_err / total_num_samples
        train_log.append(normalized_cost)
        if save_cost_func_comps==True and which.upper()=='LOCAL':
            comps_log.append((normalized_vel_err, normalized_dec_err))
        
        return normalized_cost
    

    def calculate_intended_vels(self, ref, pos, dt):
        '''
        ref = 1 x 2
        pos = 1 x 2
        dt = scalar
        '''

        if self.use_vint_direct:
            raise ValueError("calculate_intended_vels() was called! Use vint directly instead!")
        
        gain = 120
        ALMOST_ZERO_TOL = 0.01
        intended_vector = (ref - pos)*dt  # I think this should be v=dp/dt, but the study that collected this data used dp*dt so we will stick with that
        if (self.apply_zero_thresholding==True) and (np.linalg.norm(intended_vector) <= ALMOST_ZERO_TOL):
            intended_norm = np.zeros((2,))
        else:
            intended_norm = intended_vector * gain
        return intended_norm
    

    def calc_time_domain_error(self, X, Y):
        """
        Args:
            X (n_time x n_dim): time-series data of position, e.g. reference position (time x dimensions)
            Y (n_time x n_dim): time-series data of another position, e.g. cursor position (time x dimensions)

        Returns:
            td_error (n_time x 1): time-series data of the Euclidean distance between X position and Y position
        """
        assert X.shape == Y.shape, f"Shape mismatch: X.shape = {X.shape}, Y.shape = {Y.shape}"
        return np.linalg.norm(X - Y, axis=1)
    

    def check_loss_status(self, loss):
        if loss > 1e5:
            raise RuntimeError(f"Cost blew up to {loss:.3e}, aborting training!")
