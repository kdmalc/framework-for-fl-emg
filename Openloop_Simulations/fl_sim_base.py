import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt

from experiment_params import *
from cost_funcs import *


class ModelBase:
    def __init__(self, ID, w, opt_method, smoothbatch_lr=1, alphaF=0.0, alphaE=1e-6, alphaD=1e-4, verbose=False, starting_update=9, PCA_comps=64, current_round=0, num_clients=14, log_init=0):
        # Not input
        self.num_updates = 19
        self.starting_update=starting_update
        self.update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        self.id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
                7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
        
        self.type = 'BaseClass'
        self.ID = ID
        self.PCA_comps = PCA_comps
        self.pca_channel_default = 64  # When PCA_comps equals this, DONT DO PCA
        if w.shape!=(2, self.PCA_comps):
            #print(f"Class BaseModel: Overwrote the provided init decoder: {w.shape} --> {(2, self.PCA_comps)}")
            self.w = np.random.rand(2, self.PCA_comps)
        else:
            self.w = w
        self.w_prev = copy.deepcopy(self.w)
        self.global_dec_log = [copy.deepcopy(self.w)]
        self.local_dec_log = [copy.deepcopy(self.w)]
        self.w_prev = copy.deepcopy(self.w)
        self.num_clients = num_clients
        self.log_init = log_init

        self.alphaF = alphaF
        self.alphaE = alphaE
        self.alphaD = alphaD

        self.local_train_error_log = []
        self.global_train_error_log = []
        self.local_test_error_log = []
        self.global_test_error_log = []
        
        self.opt_method = opt_method.upper()
        self.current_round = current_round
        self.verbose = verbose
        self.smoothbatch_lr = smoothbatch_lr

    def __repr__(self): 
        return f"{self.type}{self.ID}"
    
    def display_info(self): 
        return f"{self.type} model: {self.ID}\nCurrent Round: {self.current_round}\nOptimization Method: {self.opt_method}"
    

##############################################################################
# Zero vel boundary code:
def reconstruct_trial_fixed_decoder(ref_tr, emg_tr, Ds_fixed, time_x, fs = 60):
    time_x = time_x
    vel_est = np.zeros_like((ref_tr))
    pos_est = np.zeros_like((ref_tr))
    int_vel_est = np.zeros_like((ref_tr))

    hit_bound = 0
    vel_est[0] = Ds_fixed@emg_tr[0]  # D@s --> Kai's v_actual
    pos_est[0] = [0, 0]
    for tt in range(1, time_x):
        vel_plus = Ds_fixed@emg_tr[tt] # at time tt --> also Kai's v_actual...
        p_plus = pos_est[tt-1, :] + (vel_est[tt-1, :]/fs)
        # These are just correctives, such that vel_plus can get bounded
        # x-coordinate
        if abs(p_plus[0]) > 36:
            p_plus[0] = pos_est[tt-1, 0]
            vel_plus[0] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter
        if abs(p_plus[1]) > 24:
            p_plus[1] = pos_est[tt-1, 1]
            vel_plus[1] = 0
            hit_bound = hit_bound + 1 # update hit_bound counter
        if hit_bound > 200:
            p_plus[0] = 0
            vel_plus[0] = 0
            p_plus[1] = 0
            vel_plus[1] = 0
            hit_bound = 0
        # now update velocity and position
        vel_est[tt] = vel_plus
        pos_est[tt] = p_plus
        # calculate intended velocity
        int_vel_est[tt] = calculate_intended_vels(ref_tr[tt], p_plus, 60)
    return vel_est, pos_est, int_vel_est


def calculate_intended_vels(ref, pos, fs):
    '''
    ref = 1 x 2
    pos = 1 x 2
    fs = scalar
    '''
    
    gain = 120
    ALMOST_ZERO_TOL = 0.01
    intended_vector = (ref - pos)/fs
    if np.linalg.norm(intended_vector) <= ALMOST_ZERO_TOL:
        intended_norm = np.zeros((2,))
    else:
        intended_norm = intended_vector * gain
    return intended_norm