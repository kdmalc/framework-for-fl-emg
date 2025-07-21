import numpy as np
import copy

from experiment_params import *
from cost_funcs import *


class ModelBase:
    def __init__(self, ID, w, opt_method, smoothbatch_lr=0.75, alphaF=0.0, alphaE=1e-6, alphaD=1e-4, 
                 verbose=False, starting_update=9, PCA_comps=64, current_round=0, num_clients=14, log_init=0, 
                 apply_zero_thresholding=False):
        # Not input
        self.num_updates = 19
        self.starting_update=starting_update
        self.update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
        self.id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
                7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
        
        self.apply_zero_thresholding = apply_zero_thresholding
        
        self.type = 'BaseClass'
        self.ID = ID
        self.PCA_comps = PCA_comps
        self.pca_channel_default = 64  # When PCA_comps equals this, DONT DO PCA
        if w.shape!=(2, self.PCA_comps):
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
    