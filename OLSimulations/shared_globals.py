import numpy as np
from datetime import datetime
np.random.seed(100)

# This data is not publicly available
path = "C:\\Users\\kdmen\\Box\\Yamagami Lab\\Data\\2024_IMWUT_EMG_FL\\CPHS\\CPHS_EMG"
model_saving_dir = "C:\\Users\\kdmen\\Repos\\personalization-privacy-risk\\Main_PythonVersion\\Main_Final\\models"
cond0_filename = '\\cond0_dict_list.p'
cond3_filename = '\\cond3_dict_list.pkl'
cond3vint_filename = '\\cond3_full_lst.pkl'

all_decs_init_filename = '\\all_decs_init.p'
nofl_decs_filename = '\\nofl_decs.p'

id2color = {0:'lightcoral', 1:'maroon', 2:'chocolate', 3:'darkorange', 4:'gold', 5:'olive', 6:'olivedrab', 
            7:'lawngreen', 8:'aquamarine', 9:'deepskyblue', 10:'steelblue', 11:'violet', 12:'darkorchid', 13:'deeppink'}
implemented_client_training_methods = ['GD', 'FullScipyMin', 'MaxIterScipyMin']
NUM_USERS = 14
D_0 = np.random.rand(2,64)

COLORS_LST = ['red', 'blue', 'magenta', 'orange', 'darkviolet', 'lime', 'cyan', 'yellow']
TRANSPARENCY = 0.7

CURRENT_DATETIME = str(datetime.now().strftime("%m-%d_%H-%M"))

STARTING_UPDATE = 2
DATA_STREAM = 'streaming'
NUM_KFOLDS = 7
PLOT_EACH_FOLD = False  # This plots the results from each fold after each fold has run, as opposed to all at once at the end

NUM_GLOBAL_ROUNDS = 300
NUM_KFCV_GLOBAL_ROUNDS = 500
NOFL_NUM_ROUNDS = 10
SAVED_DATASET = cond3vint_filename

SMOOTHBATCH_LR = 0.0
LR = 0.0001
BETA = 0.0001
NUM_FL_STEPS = 25
C = 0.35
LRT = 25