import numpy as np
import time
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model

keys = ['METACPHS_S106', 'METACPHS_S107', 'METACPHS_S108', 'METACPHS_S109', 'METACPHS_S110', 'METACPHS_S111', 'METACPHS_S112', 'METACPHS_S113', 'METACPHS_S114', 'METACPHS_S115', 'METACPHS_S116', 'METACPHS_S117', 'METACPHS_S118', 'METACPHS_S119']
key_to_num = dict()
num_to_key = dict()
for idx, key in enumerate(keys):
    key_to_num[key] = idx
    num_to_key[idx] = key

random.seed(a=1)
# Default number of k-folds
cv = 5 # Changed to 5 from 10 because the smallest class in cross val only has (had?) 7 instances
my_metrics_cols=['Algorithm', 'One Off Acc', 'CV Acc', 'K Folds']
#update_ix = np.load('C:\\Users\\kdmen\\Desktop\\Research\\Data\\update_ix.npy')
update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]

def fit_ml_algo(algo, X_train, y_train, cv, verbose=False, num_decimals=3, testing=False):
    '''Runs given algorithm and returns the accuracy metrics'''
    
    model = algo.fit(X_train, y_train)
    
    # Notice that this is tested on the data it just trained on, so this is a bad accuracy metric to use
    acc = round(model.score(X_train, y_train) * 100, 3)
    
    # Cross Validation - this fixes that issue of validating on the data that the model was trained on
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs=-1)
    # Cross-validation metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, num_decimals)
    #pre_cv = round(metrics.precision_score(y_train, train_pred) * 100, num_decimals)
    #rec_cv = round(metrics.recall_score(y_train, train_pred) * 100, num_decimals)
    
    if verbose:
        print("Training predictions:")
        print(train_pred)
        print("Ground Truth:")
        print(y_train)
        print(f"One Off Accuracy: {acc}")
        print(f"CV Accuracy: {acc_cv}")
    
    if testing:
        return train_pred, acc, acc_cv, model
    
    return train_pred, acc, acc_cv
    
    
def train_test_val_split(input_df, label_df, rng_seed=2, stratification=False, validation=False, test_percent=0.3, val_percent=0.3):
    '''
    I don't think I need a validation set if I'm just doing cross_validation since that should take care of it for me
    '''
    
    x_train = input_df.copy(deep=True)
    y_train_reg = label_df

    ## TRAIN / TEST
    # Stratify might be good to ensure that all classes are represented, I'm not sure if it'll do that by default
    my_strat = y_train_reg if stratification else None
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train_reg, stratify=my_strat, test_size=test_percent, random_state=rng_seed, shuffle=True)
    # Not sure how shuffle and random state interact...

    # Should just use cross val instead of manually making val split
    if validation:
        ## TRAIN / VAL
        X_train_pv, X_val, y_train_pv, y_val = train_test_split(
            X_train, y_train, test_size=val_percent, random_state=rng_seed, shuffle=True)

        return X_train_pv, y_train_pv, X_test, y_test, X_val, y_val
    else:
        return X_train, y_train, X_test, y_test, 0, 0
        

def train_model(my_model, X_train, y_train, cv, res_df, verbose=False, dec_num='NA'):
    ''''''
    
    train_pred_log, acc, acc_cv = fit_ml_algo(my_model, X_train, y_train, cv)
    if verbose:
        print(f"{str(my_model)}")
        print(f"Accuracy: {acc}")
        print(f"Accuracy CV 10-Fold: {acc_cv}")
        print()

    my_metrics_cols = ['Algorithm', 'One Off Acc', 'CV Acc', 'K Folds', 'N']
    temp_df = pd.DataFrame([[str(my_model), acc, acc_cv, cv, dec_num]], columns=my_metrics_cols)
    res_df = pd.concat((res_df, temp_df))
    
    return res_df
    

def test_model(model_name, X_train, y_train, X_test, y_test, test_df, cv, num_decimals=3, verbose=False, my_cols=['Algorithm', 'CV Acc', 'Test Acc', 'K Folds', 'N'], dec_num='NA'):
    '''
    Inputs
    Accepts a model NAME (not object), training and test data (eg post split) and trains a model then tests it
    '''
    
    _, _, acc_cv, trained_model = fit_ml_algo(model_name, X_train, y_train, cv, testing=True)
    y_pred = trained_model.predict(X_test)
    test_acc = round(metrics.accuracy_score(y_test, y_pred) * 100, num_decimals)
    
    if verbose:
        print(str(model_name))
        print(f"CV Accuracy: {acc_cv}")
        print(f"Test Accuracy: {test_acc}")
        print()
        
    temp_df = pd.DataFrame([str(model_name), acc_cv, test_acc, cv, dec_num], index=my_cols).T
    test_df = pd.concat((test_df, temp_df))
    return test_df


def nth_decoder_model(flat_dec_expanded_df, n, my_models, stratification=False, key_to_num_dict=key_to_num, my_metrics_cols=['Algorithm', 'One Off Acc', 'CV Acc', 'K Folds', 'N'], cv=5, test=False):
    '''
    INPUTS
    flat_dec_expanded_df: Dataframe containing all input decoder data in the form of [Subject, Update Number, Flattened Dec]
        - Flattened Dec needs to be the decoder in array form (eg use np.ravel() on the dec)
    n: The update number we are interested in.  Presumably 1-19
    my_models: List of different model objects (NOT NAMES) to test
    
    OUTPUTS
    dec_res_df:
    test_df:
    '''
    
    # Look at just update number n
    dec_df = flat_dec_expanded_df[(flat_dec_expanded_df.loc[:, 'Update Number'] == n)]
    dec_labels_df = pd.DataFrame(dec_df['Subject'].map(key_to_num_dict))
    numeric_df = dec_df.drop(['Subject', 'Update Number'], axis=1)
    
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(numeric_df, dec_labels_df, stratification=stratification)
    y_train = np.ravel(y_train)
    
    #print(f"X_train shape: {X_train.shape}")
    #print(f"y_train {y_train.shape}\n")

    dec_res_df = pd.DataFrame(columns=my_metrics_cols)
    #print("TRAINING")
    for model_num, model in enumerate(my_models):
        #print(f"{model_num} of {len(my_models)}")
        dec_res_df = train_model(model, X_train, y_train, cv, dec_res_df, dec_num=n)
        
    test_df = pd.DataFrame(columns=['Algorithm', 'CV Acc', 'Test Acc', 'K Folds'])
    if test:
        #print("TESTING")
        for model in my_models:
            #print(f"{model_num} of {len(my_models)}")
            test_df = test_model(model, X_train, y_train, X_test, y_test, test_df, cv, dec_num=n)
            
    return dec_res_df, test_df


# Did not use this one because I deleted NB 109 and redid it in 106. 106 does not shuffle rn
def shuffle_and_separate_labels(data_df, labels_df, frac=1, verbose=False):
    # First, attach labels so we can keep track of them after shuffling
    data_df['Label'] = labels_df

    # Shuffle the DF
    shuffled_df = data_df.sample(frac=1)
    frac_df = shuffled_df.iloc[int(shuffled_df.shape[0]//(1/frac)):, :]

    # Now un-attach labels
    shuffled_labels_df = frac_df['Label']
    frac_df.drop('Label', axis=1, inplace=True)
    
    if verbose==True:
        print(f"Shuffled_labels_df size: {shuffled_labels_df.shape}")
        print(f"Frac_df size: {frac_df.shape}")
        # Print it just looks bad
        #print(frac_df.head())

    return frac_df, shuffled_labels_df


###################################################################################################
# From CPHS
def calc_time_domain_error(X, Y):
    """calc_time_domain_error

    Args:
        X (n_time x n_dim): time-series data of position, e.g. reference position (time x dimensions)
        Y (n_time x n_dim): time-series data of another position, e.g. cursor position (time x dimensions)

    Returns:
        td_error (n_time x 1): time-series data of the Euclidean distance between X position and Y position
    """
    # make sure that the shapes are the same
    assert(X.shape == Y.shape)

    td_error = np.linalg.norm(X - Y, axis=1)

    return td_error