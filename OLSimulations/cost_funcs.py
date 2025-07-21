import numpy as np

# set up gradient of cost:
# d(c_L2(D))/d(D) = 2*(DF - V+)*F.T + 2*alphaD*D
def gradient_cost_l2(F, D, V, alphaD=1e-4, alphaE=1e-6, 
                     Nd=2, Ne=64, flatten=True):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder 
    alphaE is 1e-6 for all conditions
    ''' 
    
    D = np.reshape(D,(Nd, Ne))
    Vplus = V[:,1:]
    if flatten:
        return (2*(D@F - Vplus)@F.T*(alphaE) + 2*alphaD*D ).flatten()
    else:
        return 2*(D@F - Vplus)@F.T*(alphaE) + 2*alphaD*D 


# set up the cost function: 
# c_L2 = (||DF - V+||_2)^2 + alphaD*(||D||_2)^2
def cost_l2(F, D, V, alphaD=1e-4, alphaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder
    H: 2 x 2 state transition matrix
    alphaE is 1e-6 for all conditions
    ''' 

    D = np.reshape(D,(Nd,Ne))
    Vplus = V[:,1:]
    # Performance
    term1 = alphaE*(np.linalg.norm((D@F - Vplus))**2)
    # D Norm (Decoder Effort)
    term2 = alphaD*(np.linalg.norm(D)**2)
    # F Norm (User Effort)
    #term3 = alphaF*(np.linalg.norm(F)**2)

    cost  = term1 + term2

    if return_cost_func_comps:
        return cost, term1, term2
    else:
        return cost
