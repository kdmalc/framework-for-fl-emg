import numpy as np

# np.kron(a, b) is the kronecker product
# The function assumes that the number of dimensions of a and b are the same, if necessary prepending the smallest with ones. If a.shape = (r0,r1,..,rN) and b.shape = (s0,s1,...,sN), the Kronecker product has shape (r0*s0, r1*s1, ..., rN*SN) --> Thus for 2D: output is (r0*s0, r1*s1) --> In our case, must have square output since F@F.T is square and identity is square
def hessian_cost_l2(F, alphaD, alphaE=1e-6):
    # This isn't even used

    # Not sure about shape indices... should they all be 0? Or m, n, m like it is now?  
    # Dims work out in both cases...
    return 2*alphaE*np.kron((F@F.T), np.identity(F.shape[1])) + 2*alphaD*np.kron(np.identity(F.shape[0]), np.identity(F.shape[1]))


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
    if return_cost_func_comps:
        return (term1 + term2), term1, term2
    else:
        return (term1 + term2)


def estimate_decoder(F, V):
    return (V[:,1:])@np.linalg.pinv(F)