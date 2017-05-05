def CROify(Ulog,tau,rseed = 12321):
    """ The CROify function allows for generating a CRO map with specific U and tau values. 
    Ulog is the log of the hash universe size. tau is the desired number of 
    non-zero elements. P is a random permutation of 1:U chosen once and for all
    and used in all CROify calculations. A is the input vector. 
    An example call would look like:
    CRO_12_500 = CROify(12,500)
    after the CRO map is defined it can be used to apply the map to any input vector, e.g.,
    A_CROified = CRO_12_500(A) 
    """
    
    import numpy as np
    from scipy.fftpack import dct
    np.random.seed(rseed)
    U = 2**Ulog
    P = np.random.permutation(U) 
        
    def CRO_map(A):
        E = np.tile(np.concatenate([A,-A]),int(np.floor(U/len(A)/2)))
        E = np.append(E,np.zeros(U-len(E)))[P]
        return np.argpartition(dct(E,norm='ortho'),-tau)[:tau]+1
        
    return CRO_map
