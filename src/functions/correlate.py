import numpy as np

def corrcoeff_1d(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1,keepdims=1)
    B_mB = B - B.mean(-1,keepdims=1)

    # Sum of squares
    ssA = np.einsum('i,i->',A_mA, A_mA)
    ssB = np.einsum('i,i->',B_mB, B_mB)

    # Finally get corr coeff
    return np.einsum('i,i->',A_mA,B_mB)/np.sqrt(ssA*ssB)

# https://stackoverflow.com/a/40085052/ @ Divakar
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

# https://stackoverflow.com/a/41703623/ @Divakar
def corr2_coeff_rowwise(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(-1,keepdims=1)
    B_mB = B - B.mean(-1,keepdims=1)

    # Sum of squares across rows
    ssA = np.einsum('ij,ij->i',A_mA, A_mA)
    ssB = np.einsum('ij,ij->i',B_mB, B_mB)

    # Finally get corr coeff
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)

def zero_lag_correlate(arr1, arr2, wind_s, sps = 200):
    # Cross Correlate
    lcl = sps*wind_s
    N = len(arr1)
    out = np.zeros(N)
    for i in range(N):
        out[i] = corrcoeff_1d(arr1[i:i+lcl], arr2[i:i+lcl])

    return out

from scipy.signal import correlate, correlation_lags
from filtering import freq_filt

def norm_correlate(arr1, arr2):
    arr1 = freq_filt(arr1, 1, "highpass")
    arr2 = freq_filt(arr2, 1, "highpass")
    c = correlate((arr1 - np.mean(arr1))/np.std(arr1), (arr2 - np.mean(arr2))/np.std(arr2), 'full') / min(len(arr1), len(arr2))
    l = correlation_lags(arr1.size, arr2.size,)
    return c, l