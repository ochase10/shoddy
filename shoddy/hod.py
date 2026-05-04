


def avg_NcNs(M, M_min, sig_logM, M0, M1, alpha, dc=1.):
    return N_central(M, M_min, sig_logM)**2 * N_sat_per_central(M, M0, M1, alpha) * dc**2

def avg_Ns2(M, M_min, sig_logM, M0, M1, alpha, dc=1.):
    return (N_central(M, M_min, sig_logM) * N_sat_per_central(M, M0, M1, alpha))**2 * dc**2

def n_gal(M_min, sig_logM, M0, M1, alpha, z=z0, Ms=Ms, hmf=behroozi_hmf):

    dlogM = np.log10(Ms[1] / Ms[0])
    
    dndM = hmf(Ms, dlogM, z)
    N_of_M = N_hod(Ms, M_min, sig_logM, M0, M1, alpha)

    return np.trapz(dndM*N_of_M, Ms)



'zhang10'

def zheng07_central(M, M_min, sig_logM):
    return 0.5 * (1 + sci.special.erf((np.log10(M) - np.log10(M_min)) / sig_logM))

def zheng07_satellite(M, M0, M1, alpha):
    return (np.where(M>M0,(M - M0),0) / M1)**alpha

def N_hod(M, M_min, sig_logM, M0, M1, alpha, dc=1.):
    return dc * N_central(M, M_min, sig_logM) * (1 + zheng07(M, M0, M1, alpha))