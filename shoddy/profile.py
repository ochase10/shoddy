
import numpy as np



def conc(M, z=z0, cnorm=7.85, alpha=0.71, beta=-0.081, m0=2e12):
    return cnorm/(1+z)**(alpha) * (M / m0)**(beta)

def Ac(c):
    return np.log(1+c) - c/(1+c)

def rvir(M, z=z0, delta=330.):
    return M2R(M, z) / delta**(1/3)



def u_func(ks, M, hm_prof=nfw_norm, z=z0, delta=delta):
    if not isinstance(M, np.ndarray):
        M = np.array(M)
    rvir = ((3.0 * M / (4.0 * np.pi * get_rhocrit(z) * delta))**(1/3))
    logrvir = np.log10(rvir)
    rs = np.logspace(-4, logrvir, 500).T # m x r
    #mink = 10**(np.log10(2.0 * np.pi) - logrvir - 3)
    #scalemask = ks[:,None] < mink[None,:]
    #ks = np.logspace(-1, 3, 100) # k
    kr = ks[:,None,None] * rs # k x m x r
    kr_part = np.sin(kr) / kr
    #kr_part[scalemask,:] = 0
    rs2 = (rs**2) # m x r
    profs = hm_prof(rs, M, z=z) # m x r

    # this array is k x m x r dimension
    igrand = 4.0 * np.pi * rs2[None,:,:] * profs[None,:,:] * kr_part

    # integrate along the r axis (last axis) to get k x m array
    u_km = np.trapz(igrand, rs[None,:,:], axis=2)

    return u_km


def nfw(r:np.ndarray, M:np.ndarray, z=z0, delta=delta):
    M = M[:,None]
    cv = conc(M, z=z)
    A = Ac(cv)
    r_vir = rvir(M, delta, z)
    valid = r < r_vir

    hm_prof = M * cv**3 / (4.0 * np.pi * r_vir**3 * A * (cv * r / r_vir) * (1 + cv * r / r_vir)**2)
    hm_prof[~valid] = 0.

    #should be n_M x n_r
    return hm_prof / M

def u_nfw(ks, M, z=z0, delta=delta):
    if not isinstance(M, np.ndarray):
        M = np.array(M)

    r_vir = rvir(M, z=z, delta=delta)
    con = conc(M)
    r_s = r_vir/con

    krv = np.outer(ks, r_vir)
    krs = np.outer(ks, r_s)

    Si_sv, Ci_sv = sici(krv+krs)
    Si_s, Ci_s = sici(krs)
    
    cterm = np.cos(krs) * (Ci_sv - Ci_s)
    sterm = np.sin(krs) * (Si_sv - Si_s)
    exterm = np.sin(krv)/(krv+krs)
    norm = np.log(1+con) - con/(1+con)

    return (cterm + sterm - exterm) / norm