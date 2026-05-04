


def tinker08(hm, dlogm=0.1, z=z0, h_unit=False):
    sig_m = sigma_m(hm, z=z, h_unit=h_unit)
    return tinker_fsig(sig_m) * get_rho_m(z) * -dlnsig_dM(hm, dlogm, z=z, h_unit=h_unit) / hm

def behroozi13(hm, dlogm=0.1, z=z0, h_unit=False):
    a = z2a(z)
    tink = dM2dlogM(tinker_hmf(hm, dlogm, z=z, h_unit=h_unit), hm)
    a_correction = 0.144 / (1 + np.exp(14.79*(a - 0.213)))
    m_correction = (hm / 10**11.5)**(0.5 / (1+np.exp(6.5*a)))
    log_hmf = a_correction * m_correction + np.log10(tink)
    return dlogM2dM(10**log_hmf, hm)

def tinker10_bias(M, z=z0, delta_crit=d_crit, delta=200):
    v = nu(M, z2a(z), delta_crit=delta_crit)
    y = np.log10(delta)
    A = 1+0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y-0.88
    B = 0.183
    b = 1.5
    C = 0.019+0.107 * y + 0.19*np.exp(-(4/y)**4)
    c = 2.4
    return 1 - A * v**a / (v**a + delta_crit**a) + B * v**b + C * v**c
