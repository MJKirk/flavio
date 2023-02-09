import flavio
from math import pi


def Q_W(wc_obj, par, Z, N):
    scale = 91.1876
    # Run WCs down to MZ, but don't worry about RG below MZ (see section 2.3 of 1706.03783)
    wcs = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')

    # Take only the real part of the WCs, see discussion after eq. 2.3 of 1706.03783 for why
    vev = 246.2
    phiq3_11 = wcs["phiq3_11"].real * vev**2
    phiq3_12 = wcs["phiq3_12"].real * vev**2
    phiq3_22 = wcs["phiq3_22"].real * vev**2
    phiq1_11 = wcs["phiq1_11"].real * vev**2
    phiq1_12 = wcs["phiq1_12"].real * vev**2
    phiq1_22 = wcs["phiq1_22"].real * vev**2
    phiu_11 = wcs["phiu_11"].real * vev**2
    phid_11 = wcs["phid_11"].real * vev**2

    Vud = 0.97
    Vus = 0.2

    C1uNP = -0.5 * (Vud**2 * (phiq3_11 - phiq1_11) + 2*Vud*Vus*(phiq3_12 - phiq1_12) + Vus**2*(phiq3_22 - phiq1_22) - phiu_11)
    C1dNP = 0.5 * ((phiq3_11 + phiq1_11) + phid_11)

    C1uSM, C1dSM = -0.1888, 0.3419

    C1u = C1uSM + C1uNP
    C1d = C1dSM + C1dNP

    alphaEM = par["alpha_e"]
    EM_correction = (1 - alphaEM/ (2 * pi))

    Z_correction =0.00005
    N_correction = 0.00006

    theory = -2*( Z*(2*C1u+C1d+Z_correction)+N*(C1u+2*C1d+N_correction)) * EM_correction

    return theory




_process_tex = f"Q_W"
_process_taxonomy = r'Process :: low energy parity violation :: Nucelon weak charge :: $' + _process_tex + r'$'
_obs_name = f"Q_W"
_obs = flavio.classes.Observable(_obs_name, arguments=["Z", "N"])
_obs.set_description =(f"Nuclear weak charge")
_obs.tex = f"$Q_W$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, Q_W)
