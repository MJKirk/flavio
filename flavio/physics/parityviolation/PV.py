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

    # C1q is WC in L = GF/sqrt2 (q gam^mu q) (e gam^mu gam^5 e) lagrangian
    # In 1303.5522, L = (-2 / v^2) * g^eq_AV (e gam^5 gam^mu e)/2 (q gam^mu q)/2
    # => C1q = g^eq_AV
    C1uNP = -0.5 * (Vud**2 * (phiq3_11 - phiq1_11) + 2*Vud*Vus*(phiq3_12 - phiq1_12) + Vus**2*(phiq3_22 - phiq1_22) - phiu_11)
    C1dNP = 0.5 * ((phiq3_11 + phiq1_11) + phid_11)

    # Leading order SM contribution
    # C1u = -1/2 + 4/3 sW^2
    # C1d = 1/2 - 2/3 sW^2
    # shifts in eqs 106 onwards I think
    C1uSM, C1dSM = -0.1888, 0.3419

    C1u = C1uSM + C1uNP
    C1d = C1dSM + C1dNP

    alphaEM = par["alpha_e"]
    EM_correction = 1 - alphaEM/(2*pi)

    # Come from gamma-Z box results in Phys. Rev. Lett. 109, 262301 (2012), arXiv:1208.4310
    # See eq 116 of 1303.5522
    Z_correction =0.00005
    N_correction = 0.00006

    # See eq 96, 97 in 1303.5522, or eq 3.12 in 2107.13569
    theory = -2*( Z * (2*C1u +  C1d + Z_correction)
                 +N * (  C1u +2*C1d + N_correction)) * EM_correction

    return theory




_process_tex = f"Q_W"
_process_taxonomy = r'Process :: low energy parity violation :: Nucelon weak charge :: $' + _process_tex + r'$'
_obs_name = f"Q_W"
_obs = flavio.classes.Observable(_obs_name, arguments=["Z", "N"])
_obs.set_description =(f"Nuclear weak charge")
_obs.tex = f"$Q_W$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, Q_W)
