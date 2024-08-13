r"""$B^+ / B_d$ lifetime ratio"""

import flavio


def tau_Bp_over_tau_Bd_SM(par):
    r"""Sm contribution to the ratio of the B+ to Bd lifetimes."""
    B1qtilde = par["bag_lifetime_B1qtilde"]
    B2qtilde = par["bag_lifetime_B2qtilde"]
    B3qtilde = par["bag_lifetime_B3qtilde"]
    B4qtilde = par["bag_lifetime_B4qtilde"]
    deltaqq1tilde = par["bag_lifetime_deltaqq1tilde"]
    deltaqq2tilde = par["bag_lifetime_deltaqq2tilde"]
    deltaqq3tilde = par["bag_lifetime_deltaqq3tilde"]
    deltaqq4tilde = par["bag_lifetime_deltaqq4tilde"]

    # Phenomenological formula from Lenz:2022rbq
    flavio.citations.register("Lenz:2022rbq")
    ratio_SM = 1 + 0.059 * B1qtilde + 0.005 * B2qtilde - 0.674 * B3qtilde + 0.160 * B4qtilde \
                 - 0.025 * deltaqq1tilde + 0.002 * deltaqq2tilde + 0.591 * deltaqq3tilde - 0.152 * deltaqq4tilde \
                 - 0.007
    return ratio_SM

def gamma_BSM_dim6(wc_obj, par, meson):
    return 0


def tau_Bp_over_tau_Bd(wc_obj, par):
    r"""Ratio of the B+ over Bd lifetimes based on the SM estimate plus
    the NP contribution from (some) four quark operators."""
    ratio_SM = tau_Bp_over_tau_Bd_SM(par)

    tau_Bd = 1 / par['tau_B0']
    delta_ratio_BSM = gamma_BSM_dim6(wc_obj, par, "Bd") - gamma_BSM_dim6(wc_obj, par, "B+")

    return ratio_SM + delta_ratio_BSM * tau_Bd


# Observable and Prediction instance
_process_tex = r"B_q \to X"
_process_taxonomy = r'Process :: $b$ hadron decays :: Lifetimes :: $' + _process_tex + r"$"

_obs_name = "tau_B+/tau_Bd"
_obs = flavio.classes.Observable(_obs_name)
_obs.set_description(r"$B^+ / B_d$ lifetime ratio")
_obs.tex = r"$\tau_{B^+} / \tau_{B_d}$"
_obs.add_taxonomy(_process_taxonomy)
flavio.classes.Prediction(_obs_name, tau_Bp_over_tau_Bd)
