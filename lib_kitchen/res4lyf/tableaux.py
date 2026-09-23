# SPDX-License-Identifier: AGPL-3.0-or-later
# Compact rewrite of exponential integrators from ClownsharkBatwing / RES4LYF.
# Upstream may add terms beyond AGPL; see NOTICE.md and the RES4LYF LICENSE.

"""Named exponential RK tableaux from RES4LYF (`beta/rk_coefficients_beta.py`).

    ci       stage nodes, ci[0] = 0
    a[i][j]  stage i weight on stage j epsilon
    b[0][j]  final weight on stage j epsilon

phi(k) = phi_k(-h), phi(k, i) = phi_k(-h * ci[i-1]). Column 0 is filled by gen_first_col_exp.
res_2s_rkmk2e is an alias of res_2s_stable.
"""

from __future__ import annotations

from .phi import Phi, calculate_gamma, gen_first_col_exp


def _res_2s(h, c1=0.0, c2=0.5, c3=1.0):
    ci = [0, c2]
    phi = Phi(h, ci)
    a2_1 = c2 * phi(1,2)
    b2 = phi(2)/c2
    b1 = phi(1) - b2
    a = [
            [0,0],
            [a2_1, 0],
    ]
    b = [
            [b1, b2],
    ]
    return ci, a, b


def _res_2s_stable(h, c1=0.0, c2=0.5, c3=1.0):
    c2 = 1.0
    ci = [0, c2]
    phi = Phi(h, ci)
    a2_1 = c2 * phi(1,2)
    b2 = phi(2)/c2
    b1 = phi(1) - b2
    a = [
            [0,0],
            [a2_1, 0],
    ]
    b = [
            [b1, b2],
    ]
    return ci, a, b


def _res_3s(h, c1=0.0, c2=0.5, c3=1.0):
    ci = [0,c2,c3]
    phi = Phi(h, ci)
    gamma = calculate_gamma(c2, c3)
    a3_2 = gamma * c2 * phi(2,2) + (c3 ** 2 / c2) * phi(2, 3)
    b3 = (1 / (gamma * c2 + c3)) * phi(2)   
    b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, a3_2, 0],
    ]
    b = [
            [0, b2, b3],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_3s_non_monotonic(h, c1=0.0, c2=0.5, c3=1.0):
    c2 = 1.0
    c3 = 0.5
    ci = [0,c2,c3]
    phi = Phi(h, ci)
    gamma = calculate_gamma(c2, c3)
    a3_2 = gamma * c2 * phi(2,2) + (c3 ** 2 / c2) * phi(2, 3)
    b3 = (1 / (gamma * c2 + c3)) * phi(2)   
    b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, a3_2, 0],
    ]
    b = [
            [0, b2, b3],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_3s_alt(h, c1=0.0, c2=0.5, c3=1.0):
    c2 = 1/3
    c1,c2,c3 = 0, c2, 2/3
    ci = [c1,c2,c3]
    phi = Phi(h, ci)
    a = [
            [0, 0,                   0],
            [0, 0,                   0],
            [0, (4/(9*c2)) * phi(2,3), 0],
    ]
    b = [
            [0, 0, (1/c3)*phi(2)],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_3s_cox_matthews(h, c1=0.0, c2=0.5, c3=1.0):
    """Cox & Matthews; known as ETD3RK"""
    c2 = 1/2 # must be 1/2
    ci = [0,c2,1]
    phi = Phi(h, ci)
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, (1/c2) * phi(1,3), 0],  # paper said 2 * phi(1,3), but this is the same and more consistent with res_3s_strehmel_weiner
    ]
    b = [
            [0, 
            -8*phi(3) + 4*phi(2),
            4*phi(3) - phi(2)],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_3s_lie(h, c1=0.0, c2=0.5, c3=1.0):
    """Lie; known as ETD2CF3"""
    c1,c2,c3 = 0, 1/3, 2/3
    ci = [c1,c2,c3]
    phi = Phi(h, ci)
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, (4/3)*phi(2,3), 0],  # paper said 2 * phi(1,3), but this is the same and more consistent with res_3s_strehmel_weiner
    ]
    b = [
            [0, 
            6*phi(2) - 18*phi(3),
            (-3/2)*phi(2) + 9*phi(3)],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_3s_sunstar(h, c1=0.0, c2=0.5, c3=1.0):
    """https://arxiv.org/pdf/2410.00498 pg 5 (tableau 2.7)"""
    c1,c2,c3 = 0, 1/3, 2/3
    ci = [c1,c2,c3]
    phi = Phi(h, ci)
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, (8/9)*phi(2,3), 0],  # paper said 2 * phi(1,3), but this is the same and more consistent with res_3s_strehmel_weiner
    ]
    b = [
            [0, 
            0,
            (3/2)*phi(2)],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_3s_strehmel_weiner(h, c1=0.0, c2=0.5, c3=1.0):
    c2 = 1/2
    ci = [0,c2,1]
    phi = Phi(h, ci)
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, (1/c2) * phi(2,3), 0],
    ]
    b = [
            [0, 0, phi(2)],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_krogstad(h, c1=0.0, c2=0.5, c3=1.0):
    """weak 4th order, Krogstad"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a = [
            [0, 0,      0,        0],
            [0, 0,      0,        0],
            [0, phi(2,3), 0,        0],
            [0, 0,      2*phi(2,4), 0],
    ]
    b = [
            [
            0, 
            2*phi(2) - 4*phi(3),
            2*phi(2) - 4*phi(3),
            -phi(2)  + 4*phi(3)
            ],
    ]
    #a = [row + [0] * (len(ci) - len(row)) for row in a]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_krogstad_alt(h, c1=0.0, c2=0.5, c3=1.0):
    """weak 4th order, Krogstad https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a = [
            [0, 0,        0,      0],
            [0, 0,        0,      0],
            [0, 4*phi(2,2), 0,      0],
            [0, 0,        2*phi(2), 0],
    ]
    b = [
            [
            0, 
            2*phi(2) - 4*phi(3),
            2*phi(2) - 4*phi(3),
            -phi(2)  + 4*phi(3)
            ],
    ]
    #a = [row + [0] * (len(ci) - len(row)) for row in a]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_strehmel_weiner(h, c1=0.0, c2=0.5, c3=1.0):
    """weak 4th order, Strehmel & Weiner"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a = [
            [0, 0,         0,        0],
            [0, 0,         0,        0],
            [0, c3*phi(2,3), 0,        0],
            [0, -2*phi(2,4), 4*phi(2,4), 0],
    ]
    b = [
            [
            0, 
            0,
            4*phi(2) - 8*phi(3), 
            -phi(2) +  4*phi(3)
            ],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_strehmel_weiner_alt(h, c1=0.0, c2=0.5, c3=1.0):
    """weak 4th order, Strehmel & Weiner https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a = [
            [0, 0,        0,      0],
            [0, 0,        0,      0],
            [0, 2*phi(2,2), 0,      0],
            [0,  -2*phi(2), 4*phi(2), 0],
    ]
    b = [
            [
            0, 
            0,
            4*phi(2) - 8*phi(3), 
            -phi(2) +  4*phi(3)
            ],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_cox_matthews(h, c1=0.0, c2=0.5, c3=1.0):
    """weak 4th order, Cox & Matthews; unresolved issue, see below"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a2_1 = c2 * phi(1,2)
    a3_2 = c3 * phi(1,3)
    a4_1 = (1/2) * phi(1,3) * (phi(0,3) - 1) # phi(0,3) == torch.exp(-h*c3)
    a4_3 = phi(1,3)
    b1 = phi(1) - 3*phi(2) + 4*phi(3)
    b2 = 2*phi(2) - 4*phi(3)
    b3 = 2*phi(2) - 4*phi(3)
    b4 = 4*phi(3) - phi(2)
    a = [
            [0,    0,0,0],
            [a2_1, 0,0,0],
            [0, a3_2,0,0],
            [a4_1, 0, a4_3,0],
    ]
    b = [
            [b1, b2, b3, b4],
    ]
    return ci, a, b


def _res_4s_cfree4(h, c1=0.0, c2=0.5, c3=1.0):
    """weak 4th order, Cox & Matthews; unresolved issue, see below"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a2_1 = c2 * phi(1,2)
    a3_2 = c3 * phi(1,2)
    a4_1 = (1/2) * phi(1,2) * (phi(0,2) - 1) # phi(0,3) == torch.exp(-h*c3)
    a4_3 = phi(1,2)
    b1 = (1/2)*phi(1) - (1/3)*phi(1,2)
    b2 = (1/3)*phi(1)
    b3 = (1/3)*phi(1)
    b4 = -(1/6)*phi(1) + (1/3)*phi(1,2)
    a = [
            [0,    0,0,0],
            [a2_1, 0,0,0],
            [0, a3_2,0,0],
            [a4_1, 0, a4_3,0],
    ]
    b = [
            [b1, b2, b3, b4],
    ]
    return ci, a, b


def _res_4s_friedli(h, c1=0.0, c2=0.5, c3=1.0):
    """https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a3_2 = 2*phi(2,2)
    a4_2 = -(26/25)*phi(1) +  (2/25)*phi(2)
    a4_3 =  (26/25)*phi(1) + (48/25)*phi(2)
    b2 = 0
    b3 = 4*phi(2) - 8*phi(3)
    b4 =  -phi(2) + 4*phi(3)
    a = [
            [0, 0,0,0],
            [0, 0,0,0],
            [0, a3_2,0,0],
            [0, a4_2, a4_3,0],
    ]
    b = [
            [0, b2, b3, b4],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_minchev(h, c1=0.0, c2=0.5, c3=1.0):
    """https://ora.ox.ac.uk/objects/uuid:cc001282-4285-4ca2-ad06-31787b540c61/files/m611df1a355ca243beb09824b70e5e774"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a3_2 = (4/25)*phi(1,2) + (24/25)*phi(2,2)
    a4_2 = (21/5)*phi(2) - (108/5)*phi(3)
    a4_3 = (1/20)*phi(1) - (33/10)*phi(2) + (123/5)*phi(3)
    b2 = -(1/10)*phi(1) +  (1/5)*phi(2) - 4*phi(3) + 12*phi(4)
    b3 =  (1/30)*phi(1) + (23/5)*phi(2) - 8*phi(3) -  4*phi(4)
    b4 =  (1/30)*phi(1) -  (7/5)*phi(2) + 6*phi(3) -  4*phi(4)
    a = [
            [0, 0,0,0],
            [0, 0,0,0],
            [0, a3_2,0,0],
            [0, 0, a4_3,0],
    ]
    b = [
            [0, b2, b3, b4],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_4s_munthe_kaas(h, c1=0.0, c2=0.5, c3=1.0):
    """unstable RKMK4t"""
    c1,c2,c3,c4 = 0, 1/2, 1/2, 1
    ci = [c1,c2,c3,c4]
    phi = Phi(h, ci)
    a = [
            [0, 0,      0,        0],
            [c2*phi(1,2), 0,      0,        0],
            [(h/8)*phi(1,2), (1/2)*(1-h/4)*phi(1,2), 0,        0],
            [0, 0,      phi(1), 0],
    ]
    b = [
            [
            (1/6)*phi(1)*(1+h/2),
            (1/3)*phi(1),
            (1/3)*phi(1),
            (1/6)*phi(1)*(1-h/2)
            ],
    ]
    return ci, a, b


def _res_5s(h, c1=0.0, c2=0.5, c3=1.0):
    """non-monotonic #4th order"""
    c1, c2, c3, c4, c5 = 0, 1/2, 1/2, 1, 1/2
    ci = [c1,c2,c3,c4,c5]
    phi = Phi(h, ci)   
    a3_2 = phi(2,3)
    a4_2 = phi(2,4)
    a5_2 = (1/2)*phi(2,5) - phi(3,4) + (1/4)*phi(2,4) - (1/2)*phi(3,5)
    a4_3 = a4_2
    a5_3 = a5_2
    a5_4 = (1/4)*phi(2,5) - a5_2
    b4 = -phi(2) + 4*phi(3)
    b5 = 4*phi(2) - 8*phi(3)
    a = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, a3_2, 0, 0, 0],
            [0, a4_2, a4_3, 0, 0],
            [0, a5_2, a5_3, a5_4, 0],
    ]
    b = [
            [0, 0, 0, b4, b5],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_5s_hochbruck_ostermann(h, c1=0.0, c2=0.5, c3=1.0):
    """non-monotonic #4th order"""
    c1, c2, c3, c4, c5 = 0, 1/2, 1/2, 1, 1/2
    ci = [c1,c2,c3,c4,c5]
    phi = Phi(h, ci)   
    a3_2 = 4*phi(2,2)
    a4_2 = phi(2)
    a5_2 = (1/4)*phi(2) - phi(3) + 2*phi(2,2) - 4*phi(3,2)
    a4_3 = phi(2)
    a5_3 = a5_2
    a5_4 = phi(2,2) - a5_2
    b4 =  -phi(2) + 4*phi(3)
    b5 = 4*phi(2) - 8*phi(3)
    a = [
            [0, 0   , 0   , 0   , 0],
            [0, 0   , 0   , 0   , 0],
            [0, a3_2, 0   , 0   , 0],
            [0, a4_2, a4_3, 0   , 0],
            [0, a5_2, a5_3, a5_4, 0],
    ]
    b = [
            [0, 0, 0, b4, b5],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _res_6s(h, c1=0.0, c2=0.5, c3=1.0):
    """non-monotonic #4th order"""
    c1, c2, c3, c4, c5, c6 = 0, 1/2, 1/2, 1/3, 1/3, 5/6
    ci = [c1, c2, c3, c4, c5, c6]
    phi = Phi(h, ci)
    a2_1 = c2 * phi(1,2)
    a3_1 = 0
    a3_2 = (c3**2 / c2) * phi(2,3)
    a4_1 = 0
    a4_2 = (c4**2 / c2) * phi(2,4)
    a4_3 = (c4**2 * phi(2,4) - a4_2 * c2) / c3
    a5_1 = 0
    a5_2 = 0 #zero
    a5_3 = (-c4 * c5**2 * phi(2,5) + 2*c5**3 * phi(3,5))   /   (c3 * (c3 - c4))
    a5_4 = (-c3 * c5**2 * phi(2,5) + 2*c5**3 * phi(3,5))   /   (c4 * (c4 - c3))
    a6_1 = 0
    a6_2 = 0 #zero
    a6_3 = (-c4 * c6**2 * phi(2,6) + 2*c6**3 * phi(3,6))   /   (c3 * (c3 - c4))
    a6_4 = (-c3 * c6**2 * phi(2,6) + 2*c6**3 * phi(3,6))   /   (c4 * (c4 - c3))
    a6_5 = (c6**2 * phi(2,6) - a6_3*c3 - a6_4*c4)   /   c5
    #a6_5_alt = (2*c6**3 * phi(3,6) - a6_3*c3**2 - a6_4*c4**2)   /   c5**2
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    b5 = (-c6*phi(2) + 2*phi(3)) / (c5 * (c5 - c6))
    b6 = (-c5*phi(2) + 2*phi(3)) / (c6 * (c6 - c5))
    a = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, a3_2, 0, 0, 0, 0],
            [0, a4_2, a4_3, 0, 0, 0],
            [0, a5_2, a5_3, a5_4, 0, 0],
            [0, a6_2, a6_3, a6_4, a6_5, 0],
    ]
    b = [
            [0, b2, b3, b4, b5, b6],
    ]
    a, b = gen_first_col_exp(a,b,ci,phi)
    return ci, a, b


def _abnorsett_2m(h):
    c1, c2 = 0, 1
    ci = [c1, c2]
    phi = Phi(h, ci)
    a = [
            [0, 0],
            [0, 0],
    ]
    b = [
            [0, -phi(2)],
    ]
    gen_first_col_exp(a, b, ci, phi)
    return ci, a, b


def _abnorsett_3m(h):
    c1, c2, c3 = 0, 0, 1
    ci = [c1, c2, c3]
    phi = Phi(h, ci)
    a = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
    ]
    b = [
            [0, -2*phi(2) - 2*phi(3), (1/2)*phi(2) + phi(3)],
    ]
    gen_first_col_exp(a, b, ci, phi)
    return ci, a, b


def _abnorsett_4m(h):
    c1, c2, c3, c4 = 0, 0, 0, 1
    ci = [c1, c2, c3, c4]
    phi = Phi(h, ci)
    a = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
    ]
    b = [
            [0, 
            -3*phi(2) - 5*phi(3) - 3*phi(4),
            (3/2)*phi(2) + 4*phi(3) + 3*phi(4),
            (-1/3)*phi(2) - phi(3) - phi(4),
            ],
    ]
    gen_first_col_exp(a, b, ci, phi)
    return ci, a, b


# name -> builder. The dropdown label is "RES4LYF <name>"; the folder prefix
# upstream shows ("exponential/") is implicit, since everything here is one.
TABLEAUX = {
    "res_2s": _res_2s,
    "res_2s_stable": _res_2s_stable,
    "res_3s": _res_3s,
    "res_3s_non-monotonic": _res_3s_non_monotonic,
    "res_3s_alt": _res_3s_alt,
    "res_3s_cox_matthews": _res_3s_cox_matthews,
    "res_3s_lie": _res_3s_lie,
    "res_3s_sunstar": _res_3s_sunstar,
    "res_3s_strehmel_weiner": _res_3s_strehmel_weiner,
    "res_4s_krogstad": _res_4s_krogstad,
    "res_4s_krogstad_alt": _res_4s_krogstad_alt,
    "res_4s_strehmel_weiner": _res_4s_strehmel_weiner,
    "res_4s_strehmel_weiner_alt": _res_4s_strehmel_weiner_alt,
    "res_4s_cox_matthews": _res_4s_cox_matthews,
    "res_4s_cfree4": _res_4s_cfree4,
    "res_4s_friedli": _res_4s_friedli,
    "res_4s_minchev": _res_4s_minchev,
    "res_4s_munthe-kaas": _res_4s_munthe_kaas,
    "res_5s": _res_5s,
    "res_5s_hochbruck-ostermann": _res_5s_hochbruck_ostermann,
    "res_6s": _res_6s,
}

# Exponential Adams-Bashforth. Unlike everything above, these take no extra
# stages: b[j] weights the epsilon from j steps back, so one model call per step
# whatever the order. They are written for a CONSTANT h, which a geometric
# sigma schedule gives and karras/exponential schedules do not - see NOTES.
MULTISTEP_TABLEAUX = {
    "abnorsett_2m": _abnorsett_2m,
    "abnorsett_3m": _abnorsett_3m,
    "abnorsett_4m": _abnorsett_4m,
}


def build(name: str, h: float):
    """(ci, a, b) for this method at this step size."""
    builder = TABLEAUX.get(name) or MULTISTEP_TABLEAUX.get(name)
    if builder is None:
        raise KeyError(f"unknown exponential tableau: {name}")
    return builder(float(h))


def stage_count(name: str) -> int:
    """Model evaluations per step.

    One per node: stage 0 reuses the denoised output the caller already has, and
    every later stage needs its own call. reForge needs this to keep prompt
    scheduling in step, because CFGDenoiser.step counts model calls while the
    stock SamplerData.total_steps only ever doubles.
    """
    ci, _, _ = build(name, 0.5)
    return len(ci)
