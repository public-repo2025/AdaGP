import numpy as np
from CGP.utils import CGP_Loc, CGP_dist, get_threshold_shift, radius_gauss, width_gauss

def RCBase0(x,rhos,Bvec,func_proj,args=[]):
    n = len(x)
    l_tilde = []
    S11 = []
    for i in range(n):
        xi_tilde = CGP_Loc(x[i],rhos[i])
        li_tilde = func_proj(xi_tilde)
        l_tilde.append(li_tilde)
        Bvec[i] = Bvec[i]-rhos[i]
        if li_tilde < 0.0:
            S11.append(i)
    return S11, 1

def RCBase1(x,rhos,Bvec,func_proj,args=[],b_adjust=False):
    n = len(x)
    l_bar = []
    S11 = []
    for i in range(n):
        xi_tilde = CGP_dist(x[i],rhos[i],distfunc=func_proj)
        l_bar.append(xi_tilde)
        Bvec[i] = Bvec[i]-rhos[i]
        ri = get_threshold_shift(b_adjust=b_adjust,args=[np.sqrt(1.0/(2.0*rhos[i])),args])
        if xi_tilde < ri:
            S11.append(i)
    return S11, 1


def PRCIE_ind_log(x,func_proj,c,rhos,beta,Bvec,args=[],b_adjust=False):
    n = len(x)
    G_curr = [i for i in range(n)]
    G_prev = list(G_curr)
    l_bar = [0 for i in range(n)]
    c1 = 2**c-1
    S0 = []
    S1 = []
    nj_prev = 0.0
    j = 1
    dict_rho = {}
    dict_wj = {}
    for i in G_prev:
        dict_rho[i] = rhos[i]/c1
        sigma_i = np.sqrt(1.0/(2.0*dict_rho[i]))
        dict_wj[i] = width_gauss(beta/n,1.0)*sigma_i
    while (j<=c) and len(G_prev)>0:
        aj = 2**(j-1)
        nj = nj_prev+aj
        G_curr = []
        for i in G_prev:
            rho_ij = dict_rho[i]
            xi_tilde = CGP_dist(x[i],rho_ij,func_proj)
            Bvec[i] = Bvec[i]-rho_ij
            l_bar[i] = (l_bar[i]*nj_prev+aj*xi_tilde)/nj
            w_i = dict_wj[i]/np.sqrt(nj)
            if l_bar[i] < -w_i:
                S1.append(i)
            elif l_bar[i] > w_i:
                S0.append(i)
            else:
                G_curr.append(i)
        G_prev = list(G_curr)
        nj_prev = nj
        for i in G_prev:
            dict_rho[i] = dict_rho[i]*2.0
        j = j + 1
    S11 = list(S1)
    for i in G_prev:
        ri = get_threshold_shift(b_adjust=b_adjust,args=[np.sqrt(1.0/(2.0*rhos[i])*c1/nj_prev),args])
        if l_bar[i] < ri:#
            S11.append(i)
    return S11, j-1



def PRCIE_ind_loc_log(x,func_proj,c,rhos,beta,Bvec,args=[]):
    n = len(x)
    G_curr = [i for i in range(n)]
    G_prev = list(G_curr)
    x_bar = [np.array([0.0,0.0]) for i in range(n)]
    l_bar = [0.0 for i in range(n)]
    c1 = 2**c-1
    S0 = []
    S1 = []
    nj_prev = 0.0
    j = 1
    dict_rho = {}
    dict_wj = {}
    for i in G_prev:
        dict_rho[i] = rhos[i]/c1
        sigma_i = np.sqrt(1.0/(2.0*dict_rho[i]))
        dict_wj[i] = radius_gauss(beta/n,1.0)*sigma_i
    while (j<=c) and len(G_prev)>0:
        aj = 2**(j-1)
        nj = nj_prev+aj
        G_curr = []
        for i in G_prev:
            rho_ij = dict_rho[i]
            xi_tilde = CGP_Loc(x[i],rho_ij)#
            Bvec[i] = Bvec[i]-rho_ij
            x_bar[i] = (x_bar[i]*nj_prev+aj*xi_tilde)/nj
            li_tilde = func_proj(x_bar[i])
            l_bar[i] = li_tilde
            w_i = dict_wj[i]/np.sqrt(nj)
            if li_tilde < -w_i:
                S1.append(i)
            elif li_tilde > w_i:
                S0.append(i)
            else:
                G_curr.append(i)
        G_prev = list(G_curr)
        nj_prev = nj
        for i in G_prev:
            dict_rho[i] = dict_rho[i]*2.0
        j = j+1
    S11 = list(S1)
    for i in G_prev:
        if l_bar[i] < 0.0:
            S11.append(i)
    return S11, j-1



def kNNBase0(x,k,rhos,Bvec,dist_p):
    n = len(x)
    l_tilde = []
    for i in range(n):
        xi_tilde = CGP_Loc(x[i],rhos[i])
        li_tilde = dist_p(xi_tilde)
        l_tilde.append(li_tilde)
        Bvec[i] = Bvec[i]-rhos[i]
    ts = np.argsort(l_tilde)[:k]
    return ts, 1

def kNNBase1(x,k,rhos,Bvec,dist_p):
    n = len(x)
    l_bar = []
    for i in range(n):
        xi_tilde = CGP_dist(x[i],rhos[i],distfunc=dist_p)
        l_bar.append(xi_tilde)
        Bvec[i] = Bvec[i]-rhos[i]
    ts = np.argsort(l_bar)[:k]
    return ts, 1

def kPNNIE_ind_loc_log(x,k,c,rhos,beta,Bvec,dist_p):
    n = len(x)
    G_curr = [i for i in range(n)]
    G_prev = list(G_curr)
    x_bar = [np.array([0.0,0.0]) for i in range(n)]
    l_bar = [0 for i in range(n)]
    c1 = 2**c-1
    beta_j = beta/(2*k*(c-1))
    w_0j = radius_gauss(beta_j,1.0)
    dict_rho = {}
    dict_sigma = {}
    for i in G_prev:
        dict_rho[i] = rhos[i]/c1
        sigma_i = np.sqrt(1.0/(2.0*dict_rho[i]))
        dict_sigma[i] = sigma_i
    nj_prev = 0.0
    j = 1       
    while j<c and len(G_prev)>k:
        aj = 2**(j-1)
        nj = nj_prev+aj
        l_bar_right = []
        for i in G_prev:
            rho_ij = dict_rho[i]
            xi_tilde = CGP_Loc(x[i],rho_ij)#
            Bvec[i] = Bvec[i]-rho_ij
            x_bar[i] = (x_bar[i]*nj_prev+aj*xi_tilde)*1.0/nj
            l_bar[i] = dist_p(x_bar[i])
            w_ij = w_0j*dict_sigma[i]/np.sqrt(nj)
            l_bar_right.append(l_bar[i]+w_ij)
        inds = np.argsort(l_bar_right)
        ts = np.array(G_prev)[inds][:k]
        G_curr = list(ts)
        t_k = ts[k-1]
        w_tk =  w_0j*dict_sigma[t_k]/np.sqrt(nj)
        x_tk_right  = l_bar[t_k]+w_tk
        for i in G_prev:
            w_ij = w_0j*dict_sigma[i]/np.sqrt(nj)
            if l_bar[i]-w_ij <= x_tk_right and not(i in G_curr):
                G_curr.append(i)
        G_prev = list(G_curr)
        nj_prev = nj
        for i in G_prev:
            dict_rho[i] = dict_rho[i]*2.0
        j = j + 1

    aj = 2**(j-1)
    nj = nj_prev+aj
    l_bar_right = []
    for i in G_prev:
        rho_ij = dict_rho[i]
        xi_tilde = CGP_Loc(x[i],rho_ij)#
        Bvec[i] = Bvec[i]-rho_ij
        x_bar[i] = (x_bar[i]*nj_prev+aj*xi_tilde)*1.0/nj
        l_bar[i] = dist_p(x_bar[i])
        l_bar_right.append(l_bar[i])
    inds = np.argsort(l_bar_right)
    ts = np.array(G_prev)[inds][:k]
    return ts, j


def kPNNIE_ind_log(x,k,c,rhos,beta,Bvec,dist_p):
    n = len(x)
    G_curr = [i for i in range(n)]
    G_prev = list(G_curr)
    l_bar = [0 for i in range(n)]
    c1 = 2**c-1
    beta_j = beta/(2*k*(c-1))
    w_0j = width_gauss(beta_j,1.0)
    dict_rho = {}
    dict_sigma = {}
    for i in G_prev:
        dict_rho[i] = rhos[i]/c1
        sigma_i = np.sqrt(1.0/(2.0*dict_rho[i]))
        dict_sigma[i] = sigma_i
    nj_prev = 0.0
    j = 1       
    while j<c and len(G_prev)>k:
        aj = 2**(j-1)
        nj = nj_prev+aj
        l_bar_right = []
        for i in G_prev:
            rho_ij = dict_rho[i]
            xi_tilde = CGP_dist(x[i],rho_ij,distfunc=dist_p)
            Bvec[i] = Bvec[i]-rho_ij
            l_bar[i] = (l_bar[i]*nj_prev+aj*xi_tilde)/nj
            w_ij = w_0j*dict_sigma[i]/np.sqrt(nj)
            l_bar_right.append(l_bar[i]+w_ij)
        inds = np.argsort(l_bar_right)
        ts = np.array(G_prev)[inds][:k]
        G_curr = list(ts)
        t_k = ts[k-1]
        w_tk =  w_0j*dict_sigma[t_k]/np.sqrt(nj)
        x_tk_right  = l_bar[t_k]+w_tk
        for i in G_prev:
            w_ij = w_0j*dict_sigma[i]/np.sqrt(nj)
            if l_bar[i]-w_ij <= x_tk_right and not(i in G_curr):
                G_curr.append(i)
        G_prev = list(G_curr)
        nj_prev = nj
        for i in G_prev:
            dict_rho[i] = dict_rho[i]*2.0
        j = j + 1

    aj = 2**(j-1)
    nj = nj_prev+aj
    l_bar_right = []
    for i in G_prev:
        rho_ij = dict_rho[i]
        xi_tilde = CGP_dist(x[i],rho_ij,distfunc=dist_p)
        Bvec[i] = Bvec[i]-rho_ij
        l_bar[i] = (l_bar[i]*nj_prev+aj*xi_tilde)/nj
        l_bar_right.append(l_bar[i])
    inds = np.argsort(l_bar_right)
    ts = np.array(G_prev)[inds][:k]
    return ts, j



##########################################################################################################################################

def RCBase1_comp(x,rhos,Bvec,func_proj,args=[]):
    n = len(x)
    l_bar = []
    S11_0 = []
    S11_adj = []
    for i in range(n):
        xi_tilde = CGP_dist(x[i],rhos[i],distfunc=func_proj)
        l_bar.append(xi_tilde)
        Bvec[i] = Bvec[i]-rhos[i]

        if xi_tilde < 0.0:
            S11_0.append(i)
        ri = get_threshold_shift(b_adjust=True,args=[np.sqrt(1.0/(2.0*rhos[i])),args])
        if xi_tilde < ri:
            S11_adj.append(i)
    return S11_0, S11_adj, 1


def PRCIE_ind_log_comp(x,func_proj,c,rhos,beta,Bvec,args=[]):
    n = len(x)
    G_curr = [i for i in range(n)]
    G_prev = list(G_curr)
    l_bar = [0 for i in range(n)]
    c1 = 2**c-1
    S0 = []
    S1 = []
    nj_prev = 0.0
    j = 1
    dict_rho = {}
    dict_wj = {}
    for i in G_prev:
        dict_rho[i] = rhos[i]/c1
        sigma_i = np.sqrt(1.0/(2.0*dict_rho[i]))
        dict_wj[i] = width_gauss(beta/n,1.0)*sigma_i
    while (j<=c) and len(G_prev)>0:
        aj = 2**(j-1)
        nj = nj_prev+aj
        G_curr = []
        for i in G_prev:
            rho_ij = dict_rho[i]
            xi_tilde = CGP_dist(x[i],rho_ij,func_proj)
            Bvec[i] = Bvec[i]-rho_ij
            l_bar[i] = (l_bar[i]*nj_prev+aj*xi_tilde)/nj
            w_i = dict_wj[i]/np.sqrt(nj)
            if l_bar[i] < -w_i:
                S1.append(i)
            elif l_bar[i] > w_i:
                S0.append(i)
            else:
                G_curr.append(i)
        G_prev = list(G_curr)
        nj_prev = nj
        for i in G_prev:
            dict_rho[i] = dict_rho[i]*2.0
        j = j + 1
    S11_0 = list(S1)
    S11_adj = list(S1)
    for i in G_prev:
        if l_bar[i] < 0.0:
            S11_0.append(i)
        ri = get_threshold_shift(b_adjust=True,args=[np.sqrt(1.0/(2.0*rhos[i])*c1/nj_prev),args])
        if l_bar[i] < ri:#0.0:
            S11_adj.append(i)
    return S11_0, S11_adj, j-1

