import numpy as np
import matplotlib.pyplot as plt

def plot_errbar_all(names,dict_res,times,res_np,fs=14,annosize=11,wt='bold',savename='',title='',xscale='linear',yscale0='linear',basex=1,basey0=1,yscale1='linear',basey1=1,loc='best',xlim=None,ylim=None,b_leg=False):
    fig, ax = plt.subplots(1,3,figsize=(18,4))
    for i in range(3):
        name = names[i]
        data = dict_res[name]
        var = data[0]
        data2 = data[2]
        ax[i].plot(times,res_np,color='lightgreen')
        bar = ax[i].errorbar(var,data[1],yerr=data2,capsize=5,capthick=2,color='cornflowerblue',linestyle='',lw=2)
        ax[i].title.set_text(name)
        ax[i].title.set_size(fontsize=fs)
        ax[i].tick_params(axis='both',labelsize=12)
        ax[i].set_ylim(ylim)
        ax[i].set_xlabel('time (minutes)',fontsize=fs)
        ax[0].set_ylabel('distance (km)',fontsize=fs)
    plt.subplots_adjust(wspace=0.15)
    if savename != '':
        plt.savefig(savename,bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    plt.clf()


def gen_x(T):
    sc = 1
    dict_loc = {}
    dict_loc[0] = 300
    dict_loc[sc*30-1] = 290
    dict_loc[2*sc*30-1] = 280
    dict_loc[3*sc*30-1] = 225
    dict_loc[4*sc*30-1] = 175
    dict_loc[5*sc*30-1] = 160
    dict_loc[7*sc*30-1] = 95
    dict_loc[8*sc*30-1] = 35
    dict_loc[9*sc*30-1] = 15
    dict_loc[10*sc*30-1] = 5
    x = np.zeros(T)
    x[0] = dict_loc[0]
    arr_step_main = list(dict_loc.keys())
    num = len(arr_step_main)
    step0 = 0
    l0 = x[0]
    for i in range(1,num):
        step1 = arr_step_main[i]
        x[step1] = dict_loc[step1]
        l1 = dict_loc[step1]
        dd = (l0-l1)/(step1-step0)
        for j in range(1,step1-step0):
            x[step0+j] = l0-dd*j
        l0 = l1
        step0 = step1
    return x


def eta_np(x,m,times,stperh=60):
    arr_res = []
    x0 = x[0]
    step = 0
    for i in range(1,m):
        dt = times[i]
        x1 = x[dt]
        delta = dt-times[i-1]
        step = step+delta/stperh
        sp = (x0-x1)/step
        eta = x1/sp
        arr_res.append(eta)
    return np.array(arr_res)

def dist_budget(x,B,m,times,gamma,beta0=0.1,stperh=60,seed=2027):
    np.random.seed(seed=seed)
    eps = B/m/5
    arr_res = []
    arr_bar = np.zeros((2,m))
    x_tilde_0 = x[0]+np.random.laplace(0,1)/eps
    arr_res.append(x_tilde_0)
    bb = np.log(1.0/beta0)
    arr_bar[0,0]=bb/eps
    arr_bar[1,0]=bb/eps
    B_hat = B-eps
    prev_x = x_tilde_0
    for i in range(1,m):
        assert(B_hat>=0.0)
        eps = B_hat/(m-i)
        dt = times[i]
        xt_1 = x[dt]
        delta = dt-times[i-1]
        gamma1 = gamma*delta/stperh
        ss = prev_x/gamma1
        z = np.random.laplace(0,1)
        if ss > 1.0 and i < m-1:
            x_tilde_1 = xt_1+ss*z/eps
            B_hat = B_hat - eps/ss
            arr_bar[0,i]=bb/(eps/ss)
            arr_bar[1,i]=bb/(eps/ss)
        else:
            x_tilde_1 = xt_1+z/eps
            B_hat = B_hat - eps
            arr_bar[0,i]=bb/eps
            arr_bar[1,i]=bb/eps
        arr_res.append(x_tilde_1)
        prev_x = x_tilde_1
    assert(B_hat>=0.0)
    return list(times), np.array(arr_res), arr_bar

def dist_base1(x,B,m,times,gamma,beta0=0.1,stperh=60,seed=2027):
    np.random.seed(seed=seed)
    eps = B/m
    arr_res = []
    arr_bar = np.zeros((2,m))
    x_tilde_0 = x[0]+np.random.laplace(0,1)/eps
    arr_res.append(x_tilde_0)
    bb = np.log(1.0/beta0)
    arr_bar[0,0]=bb/eps
    arr_bar[1,0]=bb/eps
    B_hat = B-eps
    for i in range(1,m):
        eps = B_hat/(m-i)
        dt = times[i]
        xt_1 = x[dt]
        z = np.random.laplace(0,1)
        x_tilde_1 = xt_1+z/eps
        B_hat = B_hat - eps
        arr_res.append(x_tilde_1)
        arr_bar[0,i]=bb/eps
        arr_bar[1,i]=bb/eps
    assert(B_hat>=0.0)
    return list(times), np.array(arr_res), arr_bar


def dist_base2(x,B,m,times,gamma,beta0=0.1,stperh=60,seed=2027):
    np.random.seed(seed=seed)
    eps = B/m*3
    arr_res = []
    arr_bar = np.zeros((2,int(m/3)))
    x_tilde_0 = x[0]+np.random.laplace(0,1)/eps
    arr_res.append(x_tilde_0)
    bb = np.log(1.0/beta0)
    arr_bar[0,0]=bb/eps
    arr_bar[1,0]=bb/eps
    B_hat = B-eps
    jend = int(m/3)
    for i in range(1,m):
        if B_hat >= eps*0.99:
            dt = times[i]
            xt_1 = x[dt]
            z = np.random.laplace(0,1)
            x_tilde_1 = xt_1+z/eps
            B_hat = B_hat - eps
            arr_res.append(x_tilde_1)
            arr_bar[0,i]=bb/eps
            arr_bar[1,i]=bb/eps
        else:
            break
    assert(B_hat>=-1e-11)
    return list(times[:jend]), np.array(arr_res), arr_bar

x = gen_x(T=300)
m = 15
times = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]

seed = 2027
beta0 = 0.1
B = 1
spmax = 120

times_adapt, res_adapt, bar_adapt = dist_budget(x,B,m,times,spmax,beta0=beta0,stperh=60,seed=seed)
times_unif, res_unif, bar_unif = dist_base1(x,B,m,times,spmax,beta0=beta0,stperh=60,seed=seed)
times_agg, res_agg, bar_agg = dist_base2(x,B,m,times,spmax,beta0=beta0,stperh=60,seed=seed)

dict_res = {}
dict_res['adaptive'] = (times_adapt, res_adapt,bar_adapt)
dict_res['uniform'] = (times_unif, res_unif,bar_unif)
dict_res['small scale'] = (times_agg, res_agg,bar_agg)
names = list(dict_res.keys())
ylim = (np.min(res_adapt-bar_adapt[0])-70,np.max(res_adapt+bar_adapt[1])+30)
savename = ''

res_np = np.array(x)[times]
plot_errbar_all(names,dict_res,times,res_np,ylim=ylim,savename=savename)