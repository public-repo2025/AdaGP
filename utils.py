import numpy as np
from datetime import datetime

def clip_unit(a):
    if a > 1.0:
        return 1.0
    elif a < 0.0:
        return 0.0
    else:
        return a

def proj_to_line(u,v,x):
    a = (x[0]-v[0])*(u[0]-v[0])+(x[1]-v[1])*(u[1]-v[1])
    a = a/((u[0]-v[0])*(u[0]-v[0])+(u[1]-v[1])*(u[1]-v[1]))
    return a
    
def dist_to_line(u,v,x):
    a = proj_to_line(u,v,x)
    pa = a*u+(1.0-a)*v
    return np.linalg.norm(pa-x)

def dist_to_line_clip(a,u,v,x):
    p = clip_unit(a)
    pa = p*u+(1.0-p)*v
    return np.linalg.norm(pa-x)

def dist_to_rect(ul,ur,dl,dr,x):
    p_ulr = proj_to_line(ul,ur,x)
    p_udl = proj_to_line(ul,dl,x)
    p_udr = proj_to_line(ur,dr,x)
    p_dlr = proj_to_line(dl,dr,x)
    dist_ulr = dist_to_line_clip(p_ulr,ul,ur,x)
    dist_udl = dist_to_line_clip(p_udl,ul,dl,x)
    dist_udr = dist_to_line_clip(p_udr,ur,dr,x)
    dist_dlr = dist_to_line_clip(p_dlr,dl,dr,x)
    dist_min = min(dist_ulr,dist_udl,dist_udr,dist_dlr)
    b_out = 1.0
    if (0 <= p_ulr and p_ulr <= 1) and (0 <= p_udl and p_udl <= 1) and (0 <= p_udr and p_udr <= 1) and (0 <= p_dlr and p_dlr <= 1):
        b_out = -1.0
    return b_out*dist_min

def budget_rho_min(Bvec,j,m,n):
    rho_j = np.min(Bvec)/(m-j)
    rhos = [rho_j for i in range(n)]
    return rhos

def budget_rho_rem(Bvec,j,m,n):
    rhos = [Bvec[i]/(m-j) for i in range(n)]
    return rhos

def Bvec_init(B,n):
    Bvec = [B for i in range(n)]
    return Bvec


def compute_jaccard(s1,s2):
    s1n2 = s1.intersection(s2)
    s2u2 = s1.union(s2)
    if len(s2u2) > 0:
        jacc = len(s1n2)/len(s2u2)
    else:
        jacc = 1.0
    return jacc


def compute_knn_dist(x, res_mech, dist_p):
    k = len(res_mech)
    dist_mech = np.sum([dist_p(x[res_mech[j]]) for j in range(k)])
    return dist_mech


def get_timestamps(dsname='TDrive', hs=[12,16,20]):
    fyear = 2008
    times = []
    if dsname == 'SFCab':
        for dt in range(17,32):
            for h in hs:
                ti = datetime(fyear,5,dt,h,0,0)
                times.append(ti)
        for dt in range(1,15):
            for h in hs:
                ti = datetime(fyear,6,dt,h,0,0)
                times.append(ti)
    elif dsname == 'TDrive':
        for dt in range(2,9):
            for h in hs:
                ti = datetime(fyear,2,dt,h,0,0)
                times.append(ti)
    return times


def get_grid_verts(cw=10,x_min=1.29e7,x_max = 1.30e7,y_min = 4.8e6,y_max = 4.9e6):
    ww = (x_max-x_min)/(cw+1)
    ll = (y_max-y_min)/(cw+1)
    verts = []
    for i in range(1,cw+1):
        for j in range(1,cw+1):
            vij = np.array([i*ww+x_min, j*ll+y_min])
            verts.append(vij)
    return verts


def get_traj_dt(filename, arr_times):
    dict_traj_dt = {}
    with open(filename,'r') as f:
        for srow in f:
            row = srow.split(';')
            dt = row[0]
            if dt in arr_times:
                loc_str = row[1].split(',')
                dt_locs = [np.fromstring(loc[1:-1],sep=' ') for loc in loc_str]
                dict_traj_dt[dt] = dt_locs
    return dict_traj_dt

def select_traj_dt(dict_traj,nj,seed=0):
    arr_dt = list(dict_traj.keys())
    n0 = len(dict_traj[arr_dt[0]])
    np.random.seed(seed=seed)
    inds = np.random.randint(0,n0,size=nj)
    dict_traj_select = {}
    for dt in arr_dt:
        xt = dict_traj[dt]
        xs = np.array(xt)[inds]
        dict_traj_select[dt] = xs
    return dict_traj_select


def get_rect_p(m, cw=10, x_min=1.29e7,x_max = 1.30e7,y_min = 4.8e6,y_max = 4.9e6, seed=0):
    verts = get_grid_verts(cw=cw,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
    np.random.seed(seed=seed)
    inds = np.random.randint(low=0,high=len(verts),size=m)
    ps = np.array(verts)[inds]
    return ps

def make_rect(w, l, ps):
    rects = []
    for p in ps:
        ul = p+np.array([-0.5*w,0.5*l])
        ur = p+np.array([0.5*w,0.5*l])
        dl = p+np.array([-0.5*w,-0.5*l])
        dr = p+np.array([0.5*w,-0.5*l])
        rect = (dl,dr,ul,ur)
        rects.append(rect)
    return rects


