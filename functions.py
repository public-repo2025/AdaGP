import numpy as np
from io_utils import check_folder
from utils import compute_jaccard, compute_knn_dist, dist_to_rect, make_rect


def test_rc_bymech(x,func_proj,Bvec,rhos,mech_func,args=[]):
    assert(np.sum(np.array(Bvec)>=0.0)==len(x))
    res, t = mech_func(x,rhos,Bvec,func_proj,args=args)
    return res, t

def test_rc(dict_traj, times, B, seed, ps, rect_w=3000, rect_l=3000, params_in = '', mechs = [], folder_out0 = './results/test_rc', b_interim=False, b_analysis=False):
    m = len(times)
    np.random.seed(seed=seed)
    rects = make_rect(rect_w, rect_l, ps)
    args = [rect_w, rect_l]
    n = len(dict_traj[str(times[0])])
    arr_names = []
    dict_budget_func = {}
    dict_func = {}
    dict_Bvec = {}
    dict_jacc = {}
    dict_err = {}
    dict_rho = {}
    arr_style = []
    for mech in mechs:
        name = mech[0]
        arr_names.append(name)
        dict_budget_func[name] = mech[2]
        dict_func[name] = mech[3]
        dict_Bvec[name] = mech[4](B,n)
        dict_jacc[name] = []
        dict_err[name] = []
        dict_rho[name] = []
        arr_style.append((mech[5],mech[6]))

    folder_out = check_folder(folder_out0)
    fwrite = open(folder_out+'/params.txt','w')
    params_out = params_in.replace('<m>',str(m)).replace('<n>',str(n)).replace('<pois>',str(ps))
    fwrite.write(params_out)
    fwrite.close()
    fwrite = open(folder_out+'/res.txt','w')
    headers = 'time;res_np;'
    headers = headers+';'.join(['res_'+name+'|avg_rho|t_end' for name in arr_names])
    fwrite.write(headers+'\n')
    fdtres = open(folder_out+'/res_by_dt.txt','w')
    headers_dt = 'time;count_np;'
    headers_dt = headers_dt+';'.join(['jacc_'+name for name in arr_names])+';'+';'.join(['err_'+name for name in arr_names])
    fdtres.write(headers_dt+'\n')
    for j in range(m):
        dt = times[j]
        x = np.array(dict_traj[str(dt)])
        (dl,dr,ul,ur) = rects[j]
        func_proj_p = lambda xi: dist_to_rect(ul,ur,dl,dr,xi)

        res_np = []
        for i in range(n):
            if func_proj_p(x[i]) < 0.0:
                res_np.append(i)
        count_np = len(res_np)
        line = str(dt)
        line = line+';'+'['+','.join([str(res) for res in res_np])+']'

        arr_err = []
        arr_res = []
        arr_jacc = []
        for name in arr_names:
            budget_func = dict_budget_func[name]
            rhos_mech = budget_func(dict_Bvec[name],j,m,n)
            mech_func = dict_func[name]
            res_mech, t_mech = test_rc_bymech(x,func_proj_p,dict_Bvec[name],rhos_mech,mech_func,args=args)
            line = line+';'+'['+','.join([str(res) for res in res_mech])+']'+'|'+str(np.average(rhos_mech))+'|'+str(t_mech)#line+';'+str(res_mech)+'|'+str(np.average(rhos_mech))+'|'+str(t_mech)
            err_mech = abs(len(res_mech)-count_np)
            jacc_mech = compute_jaccard(set(res_mech),set(res_np))
            arr_err.append(err_mech)
            arr_res.append(res_mech)
            arr_jacc.append(jacc_mech)
            dict_jacc[name].append(jacc_mech)
            dict_err[name].append(err_mech)
            dict_rho[name].append(np.average(rhos_mech))
        fwrite.write(line+'\n')
        fdtres.write(str(dt)+';'+str(count_np)+';'+';'.join([str(arr_jacc[s]) for s in range(len(arr_jacc))])+';'+';'.join([str(arr_err[s]) for s in range(len(arr_err))])+'\n')

    fdtres.close()

    line_final_jacc = 'Jaccard Index: '
    line_final_sumerr = 'Total Count Error: '
    for name in arr_names:
        line_final_jacc = line_final_jacc+name+'='+str(np.average(dict_jacc[name]))+';'
        line_final_sumerr = line_final_sumerr+name+'='+str(np.sum(dict_err[name]))+';'
    fwrite.write(line_final_jacc[0:-1]+'\n'+line_final_sumerr[0:-1])
    print(line_final_jacc)
    print(line_final_sumerr)
    fwrite.close()


def test_knn_bymech(x,k,dist_p,Bvec,rhos,mech_func):
    res, t = mech_func(x,k,rhos,Bvec,dist_p)
    return res, t


def test_knn(dict_traj, times, k, B, seed, ps, params_in = '', mechs = [], folder_out0 = './results/test_knn'):
    m = len(times)
    np.random.seed(seed=seed)
    n = len(dict_traj[str(times[0])])
    arr_names = []
    dict_budget_func = {}
    dict_func = {}
    dict_Bvec = {}
    dict_jacc = {}
    dict_err = {}
    dict_rho = {}
    arr_style = []
    for mech in mechs:
        name = mech[0]
        arr_names.append(name)
        dict_budget_func[name] = mech[2]
        dict_func[name] = mech[3]
        dict_Bvec[name] = mech[4](B,n)
        dict_jacc[name] = []
        dict_err[name] = []
        dict_rho[name] = []
        arr_style.append((mech[5],mech[6]))
    
    folder_out = check_folder(folder_out0)
    fwrite = open(folder_out+'/params.txt','w')
    params_out = params_in.replace('<m>',str(m)).replace('<n>',str(n)).replace('<pois>',str(ps))
    fwrite.write(params_out)
    fwrite.close()
    fwrite = open(folder_out+'/res.txt','w')
    headers = 'time;res_np;'
    headers = headers+';'.join(['res_'+name+'|avg_rho|t_end' for name in arr_names])
    fwrite.write(headers+'\n')
    fdtres = open(folder_out+'/res_by_dt.txt','w')
    headers_dt = 'time;'
    headers_dt = headers_dt+';'.join(['jacc_'+name for name in arr_names])+';'+';'.join(['err_'+name for name in arr_names])
    fdtres.write(headers_dt+'\n')
    for j in range(m):
        dt = times[j]
        x = np.array(dict_traj[str(dt)])
        dist_p = lambda xi: np.linalg.norm(xi-ps[j])
        
        res_np = np.argsort(np.linalg.norm(x-ps[j],axis=1))[:k]
        line = str(dt)
        line = line+';'+'['+','.join([str(res) for res in res_np])+']'
        dist_np = compute_knn_dist(x,res_np,dist_p)

        arr_jacc = []
        arr_err = []
        arr_res = []
        for name in arr_names:
            budget_func = dict_budget_func[name]
            rhos_mech = budget_func(dict_Bvec[name],j,m,n)
            mech_func = dict_func[name]
            res_mech, t_mech = test_knn_bymech(x,k,dist_p,dict_Bvec[name],rhos_mech,mech_func)
            line = line+';'+'['+','.join([str(res) for res in res_mech])+']'+'|'+str(np.average(rhos_mech))+'|'+str(t_mech)
            jacc_mech = compute_jaccard(set(res_mech),set(res_np))
            err_mech = compute_knn_dist(x,res_mech,dist_p)-dist_np
            arr_jacc.append(jacc_mech)
            arr_err.append(err_mech)
            arr_res.append(res_mech)
            dict_jacc[name].append(jacc_mech)
            dict_err[name].append(err_mech)
            dict_rho[name].append(np.average(rhos_mech))
        fwrite.write(line+'\n')
        fdtres.write(str(dt)+';'+str(dist_np)+';'+';'.join([str(arr_jacc[s]) for s in range(len(arr_jacc))])+';'+';'.join([str(arr_err[s]) for s in range(len(arr_err))])+'\n')

    fdtres.close()

    line_final_jacc = 'Jaccard Index: '
    line_final_sumerr = 'Total Distance Error: '
    for name in arr_names:
        line_final_jacc = line_final_jacc+name+'='+str(np.average(dict_jacc[name]))+';'
        line_final_sumerr = line_final_sumerr+name+'='+str(np.average(dict_err[name]))+';'
    fwrite.write(line_final_jacc[0:-1]+'\n'+line_final_sumerr[0:-1])
    print(line_final_jacc)
    print(line_final_sumerr)
    fwrite.close()



####################################################################################################################################

def test_rc_bymech_comp(x,func_proj,Bvec,rhos,mech_func,args=[]):
    assert(np.sum(np.array(Bvec)>=0.0)==len(x))
    res_0, res_adj, t = mech_func(x,rhos,Bvec,func_proj,args=args)
    return res_0, res_adj, t

def test_rc_comp(dict_traj, times, B, seed, ps, rect_w=3000, rect_l=3000, params_in = '', mechs = [], folder_out0 = './results/test_rc'):
    m = len(times)
    np.random.seed(seed=seed)
    rects = make_rect(rect_w, rect_l, ps)
    args = [rect_w, rect_l]
    n = len(dict_traj[str(times[0])])
    arr_names = []
    dict_budget_func = {}
    dict_func = {}
    dict_Bvec = {}
    dict_jacc = {}
    dict_err = {}
    dict_rho = {}
    for mech in mechs:
        name = mech[0]
        arr_names.append(name)
        dict_budget_func[name] = mech[2]
        dict_func[name] = mech[3]
        dict_Bvec[name] = mech[4](B,n)
        dict_jacc[name+'_0'] = []
        dict_err[name+'_0'] = []
        dict_rho[name+'_0'] = []
        dict_jacc[name+'_adj'] = []
        dict_err[name+'_adj'] = []
        dict_rho[name+'_adj'] = []

    folder_out = check_folder(folder_out0)
    fwrite = open(folder_out+'/params.txt','w')
    params_out = params_in.replace('<m>',str(m)).replace('<n>',str(n)).replace('<pois>',str(ps))
    fwrite.write(params_out)
    fwrite.close()
    fwrite = open(folder_out+'/res.txt','w')
    headers = 'time;res_np;'
    headers = headers+';'.join(['res_'+name+'_0|avg_rho|t_end;res_'+name+'_adj|avg_rho|t_end' for name in arr_names])
    fwrite.write(headers+'\n')
    fdtres = open(folder_out+'/res_by_dt.txt','w')
    headers_dt = 'time;count_np;'
    headers_dt = headers_dt+';'.join(['jacc_'+name+'_0;jacc_'+name+'_adj' for name in arr_names])+';'+';'.join(['err_'+name+'_0;err_'+name+'_adj' for name in arr_names])
    fdtres.write(headers_dt+'\n')
    for j in range(m):
        dt = times[j]
        x = np.array(dict_traj[str(dt)])
        (dl,dr,ul,ur) = rects[j]
        func_proj_p = lambda xi: dist_to_rect(ul,ur,dl,dr,xi)

        res_np = []
        for i in range(n):
            if func_proj_p(x[i]) < 0.0:
                res_np.append(i)
        count_np = len(res_np)
        line = str(dt)
        line = line+';'+'['+','.join([str(res) for res in res_np])+']'

        arr_err = []
        arr_res = []
        arr_jacc = []
        for name in arr_names:
            budget_func = dict_budget_func[name]
            rhos_mech = budget_func(dict_Bvec[name],j,m,n)
            mech_func = dict_func[name]
            res_mech_0, res_mech_adj, t_mech = test_rc_bymech_comp(x,func_proj_p,dict_Bvec[name],rhos_mech,mech_func,args=args)
            line = line+';'+'['+','.join([str(res) for res in res_mech_0])+']'+'|'+str(np.average(rhos_mech))+'|'+str(t_mech)
            line = line+';'+'['+','.join([str(res) for res in res_mech_adj])+']'+'|'+str(np.average(rhos_mech))+'|'+str(t_mech)#line+';'+str(res_mech)+'|'+str(np.average(rhos_mech))+'|'+str(t_mech)
            err_mech_0 = abs(len(res_mech_0)-count_np)
            err_mech_adj = abs(len(res_mech_adj)-count_np)
            jacc_mech_0 = compute_jaccard(set(res_mech_0),set(res_np))
            jacc_mech_adj = compute_jaccard(set(res_mech_adj),set(res_np))
            arr_err.append(err_mech_0)
            arr_err.append(err_mech_adj)
            arr_res.append(res_mech_0)
            arr_res.append(res_mech_adj)
            arr_jacc.append(jacc_mech_0)
            arr_jacc.append(jacc_mech_adj)
            dict_jacc[name+'_0'].append(jacc_mech_0)
            dict_jacc[name+'_adj'].append(jacc_mech_adj)
            dict_err[name+'_0'].append(err_mech_0)
            dict_err[name+'_adj'].append(err_mech_adj)
            dict_rho[name+'_0'].append(np.average(rhos_mech))
            dict_rho[name+'_adj'].append(np.average(rhos_mech))
        fwrite.write(line+'\n')
        fdtres.write(str(dt)+';'+str(count_np)+';'+';'.join([str(arr_jacc[s]) for s in range(len(arr_jacc))])+';'+';'.join([str(arr_err[s]) for s in range(len(arr_err))])+'\n')

    fdtres.close()

    line_final_jacc = 'Jaccard Index: '
    line_final_sumerr = 'Total Count Error: '
    for name in arr_names:
        line_final_jacc = line_final_jacc+name+'_0='+str(np.average(dict_jacc[name+'_0']))+';'
        line_final_jacc = line_final_jacc+name+'_adj='+str(np.average(dict_jacc[name+'_adj']))+';'
        line_final_sumerr = line_final_sumerr+name+'_0='+str(np.sum(dict_err[name+'_0']))+';'
        line_final_sumerr = line_final_sumerr+name+'_adj='+str(np.sum(dict_err[name+'_adj']))+';'
    fwrite.write(line_final_jacc[0:-1]+'\n'+line_final_sumerr[0:-1])
    print(line_final_jacc)
    print(line_final_sumerr)
    fwrite.close()