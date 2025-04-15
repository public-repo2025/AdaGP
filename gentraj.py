import numpy as np
from datetime import datetime
from bisect import bisect_left
from io_utils import get_all_filenames
###################################################### preprocessing only ############################################


def deg_to_rad(deg):
    rad = deg/360.*2*np.pi
    return rad

def convert_coord(lat, long, long0=0, R=6371000.0):
    x = R*(long-long0)
    y = R*np.log(np.tan(0.25*np.pi+0.5*lat))
    return [x, y]

def nearest_stamp_loc(t0, traj, name):
    ts = traj[:,2]
    inext = bisect_left(ts, t0)
    if inext != len(ts):
        if abs(ts[inext]-t0) < abs(ts[inext-1]-t0):
            inear = inext
        else:
            inear = inext-1
    else:
        inear = inext-1
    return np.array(traj[inear][:2],dtype=np.float64)

def loc_stamp_all(dict_traj, t0):
    loc_all = []
    for name in dict_traj.keys():
        traj = dict_traj[name]
        loc = nearest_stamp_loc(t0,traj,name)
        loc_all.append(loc)
    return np.array(loc_all)

def check_valid_taxi(x, y, x_min=1.28e7, x_max=1.31e7, y_min=4.7e6,y_max=5.0e6):
    if (x_min <= x and x <= x_max) and (y_min <= y and y <= y_max):
        return True
    else:
        return False
    
def extract_taxi_data_fromfile(filename,b_clean=True,time_max=86400,sp_max=3000,R=6371000.0):
    data = np.genfromtxt(filename,names=None,delimiter=',',dtype=[int,'S19',float,float])
    trace = []
    if data.size == 1:
        row = data.item()
        time = datetime.fromisoformat(row[1].decode())
        (x, y) = convert_coord(deg_to_rad(row[3]),deg_to_rad(row[2]),R=R)
        if np.isnan(x) or np.isnan(y):
            print('NAN: ',row, filename)
        else:
            if b_clean:
                if check_valid_taxi(x, y): #x>1e7:
                    trace.append([x, y,time])
            else:
                trace.append([x, y,time])
        return np.array(trace)
    for row in data:
        time = datetime.fromisoformat(row[1].decode())
        (x, y) = convert_coord(deg_to_rad(row[3]),deg_to_rad(row[2]),R=R)
        if np.isnan(x) or np.isnan(y):
            print('NAN encountered; loc exluded: ',row, filename)
        else:
            trace.append([x, y,time])
    if b_clean and len(trace)>1:
        tmp = np.array(trace)
        ind_sorted = tmp[:,2].argsort()
        n = len(ind_sorted)
        prev_time = tmp[0][2]
        prev_loc = [tmp[0][0],tmp[0][1]]
        trace1 = []
        jj = 0
        while jj<len(trace) and not(check_valid_taxi(tmp[jj][0], tmp[jj][1])): 
            jj = jj+1
        if jj == len(trace):
            return np.array(trace1)
        trace1.append(tmp[jj])
        prev_time = tmp[jj][2]
        prev_loc = [tmp[jj][0],tmp[jj][1]]
        for i in range(jj,n):
            ind = ind_sorted[i]
            row = trace[ind]
            curr_time = row[2]
            time_delta = (curr_time - prev_time).seconds
            if time_delta <= time_max and time_delta > 0:
                (x, y) = (row[0], row[1])
                if np.sqrt((x-prev_loc[0])**2+(y-prev_loc[1])**2)/time_delta*60 <= sp_max and check_valid_taxi(x, y): 
                    trace1.append(row)
                    prev_time = curr_time
                    prev_loc = (x, y)
        trace1 = np.array(trace1)
    else:
        trace1 = np.array(trace)
    return trace1


def extract_cab_data_all(folder_in,suff='new_*.txt',len_min=100,len_max=100000,R=6371000.0,extract_func=extract_taxi_data_fromfile):
    dict_traj = {}
    trip_names = get_all_filenames(folder_in,suff)
    for name in trip_names:
        x = extract_func(name,R=R)
        n = len(x)
        if n < len_min or n > len_max:
            continue
        name_short = name.replace(folder_in,'').replace('/','').replace('\\','')
        dict_traj[name_short] = x
    return dict_traj

def get_timestamps_taxi(hs=[12,16,20]):
    fyear = 2008
    times = []
    for dt in range(2,9):
        for h in hs:
            ti = datetime(fyear,2,dt,h,0,0)
            times.append(ti)
    return times

dsname = 'TDrive'
folder_out='./data/'
filename_out = folder_out+dsname+'_traj_dt.txt'

folder_in = './data/taxi_log_2008_by_id/'
extract_func = extract_taxi_data_fromfile

dict_traj = extract_cab_data_all(folder_in=folder_in,suff='*.txt',extract_func=extract_func)
arr_hs_all = [j for j in range(24)]
arr_times_all = get_timestamps_taxi(hs=arr_hs_all)
m = len(arr_times_all)

ftraj = open(filename_out,'w')
for j in range(m):
    dt = arr_times_all[j]
    x = loc_stamp_all(dict_traj,dt)
    n = len(x)
    traj_line = ','.join([str(x[i]) for i in range(n)])
    ftraj.write(str(dt)+';'+traj_line+'\n')
ftraj.close()
