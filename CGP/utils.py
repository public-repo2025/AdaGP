import numpy as np
from scipy.special import erfinv


def get_gauss_noise(d=2):
    Z = np.random.normal(0,1,size=d)
    return Z


def get_threshold_shift(b_adjust,args=[]):
    if not(b_adjust):
        return 0.0
    [sigma,[w, l]] = args
    lw = l+w
    if (l>2.0*sigma and w>2.0*sigma):
        a = 8.0*sigma*sigma
        b = 4.0*sigma*lw
        ss = 4.0*sigma*np.sqrt(lw*lw-2.0*a)
        r1 = (b-ss)/2.0/a
        r2 = (b+ss)/2.0/a
    else:
        a = 4.0*sigma*sigma
        b = 2.0*a+2.0*sigma*lw
        c = a+2.0*sigma*lw-l*w
        ss = np.sqrt(b*b-4.0*a*c)
        r1 = (b-ss)/2.0/a
        r2 = (b+ss)/2.0/a
    if not(r1>0.0 and r1<1.0 and r2>1.0):
        print('assertion failed: l=',l,'; w=',w,'; r1,r2=',r1,r2,'; sigma=',sigma)
    shift = -r1*sigma
    return shift

def width_gauss(beta,sigma):
    #v = sigma*np.sqrt(2.0*np.log(2.0/beta))
    factor = np.sqrt(2.0)*erfinv(2.0*(1.0-beta/2.0)-1.0)
    assert(factor<np.sqrt(2.0*np.log(2.0/beta)))
    v = sigma*factor
    return v

def radius_gauss(beta, sigma):
    v = sigma*np.sqrt(2.0*np.log(1.0/beta))
    return v

def CGP_dist(x, rho, distfunc):
    Z = get_gauss_noise(d=1)[0]
    dist = distfunc(x)
    y = dist + 1.0/np.sqrt(2.0*rho)*Z
    return y


def CGP_Loc(z,rho):
    noise = get_gauss_noise()
    z_p = z + 1.0/np.sqrt(2.0*rho)*noise
    return z_p
  
