from os import makedirs, listdir
from os.path import isdir
from glob import glob

def check_folder(folder):
    if not isdir(folder):
        makedirs(folder)
        return folder
    else:
        if folder[-1] == ')':
            j = int(folder[-2:-1])+1
            foldernew = folder[:-3]+'('+str(j)+')'
            foldernew = check_folder(foldernew)
            return foldernew
        else:
            foldernew = folder+'('+str(1)+')'
            foldernew = check_folder(foldernew)
            return foldernew


def get_all_filenames(dir,prefix=''):
    dirlist = glob(dir+'/'+prefix)
    return dirlist


def get_all_subfolders(dir,substr=''):
    subfolders = [f for f in listdir(dir) if (isdir(dir+f) and substr in f)]
    return subfolders


