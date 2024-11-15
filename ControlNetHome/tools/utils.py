import os
import glob 
import sys 

def ckpt_search() -> str :
    '''
    searches through lightning_logs for the models dict of the latest code version. 
    '''
    path = os.curdir + '/lightning_logs/version_*'
    versions = glob.glob(path) 
    checkpoints = []
    while len(checkpoints) == 0:
        try:
            ver = versions.pop()
        except Exception as e:
            print(f'Error : {e} occured. No existing ckpt of previous model version exists !')
            sys.exit(1)

        checkpoints = glob.glob(ver + '/checkpoints/epoch=*')
    return checkpoints[0]