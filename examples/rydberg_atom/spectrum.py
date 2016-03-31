import math
import numpy as np
import pymongo
import os
import subprocess
import socket
import time
import shutil
import glob
from timeit import default_timer as timer


for i in np.arange(5.25,5.5,0.01):

    np.savetxt('quantInput',[i],newline=' ',fmt='%f')
    subprocess.call(["../../bin/nek",'4'])
    try:
        spec_tmp = np.loadtxt('fort.11')
    except:
        print("No fort.11!")
        break

    try:
        spectrum = np.vstack((spectrum,spec_tmp))
    except:
        spectrum = spec_tmp

    np.savetxt('fullSpectrum',spectrum,fmt='%e')
