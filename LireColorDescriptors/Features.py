import subprocess
import inspect
import os
import re
import ast
import cv2
from scipy import stats
import numpy as np
import pandas as pd

class GlobalFeatures():

    def __init__(self, img):
        """Constructor"""
        self.jar = os.path.join(os.path.dirname(inspect.getfile(GlobalFeatures)), 'lirecolordescriptors.jar')
        self.img = img
        self.out = {}

    def tipical_mean(obj, sigma=1):
        return obj[(stats.zscore(obj) < sigma).all(axis=1)].mean()

    def GCH(obj, n):
        v_c = ((obj // (256 / n)).astype(str)).sum(axis=1).value_counts()
        tmp = {}
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tmp['{}.0{}.0{}.0'.format(i, j, k)] = 0

        tmp = pd.Series(tmp)
        tmp.update(v_c)
        return tmp / len(obj)

    def extract(self):
        args = ["java", "-jar", self.jar, self.img]
        print(args)
        process = subprocess.Popen(args, stdout=subprocess.PIPE)
        data = process.communicate()
        st = 0
        key = ""
        for line in data[0].decode("utf-8").splitlines():
            if (st == 1):
                self.out[key] = ast.literal_eval(line)
                st = 0
            m = re.search('(\w+):', line)
            if m:
                key = m.group(1)
                st = 1

        im = cv2.imread(self.img)
        im = im.reshape(im.shape[0] * im.shape[1], 3)
        im = im[~np.all(im < 20, axis=1)]  # remove zero lines 2-D numpy array
        im = pd.DataFrame(im)
        self.out['GCH2'] = GlobalFeatures.GCH(im, 2).tolist()
        self.out['GCH3'] = GlobalFeatures.GCH(im, 3).tolist()
        self.out['GCH4'] = GlobalFeatures.GCH(im, 4).tolist()
        self.out['Mean'] = GlobalFeatures.tipical_mean(im, 2).tolist()

        return self.out
