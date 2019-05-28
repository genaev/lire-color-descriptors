import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.append('..')
from LireColorDescriptors.Features import GlobalFeatures

import tempfile
import shutil


class Tsv2Df():

    def __init__(self, tsv):
        self.tsv = tsv
        self.out = None

    def to2dd(df_part):
        X = df_part.X - df_part.X.min()
        Y = df_part.Y - df_part.Y.min()
        array2d = np.zeros((X.max() + 1, Y.max() + 1, 3))
        array2d[X, Y] = np.array(df_part[['R', 'G', 'B']] / 255)
        return array2d

    def extract(self):
        df = pd.read_csv(self.tsv, sep="\t", names=['Class', 'Seed', 'R', 'G', 'B', 'X', 'Y', 'Cluster_x', 'Cluster_y'],
                         header=0)
        data = df.groupby(['Class', 'Seed']).apply(Tsv2Df.to2dd).reset_index()

        dirpath = tempfile.mkdtemp()
        names = []
        for i in range(len(data)):
            name = '{}/{}_{}.jpg'.format(dirpath, data.Class[i], data.Seed[i])
            names.append(name)
            plt.imsave(name, data.iloc[i, 2])

        gfs = []
        for name in names:
            gfs.append(GlobalFeatures(name).extract())
        df_gf = pd.DataFrame(gfs)
        shutil.rmtree(dirpath)

        df_gf = pd.concat([df_gf[column].apply(pd.Series).add_prefix(column) for column in df_gf.columns], axis=1)
        df_gf = pd.concat([data[['Class', 'Seed']], df_gf], axis=1)
        self.out = df_gf
        return self.out