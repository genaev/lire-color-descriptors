import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import imutils
import cv2
from sklearn.decomposition import PCA

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

    def df2img(df):
        x = df['X'] - df['X'].min()
        y = df['Y'] - df['Y'].min()
        rgb = df[['R', 'G', 'B']].values
        img = np.zeros((x.max() + 1, y.max() + 1, 3))
        img[x, y, :] = rgb
        img = img.astype('uint8')

        pca = PCA()
        pca.fit(df[['X', 'Y']])

        angle = np.arcsin(pca.components_[0, 1]) * 180 / np.pi
        if pca.components_[0, 0] < 0:
            angle = -angle

        img = imutils.rotate_bound(img, angle)
        # img[img == 0] = 255
        mask_x = img.any(axis=2).any(axis=1)
        mask_y = img.any(axis=2).any(axis=0)
        img = img[mask_x, :, :][:, mask_y, :]
        # img = cv2.resize(img, (30, 60), interpolation=cv2.INTER_AREA)
        # img[img == 0] = 255

        # Color it in gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Create our mask by selecting the non-zero values of the picture
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        # Select the contour
        contours, cont = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 1)
        # Get all the points of the contour
        contour_index = 0
        area = 0
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > area:
                area = cv2.contourArea(contours[i])
                contour_index = i
        contour = contours[contour_index].reshape(len(contours[contour_index]), 2)
        # we assume a rectangle with at least two points on the contour gives a 'good enough' result
        # get all possible rectangles based on this hypothesis
        rect = []

        for i in range(len(contour)):
            x1, y1 = contour[i]
            for j in range(len(contour)):
                x2, y2 = contour[j]
                area = abs(y2 - y1) * abs(x2 - x1)
                rect.append(((x1, y1), (x2, y2), area))

        # the first rect of all_rect has the biggest area, so it's the best solution if he fits in the picture
        all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
        # we take the largest rectangle we've got, based on the value of the rectangle area
        # only if the border of the rectangle is not in the black part
        # if the list is not empty
        if all_rect:
            best_rect_found = False
            index_rect = 0
            nb_rect = len(all_rect)

            # we check if the rectangle is a good solution
            while not best_rect_found and index_rect < nb_rect:

                rect = all_rect[index_rect]
                (x1, y1) = rect[0]
                (x2, y2) = rect[1]

                valid_rect = True

                # we search a black area in the perimeter of the rectangle (vertical borders)
                x = min(x1, x2)
                while x < max(x1, x2) + 1 and valid_rect:
                    if mask[y1, x] == 0 or mask[y2, x] == 0:
                        # if we find a black pixel, that means a part of the rectangle is black
                        # so we don't keep this rectangle
                        valid_rect = False
                    x += 1

                y = min(y1, y2)
                while y < max(y1, y2) + 1 and valid_rect:
                    if mask[y, x1] == 0 or mask[y, x2] == 0:
                        valid_rect = False
                    y += 1

                if valid_rect:
                    best_rect_found = True

                index_rect += 1

        if best_rect_found:
            # Finally, we crop the picture and store it
            img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        # if not np.all(np.array(img.shape)):
        #     #cv2.imshow("Is that rectangle ok?", gray)
        #     #cv2.waitKey(0)
        #     cv2.imshow("Is that rectangle ok?", gray)
        #     cv2.waitKey(0)
        #     print(img.shape)
        try:
            img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            print("ERROR", e)
            img = np.zeros((30, 30, 3), np.uint8)
        return img

    def extract(self):
        df = pd.read_csv(self.tsv, sep="\t", names=['Class', 'Seed', 'R', 'G', 'B', 'X', 'Y', 'Cluster_x', 'Cluster_y'],
                         header=0)
        data = df.groupby(['Class', 'Seed']).apply(Tsv2Df.df2img).reset_index()
        for i in range(len(data)):
            img = data.iloc[i, 2]
            seed = str(data.iloc[i, 1])
            category = data.iloc[i, 0]
            # Rotate 90 Degrees Clockwise
            index = data.index[-1]
            data.loc[index + 1] = [category, seed + '.1',
                                   cv2.transpose(cv2.flip(img, 1))]
            # Rotate 270 Degrees
            data.loc[index + 2] = [category, seed + '.2',
                                   cv2.transpose(cv2.flip(img, 0))]
            # Rotate 180 Degrees
            data.loc[index + 3] = [category, seed + '.3',
                                   cv2.flip(img, -1)]
            # Flip (Mirror) Vertically
            data.loc[index + 4] = [category, seed + '.4',
                                   cv2.flip(img, 0)]
            # Flip (Mirror) Vertically + Rotate 90 Degrees Clockwise
            data.loc[index + 5] = [category, seed + '.5',
                                   cv2.flip(cv2.transpose(cv2.flip(img, 1)), 0)]
            # Flip (Mirror) Vertically + Rotate 270 Degrees Clockwise
            data.loc[index + 6] = [category, seed + '.6',
                                   cv2.flip(cv2.transpose(cv2.flip(img, 0)), 0)]
            # Flip (Mirror) Vertically + Rotate 180 Degrees Clockwise
            data.loc[index + 7] = [category, seed + '.7',
                                   cv2.flip(cv2.flip(img, -1), 0)]

        # print(data)
        # exit(0)
        #dirpath = tempfile.mkdtemp()
        dirpath = "./tmp1"
        names = []
        for i in range(len(data)):
            name = '{}/{}_{}.bmp'.format(dirpath, data.Class[i], data.Seed[i])
            names.append(name)
            plt.imsave(name, data.iloc[i, 2])

        gfs = []
        for name in names:
            gfs.append(GlobalFeatures(name).extract())
        df_gf = pd.DataFrame(gfs)
        #shutil.rmtree(dirpath)

        df_gf = pd.concat([df_gf[column].apply(pd.Series).add_prefix(column) for column in df_gf.columns], axis=1)
        df_gf = pd.concat([data[['Class', 'Seed']], df_gf], axis=1)
        self.out = df_gf
        return self.out
