import re


class RenameClasses():

    def __init__(self, tsv, df):
        self.tsv = tsv
        self.df = df

    def rename_from_dict(self):
        di = {
            'class1': 1,
            'class2': 2,
            'class3': 3,
            'class4': 4
        }
        self.df['Class'] = self.df['Class'].map(di)
        return self.df

    def rename_from_name(self):
        # print(re.findall(r'class(\d+)',self.tsv))
        classes = re.findall(r'class(\d+)', self.tsv)
        if classes:
            di = dict(zip(list(range(0, len(classes))), classes))
            self.df['Class'] = self.df['Class'].map(di)
        return self.df
