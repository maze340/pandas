from pandas.core.dtypes.common import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import math


class Preprocessing:
    def __init__(self, df, fillna_option, classlabel, filename, normalization=False, discret_type=None, discret_bins_nb=None, dir_save=""):
        """
        :param fillna_option: option to fill missing values: A for general, B for class label
        :param classlabel: class label
        :param filename: string filename for saving clean data
        :param normalization: boolean, default False
        :param discret_type: integer  describing which discretization type desired : 0 for equal-width, 1 for equal-frequency, 2 for based entropy, default: None
        :param discret_bins_nb: integer  describing which number of bins desired, default: None
        :param dir_save: str directory path for saving files, default:""
        """
        # !- empty cell ?
        self.classlabel = classlabel
        self.df = df
        self.prepro_param = {
            "fillna": "Depending on column data" if fillna_option == 'A' else "Depending on class label",
            "normalization": normalization,
            "discret_type": "equal-width" if discret_type == 0 else "equal_frequency" if discret_type == 1 else "based entropy",
            "bins_nb": discret_bins_nb}

        if classlabel != None:  # auto: remove rows with missing classlabel value
            dfDropedNA = df.dropna(subset=[classlabel])

        self.fillNA(fillna_option)

        if normalization == True:
            self.normalize()
        if discret_type != None and discret_bins_nb != None:
            self.discretize(discret_type, discret_bins_nb)

        self.encoded_df, self.enc_dec_dict = self.encodeData(self.df)
        self.dir_save = dir_save
        self.saveCleanData(filename)

    def fillNA(self, option):
        """
        Fills missing values
        :param option: fill na option: 'A 'or 'B'
        :return:
        """
        # continuousCols = self.df._get_numeric_data().columns #?- pas bon car inclus bool dans numeric
        numeric = self.df.select_dtypes(include=np.number)  # !- checker si bonne definition de continous = int et float
        numeric_columns = numeric.columns
        categorical = self.df.select_dtypes(exclude=np.number)
        categorical_columns = categorical.columns.drop(self.classlabel)
        if option == "A":
            self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
            for c in categorical_columns:
                self.df[c] = self.df[c].fillna(self.df[c].mode()[0])
                # if self.df[c][0] == True or self.df[c][0] == False:
                #   self.df[c] = self.df[c].astype("bool")
        else:
            classlabel_set = list(set(self.df[self.classlabel]))
            for v in classlabel_set:
                mask = self.df[self.classlabel] == v
                means = self.df.loc[
                    mask, numeric_columns].mean()  # ?- mean of all numeric cols with rows with classlabel=v
                self.df.loc[mask, numeric_columns] = self.df.loc[mask, numeric_columns].fillna(means)
                modes = self.df.loc[mask, categorical_columns].mode()
                # self.df.loc[mask, categorical_columns] = self.df.loc[mask, categorical_columns].fillna(modes[categorical_columns][0])#?-marche pas, pk?
                for c in categorical_columns:
                    self.df.loc[mask, c] = self.df.loc[mask, c].fillna(modes[c][0])
        if self.df.isna().sum().sum() != 0:
            print("There are still missing values...")

    def normalize(self):
        numeric = self.df.select_dtypes(include=np.number)
        numeric_columns = numeric.columns
        minmax_scaler = MinMaxScaler()
        self.df[numeric_columns] = minmax_scaler.fit_transform(self.df[numeric_columns])

    def discretize(self, type, bins_nb):
        """
        :param type: integer  describing which discretization type desired : 0 for equal-width, 1 for equal-frequency, 2 for based entropy
        :param bins_nb: integer  describing which number of bins desired
        :return:
        """
        numeric = self.df.select_dtypes(include=np.number)
        numeric_columns = numeric.columns

        if type == 0:  # equal-range
            for c in numeric_columns:
                min = self.df[c].min()
                max = self.df[c].max()
                edge_nb = bins_nb + 1
                bins = np.linspace(min, max, edge_nb)
                self.df[c] = pd.cut(self.df[c], bins=bins, include_lowest=True)
        elif type == 1:  # equal-frequency
            for c in numeric_columns:
                self.df[c] = pd.qcut(self.df[c], q=bins_nb, precision=1)
        elif type == 2:  # entropy based
            self.discret_entropy(bins_nb)

    def discret_entropy(self, bins_nb):
        """
        :param bins_nb: integer  describing which number of bins desired
        :return:
        """
        numeric = self.df.select_dtypes(include=np.number)
        numeric_columns = numeric.columns

        info_d = self.Info(self.df[self.classlabel])
        for c in numeric_columns:
            min = self.df[c].min()
            max = self.df[c].max()
            df = self.df
            df_sorted = df.sort_values(c)
            sorted_col_set = list(set(df_sorted[c]))
            splits = [(sorted_col_set[i] + sorted_col_set[i + 1]) / 2 for i in range(len(sorted_col_set) - 1)]
            df = df[[c, self.classlabel]]
            attributes = []
            for s in splits:
                attr = str(s)
                attributes.append(attr)
                df[attr] = pd.cut(df[c], [min, s, max], include_lowest=True)
            gain_splits = self.sortedGainList(df, attributes)
            splits = list(gain_splits.keys())
            edges = [min] + splits[:bins_nb - 1] + [max]
            edges = list(map(float, edges))
            edges.sort()
            self.df[c] = pd.cut(self.df[c], bins=edges, include_lowest=True)

    def Info(self, data):
        """Return the entropy of columns data
        :param data: data column
        """
        data = list(data)
        entropySum = 0
        done = list()
        for e in data:
            if e not in done:
                done.append(e)
                pb = data.count(e) / len(data)
                entropySum += -(pb * math.log2(pb))
        return entropySum

    def gain(self, d, attribute):
        """Info Gain
        : d: samples
        : attribute: attribute name"""
        attrData = list(d[attribute])
        attrSet = list(set(attrData))
        infoDforA = 0
        for v in attrSet:
            dv = d[d[attribute] == v]
            pb_dv = len(dv) / len(d)
            infoDv = self.Info(dv[self.classlabel])
            infoDforA += pb_dv * infoDv
        infoD = self.Info(d[self.classlabel])
        gainA = infoD - infoDforA
        return gainA

    def sortedGainList(self, df, attributes):
        """
        :param df: data frame object
        :param attributes: list of string attributes name
        :return: dict of attributes name sorted on descendant gain
        """
        gains = {}
        for a in attributes:
            gains[a] = self.gain(df, a)
        return dict(sorted(gains.items(), key=lambda item: item[1], reverse=True))

    def encodeData(self, df):
        """
        : param df: dataframe
        : returns: encoded dataframe, endode-decode dictionnary: {encode: encode dictionnary, decode: decode dictionnary}
        """
        df_copy = df.copy(deep=True)
        categorical = df_copy.select_dtypes(exclude=np.number)
        categorical_columns = categorical.columns
        decode_dict, encode_dict = {}, {}
        for c in categorical_columns:
            pre = df_copy[c].astype("category")
            df_copy[c] = pre.cat.codes
            decode_dict[c] = dict(enumerate(pre.cat.categories))
            encode_dict[c] = {v: k for k, v in decode_dict[c].items()}
        enc_dec_dict = {'encode': encode_dict, 'decode': decode_dict}
        return df_copy, enc_dec_dict

    def saveCleanData(self, filename):
        """
        Saves clean data
        :param filename:  string filename
        :return: None
        """
        self.df.to_csv(self.dir_save + filename + "_clean.csv")

    def savePreproParam(self):
        """
        Save preprocessing parameters  in a test file
        :return: None
        """
        fileentry = open(self.dir_save + "Preprocessing_parameter.txt", 'a')
        fileentry.write("Fill missing values option: {0}\nNormalization: {1}\nDiscretization: {2}\nBins number: {3}"
                        .format(self.prepro_param["fillna"], self.prepro_param["normalization"],
                                self.prepro_param["discret_type"], self.prepro_param["bins_nb"]))
        fileentry.close()
