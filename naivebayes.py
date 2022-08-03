import copy
from functools import reduce
import operator
import joblib

from sklearn.naive_bayes import GaussianNB, CategoricalNB


class NaiveBayesClassifier:
    def __init__(self, df, classlabel, enc_dec_dict=None, builtin=False, dir_save=""):
        """
        : param df: dataframe
        : param classlabel: type: string, class label
        : param class_type: type: integer, classification type, {0: Naive Bayes, 1: Tree Decision}
        : param sample: type: dict, sample to classify, default= None
        : param builtin: type: boolean, determine use of built-in version or not, {True: Built-in version, False: Own version}
        :param dir_save: str directory path for saving files, default:""
        """
        self.df = df
        self.classlabel = classlabel
        self.features_X = self.df.columns.drop(self.classlabel)

        if builtin is False:
            self.model, self.predict = self.naiveBayesClassifierOwn()
        elif builtin and enc_dec_dict != None:
            self.model, self.predict = self.naiveBayesClassifierLib(enc_dec_dict)
        self.filename = dir_save + "naivebayes_lib.obj" if builtin else dir_save + "naivebayes_own.obj"

    def naiveBayesClassifierOwn(self, laplace_factor=1):
        """
        :param laplace_factor: integer factor for Laplacian correction
        :return: (naive bayes model, predict function)
        """
        # laplace_factor = 0

        classlabel_yk_set = self.df[self.classlabel].unique()  # classlabel set
        occur_yk = self.df.groupby(self.classlabel).size()  # nb of occurence for each classlabel's value-yk

        # Laplacian Correction - part 1
        prior = occur_yk.add(laplace_factor).div(len(self.df) + len(
            classlabel_yk_set) * laplace_factor)  # for each classlabels's value yk: her pb -> yk: p(yk)

        occur_xi_and_yk = {}  # for each attribute-xi: for each attribute's value-x'i, the occurence of each classlabel's value-yk -> x: #(x'i, yk)
        pb_xi_given_yk = {}  # for each attribute-xi: for each attribute's value-x'i, the pb of x'i  given classlabel's value-yk -> x: P(x'i | yk)
        pb_X_given_yk = {}  # for each classlabel's value yk: pb of sample-X given classlabel's value-yk -> yk: P(X | yk) = mul( P(x'i | yk) )
        pb_yk_and_X = {}  # for each classlabel's value yk: pb of sample-X and classlabel's value-yk -> yk: P(X, yk)

        colX = self.features_X
        xi_set = {}
        for xi in colX:
            xi_set[xi] = list(self.df[xi].unique())  # ? - set of column-xi's data
            occur_xi_and_yk[xi] = self.df.groupby([self.classlabel, xi]).size().unstack().fillna(0).unstack();
            # Laplacian Correction - part 2
            pb_xi_given_yk[xi] = occur_xi_and_yk[xi].add(laplace_factor).div(
                occur_yk + len(xi_set[xi]) * laplace_factor)

        def findClass(sampleX):
            """
            Predicts class for a given sample
            :param sampleX: dict like sample {'attribute1': value...}
            :return: string class label
            """
            for yk in classlabel_yk_set:
                pb_X_given_just_yk = []
                for xi in pb_xi_given_yk:
                    # essai2
                    if sampleX[xi] not in xi_set[xi]:
                        xi_set_copy = copy.deepcopy(xi_set)  # ?-deep copy of dict
                        xi_set_copy[xi].append(sampleX[xi])
                        pb_xi_given_yk[xi] = occur_xi_and_yk[xi].add(laplace_factor).div(
                            occur_yk + len(xi_set_copy[xi]) * laplace_factor)
                        for k in list(classlabel_yk_set):
                            pb_xi_given_yk[xi][sampleX[xi], k] = laplace_factor / (occur_yk[k] + len(xi_set_copy[
                                                                                                         xi]) * laplace_factor)  # ?- pb_xi_given_yk[xi][val col 1, val col 2] PUTAIN !
                        # print(pb_xi_given_yk)

                    pb = pb_xi_given_yk[xi][sampleX[xi]][yk]
                    pb_X_given_just_yk.append(pb)
                pb_X_given_yk[yk] = reduce(operator.mul, pb_X_given_just_yk)
                pb_yk_and_X[yk] = pb_X_given_yk[yk] * prior[yk]
            # classMax = max(pb_yk_and_X)#?-erreur: ça renvoie la clef max pas la val max ('yes'>'no')
            classMax = max(pb_yk_and_X.items(), key=operator.itemgetter(1))[0]  # ?-items(), itemgetter()

            # print(pb_yk_and_X)
            return classMax

        # print(prior)
        # print(occur_yk)
        # print(occur_xi_and_yk)
        # print(pb_xi_given_yk)
        return pb_xi_given_yk, findClass

    def naiveBayesClassifierLib(self, enc_dec_dict):
        """
        : param: endode-decode dictionnary: {encode: encode dictionnary, decode: decode dictionnary}
        : returns: (model classifier, model.predict function )
        """
        # Encoding df
        df_copy = self.df.copy(deep=True)
        encoded_df_copy = df_copy.replace(enc_dec_dict["encode"])
        # print(enc_dec_dict)
        # print(encoded_df_copy)

        # Building model
        feature_X_data = encoded_df_copy.loc[:, self.features_X].values
        target_y_data = encoded_df_copy.loc[:, self.classlabel].values
        # model = GaussianNB()
        # model.fit(feature_X_data, target_y_data)
        model = CategoricalNB(alpha=1)
        model.fit(feature_X_data, target_y_data)

        def findClass(sampleX):
            """
            Predicts class for a given sample
            :param sampleX: dict like sample {'attribute1': value...}
            :return: string class label
            """
            encoded_sample = {k: enc_dec_dict["encode"][k][v] for k, v in sampleX.items()}
            # print(encoded_sample)
            # print(encoded_sample.values())
            classvalue = model.predict([list(encoded_sample.values())])
            return enc_dec_dict["decode"][self.classlabel][classvalue[0]]

        return model, findClass

    def saveModel(self):
        """
        Saves model in the disk
        :return: None
        """
        joblib.dump(self.model, self.filename)

    def loadModel(self):
        """
        Loads Model from disk
        :return: None
        """
        self.model = joblib.load(self.filename)

