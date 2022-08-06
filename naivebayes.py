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
        self.builtin = builtin
        self.enc_dec_dict = enc_dec_dict

        if builtin is False:
            self.model, self.predict = self.naiveBayesClassifierOwn()
        elif builtin and enc_dec_dict != None:
            self.model, self.predict = self.naiveBayesClassifierLib(enc_dec_dict)
        self.filename = dir_save + "naivebayes_builtin.obj" if builtin else dir_save + "naivebayes_own.obj"

    def predictSet(self, samples):
        """
        classify each sample of set
        :param samples: train set or test set
        :returns: array of class labels
        """
        labels = []
        pbs = []

        for i in range(len(samples)):
            result = self.predict(samples.iloc[i])
            if not self.builtin:
                labels.append(result[0])
                pbs.append(result[1])
            else:
                labels.append(result)

        if self.builtin:
            enc_dict = self.enc_dec_dict["encode"]
            encoded_samples = samples.replace(to_replace=enc_dict)
            pbs = self.model.predict_proba(encoded_samples.values)

        self.pbs = pbs
        return labels

    def naiveBayesClassifierOwn(self, laplace_factor=1.0):
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

        pb_xi = {}  # for each attribute-xi: for each attribute's value-x'i, the occurence of x'i
        occur_xi_and_yk = {}  # for each attribute-xi: for each attribute's value-x'i, the occurence of each classlabel's value-yk -> x: #(x'i, yk)
        pb_xi_given_yk = {}  # for each attribute-xi: for each attribute's value-x'i, the pb of x'i  given classlabel's value-yk -> x: P(x'i | yk)
        pb_X_given_yk = {}  # for each classlabel's value yk: pb of sample-X given classlabel's value-yk -> yk: P(X | yk) = mul( P(x'i | yk) )
        pb_yk_and_X = {}  # for each classlabel's value yk: pb of sample-X and classlabel's value-yk -> yk: P(X, yk)

        colX = self.features_X
        xi_set = {}
        n = len(self.df)
        for xi in colX:
            pb_xi[xi] = self.df.groupby(xi).size()
            pb_xi[xi] = pb_xi[xi].div(n)

            xi_set[xi] = list(self.df[xi].unique())  # ? - set of column-xi's data
            occur_xi_and_yk[xi] = self.df.groupby([self.classlabel, xi]).size().unstack().fillna(0).unstack()

            # Laplacian Correction - part 2
            pb_xi_given_yk[xi] = occur_xi_and_yk[xi].add(laplace_factor).div(
                occur_yk + len(xi_set[xi]) * laplace_factor)

        def findClass(sampleX):
            """
            Predicts class for a given sample
            :param sampleX: dict like sample {'attribute1': value...}
            :return: (string class label, pb)
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

            ##-1
            # pas valeure utilisée par predict_proba()
            # pb_X = reduce(operator.mul, [pb_xi[xi][sampleX[xi]] for xi in colX])
            # pb_yk_given_X = {k: v/pb_X for k,v in pb_yk_and_X.items()}

            # valeur utilisée par predict_proba()
            s = sum([v for k, v in pb_yk_and_X.items()])  # sum of all pb_yk_and_X
            pb_yk_and_X_weighted = {k: v / s for k, v in pb_yk_and_X.items()}
            ##-1

            return classMax, pb_yk_and_X_weighted

        return pb_xi_given_yk, findClass

    def naiveBayesClassifierLib(self, enc_dec_dict):
        """
        : param: endode-decode dictionnary: {encode: encode dictionnary, decode: decode dictionnary}
        : returns: (model classifier, model.predict function )
        """
        # Encoding df
        df_copy = self.df.copy(deep=True)
        encoded_df_copy = df_copy.replace(enc_dec_dict["encode"])

        # Building model
        feature_X_data = encoded_df_copy.loc[:, self.features_X].values
        target_y_data = df_copy.loc[:, self.classlabel].values
        model = CategoricalNB()
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
            return classvalue[0]

        return model, findClass

    def predictProba(self, classval):
        pbs = self.pbs
        if not self.builtin:
            pbs_classval = [e[classval] for e in pbs]
        else:
            classes = list(self.model.classes_)
            pbs_classval = list(pbs[:, classes.index(classval)])

        return pbs_classval

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

