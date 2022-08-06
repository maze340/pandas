from pandas.core.frame import DataFrame
from sklearn.tree import DecisionTreeClassifier
import math
import joblib


class Node:
    def __init__(self, attrVal=None, value=None, samplesID=None, next=None):
        """
        :param attrVal: string, attribute value of the node
        :param value: string, or name attribute or classlabel
        :param samplesID: array of data frame samples id specific to the node
        :param next: reference object of the next node
        """
        self.attrVal = attrVal  # attribute value
        self.value = value  # attribute name or label
        self.samplesID = samplesID  # IDs of node's samples
        self.pb = None
        self.next = next


class TreeDecisionClassifier:
    def __init__(self, dataFrame, classLabel, enc_dec_dict=None, pep=False, max_depth=None, min_samples_leaf=1,
                 builtin=False, dir_save=""):
        """
        :param dataFrame: data frame
        :param class label: string
        :param pep: boolean, default: None
        :param max_depth: the maximum depth of the tree, integer, default: None (no limit)
        :param min_samples_leaf: the minimum sample at leaf ,or integer or float number in [0,1], default: 1
        :param builtin: boolean {True: builtin version, False: own version}
        :param dir_save: str directory path for saving files, default:""
        """
        self.dataFrame = dataFrame
        self.features_X = self.dataFrame.columns.drop(classLabel)
        self.classLabel = classLabel
        # self.dataLabel = dataFrame[classLabel]#test si ok
        self.labelset = dataFrame[classLabel].unique()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf if type(min_samples_leaf) != float else math.ceil(
            len(self.dataFrame) * min_samples_leaf)
        self.builtin = builtin
        self.enc_dec_dict = enc_dec_dict
        # Building model
        if builtin and enc_dec_dict != None:
            self.model, self.predict = self.treeDecisionClassifierLib(enc_dec_dict)
        else:
            ids = list(self.dataFrame.index)
            attributes = list(self.dataFrame.columns.drop(self.classLabel))
            self.model = self.id3(attributes, ids, 0)
            self.predict = self.findClass
            # Post-pruning
            if pep == True:
                self.postPruningPEP(self.model, 0.5)
        self.filename = dir_save + "tree_builtin.obj" if builtin else dir_save + "tree_own.obj"

    def Info(self, data):
        """Entropy
        :param data: array like of data
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
        :param d: samples
        :param attribute: attribute name"""
        attrData = list(d[attribute])
        attrSet = list(set(attrData))
        infoDforA = 0
        for v in attrSet:
            dv = d[d[attribute] == v]
            pb_dv = len(dv) / len(d)
            infoDv = self.Info(dv[self.classLabel])
            infoDforA += pb_dv * infoDv
        infoD = self.Info(d[self.classLabel])
        gainA = infoD - infoDforA
        return gainA

    def findBestGain(self, samplesID, attributes):
        """
        Finds the attribute with the greatest gain
        :param samplesID: array of dataframe samples id of the node
        :param attributes: list of string attributes name
        :return: string, name of the best attribute
        """
        bestAttr = None
        bestGain = 0
        # df = self.dataFrame.iloc[samplesID]#?- iloc c'est selon l'index relatif a l'array, loc c'est selon l'index du dataframe
        df = self.dataFrame.loc[samplesID]
        for a in attributes:
            gain = self.gain(df, a)
            # print("{0}:{1}".format(a, gain))
            if gain > bestGain:
                bestAttr, bestGain = a, gain
        return bestAttr

    def id3(self, attributes, samplesID, attrVal=None, level=0):
        """
        id3 algorythm
        :param attributes: list of string attributes name
        :param samplesID: array of dataframe samples id of the node
        :param attrVal: string, attribute value
        :param level: int, tree level of the node
        :return: tree decision node root
        """
        infoD = self.Info([self.dataFrame[self.classLabel][i] for i in samplesID])
        node = Node()
        node.samplesID = samplesID
        node.attrVal = attrVal
        # classlabel pb
        df_sampled = self.dataFrame.loc[samplesID]
        node.pb = {v: len(df_sampled[df_sampled[self.classLabel] == v]) / len(samplesID) for v in self.labelset}
        if infoD > 0:
            if len(attributes) > 0 and (
                    self.max_depth is None or level < self.max_depth):  # prepruning max_depth condition
                bestAttr = self.findBestGain(samplesID, attributes)
                futur_min_samples_leaf = self.dataFrame.loc[samplesID][bestAttr].value_counts().min()
                if futur_min_samples_leaf >= self.min_samples_leaf:  # prepruning min_samples_leaf condition
                    node.value = bestAttr
                    node.next = []
                    attrSet = list(set([self.dataFrame[bestAttr][i] for i in samplesID]))
                    for v in attrSet:
                        samplesChild = [i for i in samplesID if self.dataFrame[bestAttr][i] == v]
                        attrChild = [e for e in attributes if e != bestAttr]
                        # print(v)
                        node.next.append(self.id3(attrChild, samplesChild, v, level + 1))
                else:
                    node.value = self.majorityClass(node)[0]
                    return node
            else:  # majority rule
                node.value = self.majorityClass(node)[0]
                return node
        else:
            # sample = self.dataFrame.loc[samplesID]#test si ok
            uniqueLabel = list(df_sampled[self.classLabel])[0]
            node.value = uniqueLabel
            return node
        return node

    def printTree(self):
        def printTreeRecurse(node, level=0):
            """
            Print a given tree
            : param node: node root of the tree model
            : param level: integer, level of the tree, default value : 0
            """
            print("{0}-{1}-{2}-{3}".format("\t" * level, node.attrVal, node.value, node.samplesID))
            if node.next:
                for child in node.next:
                    printTreeRecurse(child, level + 1)

        printTreeRecurse(self.model)

    def majorityClass(self, node):
        """
        Determines majority class label of a node
        :param node: Node object
        :returns : (major class label, his frequency)
        """
        majorLabel = None
        majorCount = 0
        labels = list(self.dataFrame.loc[node.samplesID][self.classLabel])
        for l in list(set(labels)):
            if labels.count(l) > majorCount:
                majorLabel, majorCount = l, labels.count(l)
        return majorLabel, majorCount

    def postPruningPEP(self, node, penality=0):
        """
        Computes post pruning PEP
        :param node: Node object, root node of the tree
        :param penality: float, penality for PEP
        :return: None
        """
        if node.next is not None:
            leafs = [0, 0]  # numerator, denominator
            for child in node.next:
                if child.next is not None:  # child is not a leaf
                    self.postPruningPEP(child, penality)
                if child.next is None:  # child is a leaf
                    majCount = self.majorityClass(child)[1]  # number of samples with major class
                    sampCount = len(child.samplesID)  # number of all samples
                    leafs[0] += sampCount - majCount + penality
                    leafs[1] += sampCount

            if leafs[1] == len(node.samplesID):  # if all node's sons are leafs
                if self.estimateError(node, leafs, penality):
                    node.next = None  # remove leafs
                    node.value = self.majorityClass(node)[0]

    def estimateError(self, node, leafs, penality=0):
        """
        Check if error at leaf is greater than error at node
        :param node: Node object
        :param leafs: array [numerator, denominator] for computing error at leaf
        :param penality: float penality
        :return: boolean
        """
        majCount = self.majorityClass(node)[1]  # number of samples with major class
        sampCount = len(node.samplesID)  # number of all samples
        pError_node = (sampCount - majCount + penality) / sampCount
        pError_leafs = leafs[0] / leafs[1]
        return pError_leafs >= pError_node

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

    def findClass(self, sample):
        """
        Predicts class for a given sample
        :param sample: dict like sample {'attribute1': value...}
        :return: string classlabel
        """
        node = self.model

        while node.next is not None:
            attrVal_found = False
            for child in node.next:
                if child.attrVal == sample[node.value]:
                    node = child
                    attrVal_found = True
                    break
            if attrVal_found == False:
                return self.majorityClass(node)[0], node.pb
        return node.value, node.pb

    def predictProba(self, classval):
        pbs = self.pbs
        if not self.builtin:
            pbs_classval = [e[classval] for e in pbs]
        else:
            classes = list(self.model.classes_)
            pbs_classval = list(pbs[:, classes.index(classval)])
        return pbs_classval

    def treeDecisionClassifierLib(self, enc_dec_dict):
        """
        Tree Decision Built-in version
        : param: endode-decode dictionnary: {encode: encode dictionnary, decode: decode dictionnary}
        : returns: (model classifier, model.predict function )
        """
        # Encoding
        df_copy = self.dataFrame.copy(deep=True)
        d_copy = df_copy.replace(enc_dec_dict["encode"])
        # print(d_copy)

        # Building Model
        features_X_data = d_copy.loc[:, self.features_X].values
        target_y_data = df_copy.loc[:, self.classLabel].values
        model = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                       random_state=42, criterion="entropy")
        model.fit(features_X_data, target_y_data)

        def findClass(sampleX):
            """
            Predicts class for a given sample
            :param sampleX: dict like sample {'attribute1': value...}
            :return: string class label
            """
            # encoded_sample = {k: enc_dec_dict["encode"][k][v] for k, v in sampleX.items()}#ne check pas si col avec valeures continues
            encoded_sample = {k: enc_dec_dict["encode"][k][v] if k in enc_dec_dict["encode"].keys() else v for k, v in
                              sampleX.items()}

            encoded_sample = list(encoded_sample.values())
            classvalue = model.predict([encoded_sample])
            return classvalue[0]

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

