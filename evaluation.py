from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, classlabel, train, test, classifier, dir_save=""):
        """
        :param classlabel: string, class label
        :param train: array like of train data
        :param test: array like of test data
        :param predict: predict function object which returns for a given sample a class label
        :param dir_save: str directory path for saving files, default:""
        """
        features_X = train.columns.drop(classlabel)
        target_y = classlabel
        self.predict_proba = classifier.predictProba
        self.model = classifier.model
        self.feature_train, self.target_train = train.loc[:, features_X], train.loc[:, target_y]
        self.feature_test, self.target_test = test.loc[:, features_X], test.loc[:, target_y]
        self.dir_save = dir_save

        # Prediction
        self.target_train_pred = classifier.predictSet(self.feature_train)
        self.target_test_pred = classifier.predictSet(self.feature_test)
        self.target_major_pred = self.predictMaj(self.feature_test, self.target_train)

    def script(self):
        """
        executes a serie of functions for complete evaluation
        :return: None
        """
        # Predictions
        labels_pred_train = self.target_train_pred
        labels_pred_test = self.target_test_pred
        labels_pred_major = self.target_major_pred

        # Evaluation
        cm_train = self.confusionMatrix(self.target_train, labels_pred_train)
        cm_test = self.confusionMatrix(self.target_test, labels_pred_test)
        cm_major = self.confusionMatrix(self.target_test, labels_pred_major)
        report_train = self.metricsReport(self.target_train, labels_pred_train)
        report_test = self.metricsReport(self.target_test, labels_pred_test)
        report_major = self.metricsReport(self.target_test, labels_pred_major)

        # Print Evaluations
        print("Train Evaluation\n{0}\n{1}".format(cm_train, report_train))
        print("Test Evaluation\n{0}\n{1}".format(cm_test, report_test))
        print("Major Evaluation\n{0}\n{1}".format(cm_major, report_major))

        # Saving Evaluation
        self.save("Train Confusion Matrix", cm_train)
        self.save("Train Evaluation Report", report_train)
        self.save("Test Confusion Matrix", cm_test)
        self.save("Test Evaluation Report", report_test)
        self.save("Majority Rule Confusion Matrix", cm_major)
        self.save("Majority Rule Evaluation Report", report_major)

    def confusionMatrix(self, labels_test, labels_pred):
        """
        :param labels_test / y_test: array of labels from test set
        :param labels_pred / y_pred: array of labels from classifier
        :returns: confusion matrix: [0][0]: TP, [0][1]: FP, [1][0]: TN, [1][1]: FN
        """
        return confusion_matrix(labels_test, labels_pred)

    def predictMaj(self, feature_test, target_train):
        """
        Predicts data based on majority rule
        :param feature_test: array like of test features data
        :param target_train: array like of train target data
        :return: array of class labels for test data
        """
        class_maj = max(dict(target_train.value_counts()))
        return [class_maj for i in range(len(feature_test))]

    def metricsReport(self, labels_test, labels_pred):
        """
        : param labels_test / y_test: array of labels from test set
        : param labels_pred / y_pred: array of labels from classifier
        : returns: report of metrics evaluation
        """
        return classification_report(labels_test, labels_pred, zero_division=0)  # !- changer zero_division ?

    def save(self, title, value):
        """
        :param title: string, title to write in the file
        :param value: value to write in the file
        :return: None
        """
        filename = self.dir_save + "eval.txt"
        f = open(filename, 'a')  # ?- 'a': text added at current file stream position, 'w': erased then inserted
        f.write(title + "\n")
        f.write(str(value) + "\n")
        f.close()

    def drawROCCurve(self, classval):
        pbs = self.predict_proba(classval)

        y_pred1 = pbs
        y_test = self.target_test

        fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label=classval)

        plt.figure(figsize=(6, 4))

        plt.plot(fpr, tpr, linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--')

        plt.rcParams['font.size'] = 12

        plt.title('ROC curve')

        plt.xlabel('False Positive Rate (1 - Specificity)')

        plt.ylabel('True Positive Rate (Sensitivity)')

        plt.show()

    def getAUC(self, classval):
        pbs = self.predict_proba(classval)
        y_pred1 = pbs
        y_test = list(self.target_test)
        auc = roc_auc_score(y_test, y_pred1)
        return auc
