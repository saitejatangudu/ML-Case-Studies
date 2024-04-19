import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

class KNN:
    def __init__(self,encoder="ResNet",k=9,metric="manhattan"):
        self.encoder = encoder
        self.k = k
        self.metric = metric
    def fit(self,dataset):
        self.New_class = np.unique(dataset[3])
        X,y = self.preprocess(dataset)
        self.X = X
        self.y = y
        return self
    def preprocess(self,dataset):
        dataset = dataset.drop([4],axis=1)
        dataset.columns = ["id","ResNet","VIT","class"]
        Encoded_class = []
        for i in dataset["class"]:
            Encoded_class.append(*np.where(self.New_class==i)[0])
        return (dataset[self.encoder],pd.Series(Encoded_class))
    def __str__(self):
        return ("Encoder="+str(self.encoder),
                "k="+str(self.k),"metric="+str(self.metric))
    def distance(self,sample1,sample2):
        if self.metric == "euclidean":
            return np.sqrt(np.sum(np.square(sample1-sample2)))
        if self.metric == "manhattan":
            return np.sum(np.absolute(sample1-sample2))
        if self.metric == "chebyshev":
            return np.max(np.absolute(sample1-sample2))
        if self.metric =="cosine":
            return -1*(np.dot(sample1,sample2.T)/(np.sqrt(np.sum(np.square(sample1)))*np.sqrt(np.sum(np.square(sample2)))))
    def predict(self,samples):
        val_data,val_label = self.preprocess(samples)
        predictions = [pd.DataFrame(self.y[np.argsort(np.array([self.distance(i,sample)for i in self.X]).flatten())[:self.k]]).value_counts().index[0][0] for sample in val_data]
        return self.evaluate(val_label,predictions)
    def evaluate(self,val_label,predictions):
        accuracy = accuracy_score(val_label,predictions)
        precision_micro = precision_score(val_label,predictions,average="micro",zero_division = 1)
        recall_micro= recall_score(val_label,predictions,average="micro",zero_division = 1)
        f1score_micro= f1_score(val_label,predictions,average="micro",zero_division = 1)
        precision_macro= precision_score(val_label,predictions,average="macro",zero_division = 1)
        recall_macro= recall_score(val_label,predictions,average="macro",zero_division = 1)
        f1score_macro= f1_score(val_label,predictions,average="macro",zero_division = 1)
        return {"accuracy":accuracy,"micro_precision":precision_micro,"micro_recall":recall_micro,"micro_f1score":f1score_micro,"macro_precision":precision_macro,"macro_recall":recall_micro,"macro_f1score":f1score_micro}