# %%
print("Hello Naive-Bayes!")

# %%
from collections import Counter
import os
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir (train_dir)]
    all_words = []
    for mail in emails:
        with open (mail) as m:
            for i,line in enumerate (m) :
                if i == 2:
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    # Write code for non-word removal here
    list_to_remove = list(dictionary.keys())
    # print(list_to_remove)
    for item in list_to_remove:
        if item.isalpha() == False: # 判断一个字符串是否所有都为字母
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    
    return dictionary

# To show the most frequent words in train-mails
dictionary = make_Dictionary('ling-spam/train-mails')
dictionary

# %%
import os
import numpy as np
def extract_features(mail_dir):
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0
    _i = 0
    print(len(files))
    for fil in files:
        _i+=1
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID,wordID]+=1
            docID = docID + 1
        print('\r','done {} files'.format(_i),flush=True,end='')
    return features_matrix

# %%
# # Create a dictionary of words with its frequency
# features_matrix = extract_features("ling-spam/train-mails")
# features_matrix

# %% [markdown]
# ### 4) Training the Classifiers
# Here you should write your Naïve Bayes classifiers after fully understanding its principle.
# 

# %%
########### Write Your Code Here ###########
# Prepare feature vectors per training mail and its labels
# 通过标签划分垃圾邮件 "sp"为垃圾邮件，label 为 1，否则 label 为 0

path = "ling-spam/train-mails"
X_train = extract_features(path)
f = os.listdir(path)
y_train = []
for file_name in f:
    if('sp' in file_name):
        y_train.append(1)
    else:
        y_train.append(0)

    # print((file_name, y_train[len(y_train) - 1]))




############################################

# %% [markdown]
# ### 5) Implementation of the Naive Bayes algorithm
# 
# Complete the code for naive Bayes algorithm in `predict` function.

# %%
########### Write Your Code Here ###########
# Prepare feature vectors per testing mail and its labels
# 通过标签划分垃圾邮件 "sp"为垃圾邮件，label 为 1，否则 label 为 0

path = "ling-spam/test-mails"
X_test = extract_features(path)
f = os.listdir(path)
y_test = []
for file_name in f:
    if('sp' in file_name):
        y_test.append(1)
    else:
        y_test.append(0)
    # print((file_name, y_train[len(y_train) - 1]))

############################################

# %%
y_train = np.array(y_train)
y_test = np.array(y_test)
X_test = np.array(X_test)
X_train = np.array(X_train)
X_train.shape

# %%

########### Write Your Code Here ###########

import enum
from xml.sax.saxutils import prepare_input_source


class NaiveBayes():

    def fit(self, X, y):
        # 防止后续取对数的时候值为0，此处加上一个极小数
        epsilon = 1e-8
        y = y.astype(int)
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        
        self.parameters = {}
        
        for i, c in enumerate(self.classes):
            # Calculate the mean, variance, prior probability of each class
            # 获取对应类的列
            X_Index_c = X[np.where(y == c)]
            
            X_index_c_mean = np.mean(X_Index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_Index_c, axis=0, keepdims=True)
            X_index_c_var += epsilon
            parameters = {"mean": X_index_c_mean, "var": X_index_c_var, "prior": X_Index_c.shape[0] / X.shape[0]}
            
            self.parameters["class" + str(c)] = parameters
        

        # print(self.parameters)
    @staticmethod
    def get_first(var):
        return -0.5 * np.log(2 * np.pi * var)
    @staticmethod
    def get_second(x, var, mean):
        return -np.square(x - mean) / (2 * var)

    def predict(self, X):
        # return class with highest probability，给X的每条记录都进行预测
      
        prediction = []
        for index in range(len(X)):
            proba_array = []
            for class_idx in self.parameters.keys():
                tmp = []
                parameter_dic = self.parameters[class_idx]
                for feature in range(len(X[index])):
                    var = parameter_dic["var"][0][feature]
                    mean = parameter_dic["mean"][0][feature]
                    tmp.append(NaiveBayes.get_first(var=var))
                    tmp.append(NaiveBayes.get_second(X[index][feature], var=var, mean=mean))
                    tmp.append(np.log(parameter_dic['prior']))
                proba_array.append(np.array(tmp).sum())

            # print("proba_array = ", proba_array)
            if proba_array[0] > proba_array[1]:
                prediction.append(0)
            else:
                prediction.append(1)

                
        return prediction



############################################

# %%
from sklearn.metrics import confusion_matrix

#Call the Naive Bayes algorithm, which we wrote ourselves
model = NaiveBayes()
model.fit(X_train,y_train) 

result = model.predict(X_test)
# result
print (confusion_matrix(y_test, result))

# %% [markdown]
# ### 6) Checking the results on test set
# The test set contains 130 spam emails and 130 non-spam emails. Please compute accuracy, recall, F-1 score to evaluate the performance of your spam filter.

# %%
########### Write Your Code Here ###########
from sklearn.metrics import accuracy_score, recall_score, f1_score
print("The accuracy score of my Bayes model is ", accuracy_score(y_test, result))
print("The recall score of my Bayes model is ", recall_score(y_test, result))
print("The f1 score of my Bayes mode is ", f1_score(y_test, result))

############################################

# %% [markdown]
# ### Exercise 2 Compare your Naïve Bayes algorithm with GassianNB from Sklearn, which one does better? Where is the gap?

# %%
########### Write Your Code Here ###########
from sklearn.naive_bayes import GaussianNB
model_GaussanNB = GaussianNB()
model_GaussanNB.fit(X_train,y_train)
y_predict = model_GaussanNB.predict(X_test)
print("accuracy of sklearn is", accuracy_score(y_test,y_predict))

############################################

# %% [markdown]
# The accuracy of my model is the same as the sklearn, both are 0.9615384615384616

# %% [markdown]
# ### Exercise 3 Questions
# 1. Describe another real-world application where the naïve Bayes method can be applied
# 
#     The Naive Bayes method can also be use in predicting the price of house considering the features such as location, area, floor, etc..
# 
# 2. What are the strengths of the naïve Bayes method; when does it perform well?
# 
#     The strengths of the naive Bayes method is that the efficiency is very stable, and it's not sensible to the missing data.
# 
#     When the relationship between features is independent, and the sample size is small, the performance is well.
# 
# 3. What are the weaknesses of the naïve Bayes method; when does it perform poorly?
# 
#     The weaknesses of the naive Bayes method is that it is strongly dependent on the independency of features. When the features have strong dependency with each other, the performance is poor. 
# 
# 
# 4. What makes the naïve Bayes method a good candidate for the classification problem, if you have enough knowledge about the data?
# 
#     The Naive Bayes has strong interpretability and it is of strong use when the sample size is small and the features are independent.


