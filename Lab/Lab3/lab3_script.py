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
        y = y.astype(int)
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        
        self.parameters = {}
        self.no_cnt = 0
        self.yes_cnt = 0

        self.no2word = np.array([0]*3000)
        self.yes2word = np.array([0]*3000)   

        for i, c in enumerate(self.classes):
            # Calculate the mean, variance, prior probability of each class
            X_Index_c = X[np.where(y == c)]
            for x in X_Index_c:
                for j in range(3000):
                    if i == 0: # No
                        self.no2word[j] += 1
                        self.no_cnt += 1
                    else: # yes
                        self.yes2word[j] += 1
                        self.yes_cnt += 1
            

            X_index_c_mean = np.mean(X_Index_c, axis=0, keepdims=True)
            X_index_c_var = np.var(X_Index_c, axis=0, keepdims=True)
            parameters = {"mean": X_index_c_mean, "var": X_index_c_var, "prior": X_Index_c.shape[0] / X.shape[0]}
            
            self.parameters["class" + str(c)] = parameters
        

        # print(self.parameters)


    def predict(self, whole_X):
        # return class with highest probability，给X的每条记录都进行预测
      
        prediction = []
        
        for X in whole_X:
            yes_prob = 0.5
            no_prob = 0.5
            for i in range(3000):
                if X[i] == 1:
                    yes_prob *= (self.yes2word[i] + 0.5)/(self.yes_cnt + 1)
                    no_prob *= (self.no2word[i] + 0.5)/(self.no_cnt + 1)

            if yes_prob > no_prob:
                prediction.append(1)
            else:
                prediction.append(0)

        return prediction



############################################

# %%
from sklearn.metrics import confusion_matrix

#Call the Naive Bayes algorithm, which we wrote ourselves
model = NaiveBayes()
model.fit(X_train,y_train) 

result = model.predict(X_test)
print (confusion_matrix(y_test, result))


