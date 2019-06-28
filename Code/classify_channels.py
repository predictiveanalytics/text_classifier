import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# read in data
train_df = pd.read_csv("../Data/2017 Store to channel training.csv")
train_data = train_df['STORE_NAME']


# bag of words
count_vect = CountVectorizer()

# use bag of words to transform each phrase into a vector
X_train_counts = count_vect.fit_transform(train_data)

# give less weight to common words
# equalize short and long descriptions
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



# convert group variable to list
train_groups = train_df['Channel'].tolist()

# build group dictionary
groups = train_df['Channel'].tolist()
uq_groups = set(groups)

group_dict = {}
i=0
for x in uq_groups:
  group_dict[x] = i
  i = i+1

inv_group_dict = {}
j = 0
for x in uq_groups:
  inv_group_dict[j] = x
  j = j+1

# label groups for training, test dataset
Y_train = []
for x in train_groups:
  num = group_dict.get(x)
  Y_train.append(num)


# classify using naive bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, Y_train)





test_df = pd.read_csv("../Data/sdge.csv")
test_df_data = test_df['Store Name']
test_df_data = test_df_data.replace(np.nan, '', regex=True)


test_counts = count_vect.transform(test_df_data)

# use the classifier to predict
test_predicted = clf.predict(test_counts)


test_groups = []
for x in test_predicted:
  groupname = inv_group_dict.get(x)
  test_groups.append(groupname)


test_df.insert(loc=len(test_df.columns),column='Predicted Channel', value=test_groups)


test_df.to_csv("../Output/sdge_out.csv")