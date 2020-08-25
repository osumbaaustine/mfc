#The next step is to represent each and every term and document as a vector.
# I will use the document-term matrix and decompose it into multiple matrices.
# then use sklearn’s TruncatedSVD to perform the task of matrix decomposition.
#Since the data comes from 20 different newsgroups, let’s try to have 20 topics for our text data.

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

len(svd_model.components_)

#The components of svd_model are our topics, and we can access them using svd_model.components_
#Finally, let’s print a few most important words in each of the 20 topics
# and see how our model has done.