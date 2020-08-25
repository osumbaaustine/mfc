#To find out how distinct our topics are, we should visualize them.
# Of course, we cannot visualize more than 3 dimensions,
# but there are techniques like PCA and t-SNE which can help us visualize high dimensional data
# into lower dimensions.
# Here we will use a relatively new technique
# called UMAP (Uniform Manifold Approximation and Projection).

import umap

X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1],
c = dataset.target,
s = 10, # size
edgecolor='none'
)
plt.show()