# %%
from sklearn.datasets import fetch_lfw_people

import torch
from torchvision import transforms
from torch.utils.data import random_split

device = "cpu"

# %%
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# %%
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]
# the label to predict is the id of the person
y = lfw_people.target

target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# %%
preprocessing = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ]
)

X = preprocessing(X)
y = torch.tensor(y)

# %%
train_indices, test_indices = random_split(torch.arange(X.shape[1]), [0.7, 0.3])
X_train, y_train = X[:, train_indices, :], y[train_indices]
X_test, y_test = X[:, test_indices, :], y[test_indices]

# %%
# Compute PCA on the dataset
n_components = 84

mean = torch.mean(X_train, axis=1)
X_train -= mean
X_test -= mean

# Eigen decomposition using SVD
U, S, V = torch.linalg.svd(X_train[0, ], full_matrices=False)

# %%
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

# %%
X_transformed = torch.matmul(X_train,components.T)
X_test_transformed = torch.matmul(X_test, components.T)

print(X_transformed.shape)
print(X_test_transformed.shape)

# %%
#Finally, plot the resulting eigen-vectors of the face PCA model, AKA the eigenfaces
import matplotlib.pyplot as plt
# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# %%
# Faces
plot_gallery(X[0, :], ["Face "+str(i) for i in range(12)], h, w)

# %%
# eigenfaces.shape

# %%
# Eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
# plt.show()

# %%
explained_variance = (S ** 2) / (n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0)
eigenvalueCount = torch.arange(n_components)
plt.figure()
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
# plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import numpy as np
#build random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed[0, :], y_train) #expects X as [n_samples, n_features]

# %%
predictions = estimator.predict(X_test_transformed[0, :])
correct = predictions==y_test
total_test = len(X_test_transformed[0, :])
#print("Gnd Truth:", y_test)
print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct:",correct)
print("Total Correct:",np.sum(correct))
print("Accuracy:",np.sum(correct)/total_test)
print(classification_report(y_test, predictions, target_names=target_names))

# %%
plt.show()

