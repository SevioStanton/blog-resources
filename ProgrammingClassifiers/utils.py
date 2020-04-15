import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, X, y, title=""):
    xlabel = X.columns[0]
    ylabel = X.columns[1]
    X, y = X.to_numpy(), y.to_numpy()
    X = X[:, :2]
    decision_function = clf.decision_function(X)

    support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
    support_vectors = X[support_vector_indices]

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    c_neg, c_pos = plt.cm.Paired.colors[0], plt.cm.Paired.colors[-1]
    plt.contour(xx, yy, Z, colors=[c_neg, 'black', c_pos], levels=[-1, 0, 1], alpha=1,
                linestyles=['--', '-', '--'])
#     plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
#                 linewidth=1, facecolors='none', edgecolors='k')
    plt.tight_layout()
    plt.title(title)
    plt.show()

def cross_validate_SVM(C, X, y, k=5):
    X = X.to_numpy()
    y = y.to_numpy()
    
    test_size = int(len(y)/k)
    inds = np.arange(len(y))
    np.random.shuffle(inds)
    partitions = np.array_split(inds, k)
    train_scores, test_scores = [], []
    for i in range(k):
        train_inds = np.concatenate(partitions[:i] + partitions[i+1:])
        val_inds = partitions[i]

        train_X = X[train_inds]
        train_Y = y[train_inds]
        test_X = X[val_inds]
        test_Y = y[val_inds]

        clf = svm.LinearSVC(C=C, max_iter=100000)
        clf.fit(train_X, train_Y)
        
        train_scores.append(clf.score(train_X, train_Y))
        test_scores.append(clf.score(test_X, test_Y))
    train_score = np.mean(train_scores)
    test_score = np.mean(test_scores)
    return train_score, test_score

def generate_random_data():
    n = 100
    X1a = np.random.multivariate_normal([-0.3,2], np.array([[0.7, 0.7], [0.7, 1]]), int(3/4*n))
    X1b = np.random.multivariate_normal([0,1.2], np.array([[1, -0.5], [-0.5, 0.5]]), int(1/4*n))
    X2a = np.random.multivariate_normal([1,-1], np.array([[2, 0], [0, 1.2]]), int(2.5/4*n))
    X2b = np.random.multivariate_normal([1.3,1.2], np.array([[0.3, -0.2], [-0.2, 1]]), int(0.5/4*n))
    X2c = np.random.multivariate_normal([-0.5,-1.2], np.array([[1, -0.1], [-0.1, 0.3]]), int(2.5/4*n))
    X1 = np.concatenate((X1a, X1b))
    X2 = np.concatenate((X2a, X2b, X2c)) + np.array([0,-0.5])
    Y1 = np.zeros(len(X1))
    Y2 = np.ones(len(X2))

    X = np.concatenate((X1, X2))
    X = X*(10/(10+(np.linalg.norm(X, axis=1, ord=2, keepdims=True) + np.linalg.norm(X, axis=1, ord=np.inf, keepdims=True)**2)))
    y = np.concatenate((Y1, Y2))
    rand_data = pd.DataFrame({"age": X[:,0], "chol": X[:,1], "target": y})
    return rand_data