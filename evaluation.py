from __future__ import division
import numpy as np
import sklearn.cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# compute chi square distance between two histograms of LBP
def chi_square_dist(x, y):
    num = pow(x-y, 2)
    denom = x+y
    # when two bins are zero, ignore it so it does not contribute to distance
    with np.errstate(divide='ignore'):
        result = num / denom
        result[denom == 0] = 0
    return np.sum(result)

# train and run cross validation on YALE dataset
def eval_yale():

    data_lbp = np.load('../data/dataset/lbp_norm.npy')
    data_lbp_nonorm = np.load('../data/dataset/lbp.npy')
    data_eigenface = np.load('../data/dataset/lbp_pca.npy')
    label = np.load('../data/dataset/label.npy')

    clf_knn = KNeighborsClassifier(3, metric=chi_square_dist)
    clf_svm = SVC(1, kernel='linear')
    crossval = cv.StratifiedKFold(label, 10)

    # evaluate for LBP with normalization
    acc = cv.cross_val_score(clf_knn, data_lbp, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP-Norm (kNN): %f +- %f' % (np.mean(acc), np.std(acc)))
    acc = cv.cross_val_score(clf_svm, data_lbp, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP-Norm (SVM): %f +- %f' % (np.mean(acc), np.std(acc)))

    # evaluate for LBP without normalization
    acc = cv.cross_val_score(clf_knn, data_lbp_nonorm, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (kNN): %f +- %f' % (np.mean(acc), np.std(acc)))
    acc = cv.cross_val_score(clf_svm, data_lbp_nonorm, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (SVM): %f +- %f' % (np.mean(acc), np.std(acc)))

    # evaluate for eigenface
    acc = cv.cross_val_score(clf_knn, data_eigenface, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (kNN): %f +- %f' % (np.mean(acc), np.std(acc)))
    acc = cv.cross_val_score(clf_svm, data_eigenface, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (SVM): %f +- %f' % (np.mean(acc), np.std(acc)))

# train and run cross validation on LFW dataset
def eval_lfw():
    data_lbp = np.load('../data/dataset/lfw_lbp_norm.npy')
    data_lbp_nonorm = np.load('../data/dataset/lfw_lbp.npy')
    data_eigenface = np.load('../data/dataset/lfw_pca.npy')
    label = np.load('../data/dataset/lfw_label.npy')

    clf_knn = KNeighborsClassifier(3, metric=chi_square_dist)
    clf_svm = SVC(1, kernel='linear')
    crossval = cv.StratifiedKFold(label, 10)

    # evaluate for LBP with normalization
    acc = cv.cross_val_score(clf_knn, data_lbp, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP-Norm (kNN): %f +- %f' % (np.mean(acc), np.std(acc)))
    acc = cv.cross_val_score(clf_svm, data_lbp, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP-Norm (SVM): %f +- %f' % (np.mean(acc), np.std(acc)))

    # evaluate for LBP without normalization
    acc = cv.cross_val_score(clf_knn, data_lbp_nonorm, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (kNN): %f +- %f' % (np.mean(acc), np.std(acc)))
    acc = cv.cross_val_score(clf_svm, data_lbp_nonorm, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (SVM): %f +- %f' % (np.mean(acc), np.std(acc)))

    # evaluate for eigenface
    acc = cv.cross_val_score(clf_knn, data_eigenface, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (kNN): %f +- %f' % (np.mean(acc), np.std(acc)))
    acc = cv.cross_val_score(clf_svm, data_eigenface, label,
                             scoring='accuracy', cv=crossval)
    print('accuracy for LBP (SVM): %f +- %f' % (np.mean(acc), np.std(acc)))


if __name__ == '__main__':
    eval_lfw()

