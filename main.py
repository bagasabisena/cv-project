import feature
import evaluation


# extract features for YALE
# extract Tan-Triggs normalized LBP
feature.lbp_uni_feature(True, 'lbp_norm.npy')
# extract non-normalize LBP
feature.lbp_uni_feature(False, 'lbp.npy')
# extract eigenface, 10 components
feature.eigenface_feature(10, 'lbp_pca.npy')
# get label
feature.save_label()

# extract features for LFW
# extract Tan-Triggs normalized LBP, get only min of person with 20 images
feature.lfw_lbp(True, 'lfw_lbp_norm.npy', 20)
# extract non normalized LBP, get only min of person with 20 images
feature.lfw_lbp(False, 'lfw_lbp.npy', 20)
# extract eigenface, 100 components, get only min of person with 20 images
feature.lfw_eigenface('lfw_pca.npy', 100, 20)
# get label
feature.lfw_label('lfw_label.npy', 20)

# train and evaluate
evaluation.eval_yale()
evaluation.eval_lfw()
