import numpy as np
import sklearn.metrics as sk


def right_wrong_distinction(model, test_images, test_labels):

    softmax_all, _ = model.predict(test_images)
    right_all, wrong_all = split_right_wrong(softmax_all, test_labels)

    (s_prob_all, kl_all, mean_all, var_all) = entropy_stats(softmax_all)
    (s_prob_right, kl_right, mean_right, var_right) = entropy_stats(right_all)
    (s_prob_wrong, kl_wrong, mean_wrong, var_wrong) = entropy_stats(wrong_all)

    correct_cases = np.equal(np.argmax(softmax_all, 1), test_labels)
    accuracy = 100 * np.mean(np.float32(correct_cases))
    err = 100 - accuracy

    print("\n[Error and Success Prediction]")
    print('\nPrediction Prob (mean, std) | PProb Right (mean, std) | PProb Wrong (mean, std)')
    print(np.mean(s_prob_all), np.std(s_prob_all), '|',
          np.mean(s_prob_right), np.std(s_prob_right), '|',
          np.mean(s_prob_wrong), np.std(s_prob_wrong))

    print('\nSuccess base rate (%):', round(accuracy,2),
          "({}/{})".format(len(right_all), len(softmax_all)))
    print('KL[p||u]: Right/Wrong classification distinction')
    print_curve_info(kl_right, kl_wrong)
    print('Prediction Prob: Right/Wrong classification distinction')
    print_curve_info(s_prob_right, s_prob_wrong)

    print('\nError base rate (%):', round(err,2),
          "({}/{})".format(len(wrong_all), len(softmax_all)))
    print('KL[p||u]: Right/Wrong classification distinction')
    print_curve_info(-kl_right, -kl_wrong, True)
    print('Prediction Prob: Right/Wrong classification distinction')
    print_curve_info(-s_prob_right, -s_prob_wrong, True)

    return (s_prob_right, s_prob_wrong, kl_right, kl_wrong)


def in_out_distinction(model, model2, indist_images, outdist_images):
    
    softmax_indist, _ = model.predict(indist_images)
    softmax_outdist, _ = model.predict(outdist_images)
    
    pseudo_indist, _ = model2.predict(indist_images)
    pseudo_outdist, _ = model2.predict(outdist_images)
    
    (s_prob_in, _, _, _) = entropy_stats(softmax_indist)
    (s_prob_out, _, _, _) = entropy_stats(softmax_outdist)
    
    (pseudo_prob_in, _, _, _) = entropy_stats(pseudo_indist)
    (pseudo_prob_out, _, _, _) = entropy_stats(pseudo_outdist)
    
    print("\n[In- and Out-of-Distribution Detection (MSP)]\n")
    
    print('In-dist max softmax distribution (mean, std):')
    print(np.mean(s_prob_in), np.std(s_prob_in))
    print('Out-of-dist max softmax distribution(mean, std):')
    print(np.mean(s_prob_out), np.std(s_prob_out))
    
    print('\nNormality Detection')
    print('Normality base rate (%):',
          round(100*indist_images.shape[0]/(outdist_images.shape[0]+indist_images.shape[0]),2))
    print('Prediction Prob: Normality Detection')
    print_curve_info(s_prob_in, s_prob_out)
    
    print('\nAbnormality Detection')
    print('Abnormality base rate (%):',
          round(100*outdist_images.shape[0]/(outdist_images.shape[0]+indist_images.shape[0]),2))
    print('Prediction Prob: Abnormality Detection')
    print_curve_info(-s_prob_in, -s_prob_out, True)
    
    
    print("\n[In- and Out-of-Distribution Detection (OE)]\n")
    
    print('In-dist max softmax distribution (mean, std):')
    print(np.mean(pseudo_prob_in), np.std(pseudo_prob_in))
    print('Out-of-dist max softmax distribution(mean, std):')
    print(np.mean(pseudo_prob_out), np.std(pseudo_prob_out))
    
    print('\nNormality Detection')
    print('Normality base rate (%):',
          round(100*indist_images.shape[0]/(outdist_images.shape[0]+indist_images.shape[0]),2))
    print('Prediction Prob: Normality Detection')
    print_curve_info(pseudo_prob_in, pseudo_prob_out)
    
    print('\nAbnormality Detection')
    print('Abnormality base rate (%):',
          round(100*outdist_images.shape[0]/(outdist_images.shape[0]+indist_images.shape[0]),2))
    print('Prediction Prob: Abnormality Detection')
    print_curve_info(-pseudo_prob_in, -pseudo_prob_out, True)
    
    return (s_prob_in, s_prob_out, pseudo_prob_in, pseudo_prob_out)


def split_right_wrong(softmax_all, label):
    mask_right = np.equal(np.argmax(softmax_all, axis=1), label)
    mask_wrong = np.not_equal(np.argmax(softmax_all, axis=1), label)
    right, wrong = softmax_all[mask_right], softmax_all[mask_wrong]
    return right, wrong


def entropy_stats(softmax):
    s_prob = np.amax(softmax, axis=1, keepdims=True)
    kl_all = entropy_from_distribution(softmax, axis=1)
    mean_all, var_all = np.mean(kl_all), np.var(kl_all)
    return s_prob, kl_all, mean_all, var_all


def entropy_from_distribution(p, axis):
    return np.log(10.) + np.sum(p * np.log(np.abs(p) + 1e-11), axis=1, keepdims=True)


def print_curve_info(safe, risky, inverse=False):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    if inverse:
        labels[safe.shape[0]:] += 1
    else:
        labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    print('AUPR (%):', round(100*sk.average_precision_score(labels, examples), 2))
    print('AUROC (%):', round(100*sk.roc_auc_score(labels, examples), 2))

    
def reconstruction_error(images, risk):
    y_true=images.reshape(len(images), -1)
    y_pred=risk.reshape(len(risk), -1)
    re=np.mean((y_true - y_pred) ** 2, axis=-1)
    return re