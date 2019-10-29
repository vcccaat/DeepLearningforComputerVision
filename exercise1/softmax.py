"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_train_sample = X.shape[0]  #row of train data
    num_class = W.shape[1] #column of weight, plane,horse..
    for i in range(num_train_sample):
        p_score = X[i].dot(W)    #a row of score corresponding to each class
        p_score -= np.max(p_score)  #normalize, highest is 1

        ###compute softmax loss
        # sum of scores corresponding to different classes of a sample 
        sum_score = np.sum(np.exp(p_score))  
        # each class's score over sum_score of a sample 
        score_i = lambda k: np.exp(p_score[k]) / sum_score
        # for the correct label in each sample, find softmax loss over sum
        # iteration make loss sum up all samples
        loss = loss - np.log(score_i(y[i]))

        for k in range(num_class):
            p_k = score_i(k)
            # gradient of softmax
            dW[:, k] += (p_k - (k == y[i])) * X[i]

    loss /= num_train_sample
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train_sample
    dW += reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_train_sample = X.shape[0]  
    num_class = W.shape[1]
    # matrix of score of all samples and all class
    p_score = X.dot(W)
    # normalize
    p_score -= np.max(p_score,axis = 1,keepdims = True)
    # vector
    sum_score = np.sum(np.exp(p_score), axis=1, keepdims=True)
    # element-wise division
    score_i = np.exp(p_score)/sum_score
    # loss = -log(P(y))   y is all sample label, P(y) is their scores
    loss = np.sum(-np.log(score_i[np.arange(num_train_sample), y]))

    ind = np.zeros_like(score_i)
    ind[np.arange(num_train_sample), y] = 1
    # X:n*m  W:m*k    score: n*k  formular to find X*score---x transpose dot score
    dW = X.T.dot(score_i - ind)

    loss /= num_train_sample
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train_sample
    # dW += reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    num = 3000
    # learning_rates = np.geomspace(1e-6, 5e-8, num=3)
    # regularization_strengths = np.geomspace(2.5e4, 5e4, num=2)

    for lr in learning_rates:
        # classifier_i =[] 
        for rs in regularization_strengths:
            classifier = SoftmaxClassifier()
            classifier.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=num)
            train_pred = classifier.predict(X_train)
            train_accu = np.mean(train_pred == y_train)
            val_pred = classifier.predict(X_val)
            vali_accu = np.mean(val_pred == y_val)
            results[(lr, rs)] = (train_accu, vali_accu)

            if best_val < vali_accu:
                best_val = vali_accu
                best_softmax = classifier
        #      classifier_i.append(classifier)
        # all_classifiers.append(classifier_i)
                
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
