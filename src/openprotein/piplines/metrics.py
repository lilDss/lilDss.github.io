from typing import *
from functools import partial
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import spearmanr

from openprotein.core import Components

# __all__ = ["Accuracy"]


class _Metric(metaclass=Components):
    """
    A basic interface for metrics, subclasses need to implement this interface

    Args:
        true (list): 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.
        pred (list): 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

    Returns:
        Current result: score
    """
    """
    Interface class for metrics including three methods (compute_once, update, reset),
    and compute_once is the interface that needs to be implemented by other metrics subclasses.

    Args:
        trues : True values or ground-truth.
        preds : Predicted values generated by the model.
        score : Calculated results or scores for one specific metrics.

    """
    def __init__(self):
        self.preds = []
        self.trues = []
        self.score = 0
        if self._backend == "pt":
            import torch
            self.is_tensor = torch.is_tensor
            self._conver_to_list = lambda x: x.tolist()
        elif self._backend == "ms":
            import mindspore
            self.is_tensor = partial(isinstance, A_tuple=mindspore.Tensor)
            self._conver_to_list = lambda x: x.asnumpy().tolist()
            # TODO: 判断是否为Tensor

    def compute_once(self, true, pred) -> float:
        """
        计算当前输入的计算结果。
        Compute once the metrics.
        Args:
            true (torch.Tensor): True values or ground-truth.
            pred : Predicted values generated by the model.
        """
        raise NotImplemented

    def update(self):
        """
        Calculate and then update one metrics results.
        """
        self.score = self.compute_once(self.trues, self.preds)

    def reset(self):
        """
        reset the parament (preds, trues, score)

        Args:
            No Args.
        """
        """
        Empty internal variables, set preds, trues and score to 0.
        """
        self.preds.clear()
        self.trues.clear()
        self.score = 0

    def __call__(self, true, pred) -> float:
        true = self._conver_to_list(true)
        pred = self._conver_to_list(pred)
        self.trues.extend(true)
        self.preds.extend(pred)
        self.update()
        return self.score

    def __str__(self):
        return "Current result: {}".format({repr(self): self.score})


class MetricUnion(object):
    """
    A class to uniformly calculate the multiple sets of metrics

    Args:
        metrics_list: multiple sets of metrics
    """
    """
    calculates the metrics for model
    """
    """
    Union multiple metrics subclasses, and calculates these metrics results of the model.

    Args:
        metrics_list: list of the union metrics.
        **kwargs: 
    """
    # operators = {"acc": globals()["accuracy"]}
    # _metrics_dict = {"acc": accuracy, "f1": globals()["f1"]}

    def __init__(self, metrics_list: List[_Metric], **kwargs):
        self.preds = []
        self.trues = []
        self.result = {}
        self.load(metrics_list)
        # self.operator = {"acc": globals()["accuracy"]}
        self.operator = {"acc": Accuracy()}

    def load(cls, metrics_list, *args, **kwargs):
        # TODO: user add metrics are support
        # for
        #
        pass

    def compute_once(self, true, pred) -> dict:
        """
        return a dict containing  multiple sets of items like {"acc": Accuracy(true ,pred)}

        Args:
            true (list): 1d array-like, or label indicator array / sparse matrix
                Ground truth (correct) labels.
            pred (list): 1d array-like, or label indicator array / sparse matrix
                Predicted labels, as returned by a classifier.

        Returns:
            a dict of result
        """
        """
        Compute once these union metrics results.

        Args:
            true : True values or ground-truth.
            pred : Predicted values generated by the model.

        Returns:
            results (dict): Dictionary of the union metrics calculate results.
        """
        result = {}
        for key, item in self.operator.items():
            result[key] = item(true, pred)
        return result

    def update(self):
        """
        update the result
        """
        """
        Calculate and then update metrics results.
        """
        self.result = self.compute_once(self.trues, self.preds)

    def reset(self):
        """
        reset the parament (pred、 trues、 result)

        Args:
            No Args.
        """
        """
        Empty internal variables, clear preds, trues and score.
        """
        self.preds.clear()
        self.trues.clear()
        self.result.clear()

    def __call__(self, true, pred) -> dict:
        self.trues.extend(true)
        self.preds.extend(pred)
        self.update()
        return self.result

    def __str__(self) -> str:
        info_template = "Metrics: {}"
        output_template = "{}: {:0.3f}"
        value = "\n".join(output_template.format(key, item) for key, item in self.result.items())
        return info_template.format({value})


class Accuracy(_Metric):
    """
    A class to calculate accuracy classification score.

    Args:
        normalize (bool): default=True.
            If ``False``, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
        sample_weight (n_samples)： default=None，array-like of shape (n_samples,)

    Returns:
        Acc
    """
    """
    Metrics subclass used to calculate the Accuracy classification score.
    The Accuracy classification score is the percentage of all classifications that are correct.

    Args:
        normalize (bool):
            Define return the number(False) or fraction(True) of correctly classified samples.
        sample_weight (array-like of shape n_samples):
            Sample weights.

    """
    def __init__(self, normalize=True, sample_weight=None):
        super().__init__()
        self.normalize = normalize
        self.sample_weight = sample_weight

    def compute_once(self, true: list, pred: list) -> float:
        """
        Accuracy classification score.
        In multilabel classification, this function computes subset accuracy:
        the set of labels predicted for a sample must *exactly* match the
        corresponding set of labels in true.
        Read more in the :ref:`User Guide <accuracy_score>`.

        Args:
            true (list): 1d array-like, or label indicator array / sparse matrix
                Ground truth (correct) labels.
            pred (list): 1d array-like, or label indicator array / sparse matrix
                Predicted labels, as returned by a classifier.

        Returns:
            score (float):
                If ``normalize == True``, return the fraction of correctly
                classified samples (float), else returns the number of correctly
                classified samples (int).

                The best performance is 1 with ``normalize == True`` and the number
                of samples with ``normalize == False``.

        See Also:
            balanced_accuracy_score : Compute the balanced accuracy to deal with
                imbalanced datasets.
            jaccard_score : Compute the Jaccard similarity coefficient score.
            hamming_loss : Compute the average Hamming loss or Hamming distance between
                two sets of samples.
            zero_one_loss : Compute the Zero-one classification loss. By default, the
                function will return the percentage of imperfectly predicted subsets.

        Notes:
            In binary classification, this function is equal to the `jaccard_score`
            function.
        """
        """
        Compute once the Accuracy classification score.

        Args:
            true : True values or ground-truth.
            pred : Predicted values generated by the model.

        Returns:
            Accuracy classification score.
            the fraction of correctly classified samples (float),
            or the number of correctly classified samples (int).

        Examples:
            >>> from openprotein.piplines import Accuracy
            >>> a=Accuracy()
            >>> a.compute_once([0, 2, 1, 3], [0, 1, 2, 3])
            0.5
        """
        return accuracy_score(true, pred, normalize=self.normalize, sample_weight=self.sample_weight)

    def __repr__(self):
        return "Acc"



    # TODO: 待实现
    # rho, _ = stats.spearmanr(predicted, labels)  # spearman
    # mse = mean_squared_error(predicted, labels)  # MSE

    # def __call__(self, true, pred, *, normalize=True, sample_weight=None):
    #     return accuracy_score(true, pred, normalize=normalize, sample_weight=sample_weight)


class MeanSquaredError(_Metric):
    """
    A class to calculate mean squared error regression loss.

    Args:
        sample_weight (n_samples)： default=None，array-like of shape (n_samples,)
        multioutput ：default='uniform_average'
            {'raw_values', 'uniform_average'} or array-like of shape (n_outputs,)
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            'raw_values' :
                Returns a full set of errors in case of multioutput input.
            'uniform_average' :
                Errors of all outputs are averaged with uniform weight.
        squared (bool): default=True
            If True returns MSE value, if False returns RMSE value.

    Returns:
        Mse
    """
    """
    Metrics subclass used to calculate the Mean squared error regression(MSE) loss.
    MSE is the average of the sum of squares of the differences between the true and predicted values

    Args:
        sample_weight (array-like of shape n_samples):
            Sample weights.
        multioutput ({'raw_values', 'uniform_average'} or array-like of shape n_outputs):
            Defines aggregating of multiple output values.
        squared (bool):
            If True returns MSE value, if False returns RMSE value.
    """
    # mse = mean_squared_error(predicted, labels)  # MSE
    def __init__(self, sample_weight=None, multioutput="uniform_average", squared=True):
        super().__init__()
        # Q: 一些参数，具体含义不太懂，需要后续讨论决定是否这样处理，spm类似
        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.squared = squared

    #

    def compute_once(self, true, pred) -> float:
        """
        Mean squared error regression loss.
        Read more in the :ref:`User Guide <mean_squared_error>`.

        Args:
            true: array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.
            pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.

        Returns:
            loss (float or ndarray of floats):
                A non-negative floating point value (the best value is 0.0), or an
                array of floating point values, one for each individual target.
        """
        """
        Compute once the Mean squared error regression(MSE) loss.

        Args:
            true : True values or ground-truth.
            pred : Predicted values generated by the model.

        Returns:
            MSE loss results (float or ndarray of floats).

        Examples:
            >>> from openprotein.piplines import MeanSquaredError
            >>> m=MeanSquaredError()
            >>> m.compute_once([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
            0.375
        """

        # Q:typing需要完善 return float or ndarray of floats
        # Q: mse支持多维输入输出，但是由于前面是extend，上下两次调用之间维数不一致时会报错
        return mean_squared_error(true, pred, sample_weight=self.sample_weight,
                                  multioutput=self.multioutput, squared=self.squared)

    def __repr__(self):
        return "Mse"


class Spearman(_Metric):
    """
    A class to calculate a Spearman correlation coefficient with associated p-value.

    Args:
        axis (int or None): int or None, optional
            If axis=0 (default), then each column represents a variable, with
            observations in the rows. If axis=1, the relationship is transposed:
            each row represents a variable, while the columns contain observations.
            If axis=None, then both arrays will be raveled.
        nan_policy : {'propagate', 'raise', 'omit'}, optional
            Defines how to handle when input contains nan.
            The following options are available (default is 'propagate'):
            * 'propagate': returns nan
            * 'raise': throws an error
            * 'omit': performs the calculations ignoring nan values
        alternative : {'two-sided', 'less', 'greater'}, optional
            Defines the alternative hypothesis. Default is 'two-sided'.
            The following options are available:
            * 'two-sided': the correlation is nonzero
            * 'less': the correlation is negative (less than zero)
            * 'greater':  the correlation is positive (greater than zero)

    Returns:
        Spm
    """
    """
    Metrics subclass used to calculate a Spearman correlation coefficients.
    The Spearman rank-order correlation coefficient is a nonparametric measure
    of the monotonicity of the relationship between two datasets.

    Args:
        axis (int or None, optional):
            Define the 2-D array relationship remains, or be transposed or be raveled.
        nan_policy ({'propagate', 'raise', 'omit'}, optional):
                Defines how to handle when input contains nan.
        alternative ({'two-sided', 'less', 'greater'}, optional):
            Defines the alternative hypothesis. Default is 'two-sided'.
    """
    # rho, _ = stats.spearmanr(predicted, labels)  # spearman
    #Q: 同样存在默认参数和输入输出维数的问题

    def __init__(self, axis=0, nan_policy='propagate', alternative='two-sided'):
        super().__init__()
        self.axis = axis
        self.nan_policy = nan_policy
        self.alternative = alternative

    def compute_once(self, true, pred) -> float:
        """
        Calculate a Spearman correlation coefficient with associated p-value.

        The Spearman rank-order correlation coefficient is a nonparametric measure
        of the monotonicity of the relationship between two datasets. Unlike the
        Pearson correlation, the Spearman correlation does not assume that both
        datasets are normally distributed. Like other correlation coefficients,
        this one varies between -1 and +1 with 0 implying no correlation.
        Correlations of -1 or +1 imply an exact monotonic relationship. Positive
        correlations imply that as x increases, so does y. Negative correlations
        imply that as x increases, y decreases.

        The p-value roughly indicates the probability of an uncorrelated system
        producing datasets that have a Spearman correlation at least as extreme
        as the one computed from these datasets. The p-values are not entirely
        reliable but are probably reasonable for datasets larger than 500 or so.

        Args:
            true、pred (list): 1D or 2D array_like, pred is optional
                One or two 1-D or 2-D arrays containing multiple variables and
                observations. When these are 1-D, each represents a vector of
                observations of a single variable. For the behavior in the 2-D case,
                see under ``axis``, below.
                Both arrays need to have the same length in the ``axis`` dimension.

        Returns:
            rho : float or ndarray (2-D square)
                Spearman correlation matrix or correlation coefficient (if only 2
                variables are given as parameters. Correlation matrix is square with
                length equal to total number of variables (columns or rows) in ``true``
                and ``pred`` combined.
        Warns:
            `~scipy.stats.ConstantInputWarning`
                Raised if an input is a constant array.  The correlation coefficient
                is not defined in this case, so ``np.nan`` is returned.

        References:
            .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
                Probability and Statistics Tables and Formulae. Chapman & Hall: New
                York. 2000.
                Section  14.7
        """
        """
        Compute once the Spearman correlation coefficients.

        Args:
            true : True values or ground-truth.
            pred : Predicted values generated by the model.

        Returns:
            rho (float): Spearman correlation matrix or correlation coefficient.

        Examples:
            >>> from openprotein.piplines import Spearman
            >>> s=Spearman()
            >>> s.compute_once([1, 2, 3, 4, 5], [5, 6, 7, 8, 7])
            0.8207826816681233
        """

        rho, _ = spearmanr(true, pred, axis=self.axis,
                           nan_policy=self.nan_policy, alternative=self.alternative)

        # Q: 如何简化去掉rho中间变量
        return rho

    def __repr__(self):
        return "Spm"


def accuracy(true, pred, *, normalize=True, sample_weight=None):
    """
    Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in true.
    Read more in the :ref:`User Guide <accuracy_score>`.

    Args:
        true (list): 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.
        pred (list): 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.

    Returns:
        score (float):
            If ``normalize == True``, return the fraction of correctly
            classified samples (float), else returns the number of correctly
            classified samples (int).

            The best performance is 1 with ``normalize == True`` and the number
            of samples with ``normalize == False``.

    See Also:
        balanced_accuracy_score : Compute the balanced accuracy to deal with
            imbalanced datasets.
        jaccard_score : Compute the Jaccard similarity coefficient score.
        hamming_loss : Compute the average Hamming loss or Hamming distance between
            two sets of samples.
        zero_one_loss : Compute the Zero-one classification loss. By default, the
            function will return the percentage of imperfectly predicted subsets.

    Notes:
        In binary classification, this function is equal to the `jaccard_score`
        function.
    """
    return accuracy_score(true, pred, normalize=normalize, sample_weight=sample_weight)


def f1(true, pred, *, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn", ):
    """
    Compute the F1 score, also known as balanced F-score or F-measure.
    The F1 score can be interpreted as a harmonic mean of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    In the multi-class and multi-label case, this is the average of
    the F1 score of each class with weighting depending on the ``average``
    parameter.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Args:
        true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) target values.
        pred : 1d array-like, or label indicator array / sparse matrix
            Estimated targets as returned by a classifier.
        labels : array-like, default=None
            The set of labels to include when ``average != 'binary'``, and their
            order if ``average is None``. Labels present in the data can be
            excluded, for example to calculate a multiclass average ignoring a
            majority negative class, while labels not present in the data will
            result in 0 components in a macro average. For multilabel targets,
            labels are column indices. By default, all labels in ``y_true`` and
            ``y_pred`` are used in sorted order.
            .. versionchanged:: 0.17
            Parameter `labels` improved for multiclass problem.
        pos_label : str or int, default=1
            The class to report if ``average='binary'`` and the data is binary.
            If the data are multiclass or multilabel, this will be ignored;
            setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
            scores for that label only.
        average : {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, \
                default='binary'
            This parameter is required for multiclass/multilabel targets.
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``:
                Only report results for the class specified by ``pos_label``.
                This is applicable only if targets (``y_{true,pred}``) are binary.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
            ``'samples'``:
                Calculate metrics for each instance, and find their average (only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division, i.e. when all
            predictions and labels are negative. If set to "warn", this acts as 0,
            but warnings are also raised.

    Returns:
        f1_score : float or array of float, shape = [n_unique_labels]
            F1 score of the positive class in binary classification or weighted
            average of the F1 scores of each class for the multiclass task.

    See Also:
        fbeta_score : Compute the F-beta score.
        precision_recall_fscore_support : Compute the precision, recall, F-score,
            and support.
        jaccard_score : Compute the Jaccard similarity coefficient score.
        multilabel_confusion_matrix : Compute a confusion matrix for each class or
            sample.

    Notes:
        When ``true positive + false positive == 0``, precision is undefined.
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.

    References:
        .. [1] `Wikipedia entry for the F1-score
                <https://en.wikipedia.org/wiki/F1_score>`_.
    """
    return f1_score(true, pred, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight,
                    zero_division=zero_division)
