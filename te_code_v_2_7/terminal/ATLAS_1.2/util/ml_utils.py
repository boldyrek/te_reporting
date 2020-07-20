###################################################################################################
# Cerebri AI CONFIDENTIAL
# Copyright (c) 2017-2020 Cerebri AI Inc., All Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of Cerebri AI Inc.
# and its subsidiaries, including Cerebri AI Corporation (together “Cerebri AI”).
# The intellectual and technical concepts contained herein are proprietary to Cerebri AI
# and may be covered by U.S., Canadian and Foreign Patents, patents in process, and are
# protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from Cerebri AI. Access to the
# source code contained herein is hereby forbidden to anyone except current Cerebri AI
# employees or contractors who have executed Confidentiality and Non-disclosure agreements
# explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended publication or
# disclosure of this source code, which includes information that is confidential and/or
# proprietary, and is a trade secret, of Cerebri AI. ANY REPRODUCTION, MODIFICATION,
# DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE OF THIS SOURCE
# CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF CEREBRI AI IS STRICTLY PROHIBITED, AND IN
# VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF
# THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR SELL
# ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
###################################################################################################
#!/usr/bin/env python3
"""
Consists of functions to measure ml model metrics
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, brier_score_loss
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import OrderedDict
import shap
import matplotlib.pyplot as plt
from itertools import product


def classifier_metrics(y_train, train_prob, y_test, test_prob, test_pred):
    """
    Calculates model performance metrics using the true labels, predictions, and probabilities.

    Parameters:
            y_train (1d array-like): array contains train label.
            train_prob (1d array-like): array contains predicted prob for train.
            y_test (1d array-like): array contains test label.
            test_prob (1d array-like): array contains predicted prob for test.
            test_pred (1d array-like): array contains prediction for test.

    Returns:
            df: dataframe containing performance metrics (accuracy, precision, recall, f1, roc_auc),
            normalized confusion matrix (tpr, tnr, 1-fpr, 1-fnr), Brier gain and KS score,
            lift statistic and lift statistic for two decile.
    """
    # Traditional metrics
    d = OrderedDict()
    d["accuracy"] = accuracy_score(y_test, test_pred)
    d["precision"] = precision_score(y_test, test_pred)
    d["recall"] = recall_score(y_test, test_pred)
    d["f1"] = f1_score(y_test, test_pred)
    d["roc_auc"] = roc_auc_score(y_test, test_prob)
    # Normalized confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    norm_cm  = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    tnr, fpr, fnr, tpr = norm_cm.ravel()
    d["tpr"] = tpr
    d["tnr"] = tnr
    d["1-fpr"] = 1 - fpr
    d["1-fnr"] = 1 - fnr
    # Brier gain and KS score
    d["brier_gain"] = 1 - brier_score_loss(y_test, test_prob)
    ks = stats.ks_2samp(test_prob[y_test == 1], train_prob[y_train == 1])
    d["1-ks"] = 1 - ks[0]
    # Create df with test labels and probabilities and then create lift table from them
    y_test = y_test.rename('label')
    df_y = pd.DataFrame(y_test, columns=["label"])
    df_y["probability"] = test_prob
    lift_df = lift_metric_table(df_y, "label", "probability")
    # Convert gain back to float, use np.trapz to find gain AUC, and then normalize
    gain = lift_df["Cumulative Percent of All Positive Timelines"].str.rstrip("%").astype("float")
    gain_auc = np.trapz(gain) / (100 * len(gain))
    d["lift_statistic"] = gain_auc / (1 - (np.sum(y_test == 1) / y_test.shape[0]))

    gain_1 = lift_df["Cumulative Percent of All Positive Timelines"].str.rstrip("%").astype("float")[0:2]
    gain_auc_1 = np.trapz(gain_1) / (100 * len(gain_1))
    d["lift_statistic_two_deciles"] = gain_auc_1 / (1 - (np.sum(y_test == 1) / y_test.shape[0]))

    return pd.DataFrame([d])


def lift_metric_table(df, target, score, number_bins=10):
    """
    Creates a lift table for the evaluation of a predictive model.

    Parameters:
            df (dataframe):  dataframe with true test labels and predicted probabilities.
            target (str): name of target column in the df.
            score (str):  name of sorting column in the df (e.g. probability of predicted target).
            number_bins (int): number of bins for table (default value = 10).

    Returns:
            lift table: dataframe with lift metrics.
    """
    # Group the data into n equal sized groups
    # The grouping is done by the predicted probability
    df["negative"] = 1 - df[target]
    df.sort_values(score, ascending=False, inplace=True)
    df["idx"] = range(1, len(df) + 1)
    df["bins"] = pd.cut(df["idx"], bins=number_bins, right=True, retbins=False, precision=3)
    # Obtain summary information for each group
    aggregated = df.groupby("bins", as_index=False)
    lift_table = pd.DataFrame(np.vstack(aggregated.min()[score]), columns=["min_score"])
    lift_table.sort_values("min_score", ascending=False, inplace=True)
    lift_table["Decile"] = np.arange(1, (len(df.bins.unique()) + 1))
    # Add probabilities and timeline count
    lift_table["Minimum Probability"] = (100 * aggregated.min()[score]).map("{:,.0f}%".format)
    lift_table["Maximum Probability"] = (100 * aggregated.max()[score]).map("{:,.0f}%".format)
    timelines = aggregated.sum()[target] + aggregated.sum()["negative"]
    lift_table["Total Timelines"] = timelines.map("{:,}".format)
    # Calculate positive class proportions and percent of positive timelines
    lift_table["Positive Class Proportion"] = (100 * aggregated.sum()[target] / timelines).map("{:,.0f}%".format)
    pct_positive_all_timelines = aggregated.sum()[target] / aggregated.sum()[target].sum()
    lift_table["Percent Of All Positive Timelines"] = (100 * pct_positive_all_timelines).map("{:,.0f}%".format)
    # Calculate cumulative positve class proportion (gain) and lift
    cum_pct_positive = 100 * pct_positive_all_timelines.cumsum()
    lift_table["Cumulative Percent of All Positive Timelines"] = (cum_pct_positive).map("{:,.0f}%".format)
    lift_table["Lift"] = cum_pct_positive / (lift_table["Decile"] * (100 / number_bins))

    return lift_table.drop("min_score", axis=1)


def lift_table(results_df, bins=10):
    """
    Creates a lift table for the evaluation of a predictive model.

    Parameters:
            results_df (dataframe): dataframe with 4 columns - account id, actual label('label'),
            predicted probability ('probability') and classfier prediction ('prediction').
            bins (int): number of bins for table (default value = 10).

    Returns:
            lift_table: dataframe with lift metrics.
    """

    results = results_df[["label", "probability"]].sort_values(by="probability", ascending=False).copy(deep=True)

    labels = [
        "0 - 10",
        "11 - 20",
        "21 - 30",
        "31 - 40",
        "41 - 50",
        "51 - 60",
        "61 - 70",
        "71 -80",
        "81 - 90",
        "91 - 100",
    ]

    Total_rate = results["label"].sum() / results["label"].count() * 100

    results["prob_bin"] = pd.qcut(results["probability"].rank(method="first"), bins, labels=labels)

    lift_table = pd.DataFrame(index=results["prob_bin"].unique())

    lift_table["Minimum_Probability"] = results["probability"].groupby(results["prob_bin"]).min()
    lift_table["Maximum_Probability"] = results["probability"].groupby(results["prob_bin"]).max()

    lift_table["Total_Timelines"] = results["probability"].groupby(results["prob_bin"]).size()
    lift_table["act_positives"] = results["label"].groupby(results["prob_bin"]).sum()
    lift_table["Positive_Class_Proportion"] = round(
        lift_table["act_positives"] / lift_table["Total_Timelines"] * 100, 1
    ).astype(str)

    lift_table["Percent_Of_All_Positive_Timelines"] = round(
        lift_table["act_positives"] / results["label"].sum() * 100, 1
    ).astype(str)

    lift_table["cum_positives"] = lift_table["act_positives"].cumsum()
    lift_table["cum_timeline"] = lift_table["Total_Timelines"].cumsum()
    lift_table["cum_rate"] = round(lift_table["cum_positives"] / lift_table["cum_timeline"] * 100, 1).astype(str)
    lift_table["Cumulative_Percent_Of_All_Positive_Timelines"] = round(
        lift_table["Percent_Of_All_Positive_Timelines"].astype(float).cumsum(), 1
    ).astype(str)

    lift_table["lift_baseline"] = lift_table["Positive_Class_Proportion"].astype(float) / Total_rate
    lift_table["lift_cumulative"] = lift_table["cum_rate"].astype(float) / Total_rate

    return lift_table.drop(["cum_positives", "cum_timeline", "cum_rate"], axis=1)


def plot_confusion_matrix(test_df, labels, savepath):
    """
    Plot confusion matrix from the 2x2 matrix of values.

    Parameters:
            test_df (dataframe): test dataframe contains label and prediction.
            labels (str): name of target column in the test_df.
            savepath (str): path to where the confusion matrix plot should be saved.

    Returns:
            none.
    """

    tn, fp, fn, tp = confusion_matrix(test_df["label"], test_df["prediction"]).ravel()
    cm = np.array([[tp, fp], [fn, tn]])

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()
    plt.xticks(np.arange(2), labels)
    plt.yticks(np.arange(2), labels)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:,.0f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() * 2 / 3 else "black",
        )
    plt.xlabel("True Class", fontsize=12)
    plt.ylabel("Predicted Class", fontsize=12)
    plt.savefig(savepath+'_confusion_matrix.png', bbox_inches="tight")
    plt.close()


def plot_ks_compare_train_test(y_train, train_proba, y_test, test_proba, savepath, bins=30, fig_sz=(10, 8)):
    """
    Plot KS test/train overtraining for classifier output.

    Parameters:
        y_train (1d array-like): array containing train label.
        train_prob (1d array-like): array containing predicted probability for train.
        y_test (1d array-like): array containing test actual label.
        test_prob (1d array-like): array containing predicted probability for test.
        savepath (str): path to where the confusion matrix plot should be saved.
        bins (int): number of bins for viz  (default value = 30).
        fig_sz (float,float): width and height of the plotted figure in inches (default values = (10, 8)).

    Returns:
            none.
    """

    train = pd.DataFrame(y_train, columns=["label"])
    test = pd.DataFrame(y_test, columns=["label"])
    train["probability"] = train_proba
    test["probability"] = test_proba

    decisions = []
    for df in [train, test]:
        d1 = df["probability"][df["label"] == 1]
        d2 = df["probability"][df["label"] == 0]
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    fig = plt.figure(figsize=fig_sz)

    train_pos = plt.hist(
        decisions[0],
        color="r",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="+ (train)",
    )

    train_neg = plt.hist(
        decisions[1],
        color="b",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="- (train)",
    )

    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    test_pos = plt.errorbar(center, hist, yerr=err, fmt="o", c="r", label="+ (test)")

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    test_neg = plt.errorbar(center, hist, yerr=err, fmt="o", c="b", label="- (test)")
    # get the KS score
    ks = stats.ks_2samp(decisions[0], decisions[2])
    plt.xlabel("Classifier Output", fontsize=12)
    plt.ylabel("Arbitrary Normalized Units", fontsize=12)
    plt.xlim(0, 1)
    plt.plot([], [], " ", label="KS Statistic (p-value) :" + str(round(ks[0], 2)) + "(" + str(round(ks[1], 2)) + ")")
    plt.legend(loc="best", fontsize=12)
    plt.savefig(savepath+'_ks_compare_train_test.png', bbox_inches="tight")
    plt.close()


def plot_spider_diagram(metrics_dict, savepath):
    """
    Visualizes all model performance metrics on a spider diagram.

    Parameters:
        metrics_dict (dict): keys are the names of the models and values are the dataframes with metrics names and values.
        savepath (str): path to where the spider plot should be saved.

    Returns:
            none.
    """
    # Get count of metrics
    df = next(iter(metrics_dict.values()))
    N = df.shape[1]
    metric_names = df.columns

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    # Initialize the spider plot
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, polar=True)
    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], metric_names, fontsize=12)
    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.tick_params(axis="x", pad=20)
    plt.yticks([0.0, 0.5, 0.75, 1.0], ["0.0", "0.50", "0.75", "1.0"], color="grey", fontsize=10)
    plt.ylim(0.0, 1.0)
    # add each model to the plot
    for model, df in metrics_dict.items():
        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values = df.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1, label=model)
        # Fill area
        ax.fill(angles, values, alpha=0.1)
    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), prop={"size": 14})
    plt.savefig(savepath+'_spider.png', bbox_inches="tight")
    plt.close()


def shap_feature_importances(clf, df_x, savepath):
    """
    Plot shap importances.

    Parameters:
        clf (model object): trained model
        df_x (dataframe): dataframe containing features in test set
        savepath (string): path to where the confusion matrix plot should be saved

    Returns:
            none.
    """

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df_x.values)
    fig = shap.summary_plot(shap_values, df_x, feature_names=df_x.columns, show=False)    
    plt.savefig(savepath+'_shap_summary.png', bbox_inches="tight")
    plt.close()
