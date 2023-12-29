import gc
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    plot_confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)

from atomai.utils import cv_thresh
from atomai.predictors import EnsemblePredictor

plot_out_path = "output/plots"


def plot_pr_multiclass(name, labels, predictions, n_classes=3, **kwargs):
    precision = dict()
    recall = dict()
    thresholds = dict()
    mAP = dict()

    for i in range(n_classes):
        y_test = (labels == i).astype(int)
        # y_pred = (predictions == i).astype(int)
        y_pred = predictions[:, :, :, i].flatten()
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test, y_pred)
        mAP[i] = average_precision_score(y_test, y_pred)
        # pr_display = sklearn.metrics.PrecisionRecallDisplay(precision=precision[i], recall=recall[i]).plot()

    plt.figure(figsize=(4, 4))
    no_skill = len(labels[labels == 1]) / len(labels)
    plt.plot([0, 1], [no_skill, no_skill], "k--", linestyle="--", label="No Skill")

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate(
            "f1={0:0.1f}".format(f_score),
            xy=(0.85, y[45] + 0.02),
            alpha=0.66,
            fontsize=14,
        )

    lw = 2
    colors = cycle(["blue", "red", "green", "purple"])
    for i in range(n_classes):
        plt.plot(
            recall[i],
            precision[i],
            lw=lw,
            label="Class {0} (mAP = {1:0.2f})".format(i, mAP[i]),
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Percision", fontsize=14)
    # plt.title("Percision-Recall Curve".format(name))
    # add the legend for the iso-f1 curves
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    handles.extend([l])
    legend_labels.extend(["iso-f1 curves"])
    plt.legend(handles=handles, labels=legend_labels, loc="best", fontsize=12)
    plt.gca().set_aspect("equal")

    plt.savefig(plot_out_path + "/PR.eps", bbox_inches="tight", format="eps", dpi=900)
    plt.savefig(plot_out_path + "/PR.svg", bbox_inches="tight", format="svg", dpi=900)

    plt.show()

    del precision
    del recall
    del thresholds
    del mAP
    gc.collect()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect("equal")


def plot_roc_multiclass(name, labels, predictions, n_classes=3, **kwargs):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(4, 4))
    for i in range(n_classes):
        y_test = (labels == i).astype(int)
        # y_pred = (predictions == i).astype(int)
        y_pred = predictions[:, :, :, i].flatten()
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    #         roc_display = sklearn.metrics.RocCurveDisplay(fpr=fpr[i], tpr=tpr[i]).plot()

    lw = 2
    colors = cycle(["blue", "red", "green", "purple"])
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="Class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label="No Skill")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("False Positive Rate (%)", fontsize=14)
    plt.ylabel("True Positive Rate (%)", fontsize=14)
    # plt.title(
    #     "Receiver Operating Characteristic".format(name),
    #     fontsize=14,
    # )
    plt.legend(loc="lower right", fontsize=12)
    plt.gca().set_aspect("equal")
    plt.grid(color="gray", alpha=0.2)

    plt.savefig(plot_out_path + "/roc.svg", bbox_inches="tight", format="svg", dpi=900)
    plt.savefig(plot_out_path + "/roc.eps", bbox_inches="tight", format="eps", dpi=900)
    plt.show()

    del fpr
    del tpr
    del roc_auc
    gc.collect()


def eval_ensemble(smodel, ensemble, images_test, labels_test, thresh=0.5):
    predictor = EnsemblePredictor(
        smodel, ensemble, nb_classes=3, output_shape=(1, 3, 256, 256)
    )

    y_pred = []
    for image in images_test:
        nn_output, nn_out_var = predictor.predict(image)
        nn_output = nn_output.squeeze()
        y_pred.append(nn_output)

    y_pred = np.array(y_pred)

    y_class = []
    for pred in y_pred:
        decoded_thresh = []
        for ch in range(pred.shape[2]):
            decoded_img_c = cv_thresh(pred[:, :, ch], thresh)
            decoded_img_c[decoded_img_c == 1] = ch
            # print(np.max(decoded_img_c))
            decoded_thresh.append(decoded_img_c)

        # pred[:,:,0] = 1-pred[:,:,0]
        # pred = np.sum(pred, axis=2)
        dec_class = np.array(decoded_thresh)
        # print(dec_class.shape)
        dec_class = np.sum(dec_class, axis=0)
        # print(np.max(dec_class), np.unique(dec_class))
        y_class.append(dec_class)

    y_class = np.array(y_class)

    flat_classes = np.concatenate(y_class).flatten()
    flat_test = labels_test.flatten()

    ConfusionMatrixDisplay.from_predictions(
        flat_test,
        flat_classes,
        normalize="true",
        display_labels=["Background", "Troughs", "Peaks"],
        cmap="Blues",
    )
    plt.savefig("output/plots/CM-full.svg", format="svg", bbox_inches="tight", dpi=900)
    plt.show()

    cm = confusion_matrix(flat_test, flat_classes, labels=[1, 2], normalize="true")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Troughs", "Peaks"]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.rcParams.update({"font.size": 14})
    disp.plot(cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    plt.savefig(plot_out_path + "/CM.svg", format="svg", bbox_inches="tight", dpi=1200)
    plt.show()

    # plot_roc_multiclass("Test", flat_test, y_pred)
    # plot_pr_multiclass("Test", flat_test, y_pred)

    print(
        classification_report(
            flat_test, flat_classes, target_names=["background", "troughs", "peaks"]
        )
    )

    del y_pred
    del y_class
    del flat_classes
    del flat_test
    gc.collect()
