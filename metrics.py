import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def calculate_metrics(eval_out, eval_truth, args):
    """

    Function to calculate metrics for classifiers (Accuracy, Precision, Recall, F1 score)

    """

    acc = accuracy_score(eval_truth, eval_out)

    # Calculate precision, recall, and F1 score
    # Different methods based on binary or multiclass classification
    if not args.binary:
        prec = precision_score(eval_truth, eval_out, average="macro", zero_division=0)
        recall = recall_score(eval_truth, eval_out, average="macro", zero_division=0)
        f1 = f1_score(eval_truth, eval_out, average="macro", zero_division=0)
    else:
        prec = precision_score(eval_truth, eval_out, zero_division=0)
        recall = recall_score(eval_truth, eval_out, zero_division=0)
        f1 = f1_score(eval_truth, eval_out, zero_division=0)

    return acc, prec, recall, f1


def plot_metrics(eval_out, eval_truth, args, train_arr, valid_arr, best_epoch, epoch):
    """

    Function to plot training/validation losses and confusion matrix

    """

    acc, prec, recall, f1 = calculate_metrics(eval_out, eval_truth, args)

    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Recall:', recall)

    # Training and validation losses
    sns.lineplot(x=range(1, epoch + 1), y=train_arr, label="Training Loss")
    sns.lineplot(x=range(1, epoch + 1), y=valid_arr, label="Validation Loss")
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Early stopping')
    plt.title(f'Losses | binary: {args.binary} | batch_norm: {args.batch_norm}')
    plt.show()

    # Training and validation losses on a logarithmic scale
    sns.lineplot(x=range(1, epoch + 1), y=train_arr, label="Training Loss")
    sns.lineplot(x=range(1, epoch + 1), y=valid_arr, label="Validation Loss")
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Early stopping')
    plt.title(f'Log Losses | binary: {args.binary} | batch_norm: {args.batch_norm}')
    plt.yscale("log")
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(eval_truth, eval_out)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(f'Confusion Matrix | binary: {args.binary} | batch_norm: {args.batch_norm}')
    plt.show()
