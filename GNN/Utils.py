import  itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def draw_confusion_matrix(y_test_1d,y_pred_1d, classname,filename='unnamed'):
    true_label=y_test_1d
    pred_label=y_pred_1d

    from sklearn.metrics import f1_score, accuracy_score
    f1_macro = f1_score(true_label, pred_label, average='macro')
    f1_micro = f1_score(true_label, pred_label, average='micro')
    f1_avg = f1_score(true_label, pred_label, average='weighted')
    f1_acc = accuracy_score(true_label, pred_label)
    print("F1_macro, {0}, F1_micro, {1}, F1_avg, {2}, Accuracy, {3}".format(f1_macro, f1_micro, f1_avg, f1_acc))

    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    return plot_confusion_matrix(cnf_matrix,filename, classes=classname, title="Confusion matrix")
    #return plot_confusion_matrix2(cnf_matrix, filename, classes=classname, title="Confusion matrix")

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, filename,classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    #plt.clf()

    #fig=plt.figure(figsize=(10, 10))
    fig = plt.figure()

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename,dpi=fig.dpi, bbox_inches='tight')
    plt.show()
    return plt


def plot_confusion_matrix2(cm, filename,classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename,dpi=400, bbox_inches='tight',pad_inches=0)
    plt.show()


    return plt


def plot_train_val_accuracy(path,info):
    plt.figure()
    train_accs=info['train_accs']
    val_accs=info['val_accs']

    epochs=[i for i in range(len(train_accs))]

    plt.plot(epochs,train_accs)
    plt.plot(epochs,val_accs)

    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    title = info['name'] + ": Epoch vs Accuracy"

    plt.title(title)
    plt.legend(['Train Accuracy','Validation Accuracy'])
    plt.savefig(path+info['name']+'_accuracy.png')
    #plt.show()

def plot_train_val_test_accuracy(path,info):
    plt.figure()
    train_accs=info['train_accs']
    val_accs=info['val_accs']
    test_accs = info['test_accs']

    epochs=[i for i in range(len(train_accs))]

    plt.plot(epochs,train_accs)
    plt.plot(epochs,val_accs)
    plt.plot(epochs, test_accs)

    plt.plot()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    title = info['name'] + ": Epoch vs Accuracy"

    plt.title(title)
    plt.legend(['Train Accuracy','Validation Accuracy','Test Accuracy'])
    plt.savefig(path+info['name']+'_accuracy.png')
    #plt.show()

if __name__ == '__main__':
    y_true=[0,1,2,3,4,5,6,7,8,9,10]
    y_pred=[0,1,2,3,4,5,6,7,8,9,10]

    draw_confusion_matrix(y_true, y_pred, [1,2,3,4,5,6,7,8,9,10,11], filename='testing')