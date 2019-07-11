import numpy as np 
import matplotlib.pyplot as plt


def softmax(x):
    """(matrix/vector) --> softmax(matrix/vector).

    Defines the softmax of a matris/vector along its columns
    returns matrix/vector of same size
    """
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x), axis=0)
    return ex/sum_ex


def unitNorm(x):
    """(matrix/vector) --> softmax(matrix/vector).
    Returns the L2 normalized columns
    """
    from sklearn import preprocessing
    x_normalized = preprocessing.normalize(x, norm='l2', axis=1)
    return x_normalized


def minDiagonalSwap(nums, index):
    """ Return inplace swaped.

    Swap the diagonal element with the minimum element in the array
    return inplace altered list.
    """
    ind = np.argmin(nums)
    temp = nums[index]
    nums[index] = nums[ind]
    nums[ind] = temp
    return


def GetRawDataBaseline(input):
    """ (list of dict of{'x': tensor, 'y':int, 'lengths':int}) --> (np.array(X), np.array(y))."""
    samples = []
    labels = []
    for samp in input:
        ## to do: check for tensor input and convert it into a arrray
        # samples.append(samp['word'].cpu().numpy())
        samples.append(samp["word"])
        labels.append(samp['y'])
    return samples, np.array(labels)


#  To determine number of samples per class
def NsamplesPerClass(y, isPlot=False):
    """ array of int --> list(int)

    Returns a list of ints, containing number of samples per class.
    """
    n_class = len(set(y))
    n_cls_labels = []
    for cls in range(n_class):
        n_cls_labels.append(sum(y == cls))
    if isPlot:
        plt.figure()
        plt.bar(np.linspace(0,n_class-1,n_class),n_cls_labels)
        plt.xlabel('Class')
        plt.ylabel('Number of labels')
        plt.title('Number of samples per class in clean dataset')
        plt.show()
    return n_cls_labels


def confusionMatrix(y_true, y_pred, isPlot=False):
	"""(array, array) --> Confusion_matrix(n_cls,n_cls)

	Returns a confusion matrix between two class arrays.
	"""
	from sklearn.metrics import classification_report, confusion_matrix
	import seaborn as sn
	import pandas  as pd
	n_cls = len(set(y_true))

	# for ix in range(n_cls):
	#    print(ix, confusion_matrix(y_true, y_pred)[ix].sum())
	cm = confusion_matrix(y_true, y_pred)
	cm = cm/cm.astype(np.float).sum(axis=1)[:, np.newaxis]
	# print(cm)

	if isPlot:
		# Visualizing of confusion matrix
		df_cm = pd.DataFrame(cm, range(n_cls), range(n_cls))
		plt.figure(figsize=(5, 4))
		sn.set(font_scale=1.4)  # for label size
		sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
		plt.ylabel("Prediction")
		plt.xlabel('True')
		plt.show()
	return cm

def plotmat(mat_input, labels=[]):
	import pandas as pd
	import seaborn as sn
	import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 5))
	if labels:
		df_cm = pd.DataFrame(np.around(mat_input, decimals=2), labels, labels)
	else:
		df_cm = pd.DataFrame(np.around(mat_input, decimals=2), range(mat_input.shape[0]), range(mat_input.shape[1]))
	sn.set(font_scale=1.4)  # for label size
	cmap = sn.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap=cmap)  # font size
	plt.yticks(rotation=45)
	plt.xticks(rotation=45)
	plt.show()
	return df_cm



def getStamp():
	""" Returns the current time.
	To append to filename."""
	import time
	val = time.localtime(time.time())
	app = "{}{}{}{}".format(val[3],val[4],val[5], val[6])
	return app
