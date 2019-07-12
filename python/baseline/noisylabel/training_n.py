import os
import numpy as np
import pickle
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import baseline as bl
from baseline.pytorch import *
import logging
import mead
from baseline.utils import read_config_stream, normalize_backend
from mead.utils import convert_path, parse_extra_args, configure_logger

from utilities import GetRawDataBaseline, confusionMatrix, NsamplesPerClass
from noise_model import inject_noise

logger = logging.getLogger('mead')
class e2e_noise_model(object):
    """ Describes the end to end model. """

    def __init__(self, Bmodel_params, Nmodel_params):
        """Initialize the nmosie model parmeters:
        Bmodel_params: Json parameters for base model such as sst2-pytorch.json
        Nmodel_params: Json parameters for all the nosie models such as Noise_model_pysst2.json
        """
        self.Bmodel_params = Bmodel_params
        self.Bmodel_params = Bmodel_params
        self.Nmodel_params = Nmodel_params

    @staticmethod
    def classSpecificData(data, classes_tk):
        """ Used by combat skew to select samples from a set of classes."""
        ext = []
        for xx in data:
            if xx['y'] in classes_tk:
                xx['y'] = classes_tk.index(xx['y'])
                ext.append(xx)
        return ext

    def creatDataProfile(self, BLconfigFile, Cfile_dir, jsonFileToLoad, combatSkew=False):
        """
        To ensure the reproducibility of results store the processed data file at location BLconfigFile
        type(Cfile_dir): file location: where all the config files available

        :param BLconfigFile: <path> to store the processed file.
        :param Cfile_dir:  <path> of the config file
        :param jsonFileToLoad: <str> config file
        :param combatSkew: <bool> for now keep it False
        :return:
        """

        if os.path.isfile(BLconfigFile):
            print('Loading from the disk')
            self.embeddings, self.train_data, self.valid_data, self.test_data, self.labels = pickle.load(
                open(BLconfigFile, "rb"))
            _, self.y_train = GetRawDataBaseline(self.train_data.examples)
        else:
            config_params = bl.utils.read_config_file(Cfile_dir + jsonFileToLoad)
            # task_name = config_params.get('task', 'classify')
            # print('Task: [{}]'.format(task_name))
            # # Cfile_dir = '/home/ijindal/dev/work/baseline/python/mead/config/'
            # # print(bl.utils.read_config_file(file_dir + 'sst2.json'))
            # confi_task = md.Task.get_task_specific(task_name, Cfile_dir + 'logging.json',
            #                                        Cfile_dir + 'mead-settings.json')
            # confi_task.read_config(config_params, Cfile_dir + 'datasets-classify.json')
            # # confi_task.initialize( '/home/ijindal/dev/work/vaseline/python/mead/config/embeddings.json')
            # confi_task.initialize('/home/ijindal/dev/work/baseline/python/mead/config/' + 'embeddings.json')
            # print(confi_task)

            dataset_params = bl.utils.read_config_stream(Cfile_dir + 'datasets.json')
            settings_params = bl.utils.read_config_stream(Cfile_dir + 'mead-settings.json')
            embeddings_params = bl.utils.read_config_stream(Cfile_dir + 'embeddings.json')
            logging_params = bl.utils.read_config_stream(Cfile_dir + 'logging.json')

            logging_params = read_config_stream(logging_params)
            configure_logger(logging_params)
            dataset_params = read_config_stream(dataset_params)
            embeddings_params = read_config_stream(embeddings_params)
            settings_params = read_config_stream(settings_params)
            # config_params['backend'] = 'pytorch'
            task_name = config_params['task']
            print('Task: [{}]'.format(task_name))
            logger.info('Task: [{}]'.format(task_name))
            confi_task = mead.Task.get_task_specific(task_name, settings_params)

            config_params['reporting'] = {}
            config_params['model']['gpus'] = 1
            confi_task.read_config(config_params, dataset_params, reporting_args=[])
            confi_task.initialize(embeddings_params)


            confi_task._load_dataset()
            self.embeddings = confi_task.embeddings
            self.train_data = confi_task.train_data
            self.valid_data = confi_task.valid_data
            self.test_data = confi_task.test_data
            self.labels = confi_task.labels

            _, self.y_train = GetRawDataBaseline(self.train_data.examples)

            pickle.dump([confi_task.embeddings,
                         confi_task.train_data,
                         confi_task.valid_data,
                         confi_task.test_data,
                         confi_task.labels], open(BLconfigFile, "wb"))

            # print(confi_task.train_data.examples.example_list[0])

        if combatSkew:
            # Needs further investigation.
            print("Skew Combat Training is OOONNN...")
            skew_dist = NsamplesPerClass(self.y_train, isPlot=False)
            classes_tk = [x for x in range(len(skew_dist)) if
                          (skew_dist[x] < max(skew_dist) / 15) or (skew_dist[x] > max(skew_dist) / 7)]
            # (skew_dist[x] < max(skew_dist)/20) or (skew_dist[x] > max(skew_dist)/5)
            print(classes_tk)

            self.train_data.examples.example_list = self.classSpecificData(self.train_data.examples.example_list,
                                                                           classes_tk)
            self.test_data.examples.example_list = self.classSpecificData(self.test_data.examples.example_list,
                                                                          classes_tk)
            self.valid_data.examples.example_list = self.classSpecificData(self.valid_data.examples.example_list,
                                                                           classes_tk)
            self.labels = [self.labels[classes_tk[i]] for i in range(len(classes_tk))]
            _, y_test = GetRawDataBaseline(self.test_data.examples)
            self.test_data.steps = len(y_test)
            _, self.y_train = GetRawDataBaseline(self.train_data.examples)

        return

    def creatNoiseProfile(self, file_name, noisetype, noiselvl):
        """
        Creates noise distribution and save noisy labels to the disk for reproducibility.
        :param file_name: <str> generate the noisyu labels and save the files to reproduce the results
        :param noisetype: <str> type of label noise.
        :param noiselvl: <float> varies [0,0, 0.9]
        :return:
        """
        if noiselvl == 0:
            self.y_noisy_train = self.y_train
        else:
            print(file_name)
            if os.path.isfile(file_name):
                print('Noisy data exist \n Loading from the disk....')
                self.N_train_data, self.N_valid_data = pickle.load(open(file_name, "rb"))

            else:
                _, y_valid = GetRawDataBaseline(self.valid_data.examples)

                train_noise = inject_noise(self.y_train, n_type=noisetype, noiselvl=noiselvl)
                valid_noise = inject_noise(y_valid, n_type=noisetype, noiselvl=noiselvl,
                                           noise_mat=train_noise.noise_matrix)

                for i in range(len(self.train_data.examples)):
                    self.train_data.examples[i]['y'] = int(train_noise.noisy_label[i])
                for i in range(len(self.valid_data.examples)):
                    self.valid_data.examples[i]['y'] = int(valid_noise.noisy_label[i])

                pickle.dump([self.train_data, self.valid_data], open(file_name, "wb"))

                for i in range(len(self.train_data.examples)):
                    self.train_data.examples[i]['y'] = int(train_noise.true_label[i])
                for i in range(len(self.valid_data.examples)):
                    self.valid_data.examples[i]['y'] = int(valid_noise.true_label[i])

                self.N_train_data, self.N_valid_data = pickle.load(open(file_name, "rb"))

            _, self.y_noisy_train = GetRawDataBaseline(self.N_train_data.examples)
        # X_noisy_valid, y_noisy_valid = GetRawDataBaseline(N_valid_data.examples)
        self.out_dist = confusionMatrix(self.y_train, self.y_noisy_train)

        return

    def creatNoiseModel(self, noisemodel, ndistribution, diagPen=1):
        """
        Creates end-end-end deep network model given the noise model and noise distribution.
        :param noisemodel: str
        :param ndistribution: noise distribution matrix
        :param diagPen: specify the k*n_class initialization of noise model
        :return: baseline model
        """

        from classify_conv_noisylabel import ConvNoiseModel
        from baseline.pytorch.classify import ConvModel

        n_paramaters = self.Nmodel_params[noisemodel]
        n_cls = len(ndistribution)

        if noisemodel == "WoNM":
            self.Bmodel_params["model_type"] = "default"
            self.Bmodel_params["nm"] = False
            model = ConvModel.create(self.embeddings, self.labels, **self.Bmodel_params["model"])
        else:
            self.Bmodel_params["model_type"] = "cnn_noisy"
            self.Bmodel_params["nm"] = noisemodel
            model = ConvNoiseModel.create(self.embeddings, self.labels, **self.Bmodel_params["model"])

            if n_paramaters["weighIni"] == "ones":
                print("Initialized to identity matrix with scaling: {}".format(diagPen))
                model.output.linear2.weight.data.copy_(torch.from_numpy(diagPen * np.eye(n_cls)))

            if n_paramaters["weighIni"] == "dist":
                print("Last layer weights are initialized to noise distribution")
                model.output.linear2.weight.data.copy_(torch.from_numpy(ndistribution))

            if n_paramaters["weighIni"] == "rand":
                print("Last layer weights are initialized to normal random")
                model.output.linear2.weight.data.copy_(torch.from_numpy(np.random.rand(n_cls, n_cls)))

            print("Noise model initialization weights: ", list(model.parameters())[-1])

        self.model = model
        return model

    def modelTrain(self, noiselvl, noisemodel):
        """
        Train the end-to-end model.

        :param noiselvl: <float> level of label noise
        :param noisemodel: <str> type of noise model.
        :return:
        """

        from baseline.pytorch.classify import fit

        n_paramaters = self.Nmodel_params[noisemodel]
        self.Bmodel_params['train']['weight_decay'] = n_paramaters["penality"]

        if noiselvl == 0:
            fit(self.model, self.train_data, self.valid_data, self.test_data, nm=n_paramaters["noise_model"],
                **self.Bmodel_params['train'])
        else:
            fit(self.model, self.N_train_data, self.N_valid_data, self.test_data, nm=n_paramaters["noise_model"],
                **self.Bmodel_params['train'])
        return

    def modelTest(self, noisemodel):
        """
        Test the learned model.

        :param noisemodel: <str> type of noise model
        :return: weight of the noise model and test accuracy and f1 score
        """

        last_layer_weights = 0

        if noisemodel != 'WoNM':
            last_layer_weights = self.model.output.linear2.weight.detach().cpu().numpy()
            del list(self.model.children())[-1][2:]

        X_test, y_test = GetRawDataBaseline(self.test_data.examples)

        from baseline.train import EpochReportingTrainer, create_trainer
        from baseline.pytorch.classify import ClassifyTrainerPyTorch
        from baseline.utils import listify, to_spans, f_score, revlut, get_model_file

        args = {**self.Bmodel_params['model'], **self.Bmodel_params['train']}
        # trainer = create_trainer(ClassifyTrainerPyTorch, self.model, **args)
        trainer = create_trainer(self.model, **args)

        trn = trainer.test(self.test_data, reporting_fns=[])

        # To do--required when the dataset size is preety big and we need to obtain the last layer activations.

        # ditss = {'x': np.array(X_test)}  # x.astype(dtype = int)}
        # self.y_pred = np.argmax(self.model.classify_prob(ditss), axis=1)
        # y_pred = np.zeros((1,len(X_test)))[0]
        # div = 11
        # for i in range(len(X_test)//div):
        # 	ditss ={'x': np.array(X_test[div*i+0:div*(i+1)])}  # x.astype(dtype = int)}
        # 	y_pred[div*i+0:div*(i+1)] = np.argmax(self.model.classify_prob(ditss), axis=1)
        # self.y_pred = y_pred
        # Acc_val = 100*sum(self.y_pred == y_test)/len(y_test)
        # print("Accuracy: %.2f%%" % Acc_val)
        # f1score = f1_score(y_test, self.y_pred, average='macro')
        # print("F1 Score: %.2f%%" % f1score)

        return last_layer_weights, trn['acc'] * 100, trn['f1'] * 100

    def lastLayerAct(self):

        """
        Returns the last FC layer activations for training and test data.
        :return:
        """

        X_test, y_test = GetRawDataBaseline(self.test_data.examples)
        X_train, y_train = GetRawDataBaseline(self.train_data.examples)

        del list(self.model.children())[-1][0:]
        print(self.model)
        layer_sz = self.Bmodel_params["model"]["cmotsz"] * len(self.Bmodel_params["model"]["filtsz"])
        x_train = np.zeros((len(self.y_train), layer_sz))

        for i in range(len(self.y_train) // 10):
            ditss = {'x': np.array(X_train[10 * i + 0:10 * (i + 1)])}  # x.astype(dtype = int)}
            x_train[10 * i + 0:10 * (i + 1), :] = self.model.predict_lastlayer(ditss)
        print("Size of Training Matrix: ", x_train.shape)

        ditss = {'x': np.array(X_test)}
        x_test = self.model.classify_lastlayer(ditss)
        print("Size of test Matrix: ", x_test.shape)
        return x_train, x_test

    def lastLayerSVM(self, onNoisy=False):
        """Train Linear SVM classifier on Last feature layer of the trained model.
        """
        _, y_test = GetRawDataBaseline(self.test_data.examples)
        x_train, x_test = self.lastLayerAct()
        print("Start Training")

        from sklearn.svm import LinearSVC, SVC
        import sklearn
        print ("Method = Linear SVM")
        svmmodel = LinearSVC(penalty='l2', random_state=0, dual=False)
        if onNoisy:
            svmmodel.fit(x_train, self.y_noisy_train)
        else:
            svmmodel.fit(x_train, self.y_train)

        print("Start Testing")
        results = svmmodel.predict(x_test)
        acc = svmmodel.score(x_test, y_test)
        print("Accuracy = " + repr(sklearn.metrics.accuracy_score(y_test, results)))
        print(sklearn.metrics.classification_report(y_test, results))
        return results, acc, x_test

    def LowDimVisualization(self, x_train=np.array(0), **kwargs):
        """To visualize the last layer activations. tSNE(high time complexity) or Isomap"""
        # get equal number of samples per class
        sampPerClas = kwargs.get('all_samples')
        nSampls = kwargs.get('nSampls', 50)
        allsamp = kwargs.get('allsamp', False)
        embedder = kwargs.get('embedder', 'pca')
        n_comp = kwargs.get('n_comp', 2)
        filename = kwargs.get('filename', '')
        embdClrSett = kwargs.get('embdClrSett')

        if not x_train.any():
            x_train, _ = self.lastLayerAct()

        if not allsamp:
            n_cls = len(sampPerClas)
            n_samp = [x * nSampls // 100 for x in sampPerClas]
            print("Choose {} number of samples per class.".format(n_samp))
            samp_index = []
            for acls in range(n_cls):
                y_cls = np.array([i for i in range(len(self.y_train)) if self.y_train[i] == acls])
                for i in range(n_samp[acls]):
                    samp_index.append(y_cls[i])
            # print(len(samp_index))
            x_tr = x_train[samp_index, :]
            y_T_tr = self.y_train[samp_index]
            y_N_tr = self.y_noisy_train[samp_index]
        else:
            x_tr = x_train
            y_T_tr = self.y_train
            y_N_tr = self.y_noisy_train

        from sklearn import manifold

        if embedder == 'tsne':
            tsne = manifold.TSNE(n_components=n_comp, init='pca', random_state=0)
            Y = tsne.fit_transform(x_tr)
        elif embedder == 'isomap':
            Y = manifold.Isomap(n_neighbors=10, n_components=n_comp).fit_transform(x_tr)
        elif embedder == 'umap':
            import umap
            # tran = umap.UMAP(n_neighbors=5, random_state=42).fit(x_tr)
            # Y = tran.embedding_
            Y = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=n_comp, random_state=42).fit_transform(x_tr)
        elif embedder == 'pca':
            print('doing PCA')
            from sklearn import decomposition
            pca = decomposition.PCA(n_components=n_comp)
            pca.fit(x_tr)
            Y = pca.transform(x_tr)
        else:
            print('{} Low Dimensional Embedder not known. Choose from the list: [tsne, isomap, umap, pca]'.format(
                embedder))

        clr = embdClrSett[0]  # ['r','b','g','c','k','m']
        markers = embdClrSett[1]  # ['o','<','+','s','d','x']
        s = embdClrSett[2]
        alpha = embdClrSett[3]

        if n_comp == 2:
            # if embedder=='umap':
            # 	print("Umap embedder can output 2D.\n Showing 2D representation")
            if sum(self.y_noisy_train == self.y_train) == len(self.y_train):
                for i in range(len(self.labels)):
                    y_out = Y[y_T_tr == i, :]
                    plt.scatter(y_out[:, 0], y_out[:, 1], c=clr[i], s=s, alpha=alpha, label=clr[i], marker=markers[i])
                    plt.legend()
                    plt.title("Superimposing True Labels only")
            else:
                plt.figure(figsize=(5, 7))
                plt.subplot(2, 1, 1)
                for i in range(len(self.labels)):
                    y_out = Y[y_T_tr == i, :]
                    plt.scatter(y_out[:, 0], y_out[:, 1], c=clr[i], s=s, alpha=alpha, label=clr[i], marker=markers[i])
                    plt.legend()
                    plt.title("Superimposing True Labels")
                plt.subplot(2, 1, 2)
                for i in range(len(self.labels)):
                    y_out = Y[y_N_tr == i, :]
                    plt.scatter(y_out[:, 0], y_out[:, 1], c=clr[i], s=s, alpha=alpha, label=clr[i], marker=markers[i])
                    plt.title("Superimposing Noisy Labels")
                    plt.legend()
        else:
            assert n_comp == 3
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(5, 7))
            ax = fig.add_subplot(2, 1, 1, projection='3d')
            for i in range(len(self.labels)):
                y_out = Y[y_T_tr == i, :]
                ax.scatter(y_out[:, 0], y_out[:, 1], y_out[:, 2], c=clr[i], s=10, alpha=0.4, label=clr[i])
                ax.legend()
                ax.set_title("Superimposing True Labels")
            ax = fig.add_subplot(2, 1, 2, projection='3d')
            for i in range(len(self.labels)):
                y_out = Y[y_N_tr == i, :]
                ax.scatter(y_out[:, 0], y_out[:, 1], y_out[:, 2], c=clr[i], s=10, alpha=0.4, label=clr[i])
                ax.set_title("Superimposing Noisy Labels")
                ax.legend()

        if filename:
            # plt.suptitle(filename)
            # plt.suptitle(filename)
            plt.savefig(filename)
        plt.show()

        return Y

    def trainValidLossAcc(self):
        import numpy as np
        epochs = self.Bmodel_params['train']['epochs']
        plt.figure(figsize=(9, 7))
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, epochs - 1, epochs), np.array(self.model.train_loss), color='b', label='Training Loss')
        plt.plot(np.linspace(0, epochs - 1, epochs), np.array(self.model.valid_loss), color='g',
                 label='Validation Loss')
        plt.ylabel("Average Loss")
        plt.legend()
        plt.title("Average loss per epoch")
        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, epochs - 1, epochs), np.array(self.model.train_acc), color='b', label='Training acc')
        plt.plot(np.linspace(0, epochs - 1, epochs), np.array(self.model.valid_acc), color='g', label='Validation acc')
        plt.ylabel("Classification Accuracy")
        plt.xlabel("Number of Epochs")
        plt.legend()
        plt.title("Classification Accuracy per epoch")
        plt.show()
        return

    # Not Complete yet.
    def class_specific_testing(self, h):

        if Dataset == 'dbpedia':
            y_pred = np.zeros((1, len(X_test)))[0]
            for i in range(len(X_test) // 100):
                ditss = {'x': np.array(X_test[100 * i + 0:100 * (i + 1)])}  # x.astype(dtype = int)}
                y_pred[100 * i + 0:100 * (i + 1)] = np.argmax(self.model.classify_prob(ditss), axis=1)

            Acc_val = 100 * sum(y_pred == y_test) / len(y_test)
            print("Accuracy: %.2f%%" % Acc_val)
            accuracies.append(Acc_val)

            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        else:

            ditss = {'x': np.array(X_test)}  # x.astype(dtype = int)}
            y_pred = model.classify_prob(ditss)

            # n_cls = len(set(y_test))
            # y_pred = np.zeros([len(out), n_cls])
            # for i in range(len(out)):
            #        for j in range(n_cls):
            #                 y_pred[i][j] = out[i][j][1].cpu().numpy()
            Acc_val = 100 * sum(np.argmax(y_pred, axis=1) == y_test) / len(y_test)
            print("Accuracy: %.2f%%" % Acc_val)
