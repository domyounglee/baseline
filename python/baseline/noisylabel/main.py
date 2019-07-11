import numpy
import string
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from utilities import GetRawDataBaseline, softmax, NsamplesPerClass, confusionMatrix
import baseline as bl
from mead.utils import convert_path
from training_n import e2e_noise_model


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier with noisy labels')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--noisemodel', help='JSON for different Noise model', required=True, type=convert_path)
    parser.add_argument('--noisetype', help=['uni', 'rand', 'cc'], type=str, default="uni")
    parser.add_argument('--noiselvl', help='Noise levels to run', nargs='+', type=float, default=[0.0])
    parser.add_argument('--optag', help='Any optional tag for data dumping', nargs='?', default='')
    parser.add_argument('--nmScale', help='defines the scaling factor for noise model initialization, default=n_cls', default=1)
    args = parser.parse_known_args()[0]
    print(args)

    Ntype = args.noisetype
    noise_input = args.noiselvl

    if Ntype == 'rand':
        rand = 'Rand'
    elif Ntype == 'cc':
        rand = 'CC'
    elif Ntype == 'uni':
        rand = ''

    print("Label noise type defined: ", Ntype)
    print('Noise levels working with: ', noise_input)

    jsonFileToLoad = args.config.split('/')[-1]
    Cfile_dir = '/'.join([x for x in args.config.split('/')[:-1]]) + '/'
    print("Further information will be stored at this directory: ", Cfile_dir)

    jdata = bl.utils.read_config_file(args.noisemodel)
    print("This is the noise model configuration:\n", jdata)
    config_params = bl.utils.read_config_file(Cfile_dir + jsonFileToLoad)
    print("Baseline training config: ", jsonFileToLoad)
    print(config_params)

    # Working Dataset
    Dataset = config_params['dataset']
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, Dataset+"_noisy")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    # Load/save the Baseline configured data and Embedding weight matrix for reproducibility
    BLconfigFile = os.path.join(final_directory, Dataset + '_config_BLTask' + config_params['backend'])
    print(BLconfigFile)

    nm = e2e_noise_model(Bmodel_params=config_params, Nmodel_params=jdata)
    nm.creatDataProfile(BLconfigFile, Cfile_dir, jsonFileToLoad)

    _, y_train = GetRawDataBaseline(nm.train_data.examples)
    _, y_test = GetRawDataBaseline(nm.test_data.examples)
    _, y_valid = GetRawDataBaseline(nm.valid_data.examples)

    n_class = len(set(y_train))
    print('Number of classes: ', n_class)
    print('Number of training samples: ', len(y_train))
    print('Number of validation samples: ', len(y_valid))
    print('Number of test samples: ', len(y_test))

    models_in = list(jdata.keys())
    # Repeat the experiments for each noise level.
    for noise_per in noise_input:

        if noise_per != 0:
            file_name = os.path.join(final_directory,
                                     Dataset + '_' + 'noisy_train_data_' + rand + config_params['backend'] + str(
                                         int(noise_per * 100)))
            nm.creatNoiseProfile(file_name, noiselvl=noise_per, noisetype=Ntype)
        else:
            nm.out_dist = np.eye(n_class)

        weight_dict = {}
        weight_dict['out_dist'] = nm.out_dist
        # Repeat the training for each noise model specified in jdata
        for Nmodel in models_in:

            weight_para = []
            accuracies = []
            f1_scores = []
            #
            for tr in range(trials):
                # create model
                nm.creatNoiseModel(noisemodel=Nmodel, ndistribution=nm.out_dist, diagPen=float(args.nmScale) * n_class)
                print( "Model Parameters --> \n" + "Noise level: ", noise_per, "||Noise Type: ", Ntype, "|| Working model: ", Nmodel, "|| Trial: ", tr, "||")
                #  Train Model
                nm.modelTrain(noiselvl=noise_per, noisemodel=Nmodel)

                last_layer_weights, Acc_val, f1score = nm.modelTest(noisemodel=Nmodel)

                weight_para.append(last_layer_weights)
                accuracies.append(Acc_val)
                f1_scores.append(f1score)

                del nm.model

            print(f1_scores)
            print(accuracies)

            if Nmodel != 'WoNM':
                val = 'yes'
                weight_dict[Nmodel] = weight_para[np.argmax(np.array(accuracies))]
            else:
                val = 'no'
            weight_dict[Nmodel + 'f1_scores'] = f1_scores
            weight_dict[Nmodel + 'accuracies'] = accuracies

            fileName = os.path.join(final_directory, Dataset + '_B' + str(config_params['batchsz']) + 'Everything' +
                                    rand + config_params['backend'] + str(int(noise_per * 100)) + args.optag)
            print("Noise model weights are stored at: ", fileName)
            pickle.dump(weight_dict, open(fileName, 'wb'))


if __name__ == "__main__":
    np.random.seed(7) # fix random seed for reproducibility
    trials = 5  # repeat the experiment
    main()