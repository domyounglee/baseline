import numpy
import string
import pickle
import torch
import torch.nn as nn
import argparse
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np


# fix random seed for reproducibility
np.random.seed(7)


from utilities import GetRawDataBaseline, softmax, NsamplesPerClass, confusionMatrix


import baseline as bl
import mead as md
from mead.utils import convert_path
from training_n import e2e_noise_model

trials = 2

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
    # Load/save the Baseline configured data and Embedding weight matrix for reproducibility
    BLconfigFile = os.path.join( Dataset + '_config_BLTask' + config_params['backend'])
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

    models_in = list(jdata.keys())  # ['WoNM', 'NMWoRegu', 'NMwRegu0.01', 'NMwRegu0.1']
    # fileName = os.path.join(Dataset, Dataset + '_B' +str(config_params['batchsz']) + 'Everything'+
    #     rand+config_params['backend']+ str(int(0.2*100)) + args.optag)
    # print(fileName)
    # models_in = ['NMwRegu01']
    # For reproducibility lets fix the noise distribution and save to the disk
    # 0.1,0.2,0.3,0.4,0.5
    # noise_input = [0]

    for noise_per in noise_input:

        if noise_per != 0:
            file_name = os.path.join(Dataset,
                                     Dataset + '_' + 'noisy_train_data_' + rand + config_params['backend'] + str(
                                         int(noise_per * 100)))
            nm.creatNoiseProfile(file_name, noiselvl=noise_per, noisetype=Ntype)
        else:
            nm.out_dist = np.eye(n_class)

        weight_dict = {}
        weight_dict['out_dist'] = nm.out_dist

        for Nmodel in models_in:

            weight_para = []
            accuracies = []
            f1_scores = []
            # model_features = jdata[Nmodel]
            # noise_model = model_features["noise_model"]
            # regu = model_features["regu"]
            # penality = model_features["penality"]
            # print(noise_model,regu,penality)

            for tr in range(trials):
                # creatmodle
                nm.creatNoiseModel(noisemodel=Nmodel, ndistribution=nm.out_dist, diagPen=float(args.nmScale) * n_class)
                print(
                "Model Parameters --> \n" + "Noise level: ", noise_per, "||Noise Type: ", Ntype, "|| Working model: ",
                Nmodel, "|| Trial: ", tr, "||")
                #  Train Model
                nm.modelTrain(noiselvl=noise_per, noisemodel=Nmodel)
                print('Noise level: ', noise_per, 'Working model: ', Nmodel, ' Trial: ', tr)

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

            fileName = os.path.join(Dataset, Dataset + '_B' + str(config_params['batchsz']) + 'Everything' +
                                    rand + config_params['backend'] + str(int(0.2 * 100)) + args.optag)
            print(fileName)
            pickle.dump(weight_dict, open(fileName, 'wb'))

if __name__ == "__main__":
    main()