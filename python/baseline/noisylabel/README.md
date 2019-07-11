# Classification in the presence of label noise.

Run label noise experiments via:
```
python main.py --config ~/baseline/python/mead/config/sst2.json  --noisemodel ~/baseline/noisylabel/noise_model_py.json --noisetype uni --noiselvl 0.0 0.1 0.2 0.3 0.4 0.45 
```

- `--config` specifies the Baseline configuration for a dataset.
- `--noisemodel` specifies the types of noise model one wants to try. Different models we used in our experiments include `WoNM`, `NMWoRegu`, `NMwRegu01` etc. Further information on this can be found under types of experiments. `noise_model_py.json` contains information for all the models. 
- `--noisetype` Default is uniform label noise (uni). One can choose among ['uni', 'rand', 'cc'].
- `--noiselvl` specify the level of artificial label noise injected into the datasets. can got from 0.0 --> 0.9
- `--nmScale` defines the scaling factor of noise model initialization. Default is identity matrix. 

## Types of Experiments

- **Different datasets**: Explored different types of text classification datasets ranging from binary classification to multi-class classification problem.
    + SST2 (2 classes)
    + Trec (6 classes)
    + AG-News (4 classes)
    + Dbpedia (14 classes)
- **Effect of noise types**
    + **Uniform label flipping (Uni)**: A clean label is swapped with another label from the given number of labels sampled uniformly at random.
    + **Structured label flipping (Rand)**: A clean label id swapped with another label from the given number of labels-
        * *Random order of flipping*: sampled randomly over a unit simplex. 
        * *Increasing order of flipping*: sampled randomly over a unit simplex but with increasing order of confusions.
    + **Class Conditional label flipping (CC)**: A clean label form one class is specifically swapped with label from another class. We find the most confused labels by testing the trained base model on clean dataset.
- **Effect of batch sizes**
    + Here we studied the effect of batch size on different datasets with different types of label noise.
- **Effect of noise models**
    + `WoNM` :  Without Noise Model; No noise model stacked to the network.
    + `NMWoRegu`:  Noise Model Without Regularization; Noise model is stacked on top of base model but no regularization applied and noise model's weights are initialized to identity.
    + `NMwRegu01`:  A noise model with l2 regularization (penalty 0.1) stacked and noise model's weights are initialized to identity.
    + `NMwRegu001`: A noise model with l2 regularization (penalty 0.01) stacked and noise model's weights are initialized to identity.
    + `TDwRegu001`: A noise model with l2 regularization (penalty 0.01) stacked and noise model's weights are initialized to True noise distribution injected to the labels.
    + `RandwRegu01`: A noise model with l2 regularization (penalty 0.01) stacked and noise model's weights are initialized randomly.
    + `NMwl1Regu001`: A noise model with l1 regularization (penalty 0.01) stacked and noise model's weights are initialized to identity.
    + `NMwl1l2Regu001`: A noise model with l2-l1 (elastic net) regularization (penalty 0.01) stacked and noise model's weights are initialized to identity.

































```
@article{jindal2019effective,
  title={An Effective Label Noise Model for DNN Text Classification},
  author={Jindal, Ishan and Pressel, Daniel and Lester, Brian and Nokleby, Matthew},
  journal={arXiv preprint arXiv:1903.07507},
  year={2019}
}
```