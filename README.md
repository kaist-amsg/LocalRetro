# LocalRetro
Implementation of Retrosynthesis Prediction with LocalRetro developed by prof. Yousung Jung group at KAIST (contact: ysjn@kaist.ac.kr).

## Announcements 
### 2023.12.15 update

After reading several papers claiming higher accuracy than LocalRetro using LocalRetro as backbones, I realized training LocalRetro with more epochs and average the models can actually get higher top-1, 3, 5 accuracy (without ANY change in model architecture). Therefore, I trained the model and updated the checkpoints. Also, template mask was added to remain the high accuracy at high k.

#### Exact match accuracy (%) on USPTO-50K dataset 

| Stereo  | Top-1 | Top-3 | Top-5 | Top-10 | Top-50 |
| --------| ---- | ---- | ---- | ---- | ---- |
| Unaware | 54.2 | 76.8 | 84.0 | 90.3 | 95.5 |
|  Aware  | 55.8 | 78.9 | 86.2 | 92.7 | 98.1 |

### 2023.07.10 update
To address the issue raised from the coommnuty (see also [#15](https://github.com/kaist-amsg/LocalRetro/issues/15))., the function `get_atom_pair` in `model_utils.py` is updated. 

Also, we change the activation function from ReLU to GeLU and recalculate the accuracy using both *stereo-aware* and *stereo-unaware* metrics, showing at the bottom of README.md (see [#12](https://github.com/kaist-amsg/LocalRetro/issues/12)).

For example, following problem (reaction #270 in test set) is an ester hydrolysis reaction, which has nothing to do with the single bond highlighed in red but somehow changed in the ground truth. The prediction of this retrosynthesis is identified as correct by the *stereo-aware* metric but wrong by the *stereo-unaware* metric.

![](https://hackmd.io/_uploads/rJPY99iFh.png)

#### Exact match accuracy (%) on USPTO-50K dataset 

| Stereo  | Top-1 | Top-3 | Top-5 | Top-10 | Top-50 |
| --------| ---- | ---- | ---- | ---- | ---- |
| Unaware | 52.6 | 75.3 | 83.5 | 90.2 | 95.7 |
|  Aware  | 54.0 | 77.3 | 85.7 | 92.5 | 98.4 |


### 2022.02.09 update
We cleaned the code and made the template more simplied, which yields 658 local reaction templates for USPTO_50K dataset and 20,221 local reaction templates for USPTO_MIT dataset. Therefore we tested the top-k accuracy again and the results are updated at the bottom of README.md.
The training takes around 100 minutes on NVIDIA GeForce RTX 3090

## Developer
Shuan Chen (contact: shuankaist@kaist.ac.kr)<br>

## Requirements
* Python (version >= 3.6) 
* Numpy (version >= 1.16.4) 
* PyTorch (version >= 1.0.0) 
* RDKit (version >= 2019)
* DGL (version >= 0.5.2)
* DGLLife (version >= 0.2.6)

## Requirements
Create a virtual environment to run the code of LocalRetro.<br>
Install pytorch with the cuda version that fits your device.<br>
```
cd LocalRetro
conda create -c conda-forge -n rdenv python=3.7 -y
conda activate rdenv
conda install pytorch cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge rdkit -y
pip install dgl
pip install dgllife
```


## Publication
Shuan Chen and Yousung Jung. Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention, [JACS Au 2021](https://pubs.acs.org/doi/10.1021/jacsau.1c00246).



## Usage
### [1] Download the raw data of USPTO-50K or USPTO-MIT dataset
See the README in `./data` to download the raw data files for training and testing the model.

### [2] Data preprocessing
A two-step data preprocessing is needed to train the LocalRetro model.

#### 1) Local reaction template derivation 
First go to the data processing folder
```
cd preprocessing
```
and extract the reaction template with specified dataset name (default: USPTO_50K).
```
python Extract_from_train_data.py -d USPTO_50K
```
This will give you four files, including 
(1) atom_templates.csv
(2) bond_templates.csv
(3) template_infos.csv
(4) template_rxnclass.csv (if train_class.csv exists in data folder)<br>

#### 2) Assign the derived templates to raw data
By running
```
python Run_preprocessing.py -d USPTO_50K
```
You can get four preprocessed files, including 
(1) preprocessed_train.csv
(2) preprocessed_val.csv
(3) preprocessed_test.csv
(4) labeled_data.csv<br>


### [3] Train LocalRetro model
Go to the localretro folder
```
cd ../scripts
```
and run the following to train the model with specified dataset (default: USPTO_50K)
```
python Train.py -d USPTO_50K
```
The trained model will be saved at ` LocalRetro/models/LocalRetro_USPTO_50K.pth`<br>

### [4] Test LocalRetro model
To use the model to test on test set, simply run 
```
python Test.py -d USPTO_50K
```
to get the raw prediction file saved at ` LocalRetro/outputs/raw_prediction/LocalRetro_USPTO_50K.txt`<br>
Finally you can get the reactants of each prediciton by decoding the raw prediction file
```
python Decode_predictions.py -d USPTO_50K
```
The decoded reactants will be saved at 
`LocalRetro/outputs/decoded_prediction/LocalRetro_USPTO_50K.txt`<br>and 
`LocalRetro/outputs/decoded_prediction_class/LocalRetro_USPTO_50K.txt`<br>