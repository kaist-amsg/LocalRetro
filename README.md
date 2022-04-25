# LocalRetro
Implementation of Retrosynthesis Prediction with LocalRetro developed by prof. Yousung Jung group at KAIST (contact: ysjn@kaist.ac.kr).

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


## Code reuploading announcement (2022.02.09)
We cleaned the code and made the template more simplied, which yields 658 local reaction templates for USPTO_50K dataset and 20,221 local reaction templates for USPTO_MIT dataset. Therefore we tested the top-k accuracy again and the results are updated at the bottom of README.md.
The training takes around 100 minutes on NVIDIA GeForce RTX 3090

## Code cleaning announcement (2021.09.16)
Currently, we are cleaning up the codes, and the codes will be uploaded back afterwards.

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

#### Exact match accuracy (%) on USPTO-50K dataset without given reaction class 
*AT = Augmented Transformer

| Method | Top-1 | Top-3 | Top-5 | Top-10 | Top-50 |
| -------- | -------- | -------- | -------- | -------- | -------- |
| GLN         | 52.5 | 69.0 | 75.6 | 83.7 | 92.4 |
| G2Gs        | 48.9 | 67.6 | 72.5 | 75.5 |  /   |
| GraphRetro  | **53.7** | 68.3 | 72.2 | 75.5 | / |
| AT     | 53.5 | 69.4 | 81.0 | 85.7 | / |
| MEGAN | 48.1 | 70.7 | 78.4 | 86.1 | 93.2 |
| LocalRetro  | 53.4 | **77.5** | **85.6** | **92.4** | **98.4** |

#### Exact match accuracy (%) on USPTO-50K dataset with given reaction class

| Method | Top-1 | Top-3 | Top-5 | Top-10 | Top-50 |
| -------- | -------- | -------- | -------- | -------- | -------- |
| GLN         | 64.2 | 79.1 | 85.2 | 90.0 | 93.2 |
| G2Gs        | 61.0 | 81.3 | 86.0 | 88.7 |  /   |
| GraphRetro  | 63.9 | 81.5 | 85.2 | 88.1 | / |
| MEGAN | 60.7 | 82.0 | 87.5 | 91.6 | 95.3 |
| LocalRetro  | **64.2** | **86.8** | **93.0** | **96.9** | **98.6** |
