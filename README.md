# LocalMapper
Atom-map organic reactions with LocalMapper developed by prof. Yousung Jung group at KAIST (contact: ysjn@kaist.ac.kr).

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
Create a virtual environment to run the code of LocalMapper.<br>
Install pytorch with the cuda version that fits your device.<br>
```
cd LocalMapper
conda create -c conda-forge -n rdenv python=3.7 -y
conda activate rdenv
conda install pytorch cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge rdkit -y
pip install dgl
pip install dgllife
```

## Publication
TBD
"# LocalMapper" 
