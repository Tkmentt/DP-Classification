# DP EEG motor imagery classification

## About

## Installation
The installation has 2 steps, first you will need to prepare a working directory with this project 
(the source code and data). 

The second step involves installing all dependencies needed to run the project. 
There are two ways to install the dependencies. One using Conda which is the *recomended* way, since using Conda
additional dependencies to run the classification of Keras models on CUDA GPUs. If using GPUs to train the models 
is not desired the installation can be done using pure Python.

### Workspace
Before installing the project, prepare the workspace directory by either cloning the repository via
```bash
git clone https://github.com/Tkmentt/DP-Classification.git
cd DP-Classification
```

or download the source code from https://github.com/Tkmentt/DP-Classification
and unzip the folder from the downloaded zip file. Change your current directory to the extracted folder.
```bash
cd DP-Classification-main
```

Download the dataset from https://zenodo.org/records/15399490
and unzip it in the working directory to a folder named **data**.

### Conda
First download and install Conda (https://www.anaconda.com/download/).

After the installation create a new Conda environment (to avoid any dependency conflicts) for example *eeg*.
```bash
conda create -n eeg python=3.10
```
Activate the environment.
```bash
conda activate eeg
```
Install CUDA dependencies if you intend on using a GPU accelerator for the model training.
```bash
conda install -n eeg -c conda-forge cudatoolkit=11.3 cudnn=8.1.0
conda install -n eeg -c nvidia cuda-nvcc=11.3
```
Install required Python packages from requirements.txt.
```bash
pip install -r requirements.txt
```

### Python
Download and install Python version 3.10 (https://www.python.org/downloads/release/python-31011/).

If you had a version of Python installed already, make sure you are using the correct version e.g.
```
~eeg-motion-detection>python --version
Python 3.10.11
```
(Any Python 3.10 and higher version will most likely work, however during development only 3.10.11 was used)

Create a new Python virtual environment.
```bash
python -m venv venv
```
Activate the virtual environment.

- On Windows 
```
venv\Scripts\activate
```
- On Linux
```
source venv/bin/activate
```
Install required Python packages from requirements.txt.
```bash
pip install -r requirements.txt
```

### Run the project
This research project does not have one entry point, rather numerous testing scripts which can be run separately:

- debug_data
- train_MNE
- train_old
- train_raw
- train_stack
- real_time_mne
- real_time_raw
- real_time_comparator

## Configuration
Configuration is done via a config.py file
