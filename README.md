# FusionModel

## Content
- [Background and Framework](https://github.com/AadSama0404/FusionModel/blob/main/README.md#background-and-framework)
- [Installation and Usage](https://github.com/AadSama0404/FusionModel/blob/main/README.md#installation-and-usage)
- [Output and Analysis](https://github.com/AadSama0404/FusionModel/blob/main/README.md#output-and-analysis)

## Background and Framework
**Fusion Model** is an interpretable clone-based prognostic prediction model for patient cohorts comprising heterogeneous subgroups. It outputs a binary prediction indicating non-response (0) or response (1) as well as the corresponding probability of response between 0 and 1 which called the sample-score.
![](Overview.png)

## Installation and Usage
**Clone the Repository**
```sh
git clone https://github.com/AadSama0404/FusionModel.git
```
**Install Dependencies**
```sh
pip install -r requirements.txt
```
**Data Preprocessing and Model Training**
```sh
python data_preprocess.py
python main.py
```

## Output and Analysis
The test output of **Fusion Model** is two files: Output.csv and A_matrix.txt. The pictures in the [Preformence](https://github.com/AadSama0404/FusionModel/tree/main/Performance) folder are generated based on these data. The file structure and the meaning of each column are as follows.
```
Output.csv 
├── Y
├── Sample score
├── Predicted Label
├── PFS
├── Status
├── Subgroup ID
```
```
Clone_weight.txt 
├── Weights of Clones
├── Order of Clones
├── Predicted Label
├── Subgroup ID
```
