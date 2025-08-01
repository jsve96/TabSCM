# TabSCM: Strucutral Causal Models for mixed-type generation of Tabular Data

**TabSCM** is a practical framework for mixed tabular data
generation using Structural Causal Models (SCMs) that
combines causal reasoning with decision trees and diffusion models

The file structure and setup is inspired by the official implementation of [TabSyn](https://github.com/amazon-science/tabsyn).

## Setup
Tested with python 3.10.
```
conda env create -f environment.yaml
```

We recommend to set up a seperate environment for GOGGLE

```
conda env create -f environment_goggle.yaml
```

## Preparing Datasets

### Using the datasets adopted in the paper

Download raw dataset:

```
python download_dataset.py
```

Process dataset:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for you dataset [NAME_OF_DATASET] in ./data:
```
cd data
mkdir [NAME_OF_DATASET]
```

Put the tabular data in .csv format in this directory ([NAME_OF_DATASET].csv). **The first row should be the header** indicating the name of each column, and the remaining rows are records.

Then, write a .json file ([NAME_OF_DATASET].json) recording the metadata of the tabular, covering the following information:
```
{
    "name": "[NAME_OF_DATASET]",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/[NAME_OF_DATASET]/[NAME_OF_DATASET].csv"
    "test_path": null,
}
```
Put this .json file in the .Info directory.

Finally, run the following command to process the UDF dataset:
```
python process_dataset.py --dataname NAME_OF_DATASET
```

## Training Models

For baseline methods, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, beijing, early_diab, heloc, housing, magic

Options of [NAME_OF_BASELINE_METHODS]: smote, goggle*, great, stasy, codi, tabddpm, tabdiff, tabsyn, ctgan

For Tabsyn, use the following command for training:

```
# train VAE first
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train

# after the VAE is trained, train the diffusion model
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
```

For TabDiff follow the instruction in [TabDiff](https://github.com/MinkaiXu/TabDiff/tree/main).


For TabSCM you can run

```
python main.py --dataname [NAME_OF_DATASET] --method tabscm --mode train 
```

Currently, we support the following parameter ```--cd_alg``` (pc,ges,notears), ```--ci_test ``` (fisherz,chisq) only for pc, 
*for GOGGLE first activate the conda environment.


## Tabular Data Synthesis

For baseline methods, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_METHODS] --mode sample --save_path [PATH_TO_SAVE]
```

The default save path is "synthetic/[NAME_OF_DATASET]/[METHOD_NAME].csv"

We repeated this procedure and put each synthetic sample "smapleXX"  in "syn_data_samples/[NAME_OF_DATASET]/[NAME_OF_METHOD]/sampleXX.csv". It may be possible that you first need to set up the directory "syn_data_samples/[NAME_OF_DATASET]/[NAME_OF_METHOD]".

## Evaluation
We evaluate the quality of synthetic data using metrics from various aspects, we adopted the implementation provided in [TabSyn](https://github.com/amazon-science/tabsyn/tree/main).

#### Density estimation of single column and pair-wise correlation ([link](https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included))

```
python eval_final_density.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA_DIRECTORY]
```


#### Machine Learning Efficiency

```
python eval_mle_batch.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA_DIRECTORY]
```

#### Pricavy protection: Distance to Closest Record (DCR)

```
python eval_dcr_batch.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA_DIRECTORY]
```

Note: the optimal DCR score depends on the ratio between #Train and #Holdout (# Test). Ideally, DCR sore should be #Train / (#Train + #Holdout). To let the optimal score be $50\%$, you have to let the training and testing set have the same size. 

#### Detection: Classifier Two Sample Tests (C2ST)

```
python eval_detection_final.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA_DIRECTORY]
```

