# RSupCon: Robust Supervised Contrastive Learning

Our paper extends **Supervised Contrastive Learning** to the adversarial domain and modifies the code based on the PyTorch implementation of SupCon.  
Original SupCon repository: [https://github.com/HobbitLong/SupContrast/tree/master](https://github.com/HobbitLong/SupContrast/tree/master)

## Code Description

### `main_rsupcon.py`
Implements the encoder training described in our paper. This script includes several options to customize the training process:

- `--trans_order`  
  Specifies the order of augmentations to be applied.  
  Example: `--trans_order b0,s0,t0`  
  - **b (Base Augment):** Small changes from original images, preserving human-perceived features and color information.  
  - **s (Sim Augment):** Color distortion augmentations, producing images rich in shape features but containing many non-robust features.  
  - **t (Trivial Augment):** Various augmentations affecting both robust and non-robust features.

- `--atk_anchor`  
  Defines the set of images used as adversarial examples.  
  Example: `--atk_anchor b0,s0`

- `--atk_randstart`  
  Adds random noise to specific images at the start of an adversarial attack.  
  Example: `--atk_randstart b0`

- `--atk_contrast`  
  Specifies the contrast set during the adversarial attack.  
  Example: `--atk_contrast b0,s0`

- `--cln_anchor`  
  Selects clean anchor images for RSupCon loss.  
  Example: `--cln_anchor t0`

#### Usage Examples
1.
```
--trans_order b0,s0,t0 --atk_anchor b0,s0 --atk_randstart b0 --atk_contrast b0,s0 --cln_anchor t0
```
Train encoder with three types of augmentations.

2.
```
--trans_order b0,s0,t0 --atk_anchor s0,t0 --atk_contrast b0,s0 --cln_anchor b0
```
Another variant with three augmentation types.

3.
```
--trans_order s0,s1,s2 --atk_anchor s0,s1 --atk_contrast s0,s1 --cln_anchor s2
```
Train encoder with a single augmentation method type.

### `main_linear.py`
Trains a linear classifier with a pre-trained encoder.

### `main_eval.py`
Evaluates robustness.  
- `--test_option`: `'all'` or `'partial'`
  - `'all'`: Evaluate overall robustness.
  - `'partial'`: Evaluate specific robustness types with `--test_type`:
    - `'adversarial'`
    - `'corrupt'`
    - `'ood'`  
- Results are stored as CSV files in the `robustness/` folder.

### `main_vis.py`
Performs visualization-based evaluations:  
- t-SNE  
- Activation Maximization by Optimization  
- Maximally Activating images  
- Results saved in the `vis/` folder.

## Installation
Install all required packages:
```
pip install -r requirements.txt
```

## Libraries Used for Evaluation
- **torchattacks**: For PGD-20.  
  Repo: [https://github.com/Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- **robustbench**: Robustness benchmark library, includes AutoAttack and corruption evaluation.  
  Repo: [https://github.com/RobustBench/robustbench/tree/master](https://github.com/RobustBench/robustbench/tree/master)
- **pytorch-ood** (v0.1.6, Python ≥3.9.x): For out-of-distribution detection.  
  Repo: [https://github.com/kkirchheim/pytorch-ood](https://github.com/kkirchheim/pytorch-ood)

## Note on Downloading Corruption Data
If downloading corruption data with `robustbench` fails, use these alternative sources:
- CIFAR-10-C / CIFAR-100-C repo: [https://github.com/hendrycks/robustness](https://github.com/hendrycks/robustness)  
- CIFAR-10-C direct: [https://zenodo.org/record/2535967](https://zenodo.org/record/2535967)  
- CIFAR-100-C direct: [https://zenodo.org/record/3555552](https://zenodo.org/record/3555552)

Place the extracted datasets in:
- `./data/CIFAR-10-C/`
- `./data/CIFAR-100-C/`

## Execution Scripts
- **run_rsup_cifar10.sh**  
  Train encoder from scratch on CIFAR-10, train linear classifier, evaluate robustness, and visualize results.
- **run_rsup_cifar100.sh**  
  Same as above but for CIFAR-100 (no visualization).

