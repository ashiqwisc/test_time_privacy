# Inducing Uncertainty on Open-Weight Models for Test-Time Privacy in Image Recognition

[Muhammad H. Ashiq](https://github.com/ashiqwisc)¹ · [Peter Triantafillou](https://warwick.ac.uk/fac/sci/dcs/people/peter_triantafillou/)² · [Hung Yun Tseng](https://openreview.net/profile?id=~Hung_Yun_Tseng1)¹ · [Grigoris G. Chrysos](https://grigorisg9gr.github.io/_pages/about/)¹  
¹University of Wisconsin-Madison · ²University of Warwick

[**Paper**](https://www.arxiv.org/abs/2509.11625)

---

A key concern for AI safety remains understudied in the machine learning (ML) literature: how can we ensure users of ML models do not leverage predictions on incorrect personal data to harm others? This is particularly pertinent given the rise of open-weight models, where simply masking model outputs does not suffice to prevent adversaries from recovering harmful predictions. To address this threat, which we call *test-time privacy*, we induce maximal uncertainty on protected instances while preserving accuracy on all other instances. Our proposed algorithm uses a Pareto optimal objective that explicitly balances test-time privacy against utility. We also provide a certifiable approximation algorithm which achieves $(\varepsilon, \delta)$ guarantees without convexity assumptions. We then prove a tight bound that characterizes the privacy-utility tradeoff that our algorithms incur. Empirically, our method obtains at least 3x stronger uncertainty than pretraining with marginal drops in accuracy on various image recognition benchmarks. Altogether, this framework provides a tool to guarantee additional protection to end users.

The paper was accepted as a long paper in the NeurIPS'25 workshop on Regulatable ML and the NeurIPS'25 workshop on Reliable ML for Unreliable Data. 

---

## Table of Contents
- [Installation](#installation)
- [File Guide](#fileguide)
- [Usage](#usage)

## Installation
To install the package, simply clone the repository and install dependencies through conda: 
```bash
conda env create -f env.yml
conda activate beyond_certified_unlearning
```

## Guide
Here is an overview of the files/folders and their functionalities: 
- `data`: A folder containing data; if no data is contained, it will be loaded automatically during training. 
- `experiments`: A folder containing `configs` which contains hyperparameter configurations for our various experiments. Please see `experiments/configs/CIFAR100/pareto/CIFAR100_ResNet50_75.yaml` for an example config for the main Alg. 1 experiments. Furthermore, contains experiments.txt, which contains all the commands (and more) for our experiments in the paper. Configs for the synthetic, LabelDP, and Gaussian baselines are in this folder as well. 
- `label_dp`: A link to the label_dp repository, which implements the paper by Ghazi et al. 2021. We use this a backend for frontend `labeldp.py`, which is integrated into our experimental pipeline so that we can use LabelDP as a baseline. Please see `experiments/configs/labeldp/CIFAR10/ResNet.yaml` for an example config file for `labeldp.py`. 
- `evaluator.py`: Contains code to evaluate accuracy and forget set metrics.
- `load_dataset.py`: Standard dataset loading code. 
- `models.py`: Standard code specifying models like logistic regression, MLP, ResNet18, and ResNet50. 
- `main.py`: Main function, entry point to our experimental pipeline. 
- `synthetic.py`: Implements our synthetic baseline, which is discussed in the Appendix. 
- `train.py`: Implements training, retraining, and Pareto finetuning with and without gradient surgery (Algorithm 1)
- `uniformity_exact.py`: Computes Algorithm 2.
- `uniformity_helper.py`: Additional code used in `uniformity_exact.py`.
- `visualization.py`: Saves softmax forget set outputs after uniformity has been induced, for inspection. All of this is done automatically. 
- `gaussian_datset.py`: Computes the GaussianUniform baseline discussed in the Appendix. 
- `testsampling.py`: Computes finetuning on corrupted test samples, as discussed in the Appendix. 
- `nearestneighbors.py`: Computes accuracy and confidence distance for nearest neighbors to the forget set, in the test set, as discussed in the Appendix. 
- `attack.py`:  Computes Gaussian, FGSM, and PGD TTP attacks, as discussed in the Appendix.
- `bound_value.py`: Computes tightness of bound in Theorem 3.5, as discussed in the Appendix. 
- `vit_trainer_cifar.py` and `vit_trainer_tinyimagenet.py`: Finetuning code for ViT pretrained on ImageNet, as discussed in the Appendix.

## Usage
First, please be sure to make empty `results`, `logs`, and `images` directories after cloning before running any experiments. 

Then, to reproduce experiments, please take a look at `experiments/experiments.txt` and run the appropriate commands. Check the config files in `experiments/configs` first to ensure that you are running the right experiment. Please remove the load model paths before running for the first time.

Pretraining and retraining baselines are implemented in the *_0.yaml file in `experiments/configs`; please run these to obtain the pretrained model before running any additional experiments. 

Note that for the LabelDP baseline, it is best practice to use only one GPU. For example: 
```bash
export CUDA_VISIBLE_DEVICES=1
```

Otherwise, due to deprecated code used in the baseline repository, one may run into errors with tensor shapes. 

