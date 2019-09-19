# Sampling-Bias-Active-Learning

This repository contains the code for our EMNLP '19 paper:

[Sampling Bias in Deep Active Classification: An Empirical Study]()

[Ameya Prabhu](https://drimpossible.github.io/)\*, [Charles Dognin](https://www.linkedin.com/in/charlesdognin)\* and [Maneesh Singh](https://www.linkedin.com/in/maneesh-singh-3523ab9)  (\* Authors contributed equally).

### Citation
If you find our work useful in your research, please consider citing:

	Coming soon..

## Introduction


## Installation and Dependencies

Install all requirements required to run the code by:
	
	# Activate a new virtual environment
	$ pip install -r requirements.txt

## Usage

Step 1: Download all datasets for the main study experiments, 
	
	$ bash download_datasets.sh

Code is present in `src` folder.

To reproduce all experimental results,

Step 1: To train a model with our AL framework, please use:

	$ cd src 
	$ python main.py [Your arguments]
	
In particular, you can choose:
- `--dataset`: The dataset (among the 8 available in our paper)
- `--model`: the model (FastText or Multinomial Bayes)

You can also choose all the hyperparameters of your model as well as many active learning related hyperparameters (acquisition function, number of initial data points, number of points queries at each iterations...)
For a comprehensive list of arguments, please look at the opts.py function. 	

Step 2: To replicate our results with your logs or our logs, please use: 

    $ compute_intersection.py --dataset [DATASET] --logs_dir [PATH_TO_LOGS] --model [MODEL] --same_seed [BOOL] --dif_seed [BOOL]
    $ compute_intersection_support.py --dataset [DATASET] --logs_dir [PATH_TO_LOGS] --model [MODEL] --same_seed [BOOL] --dif_seed [BOOL]
    $ compute_label_entropy.py --dataset [DATASET] --logs_dir [PATH_TO_LOGS] --model [MODEL] --same_seed [BOOL] --dif_seed [BOOL]


## Contact

If facing any problem with the code, please open an issue here. Please do get in touch with us by email for any questions, comments, suggestions regarding the paper!
ameya.prabhu@oxford.edu, charles.dognin@verisk.com. 

Code stubs and formatting borrowed from [Deep Expander Networks](https://github.com/drimpossible/Deep-Expander-Networks) repository.
