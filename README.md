# Sampling-Bias-Active-Learning

This repository contains the code for our EMNLP '19 paper:

[Sampling Bias in Deep Active Classification: An Empirical Study](https://arxiv.org/pdf/1909.09389.pdf)

[Ameya Prabhu](https://drimpossible.github.io/)\*, [Charles Dognin](https://www.linkedin.com/in/charlesdognin)\* and [Maneesh Singh](https://www.linkedin.com/in/maneesh-singh-3523ab9)  (\* Authors contributed equally).

### Citation
If you find our work useful in your research, please consider citing:

	@misc{prabhu2019sampling,
    		title={Sampling Bias in Deep Active Classification: An Empirical Study},
    		author={Ameya Prabhu and Charles Dognin and Maneesh Singh},
    		year={2019},
   		eprint={1909.09389},
    		archivePrefix={arXiv},
    		primaryClass={cs.CL}
	}

## Introduction

The exploding cost and time needed for data labeling and model training are bottlenecks for training DNN models on large datasets. Identifying smaller representative data samples with strategies like active learning can help mitigate such bottlenecks. Previous works on active learning in NLP identify the problem of sampling bias in the samples acquired by uncertainty-based querying and develop costly approaches to address it. Using a large empirical study, we demonstrate that active set selection using the posterior entropy of deep models like FastText.zip (FTZ) is robust to sampling biases and to various algorithmic choices (query size and strategies) unlike that suggested by traditional literature. We also show that FTZ based query strategy produces sample sets similar to those from more sophisticated approaches (e.g ensemble networks). 

Finally, we show the effectiveness of the selected samples by creating tiny high-quality datasets, and utilizing them for fast and cheap training of large models. Based on the above, we propose a simple baseline for deep active text classification that outperforms the state-of-the-art. We expect the presented work to be useful and informative for dataset compression and for problems involving active, semi-supervised or online learning scenarios.

## Installation and Dependencies

Install all requirements required to run the code by:
	
	# Activate a new virtual environment
	$ pip install -r requirements.txt

## Usage

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


Step 3: To run the ULMFiT model on the resulting dataset, please use the original repository: 
https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts

## Reproducing the results

Step 1: Download all datasets for the main study experiments, 
	
	$ bash download_datasets.sh

Code is present in `src` folder.

To reproduce all experimental results,

	$ python generate_all_experiments.py > exp.sh
	$ bash exp.sh

This will produce some bash files. Run all of them parallely/sequentially to reproduce all experiments as in the paper!

## Contact

If facing any problem with the code, please open an issue here. Please email for any questions, comments, suggestions regarding the paper to us on `ameya.prabhu@mailfence.com` and `charles.dognin@verisk.com`. Thanks!
