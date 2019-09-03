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

Step 1: Train all models and generate the actively learnt sample sets:

	$ python generate_all_experiments.py > experiments.sh
	$ bash experiments.sh
	$ rm experiments.sh

Step 2: To generate all the results across all tables:

	$ python generate_all_tables.py > tables.sh
	$ bash tables.sh
	$ rm tables.sh

Pretrained models available here:

## Contact

If facing any problem with the code, please open an issue here. Please do get in touch with us by email for any questions, comments, suggestions regarding the paper!

Code stubs and formatting borrowed from [Deep Expander Networks](https://github.com/drimpossible/Deep-Expander-Networks) repository.
