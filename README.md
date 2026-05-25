<div align="center">

# Poly-EPO: Training Exploratory Reasoning Models

</div>

This is the official PyTorch implementation of our paper "<strong>Poly-EPO: Training Exploratory Reasoning Models</strong>".


## Installation

In order for the installations to go smoothly, make sure you are operating from a GPU machine, typically one compatible with flash attention. It is ideal if you use the same GPU machines that you would use to run your experiments. 

Our installation is the same as that of [maxrl](https://github.com/tajwarfahim/maxrl/tree/main). In particular, follow the steps below to ensure exact match with our environment setting.

First, create a fresh conda environment

```
conda create -n polyepo python==3.10
conda activate polyepo
```

Next, install pytorch and associated dependencies. In particular, we use the following version:

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Now we should install flash-attention. To do this smoothly, we will build it from source, but feel free to use any other method of choice as long as it works. 

Run the following commands one by one (we can change MAX_JOBS based on how much CPU memory and cores we have):

```
pip install ninja
pip install packaging
pip install psutil

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
export MAX_JOBS=4
python setup.py install
```

Next, setup vllm.

```
pip install vllm==0.8.4
```

Setup additional things like wandb and math-verify.

```
pip install wandb
pip install math-verify
```

Now setup our codebase. Make sure you are inside the project folder, and run

```
pip install -e .
```

This should finish necessary installations. Note that it is possible that different packages may end up breaking since package versions keep changing, please your own judgement to fix them/reach out to us in case the above setup process leads to error. Thanks!

## Reproducing our experiments

### Qwen3-1.7B-Base and Qwen3-4B-Base experiments

1. Download and preprocess all the datasets. Change the local file paths depending on your machine.

```
# Training dataset
python examples/polyepo_data_preprocess/polaris.py --local_dir /path/to/polaris

# Evaluation dataset
python examples/polyepo_data_preprocess/aime25.py --local_dir /path/to/aime25
python examples/polyepo_data_preprocess/beyondaime.py --local_dir /path/to/beyondaime
python examples/polyepo_data_preprocess/math_500.py --local_dir /path/to/math_500
python examples/polyepo_data_preprocess/minerva.py --local_dir /path/to/minerva
```

2. Now run the following script (modify to run different algorithms/change local file paths appropriately):

```
bash qwen3_experiments/qwen3_polyepo_withdrgrpo.sh
```

Note that we use 4xH200 GPUs for our training runs, please change the hyperparameters (or system-specific environment variables) appropriately according to the number of GPUs available in your system.


## Acknowledgements
The codebase for the algorithm is built on top of [maxrl](https://github.com/tajwarfahim/maxrl/tree/main), and we express our gratitude to the authors of maxrl for providing us with an easy-to-work-with codebase!
