The source code and data for the paper: **Towards Comprehensive Expert Finding with a Hierarchical Matching Network**.



## Requirements

- python >=3.6
- torch == 1.7.0
- numpy == 1.19.5
- tokenizers == 0.8.1rc2
- sklearn == 0.20.0
- Cuda Version == 11.1
- fire



## Basic

The introduction of the dirs and files:

- `main.py`: the entrance of the project
- `utils.py`: the ranking metrics for evaluating the model performance
- `data/`: the dir to save the preprocessed data (Bioinformatics dataset)
- `config/`: the parameter config 
- `models/`: the model architecture
- `checkpoints/`: the dir for saving the best model on validation dataset



## Usage

We provide the code and a dataset `Bioinformatics` for evaluation.

Run the `main.py` for training, validation and testing; the script is:

```sh
python3 main.py run --dataset=Bioinformatics
```

A sample output:

```
*************************************************
user config:
dataset => Bioinformatics
a_num => 113
q_num => 958
tag_num => 435
*************************************************
train data: 15680; test data: 1580; dev data: 1900
start training....
2021-11-23 03:00:45  Epoch 0: train data: loss:0.9826.
dev data results
mean_mrr: 0.3047; P@1: 0.1579; ndcg@10: 0.3659;
2021-11-23 03:02:56  Epoch 1: train data: loss:0.1997.
dev data results
mean_mrr: 0.2702; P@1: 0.1158; ndcg@10: 0.3393;
2021-11-23 03:05:06  Epoch 2: train data: loss:0.1884.
dev data results
mean_mrr: 0.2680; P@1: 0.1263; ndcg@10: 0.3222;
2021-11-23 03:07:18  Epoch 3: train data: loss:0.1769.
dev data results
mean_mrr: 0.3549; P@1: 0.2211; ndcg@10: 0.3980;
2021-11-23 03:09:31  Epoch 4: train data: loss:0.1746.
dev data results
mean_mrr: 0.3582; P@1: 0.2211; ndcg@10: 0.4037;
2021-11-23 03:11:44  Epoch 5: train data: loss:0.1710.
dev data results
mean_mrr: 0.3106; P@1: 0.1789; ndcg@10: 0.3632;
2021-11-23 03:13:57  Epoch 6: train data: loss:0.1687.
dev data results
mean_mrr: 0.3346; P@1: 0.1684; ndcg@10: 0.3930;
2021-11-23 03:16:10  Epoch 7: train data: loss:0.1682.
dev data results
mean_mrr: 0.3538; P@1: 0.2000; ndcg@10: 0.4063;
2021-11-23 03:18:23  Epoch 8: train data: loss:0.1661.
dev data results
mean_mrr: 0.3299; P@1: 0.1789; ndcg@10: 0.3799;
2021-11-23 03:20:36  Epoch 9: train data: loss:0.1674.
dev data results
mean_mrr: 0.3312; P@1: 0.1789; ndcg@10: 0.3830;
****************************************************************************************************
test data results
mean_mrr: 0.4715; P@1: 0.3083; ndcg@10: 0.5475;
****************************************************************************************************
```