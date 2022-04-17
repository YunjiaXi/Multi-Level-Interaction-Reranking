# MIR
The implementation of reranking method proposed in [Multi-Level Interaction Reranking with User Behavior History]()

## Requirements
```
tensorflow-gpu >= 1.9.0,<2
numpy >= 1.16.4
scikit-learn >= 0.21.2
lightgbm >= 2.3.2
```

## Get Started

Download data from  [Ad](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56) and [PRM Public](https://github.com/rank2rec/rerank), and preprocess

```
python preprocess_ad.py
python preprocess_prm.py
```

Run initial ranker
```
python run_init_ranker.py
```
Run re-ranker
```
python run_mir.py
```

Model parameters can be set by using a config file, and specify its file path at `--setting_path`, e.g., `python run_ranker.py --setting_path config`. The config files for the different models can be found in `example/config`. Moreover, model parameters can also be directly set from the command line. The supported parameters are listed as follows.

#### Parameters of `run_init_ranker.py`

| argument          | usage                                                                                                                                                                                          |
| ----------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data_dir`      | The path to the directory where the data is stored                                                                                                                                             |
| `--save_dir`      | The path to the directory where the models and logs are stored                                                                                                                                 |
| `--model_type`    | The algorithm of reranker, including `DNN`, `DIN`, and `LambdaMART`<br />**PLEASE ATTENTION**: Before training `lambdaMART`,  you need to train  `DNN` to <br /> get the pre-trained embedding |
| `--setting_path`  | The path to the `json` config file, like files in `example\config`                                                                                                                             |
| `--max_hist_len`   | The max length of history                                                                                                                                                                      |
| `--data_set_name` | The name of the dataset, such as `ad` and `prm`                                                                                                                                                |
| `--epoch_num`     | The number of  epoch for `DNN` model                                                                                                                                                           |
| `--batch_size`    | Batch size for `DNN` model                                                                                                                                                                     |
| `--lr`            | Learning rate for `DNN` and `lambdaMART`                                                                                                                                                       |
| `--l2_reg`        | The coefficient of l2 regularization for DNN model                                                                                                                                             |
| `--eb_dim`        | The size of embedding for DNN model                                                                                                                                                            |
| `--tree_num`      | The number of trees for `lambdaMART` model                                                                                                                                                     |
| `--tree_type`     | The type of tree for `lambdaMART` model, including `lgb` and `sklearn`                                                                                                                         |



##### Parameters of `run_mir.py`

| argument           | usage                                                                                                              |
|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--data_dir`       | The path to the directory where the data is stored                                                                 |
| `--save_dir`       | The path to the directory where the models and logs are stored                                                     |
| `--setting_path`   | The path to the `json` config file, like files in `example\config`                                                 |
| `--data_set_name`  | The name of the dataset, such as `ad` and `prm`                                                                    |
| `--initial_ranker` | The name of initial ranker, including `DNN`, `lambdaMART`.                                                         |
| `--epoch_num`      | The number of  epoch                                                                                               |
| `--max_hist_len`   | The max length of history                                                                                          |
| `--batch_size`     | Batch size                                                                                                         |
| `--lr`             | Learning rate                                                                                                      |
| `--l2_reg`         | The coefficient of l2 regularization                                                                               |
| `--eb_dim`         | The size of embedding                                                                                              |
| `--hidden_size`    | The size of hidden unit, usually the hideen size of LSTM/GRU                                                       |
| `--keep_prob`      | Keep prob in dropout                                                                                               |
| `--metric_scope`   | The scope of metrics, for example when `--metric_scope=[1, 3, 5]`,  <br />MAP@1, MAP@3, and MAP@5 will be computed |
| `--max_norm`       | The max norm of gradient clip                                                                                      |


## Structure

### Data processing
We process two datasets, [Ad](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56) and [PRM Public](https://github.com/rank2rec/rerank), containing user and item features with recommendation lists for the experimentation with personalized re-ranking. 

preprocess_ad.py and preprocess_prm.py: process Ad and PRM Public, respectively.


### Initial rankers

initial_model.py implements three initial ranking algorithms:
* DNN: a naive algorithm that directly train a multi-layer perceptron network with input labels (e.g., clicks).

* LambdaMART: the implementation of the LambdaMART model in <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf">*From RankNet to LambdaRank to LambdaMART: An Overview*</a>

* DIN: the implementation of Deep Interest Network in [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)

run_init_ranker.py: the main function of initial ranker

### Re-ranking algorithm

model.py: the implementation of our proposed reranking method, MIR.

run_mir.py: the main function of reranker.


