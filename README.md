# CODE_DATA (TUDOR)
# AAAI26_CODE_DATA (TUDOR)
There's several additional data, however due to large file size for 'Supplementary Material', 'behaviors.tsv' are not available.
Instead, make README.md to describe code & data by descriptions.
** All the files in here are just the samples.

## File Descriptions
### Adressa_1w / datas
- `1w_behaviors.tsv`  
  User's click logs for one week.
- 'all_news_nyheter_splitted.tsv'
  Category 'nyheter' is replaced to its subcategories and applied to all_news.tsv for one week.
- `all_news.tsv`  
  All news met datas for one week.
- 'category2int_nyheter_splitted.tsv'
  Category 'nyheter' is replaced to its subcategories and applied to category2int.tsv for one week. 
- 'category2int.tsv'
  Map category to int(index) for one week.
- `news_publish_times.tsv`  
  Publish times of news for one week.
- 'news2int.tsv'
  Map news to int(index) for one week.
- 'user2int.tsv'
  Map user to int(index) for one week.

### codes
#### model
- `config.py`  
  Experiment settings and hyperparameter definitions.
- `main.py`  
  File for training and evaluation pipeline.
- `TUDOR.py`  
  TUDOR model architecture.

#### preprocess
- `make_1w_datas.py`  
  Preprocess data for one week experiments.
- `make_total_graph_1w.py`  
  Build the user–news interaction graph for one week.
- `test_ns_idx_1w.py`  
  Map negative samples and publish times for test.
- `tkg_negative_sampling_1w.py`  
  Generate negative samples considering news lifetime(36h). 
- `train_ns_idx_1w.py`  
  Map negative samples and publish times for train.

#### utils
- `general/attention/additive.py`  
  Collection of common utility functions.
- `EarlyStopping.py`  
  Early stopping implementation.
- `evaluate.py`  
  Calculate evaluation metrics (nDCG, MRR).
- `function.py`  
  Utility functions.
- `layer.py`  
  Definitions of  neural network layers.
- `make_test_datas_1w.py`  
  Make inputs for model test from preprocessed data.
- `make_train_datas_1w.py`  
  Make inputs for model train from preprocessed data.
- `MSA_news_encoder.py`  
  Multi-head self-attention NewsEncoder using title & category.
- `nce_loss.py`  
  Calculate negative log-likelihood(NLL) loss.
- `ns_indexing.py`  
  Handling negative sampling indices.
- `time_split_batch.py`  
  Split global graphs to time windows.



### data
- `all_news.tsv`  
  Containing all news metadata and content.
- 'scaled_down_pretrained_word_embedding.npy'
  Scaled-down version due to large size of the original data. 
- `word2int.tsv`  
  Map words to int(index).

## File Execution Order
1. make_1w_datas.py
2. tkg_negative_sampling_1w.py
3. make_total_graph_1w.py
4. train_ns_idx_1w.py
5. test_ns_idx_1w.py
6. main.py

## File Descriptions
### Adressa_1w / datas
- `1w_behaviors.tsv`  
  User's click logs for one week.
- 'all_news_nyheter_splitted.tsv'
  Category 'nyheter' is replaced to its subcategories and applied to all_news.tsv for one week.
- `all_news.tsv`  
  All news met datas for one week.
- 'category2int_nyheter_splitted.tsv'
  Category 'nyheter' is replaced to its subcategories and applied to category2int.tsv for one week. 
- 'category2int.tsv'
  Map category to int(index) for one week.
- `news_publish_times.tsv`  
  Publish times of news for one week.
- 'news2int.tsv'
  Map news to int(index) for one week.
- 'user2int.tsv'
  Map user to int(index) for one week.

### codes
#### model
- `config.py`  
  Experiment settings and hyperparameter definitions.
- `main.py`  
  File for training and evaluation pipeline.
- `TUDOR.py`  
  TUDOR model architecture.

#### preprocess
- `make_1w_datas.py`  
  Preprocess data for one week experiments.
- `make_total_graph_1w.py`  
  Build the user–news interaction graph for one week.
- `test_ns_idx_1w.py`  
  Map negative samples and publish times for test.
- `tkg_negative_sampling_1w.py`  
  Generate negative samples considering news lifetime(36h). 
- `train_ns_idx_1w.py`  
  Map negative samples and publish times for train.

#### utils
- `general/attention/additive.py`  
  Collection of common utility functions.
- `EarlyStopping.py`  
  Early stopping implementation.
- `evaluate.py`  
  Calculate evaluation metrics (nDCG, MRR).
- `function.py`  
  Utility functions.
- `layer.py`  
  Definitions of  neural network layers.
- `make_test_datas_1w.py`  
  Make inputs for model test from preprocessed data.
- `make_train_datas_1w.py`  
  Make inputs for model train from preprocessed data.
- `MSA_news_encoder.py`  
  Multi-head self-attention NewsEncoder using title & category.
- `nce_loss.py`  
  Calculate negative log-likelihood(NLL) loss.
- `ns_indexing.py`  
  Handling negative sampling indices.
- `time_split_batch.py`  
  Split global graphs to time windows.



### data
- `all_news.tsv`  
  Containing all news metadata and content.
- 'scaled_down_pretrained_word_embedding.npy'
  Scaled-down version due to large size of the original data. 
- `word2int.tsv`  
  Map words to int(index).

## File Execution Order
1. make_1w_datas.py
2. tkg_negative_sampling_1w.py
3. make_total_graph_1w.py
4. train_ns_idx_1w.py
5. test_ns_idx_1w.py
6. main.py
