# Distributed-training


## FSDP BERT

To run BERT example with FSDP
* downlaod the IMDB dataset 
* Run the script
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 
tar -xf aclImdb_v1.tar.gz
python FSDP_BERT.py

```
## DDP T5

To run T5 example with DDP for text_summerization
* downlaod the WikiHow from [here](https://github.com/mahnazkoupaee/WikiHow-Dataset) into data folder
* Run the script
```
python FSDP_T5.py

```
