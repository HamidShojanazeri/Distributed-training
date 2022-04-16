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
For running BERT with Torchrun

```
torchrun --nnodes 1 --nproc_per_node 4  FSDP_BERT_torchrun.py

```
## FSDP T5

To run T5 example with FSDP and DDP(just need to uncomment the DDP wrapping in the script) for text_summerization
* Download the two CSV files in [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) dataset as linked below:
* [wikihowAll.csv](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358) and [wikihowSep.csv](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag)
* Create "data" folder and place the two files in the folder.
* Run the script
```
python FSDP_T5.py

```
For running T5 with Torchrun

```
torchrun --nnodes 1 --nproc_per_node 4  FSDP-T5-torchrun.py

```
