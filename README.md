This project is for very small GPT pretraining from scratch. 

For single computer multiple GPUs condition, run the following code in notebook

!torchrun --standalone --nproc_per_node=n file-path/GPT_train.py

(n is the number of GPUs in your computer, file-path is the file path of GPT_train.py)