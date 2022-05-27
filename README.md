Code and data accompanying the paper **Submission 1207: Structural Characterization of Dialogues for Disentanglement**

## Model

1) download dataset from http://jkk.name/irc-disentanglement/
    
2) train our model. Example:
      ```
      python run.py \
      --output_dir ./experiments/v7_test_5e-6_bs2_50_128/ \
      --train_batch_size 2 \
      --eval_batch_size 2 \
      --do_train --do_test \
      --do_lower_case \
      --task_name bert_v7 \
      --max_seq_length 128 \
      --learning_rate 5e-6
      ```


### Requirements

Python 3.6

Pytorch 1.10.0

CUDA 10.2
