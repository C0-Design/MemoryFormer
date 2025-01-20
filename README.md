# MemoryFormer: Minimize Transformer Computation by Removing Fully-Connected Layers
Code to reproduce the work from NeurIPS2024 paper "MemoryFormer: Minimize Transformer Computation by Removing Fully-Connected Layers"\
link to paper https://arxiv.org/pdf/2411.12992
## Training
1. Prepare for the PILE dataset according to official Pythia repo:\
https://github.com/EleutherAI/pythia?tab=readme-ov-file#exploring-the-dataset

2. Download the necessary library:
```
cd MemoryFormer
git clone https://github.com/libxsmm/libxsmm.git
```

3. Run the training script:
```
python ./deepy.py train.py ./configs/memoryformer_tiny.yml
*or*
python ./deepy.py train.py ./configs/memoryformer_small.yml
```
Feel free to adjust the batch-size / num-gpu etc. in the ***.yml** file according to your own serve's configuration.

## Evaluation
Evaluate the pre-trained MemoryFormer using the builtin script:
```
python ./deepy.py eval.py -d configs your_configs.yml --eval_tasks task1 task2 ... taskn
```
More detailed instruction for evaluation found at:\
https://github.com/C0-Design/MemoryFormer/blob/master/GPT-NeoX-README.md#evaluation

## Citing
If you find this implementation helpful, please consider citing these great works

```
@inproceedings{dingmemoryformer,
title={MemoryFormer: Minimize Transformer Computation by Removing Fully-Connected Layers}, 
author={Ding, Ning and Tang, Yehui and Qin, Haochen and Zhou, Zhenli and Xu, Chao and Li, Lin and Han, Kai and Heng, Liao and Wang, Yunhe},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}, 
year={2024} 
}
```
