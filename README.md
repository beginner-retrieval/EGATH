# GCDH
Source code of "Graph Convolutional Network Discrete Hashing for Cross-Modal Retrieval" accepted by TNNLS.
## Environment:
 - pytorch 1.7.0+
## Device:
 - NVIDIA RTX-2080Ti GPU with 32 GB RAM

## Quick start
you can choose **--flag 'mir' | 'nus' | 'coco'**, for other parameters are set to same with those in our paper.

<!-- 如果有较好的服务器可以尝试加大batchsize -->
- train:
 python main.py train --flag 'mir' --batch_size 64 --device 0
 
- test:
 python main.py test --flag 'mir' --batch_size 64 --device 0
