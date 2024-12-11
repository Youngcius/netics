# Netics: GPU-based Distributed QECC Decoder 

This repository contains the implementations of algorithmic decoders for repetition codes and surface codes, especially deployed on GPU in a paradigm of distributed computing.

It is part of my internship work at Alibaba Quantum Laboratory.



**Repetition code:**

| Syndrome graph                               | Monolothic UF decoding                         | Distributed (GPU) UF decoding                  |
| -------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| ![](./experiments/results/rep-syndromes.png) | ![](./experiments/results/rep-mono-decode.png) | ![](./experiments/results/rep-dist-decode.png) |


**Surface code:**

| Syndrome graph                                | Monolothic UF decoding                          | Distributed (GPU) UF decoding                   |
| --------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| ![](./experiments/results/surf-syndromes.png) | ![](./experiments/results/surf-mono-decode.png) | ![](./experiments/results/surf-dist-decode.png) |



**CPU v.s. GPU UF decoding on surface code:**

![](./experiments/results/cpu-gpu-comparison.png)
