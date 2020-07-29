# BERT

## Benchmark Information

BERT is the
[BERT masked language modeling](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert) benchmark.

## Software

Tensorflow v1.

## Hardware

TPU v3 and TPU v4.

## Model
### Publication/Attribution

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. [BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding]
(https://arxiv.org/abs/1810.04805). NAACL 2019.

## Dataset Preparation

*   [BERT Wikipedia dataset preparation](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets)

## Research submissions

A subset of research submissions were run using Google internal infrastructure.
Contact Peter Mattson (petermattson@google.com) for more details.

## Submissions CLs
Framework	Platform	Benchmark	TPU Worker CL	TPU Green VM	FLAX Commit	JAX Commit
TF	tpu-v4-512-TF	gnmt	317991884	n/a	n/a	n/a
TF	tpu-v4-512-TF	transformer	317773961	n/a	n/a	n/a
TF	tpu-v4-512-TF	bert	318324155	n/a	n/a	n/a
TF	tpu-v4-512-TF	resnet	316295790	n/a	n/a	n/a
TF	tpu-v4-512-TF	maskrcnn	317770796	n/a	n/a	n/a
TF	tpu-v4-512-TF	ssd	316295790	n/a	n/a	n/a
TF	tpu-v4-128-TF	minigo	316295790	n/a	n/a	n/a
TF	tpu-v4-128-TF	gnmt	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	transformer	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	bert	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	resnet	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	maskrcnn	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	dlrm	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	ssd	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	gnmt	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	transformer	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	bert	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	resnet	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	maskrcnn	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	dlrm	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	ssd	317896100	n/a	n/a	n/a
TF	tpu-v3-1024-TF	gnmt	317991884	n/a	n/a	n/a
TF	tpu-v3-1024-TF	maskrcnn	317770796	n/a	n/a	n/a
TF	tpu-v3-8192-TF	transformer	317896100	n/a	n/a	n/a
TF	tpu-v3-8192-TF	bert	317585047	n/a	n/a	n/a
TF	tpu-v3-8192-TF	resnet	314149382	n/a	n/a	n/a
TF	tpu-v3-8192-TF	ssd	314149382	n/a	n/a	n/a
TF2	tpu-v3-32-TF2.0	bert	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
TF2	tpu-v3-32-TF2.0	resnet	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
TF2	gpu-v100-8-TF2.0	bert	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
TF2	gpu-v100-8-TF2.0	resnet	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
JAX	tpu-v3-8192-JAX	transformer	cl/318350991	n/a	https://github.com/google/jax/commit/c9670d50c5015f36d7569bdf204b0651c64e55fb	https://github.com/google/flax/commit/ee20ca22dda285dac41e3e65f0a2d4cedc88bea8 
JAX	tpu-v3-8192-JAX	bert	cl/318008749	n/a	https://github.com/google/jax/commit/75278309aa00cd22e6b85422215383fa9d594ecb	https://github.com/google/flax/commit/ed42d0670c293d0ac205313b19fbef9832e81304 
JAX	tpu-v3-8192-JAX	resnet	cl/316292965	n/a	https://github.com/google/flax/commit/484cc11669f773ba69d184f2f26933954797943e	https://github.com/google/flax/commit/484cc11669f773ba69d184f2f26933954797943e
JAX	tpu-v3-4096-JAX	ssd	cl/317422581	n/a	https://github.com/google/jax/commit/8f4ba7e679b889ce8b75ef8fa07a1df947d89e52	https://github.com/google/flax/commit/4578df437833933fb5e24ac732d0cf7a5ed7d0e7
