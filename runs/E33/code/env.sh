# E33 只用（#226）：服务器上跑 finetune.train 所需的环境变量（同 scripts/finetune/run.py 的 ENV）。source 它。
Y=/remote-home/xufang/YGC
export MOON_DATA=$Y/dataset/Moon MOON_WEIGHTS=$Y/weights MOON_RESULTS=$Y/results CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_HOME=$Y/weights/torch_home
PY=/opt/envs/loftr/bin/python
