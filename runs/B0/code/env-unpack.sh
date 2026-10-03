#!/bin/bash
# on a gpfs host (126/160): unpack envs + scripts to local disk, set up torch hub cache
set -e
S=/remote-home/xufang/YGC/envpack
W=/remote-home/xufang/YGC/weights
mkdir -p /opt /root/scripts /opt/torch_home/hub/checkpoints
cd /opt && nice -n 19 ionice -c3 tar -xf $S/envs.tar
cp $S/run-method.sh $S/queue.sh /root/scripts/ && chmod +x /root/scripts/*.sh
C=/opt/torch_home/hub/checkpoints
ln -sf $W/backbones/dinov2_vitl14_pretrain.pth $C/
ln -sf $W/backbones/vgg19_bn-c79401a0.pth $C/
ln -sf $W/official/superpoint/superpoint_v1.pth $C/
ln -sf $W/official/lightglue/superpoint_lightglue_v0-1_arxiv.pth $C/
/opt/envs/loftr/bin/python -c "import torch, kornia; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
df -h /opt | tail -1
