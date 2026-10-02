cd /remote-home/xufang/YGC/moon-exp-roma
F=/remote-home/xufang/YGC/results/finetune
for k in 0 1 2; do /opt/envs/loftr/bin/python -m finetune.label_from_raw $F/_relabel_M3/s$k/train.npz --out $F/_relabel_M3/s$k.labels.jsonl > $F/_relabel_M3/s$k.labels.log 2>&1 & done
wait
cat $F/_relabel_M3/s0.labels.jsonl $F/_relabel_M3/s1.labels.jsonl $F/_relabel_M3/s2.labels.jsonl > $F/labels_rm1.jsonl.tmp && mv $F/labels_rm1.jsonl.tmp $F/labels_rm1.jsonl
