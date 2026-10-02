# usage: GPU=<g> testeval.sh <run> <step>：对某个 ckpt 跑 Test 推理并写进 <run>/testrun（工作台记录，含 Val）
cd /remote-home/xufang/YGC/moon-exp-w5
export MOON_DATA=/remote-home/xufang/YGC/dataset/Moon MOON_WEIGHTS=/remote-home/xufang/YGC/weights TORCH_HOME=/opt/torch_home
F=/remote-home/xufang/YGC/results/finetune; r=$1; st=$2; R=$F/$r
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 ionice -c3 /opt/envs/loftr/bin/python -m baselines.match configs/baselines/anymatch_loftr.json --split test --weights $R/ckpt_$st.pt --name step$st --out $R/match > $R/test_$st.log 2>&1
mkdir -p $R/testrun && [ -f $R/testrun/T/exp.toml ] || /opt/envs/wb/bin/python -m workbench --runs $R/testrun new T --title "$r test" >> $R/test_$st.log 2>&1
for sp in val test; do nice -n 10 /opt/envs/wb/bin/python -m baselines.fit $R/match/step$st --split $sp --run $R/testrun/T --name step$st >> $R/test_$st.log 2>&1; done
nice -n 10 /opt/envs/wb/bin/python -m workbench --runs $R/testrun eval T 2>&1 | grep auc
