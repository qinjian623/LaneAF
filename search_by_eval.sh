for i in `seq $1 5 $2`
do
  echo $i;
  CUDA_VISIBLE_DEVICES=7 python infer_culane.py --scale $4 --dataset-dir ~/../data/culane/ --snapshot $3/ep0$i.pth --output-dir $3/sbe_$i;
  hm_eval_lane --dataset-dir ~/../data/culane/ --output-dir $3/sbe_$i/outputs/ --type split > $3/sbe_$i/eval.json
done