for i in $(seq 0 3)
do
  python preprocess.py -train_src distill_files/src-train.$i -train_tgt distill_files/tar-train.$i -valid_src demo_temp/src_valid.txt -valid_tgt demo_temp/tgt_valid.txt -save_data demo_temp/demo_$i
  python train.py -data demo_temp/demo_$i -save_model models/demo-model -gpuid 1
  cp models/demo-model_step_100000.pt distill_files/model$i.pt
  python translate.py -model distill_files/model$i.pt -src distill_files/src-train.$i -output distill_files/output.txt$i -replace_unk -verbose -gpu 0
  python new_distillation.py $i
  python input_distillation.py $i
done
for i in $(seq 0 3)
do
  python translate.py -model distill_files/model$i.pt -src distill_files/src-train.0 -output distill_files/train_output_$i.txt -replace_unk -verbose -gpu 0
done
