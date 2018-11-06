# input distillation
이 코드는 Open-nmt 기반으로 만들어진 input distillation conversation 모델 입니다.


## How to use

### step 1 : distillation datasets
```bash
sh start.sh
```

output files in 'distill_files'

* `similarity_list_{X}` : X번째 distill에서 만들어진 output 마다의 input간 유사도 리스트입니다. 
    [input_similarity]\t[remove_input_list]\t[output_sentence]
* `Response{X}.txt` : X번째 distill에 사용된 output list입니다. reward를 계산할때 사용합니다.
* `src-train.{X}` : X번째 source train dataset 입니다.
* `tgt-train.{X}` : X번째 target train dataset 입니다.
* `output.txt{X}` : X번째 모델이 X번째 dataset으로 만든 output 입니다.
* `model{X}.pt` : X번째 dataset으로 학습된 모델입니다.
* `input_emb_{X}.pkl` : input간의 유사도를 계산하기 위해 만들어진 embedding vectors입니다.
* `train_output_{X}.txt` : 강화학습을 용이하게 하기위해 X번째 모델마다 강화학습용 train dataset으로 만든 output 결과입니다.

### step 2 : train reinforcement learning model
```bash
python rein_train.py -model distill_files/model0.pt -src distill_files/src-train.0
```

training result

* `save/new_RL_model_epoch{Y}.pt` : Y번째 epoch 마다의 저장된 모델(10 epoch 마다 저장)

### step 3 : test reinforcement learning model
```bash
sh process.sh
python rein_test.py
```

testing result

* `reward_result.pkl` : test dataset에 대해 강화학습이 선택한 모델번호가 있습니다.
* `output_result.txt` : test dataset에 대해 강화학습이 만든 응답이 있습니다.
# input_distill
