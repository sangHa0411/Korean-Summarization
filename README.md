# Korean-Summarization

## Goal
  1. Encoder - Decoder 모델을 이용해서 Summarization Task 베이스라인 만들기
  2. argument에서 PLM만을 변경해서 쉽게 각 모델의 성능을 쉽게 비교할 수 있도록 하기


## Dataset
  1. Aihub 도서자료 요약 데이터 
  2. 주소 : https://aihub.or.kr/aidata/30713
  3. 크기 : 대략 500M 
  4. Huggingface Dataset
      ![image](https://user-images.githubusercontent.com/48673702/150442297-4e1d5928-cf6b-492e-90b2-35f1b75220f8.png)
      ```python
      from datasets import load_dataset
      
      dataset = load_dataset('sh110495/book-summarization')
      ```
      
## Baseline 구축
  1. AutoModel, AutoConfig, AutoTokenizer을 이용한 모델 및 토크나이저 불러오기
  2. 한국어 맞춤의 rouge score를 계산하기 위해서 rouge-score 라이브러리 활용
  3. 학습 과정 및 결과를 보기 위한 wandb 활용

## Argument
|Argument|Description|Default|
|--------|-----------|-------|
|output_dir|model saving directory|exps|
|logging_dir|logging directory|logs|
|PLM|model name(huggingface)|gogamza/kobart-base-v2|
|epochs|train epochs|5|
|lr|learning rate|5e-5|
|train_batch_size|train batch size|16|
|eval_batch_size|evaluation batch size|16|
|generation_max_length|generation max token size|128|
|beam_size|generation beam size|1|
|warmup_steps|warmup steps|2000|
|weight_decay|weight decay|1e-4|
|evaluation_strategy|evaluation strategy|steps|
|gradient_accumulation_steps|gradient accumulation steps|1|
|max_input_len|max input token size|1024|
|max_target_len|max target token size|128|
|preprocessing_num_workers|preprocessing worker size|4|


## Terminal
  ```
  # 예시
  python train.py --max_input_len 1024 --max_target_len 128 --train_batch_size 16 --eval_batch_size 16 --eval_steps 5000 --epochs 3
  ```
  
## WandB
  1. Model : gogamza/kobart-base-v2
  2. Metrics : Loss, Rouge1, RougeL
      1. Loss
        ![image](https://user-images.githubusercontent.com/48673702/150443619-951f2a3d-26af-44d4-aa16-ce972e661b85.png)
      3. Rouge1
        ![image](https://user-images.githubusercontent.com/48673702/150443525-a34718cd-239e-498e-b877-45e87ffafd57.png)
      3. RougeL
        ![image](https://user-images.githubusercontent.com/48673702/150443667-6a44610f-fa03-4c70-8c25-0531fff27b26.png)

## Result
|Model|Loss|Rouge1|RougeL|
|-----|----|------|------|
|gogamza/kobart-base-v2|1.659|50.482|39.218|
