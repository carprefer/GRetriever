# G-Retriever report

## 1. Implementation


## 2. Evaluation

- 다음의 명령어로 evaluate.py를 돌릴 수 있다.
```shell
# llama-2-7b로 simpleQA 돌리기 
nohup python evaluate.py --dataset 0 > dataset_0.txt 2>&1 &
# llama-2-7b로 CWQ 돌리기 
nohup python evaluate.py --dataset 1 > dataset_1.txt 2>&1 &
# llama-2-7b로 WebQSP 돌리기(LLM만)
nohup python evaluate.py --dataset 2 --llm > dataset_2.txt 2>&1 &
# llama-2-70b로 GrailQA 돌리기 
nohup python evaluate.py --dataset 3 --model 1 > dataset_3.txt 2>&1 &
```
- 평가 방식을 같게 하기 위하여 evaluate.py는 내려받아 사용하였다.
- Inference-only에는 llama-2-7b-chat-hf를 사용하자!!!

| Setting | ExplaGraphs | SceneGraphs | WebQsp |
|---------|-------------|-------------|--------|
| Inference-only | 0.5433 | 0.35.72 | 47.02 |

- LLM만 사용했을 때의 결과이다.

| LLM | dataset | hit ratio |
|-----|---------|-----------|
| llama-2-7b-chat-hf | SimpleQA | 26.0% |
| llama-2-7b-chat-hf | CWQ | 43% |
| llama-2-7b-chat-hf | WebQSP | 69.4% |
| llama-2-7b-chat-hf | GrailQA | 33.0% |
| llama-2-7b-chat-hf | WebQuestions | 73.0% |



## 3. Limitation

- GPT-3.5나 GPT-4로 테스트해보지 못하였다.
- LLM의 답변에서 정보를 추출하는 parser를 좀 더 보완해야 할 것 같다.
- Freebase를 사용하였지만, 비어있는 정보들이 많아, 유용한 정보를 얻기 어려웠다.
- Transformer pipeline을 사용하여 LLM을 돌리면 속도가 느리다. SGLANG을 고려해보자.
- Hit@1을 적용해서 평가를 하자.