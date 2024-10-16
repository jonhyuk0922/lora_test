import nltk
import dataset
from datasets import load_metric

# nltk 패키지의 punkt 다운
nltk.download('punkt')

# Rouge 지표 불러오기
rouge_metric = load_metric("rouge")

# compute_metrics 함수 정의
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # -100은 패딩 토큰에 할당된 값이므로 이를 무시
    labels = [[label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge 스코어 계산
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # 특정 결과값만 추출 (RougeL은 요약 작업에서 많이 사용됨)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return result