from datasets import load_dataset

# 데이터셋 로드 및 분할 함수
def load_and_split_dataset(dataset_name="beomi/KoAlpaca-v1.1a", train_size=1000, val_size=100):
    dataset = load_dataset(dataset_name)
    
    # 데이터셋 분할
    train_data = dataset['train'].select(range(0, train_size))
    validation_data = dataset['train'].select(range(train_size, train_size + val_size))
    
    return train_data, validation_data

# 전처리 함수
def preprocess_function(examples, tokenizer, input_max_length=512, label_max_length=128):
    inputs = examples['instruction']
    targets = examples['output']
    
    # 입력 토큰화
    model_inputs = tokenizer(inputs, max_length=input_max_length, truncation=True, padding="max_length")
    
    # 출력 레이블 토큰화
    labels = tokenizer(targets, max_length=label_max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# 전처리 적용 함수
def apply_preprocessing(dataset, tokenizer, preprocess_fn, batched=True):
    return dataset.map(lambda examples: preprocess_fn(examples, tokenizer), batched=batched)

def prepare_data(tokenizer):
    # 데이터셋 로드 및 분할
    train_data, validation_data = load_and_split_dataset()

    # 데이터 전처리 적용
    train_data = apply_preprocessing(train_data, tokenizer, preprocess_function)
    validation_data = apply_preprocessing(validation_data, tokenizer, preprocess_function)

    return train_data , validation_data