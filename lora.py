from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType , PeftType, get_peft_model
from transformers import TrainingArguments , Trainer

def configure_lora():
    """
    LoRA 설정을 위한 함수.
    """
    # QLoRA 설정
    lora_config = LoraConfig(
        peft_type=PeftType.LORA,  # PEFT 방법으로 LoRA 지정
        task_type=TaskType.CAUSAL_LM,  # 작업 유형: Causal LM
        inference_mode=False,  # 추론 모드 설정 여부
        r=8,  # 저랭크 행렬의 차원
        bias="none",  # 바이어스 파라미터를 학습하지 않음
        lora_alpha=32,  # 저랭크 행렬의 스케일링 계수
        lora_dropout=0.1,  # LoRA 계층에 적용되는 드롭아웃 확률
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
        ]
    )
    print("--- set lora config Done ---")
    return lora_config

def load_quantized_model(model_name: str):
    """
    양자화된 모델을 불러오고 LoRA를 적용하는 함수.
    
    Args:
    - model_name (str): 불러올 모델의 이름
    
    Returns:
    - model: 양자화 및 LoRA가 적용된 모델
    - tokenizer: 토크나이저
    """
    # 양자화를 위한 BitsAndBytesConfig 설정
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True  # 8비트 양자화 적용
    )

    # 모델과 토크나이저 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    print("--- model load Done ---")

    # LoRA 설정 적용
    lora_config = configure_lora()
    model = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 확인
    model.print_trainable_parameters()
    print("--- peft model load Done ---")
    
    return model, tokenizer

# # 함수 호출 예시
# model_name = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
# model, tokenizer = load_quantized_model(model_name)

def get_training_args(output_dir, learning_rate=1e-3, train_batch_size=4, eval_batch_size=4, epochs=2, weight_decay=0.01):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
    )

def create_trainer(model, train_data, validation_data, tokenizer, output_dir):
    training_args = get_training_args(output_dir=output_dir)
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics, # 필요시 활성화
    )