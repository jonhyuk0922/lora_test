import json

# DeepSpeed 설정 딕셔너리
ds_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": True,
        "initial_scale_power": 8  # 초기 스케일링 값 (FP16 학습 안정화)
    },
    "zero_optimization": {
        "stage": 2  # ZeRO 최적화 Stage 2 설정
    },
    "sparse_attention": {
        "mode": "fixed",
        "block": 16,
        "different_layout_per_head": True,
        "num_local_blocks": 4,
        "num_global_blocks": 1,
        "attention": "bidirectional",
        "horizontal_global_attention": False,
        "num_different_global_patterns": 4
    },
    "inference": {
        "enabled": True,
        "dtype": "fp16"  # 추론 시 FP16 사용
    }
}

# JSON 파일로 저장
with open("deepspeed_config.json", "w") as json_file:
    json.dump(ds_config, json_file, indent=4)