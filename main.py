from lora import load_quantized_model , create_trainer 
from dataset import prepare_data

def main(output_dir):
    # model - qlora
    model_name = "beomi/gemma-ko-2b"
    model, tokenizer = load_quantized_model(model_name)
    
    # pad token 추가
    tokenizer.pad_token = tokenizer.eos_token
    # 모델의 토크나이저 크기 조정
    model.resize_token_embeddings(len(tokenizer))
    # dataset
    train_data, validation_data = prepare_data(tokenizer)

    # set trainer 
    trainer = create_trainer(model, train_data, validation_data, tokenizer, output_dir)
    
    trainer.train()


if __name__ == "__main__":
    output_dir = "./save"
    main(output_dir)