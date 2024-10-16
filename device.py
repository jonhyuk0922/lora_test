import torch
print(torch.cuda.is_available())  # True가 나와야 GPU 사용 가능
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수 확인