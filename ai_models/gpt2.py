#사용 모델 : GPT 1024 토큰 
#기능 
# 1. 모델로딩 
# 2. 텍스트 입력 받음
# 3. 텍스트 슬라이싱 
# 4. 요약 



from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model_and_tokenizer():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    print("Model and tokenizer loaded successfully.")
    
    return model, tokenizer

def determine_length(input_text_length):
    if input_text_length <= 1000:
        return 50, 100
    elif input_text_length <= 2000:
        return 75, 150
    elif input_text_length <= 3000:
        return 100, 200
    elif input_text_length <= 4000:
        return 125, 250
    elif input_text_length <= 5000:
        return 150, 300
    else:
        return 200, 400

def summarize_text_with_loaded_model(model, tokenizer, input_text):
    input_text_length = len(input_text)
    min_length, max_new_tokens = determine_length(input_text_length)
    
    # 입력 텍스트를 토큰화 (GPT-2 모델은 최대 1024 토큰까지 처리 가능)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    
    # 단어 수와 토큰 수 계산 및 출력
    words = input_text.split()
    tokens = tokenizer.tokenize(input_text)
    print(f"입력 텍스트: {input_text}")
    print(f"단어 수: {len(words)}")
    print(f"토큰 수: {len(tokens)}")
    
    # 슬라이딩 윈도우 기법을 사용하여 텍스트 처리
    stride = 512  # 슬라이딩 윈도우의 스트라이드
    summarized_text = ""

    for i in range(0, len(tokens), stride):
        # 현재 윈도우의 토큰을 선택 (시작 인덱스부터 최대 1024 토큰까지)
        window = tokens[i:i + 1024]
        
        # 선택된 토큰을 토큰 ID로 변환
        input_ids = tokenizer.convert_tokens_to_ids(window)
        input_ids = torch.tensor([input_ids])
        
        # attention_mask 생성
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        # 자동 미분 비활성화, 메모리 사용량을 줄이고 연산 속도 증가
        with torch.no_grad():
            # 모델을 사용하여 요약 생성
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask,  # attention_mask 설정
                max_new_tokens=max_new_tokens,  # 요약 텍스트의 최대 길이
                num_beams=5,                    # 빔 검색(beam search)의 빔 수. 높은 값은 더 다양한 결과를 생성하지만, 계산 비용이 증가함
                length_penalty=0.1,             # 생성된 텍스트의 길이를 제어하는 패널티. 값이 높을수록 더 긴 텍스트를 선호함
                no_repeat_ngram_size=3,         # 반복되는 n-그램을 피하기 위해 사용할 n-그램의 크기. 동일한 3-그램의 반복을 피함
                early_stopping=True,            # 더 이상 유의미한 단어를 생성하지 못한다고 판단되면 조기 종료
                do_sample=False,                # 샘플링 대신 빔 검색을 사용하여 결과를 생성. 동일한 입력에 대해 항상 동일한 출력을 생성함
                pad_token_id=tokenizer.eos_token_id  # pad_token_id 설정
            )

        # 생성된 텍스트 디코딩 (요약 텍스트 생성)
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summarized_text += decoded_text + " "

    return summarized_text.strip()

def main():
    print("Starting test summary : ver 0.04")
    # 모델과 토크나이저를 로드
    model, tokenizer = load_model_and_tokenizer()

    while True:
        input_text = input("Enter the text to summarize (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break

        try:
            summary = summarize_text_with_loaded_model(model, tokenizer, input_text)
            print("****************************************************************")
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()



