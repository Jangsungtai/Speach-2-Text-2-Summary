from transformers import BartForConditionalGeneration, BartTokenizer, LEDForConditionalGeneration, LEDTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

# 라이브러리 용도:
# - transformers: 자연어 처리를 위한 사전 학습된 모델을 제공
# - torch: PyTorch, 딥러닝을 위한 오픈 소스 라이브러리
# - 입력 토큰 제한으로 슬라이딩 기법 추가 

def load_model_and_tokenizer(model_choice="bart_large"):
    """
    선택된 모델과 토크나이저를 로드하고 모델을 평가 모드로 설정합니다.
    - 모델과 토크나이저를 로딩 후 평가 모드로 설정하여 추론 준비를 완료합니다.
    """
    if model_choice == "bart_large":
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    elif model_choice == "led":
        model_name = "allenai/led-base-16384"
        model = LEDForConditionalGeneration.from_pretrained(model_name)
        tokenizer = LEDTokenizer.from_pretrained(model_name)
    elif model_choice == "gpt2":
        model_name = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("Invalid model choice")

    model.eval()  # 평가 모드 설정
    print(f"{model_choice} Model and tokenizer loaded successfully.")
    return model, tokenizer

def determine_length(input_text_length):
    """
    입력 텍스트의 길이에 따라 요약의 최소 및 최대 길이를 결정합니다.
    """
    if input_text_length > 300:
        return 150, 300  # 텍스트 길이가 300 이상일 경우 최소 및 최대 길이 설정
    else:
        return 0, 0  # 300 이하인 경우 요약을 하지 않음

def generate_summary(model, tokenizer, inputs, max_length, min_length, model_choice):
    """
    주어진 입력 텍스트를 사용하여 요약문을 생성합니다.
    GPT-2는 다른 설정을 사용합니다.
    """
    with torch.no_grad():
        if model_choice == "gpt2":
            attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
            outputs = model.generate(
                inputs['input_ids'],      # 입력 텍스트의 토큰 ID
                max_length=max_length,    # 생성할 요약 텍스트의 최대 길이
                min_length=min_length,    # 생성할 요약 텍스트의 최소 길이
                num_beams=5,              # 빔 검색의 빔 수, 더 다양한 결과 생성
                length_penalty=1.0,       # 텍스트 길이를 제어하는 패널티, 값이 높을수록 긴 텍스트 선호
                no_repeat_ngram_size=3,   # 반복되는 n-그램 방지, 동일한 3-그램 반복 피함
                early_stopping=True,      # 유의미한 단어 생성 안 될 시 조기 종료
                do_sample=False,          # 샘플링 대신 빔 검색 사용, 항상 동일한 출력 생성
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            outputs = model.generate(
                inputs['input_ids'],      # 입력 텍스트의 토큰 ID
                max_length=max_length,    # 생성할 요약 텍스트의 최대 길이
                min_length=min_length,    # 생성할 요약 텍스트의 최소 길이
                num_beams=5,              # 빔 검색의 빔 수, 더 다양한 결과 생성
                length_penalty=1.0,       # 텍스트 길이를 제어하는 패널티, 값이 높을수록 긴 텍스트 선호
                no_repeat_ngram_size=3,   # 반복되는 n-그램 방지, 동일한 3-그램 반복 피함
                early_stopping=True,      # 유의미한 단어 생성 안 될 시 조기 종료
                do_sample=False           # 샘플링 대신 빔 검색 사용, 항상 동일한 출력 생성
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text_with_loaded_model(model, tokenizer, input_text, model_choice):
    """
    입력된 텍스트를 사용하여 텍스트를 요약하고, 요약 결과와 함께 단어 및 토큰 수를 반환합니다.
    슬라이딩 기법을 활용해서 determine_length() 함수에서 정해진 길이까지 요약 진행함.
    """
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length" if model_choice == "gpt2" else False)
    token_count = len(inputs['input_ids'][0])
    input_words = input_text.split()  # 입력 텍스트를 공백 기준으로 나누어 단어 수 계산

    if token_count <= 300:
        print("The text is 300 tokens or less; skipping summarization.")
        return input_text, len(input_words), token_count, len(input_words), token_count

    min_length, max_length = determine_length(token_count)
    
    summarized_texts = []
    stride = 512
    while True:
        for i in range(0, token_count, stride):
            end_idx = i + 1024
            input_slice = inputs['input_ids'][0][i:end_idx].unsqueeze(0)
            attention_mask = inputs['attention_mask'][0][i:end_idx].unsqueeze(0) if 'attention_mask' in inputs else None

            summarized_text = generate_summary(model, tokenizer, {'input_ids': input_slice, 'attention_mask': attention_mask} if attention_mask is not None else {'input_ids': input_slice}, max_length, min_length, model_choice)
            summarized_texts.append(summarized_text)

        combined_text = " ".join(summarized_texts)
        combined_token_count = len(tokenizer(combined_text)['input_ids'])

        if combined_token_count <= max_length:
            break
        else:
            inputs = tokenizer(combined_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length" if model_choice == "gpt2" else False)
            token_count = len(inputs['input_ids'][0])
            summarized_texts = []

    summarized_text = " ".join(summarized_texts)
    summary_words = summarized_text.split()
    summary_tokens = tokenizer.tokenize(summarized_text)
    summary_token_count = len(tokenizer.convert_tokens_to_ids(summary_tokens))

    return summarized_text, len(input_words), token_count, len(summary_words), summary_token_count

def main():
    """
    사용자 입력을 반복적으로 받아 텍스트를 요약하고, 요약 결과 및 통계를 출력합니다.
    """
    model_choice = input("Select model (bart_large, led, gpt2): ").strip().lower()
    if model_choice not in ["bart_large", "led", "gpt2"]:
        print("Invalid choice. Defaulting to 'bart_large'.")
        model_choice = "bart_large"
    
    model, tokenizer = load_model_and_tokenizer(model_choice)
    while True:
        input_text = input("Enter the text to summarize (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
        
        try:
            summary, input_words, input_tokens, summary_words, summary_tokens = summarize_text_with_loaded_model(model, tokenizer, input_text, model_choice)
            print("****************************************************************")
            print(f"Summary: {summary}")
            print(f"Input Words: {input_words}, Input Tokens: {input_tokens}, Summary Words: {summary_words}, Summary Tokens: {summary_tokens}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
