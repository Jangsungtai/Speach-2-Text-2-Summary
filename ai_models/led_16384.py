#t사용 모델 : LED (Longformer Encoder-Decoder) 16384토큰

from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch

# 라이브러리 용도:
# - transformers: 자연어 처리를 위한 사전 학습된 모델을 제공
# - torch: PyTorch, 딥러닝을 위한 오픈 소스 라이브러리


def load_model_and_tokenizer():
    """
    LED 모델과 토크나이저를 로드하고 모델을 평가 모드로 설정합니다.
    - 모델과 토크나이저를 로딩 후 평가 모드로 설정하여 추론 준비를 완료합니다.
    """
    model_name = "allenai/led-base-16384"
    model = LEDForConditionalGeneration.from_pretrained(model_name)
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model.eval()  # 평가 모드 설정
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def determine_length(input_text_length):
    """
    입력 텍스트의 길이에 따라 요약의 최소 및 최대 길이를 결정합니다.
    """
    if input_text_length > 300:
        return 150, 300  # 텍스트 길이가 300 이상일 경우 최소 및 최대 길이 설정
    else:
        return 0, 0  # 300 이하인 경우 요약을 하지 않음

def summarize_text_with_loaded_model(model, tokenizer, input_text):
    """
    입력된 텍스트를 사용하여 텍스트를 요약하고, 요약 결과와 함께 단어 및 토큰 수를 반환합니다.
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    token_count = len(inputs['input_ids'][0])

    input_words = input_text.split()  # 입력 텍스트를 공백 기준으로 나누어 단어 수 계산
    #print(f"입력 텍스트 단어 수: {len(input_words)}, 입력 텍스트 토큰 수: {token_count}")


    if token_count <= 300:
        print("The text is 300 tokens or less; skipping summarization.")
        return input_text, len(input_words), token_count, len(input_words), token_count

    min_length, max_length = determine_length(token_count)
    
    with torch.no_grad():  # 자동 미분을 비활성화하여 연산 속도 및 메모리 효율을 향상
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],  # 어텐션 마스크 추가
            max_length=max_length,        # 요약 텍스트의 최대 길이
            min_length=min_length,        # 요약 텍스트의 최소 길이
            num_beams=5,                  # 빔 검색의 빔 수, 더 다양한 결과 생성
            length_penalty=1.0,           # 텍스트 길이를 제어하는 패널티, 값이 높을수록 긴 텍스트 선호
            no_repeat_ngram_size=3,       # 반복되는 n-그램 방지, 동일한 3-그램 반복 피함
            early_stopping=True,          # 유의미한 단어 생성 안 될 시 조기 종료
            do_sample=False               # 샘플링 대신 빔 검색 사용, 항상 동일한 출력 생성
        )

    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary_words = summarized_text.split()
    summary_tokens = tokenizer.tokenize(summarized_text)
    summary_token_count = len(tokenizer.convert_tokens_to_ids(summary_tokens))

    #print(f"Summary (단어수: {len(summary_words)}, 토큰수: {summary_token_count}): {summarized_text}")
    return summarized_text, len(input_words), token_count, len(summary_words), summary_token_count

def main():
    """
    사용자 입력을 반복적으로 받아 텍스트를 요약하고, 요약 결과 및 통계를 출력합니다.
    """
    model, tokenizer = load_model_and_tokenizer()
    while True:
        input_text = input("Enter the text to summarize (or 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break
        
        try:
            summary, input_words, input_tokens, summary_words, summary_tokens = summarize_text_with_loaded_model(model, tokenizer, input_text)
            print("****************************************************************")
            print(f"Summary: {summary}")
            print(f"Input Words: {input_words}, Input Tokens: {input_tokens}, Summary Words: {summary_words}, Summary Tokens: {summary_tokens}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
