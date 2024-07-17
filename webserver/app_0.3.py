#CLI 실행방법 : streamlit run app_0.3.0.py
#에러 발생 : https://www.youtube.com/watch?v=fgdIkPXMNyA
#작동 영상 : https://www.youtube.com/watch?v=NHopJHSlVo4
#썸네일 주소 형식 : https://img.youtube.com/vi/NHopJHSlVo4/mqdefault.jpg

#0.1.2 : 
# 1. 함수별 에러 처리 수행 
# 2. 썸네일 추가 
# 3. placeholder 위치 조정 

# 0.1.3 : 
# text 파일로 저장 기능 추가 
# 초기화 버튼 추가 reset 

# 0.2.0 : 
# 모델 테스트 : GPT2, LED, Bart, Bert, Kobert 
# Bart large 모델 활용 (Parameters : 406M, tokens : 1024)

# 0.3.0 : 
# GPT2, Bart large, LED 모델 선택 버튼 추가 

import streamlit as st
import tempfile
from pprint import pprint
import whisper
from pytube import YouTube, exceptions
import io  # 텍스트 저장 용
import sys
import os

# 현재 파일의 경로
current_directory = os.path.dirname(__file__)
# 상위 디렉토리로 이동
parent_directory = os.path.dirname(current_directory)
# ai_model 디렉토리의 경로를 추가
ai_models_path = os.path.join(parent_directory, 'ai_models')
sys.path.append(ai_models_path)

from LLM_summarize import load_model_and_tokenizer, summarize_text_with_loaded_model

# 비디오를 받아서 음성을 텍스트로 변환하는 함수 호출
def transcribeVideoOrchestrator(video, model_name: str):
    try:
        transcription = transcribe(video, model_name)
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# 비디오를 텍스트로 변환하는 함수
def transcribe(video: dict, model_name="medium"):
    try:
        print("Transcribing...", video['name'])
        print("Using model:", model_name)
        model = whisper.load_model(model_name)
        result = model.transcribe(video['path'])
        full_text = result['text']

        segments_info = []
        for segment in result['segments']:
            start_minutes, start_seconds = convert_to_minutes_seconds(segment['start'])
            end_minutes, end_seconds = convert_to_minutes_seconds(segment['end'])
            segment_info = {
                "id": segment['id'],
                "start": f"{start_minutes}:{start_seconds}",
                "end": f"{end_minutes}:{end_seconds}",
                "text": segment['text']
            }
            segments_info.append(segment_info)
        return {"text": full_text, "segments": segments_info}
    except Exception as e:
        st.error(f"Error during model processing: {e}")
        return None

# 초를 분과 초로 변환하는 함수
def convert_to_minutes_seconds(seconds):
    minutes = int(seconds) // 60
    remaining_seconds = int(seconds) % 60
    return minutes, remaining_seconds

# YouTube 비디오를 다운로드하는 함수
def downloadYoutubeVideo(youtube_url: str) -> dict:
    try:
        yt = YouTube(youtube_url)
        directory = tempfile.gettempdir()
        file_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(directory)
        video_length = yt.length
        return {"name": yt.title, "thumbnail": yt.thumbnail_url, "path": file_path, "duration": video_length}
    except exceptions.PytubeError as e:
        st.error(f"Error downloading video: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def main():
    try:
        st.set_page_config(page_title="Teaching2Text")  # 페이지 타이틀 설정
        st.title("Teaching2Text")
        st.markdown("ver0.3.0: Transcribe with Whisper models and summarize with Bert_Large, GPT2, or LED LLM models by AwesomeJ", unsafe_allow_html=True)

        # YouTube URL 입력
        sample_url = "https://www.youtube.com/watch?v=NHopJHSlVo4"
        st.markdown(f'<span style="color:gray;">Sample URL: {sample_url}</span>', unsafe_allow_html=True)
        url = st.text_input("Enter YouTube URL:", value=sample_url)

        st.markdown("---")

        # Speach2Text 모델 선택
        models = ["tiny", "base", "small", "medium", "large"]
        model = st.selectbox("Select Speach2Text Model:", models)
        st.write("If you take a smaller model it is faster but not as accurate, whereas a larger model is slower but more accurate.")
        
        models_mode = ["Normal mode", "English mode"]
        model_mode = st.selectbox("Select Speach2Text model_mode:", models_mode)
        st.write("Normal mode is good for any language, English mode is optimized specifically for English.")
        st.markdown("---")

        if model_mode == "English mode" and model in ["tiny", "base", "small", "medium"]:
            model = f"{model}_en"
        
        # LLM Summarize 모델 선택
        llm_models = ["bart_large", "led", "gpt2"]
        llm_model_choice = st.selectbox("Select LLM Summarize Model:", llm_models)
        st.markdown("---")

        if st.button("Transcribe"):
            thumbnail_placeholder = st.empty()
            video_info_placeholder = st.empty()
            transcription_placeholder = st.empty()

            if url:
                video_info = downloadYoutubeVideo(url)
                if video_info:
                    st.image(video_info['thumbnail'], caption=str(video_info['name'])) 
                    video_info_placeholder.subheader("Video Information:")
                    video_info_placeholder.write("Loading Videos")
                    transcript = transcribeVideoOrchestrator(video_info, model)
                    if transcript:
                        word_count = len(transcript["text"].split())
                        last_id = transcript["segments"][-1]["id"] + 1

                        video_info_placeholder.write(f"Video Name: {video_info['name']}")
                        video_info_placeholder.markdown(
                            f"Video Name: {video_info['name']}<br>Total Sentence: {last_id}<br>Total Words: {word_count}<br>Video Length: {video_info['duration']} seconds", unsafe_allow_html=True)
                        
                        transcription_placeholder.subheader("Full Text Transcription:")
                        transcription_output = ""
                        for segment in transcript["segments"]:
                            transcription_output += f"{segment['id']}, {segment['start']} ~ {segment['end']} : {segment['text']} <br>"
                        transcription_placeholder.markdown(transcription_output, unsafe_allow_html=True)

                        # 요약 기능 로드
                        llm_model, llm_tokenizer = load_model_and_tokenizer(llm_model_choice)
                        summary, input_words, input_tokens, summary_words, summary_tokens = summarize_text_with_loaded_model(llm_model, llm_tokenizer, transcript["text"], llm_model_choice)

                        # 결과 출력
                        st.subheader("Summary:")
                        st.write(f"Input words/tokens: {input_words} / {input_tokens}, Summary words/tokens: {summary_words} / {summary_tokens}")
                        st.write(summary)

                        # 텍스트 파일로 저장 버튼 추가
                        text_file_output = f"Video Name: {video_info['name']}\nTotal Sentence: {last_id}\nTotal Words: {word_count}\nVideo Length: {video_info['duration']} seconds\n\nFull Text Transcription:\n"
                        for segment in transcript["segments"]:
                            text_file_output += f"{segment['id']}, {segment['start']} ~ {segment['end']} : {segment['text']}\n"
                                                
                        text_file = io.BytesIO(text_file_output.encode('utf-8'))
                        st.download_button(
                            label="Download Transcription",
                            data=text_file,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                        # 리셋 버튼
                        if st.button("Reset"):
                            st.session_state.url = sample_url
                            st.session_state.model = 'medium'
                            st.experimental_rerun()
                    else:
                        st.error("Transcription failed. Please try again.")
                else:
                    st.error("Video download failed. Please check the URL and try again.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
