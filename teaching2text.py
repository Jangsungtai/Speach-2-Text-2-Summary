import subprocess  # 시스템 명령어를 실행하기 위한 모듈
import os          # 운영 체제와 상호 작용하기 위한 모듈

def main():
    # 현재 스크립트가 있는 디렉토리
    current_directory = os.path.dirname(__file__)
    # webserver 디렉토리
    webserver_directory = os.path.join(current_directory, 'webserver')
    # 실행할 파일 경로
    app_path = os.path.join(webserver_directory, 'app_0.3.py')

    try:
        # 스트림릿 앱 실행
        subprocess.run(['streamlit', 'run', app_path])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
