## 🕰️ 역사 챗봇 프로젝트

이 프로젝트는 사용자가 역사 관련 질문을 하면 인공지능이 답변을 제공하는 **역사 챗봇**입니다. 아래 단계를 따라 Miniconda를 사용해 프로젝트를 설정하고 실행할 수 있습니다. 누구나 쉽게 따라 할 수 있도록 정리했습니다. 😄

> **💡 참고:** Miniconda는 경량화된 설치 옵션을 제공하여 필요한 패키지와 환경만 설치할 수 있도록 도와줍니다.

### 1️⃣ Miniconda 설치
- [공식 웹사이트](https://docs.conda.io/en/latest/miniconda.html)에서 Miniconda를 다운로드하고 설치합니다.
- 운영 체제에 맞는 설치 방법을 따릅니다.

### 2️⃣ 환경 변수 설정
- 설치 과정에서 Miniconda를 시스템의 PATH 환경 변수에 추가합니다.
- 설치가 제대로 되었는지 확인하려면:
  ```bash
  conda --version
  ```
  명령어를 실행했을 때 버전 번호가 표시되면 성공적으로 설치된 것입니다. ✅

### 3️⃣ 가상환경 생성
- 다음 명령어를 실행해 새로운 Conda 가상환경을 만듭니다:

```bash
conda create --name history python=3.10
```
여기서는 Python 3.10 버전을 사용합니다.

### 4️⃣ 가상환경 목록 확인 🗂️
- 생성된 가상환경을 확인하려면:

```bash
conda env list
```
명령어를 실행하면 history라는 이름의 환경이 목록에 표시됩니다.

### 5️⃣ 가상환경 활성화 🚀

- 가상환경을 활성화하려면:

```bash
conda activate history
```
활성화되면 터미널 프롬프트에 (history)가 표시됩니다.

### 6️⃣ 필요한 패키지 설치 📦

프로젝트에 필요한 패키지를 설치하려면 `requirements.txt` 파일을 사용합니다:

```bash
pip install -r requirements.txt
```
이때 requirements.txt 파일이 현재 디렉토리에 있어야 합니다.

### 7️⃣ 애플리케이션 실행 🐍

프로젝트의 메인 Python 스크립트를 실행하려면:

```bash
python main.py
```

### 🔑 추가 설정

- **포트 설정:** 이 애플리케이션은 기본적으로 `8002` 포트를 사용합니다. 실행 후 웹 브라우저에서 다음 주소로 접속하세요:
```text
http://localhost:8002
```
- API 키: GPT API를 사용하기 위해 OpenAI API 키가 필요합니다. 환경 변수 또는 설정 파일에 API 키를 저장하세요.

### 📂 프로젝트 구조

- `app.py`: 애플리케이션의 주요 로직을 포함한 파일입니다.
- `main.py`: 챗봇의 초기 설정 및 실행을 담당하는 스크립트입니다.
- `requirements.txt`: 프로젝트에 필요한 패키지 목록이 포함되어 있습니다.
- `templates/`: HTML 템플릿 파일들이 저장된 디렉토리입니다.
- `static/`: CSS 및 JavaScript 파일 등 정적 파일들이 위치한 디렉토리입니다.

### 🎉 준비 완료!

이제 모든 설정이 완료되었으니 프로젝트를 실행할 준비가 되었습니다. 즐겁게 시작하세요! 💻
