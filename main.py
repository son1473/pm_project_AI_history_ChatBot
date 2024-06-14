from flask import Flask, request, jsonify, render_template
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# API_KEY 가져오기
api_key = os.getenv('API_KEY')

# OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = api_key

# 텍스트 파일 경로 설정
file_paths = {
    "queensunduk": './data/queensunduk.txt',
    "kingsejong": './data/kingsejong.txt',
    "ahnjunggeun": './data/ahnjunggeun.txt'
}

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 텍스트 파일 읽기
raw_texts = {}
for character, file_path in file_paths.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_texts[character] = file.read()

# ChatGPT 모델 설정
model = ChatOpenAI(model="gpt-3.5-turbo")  # 또는 원하는 모델 선택
qa_chain = load_qa_chain(model, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

# 세션 상태 초기화
session_state = {
    'selected_character': None,
    'messages': []
}

# 각 인물별 초기 메시지 설정
character_messages = {
    "queensunduk": '''
        저는 초등학생이고 당신은 신라 시대의 선덕여왕입니다. 
        당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
        앞 내용에 대해 선덕여왕으로서 학생에게 부드럽게 대화해주세요.
    ''',
    "kingsejong": '''
        저는 초등학생이고 당신은 조선 시대의 세종대왕입니다. 
        당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
        앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
    ''',
    "ahnjunggeun": '''
        저는 초등학생이고 당신은 일제 강점기 시대의 안중근입니다. 
        당신은 비장하며 근엄한 군인의 말투로 대화합니다.
        앞 내용에 대해 안중근으로서 학생에게 딱딱하게 반말로 대화해주세요. 퉁명스럽게 반말로 말해주세요. 
    '''
}

# 메인페이지 라우팅
@app.route('/')
def index():
    return render_template('main.html')

# 챗봇 라우팅 - 세종대왕
@app.route('/sejong')
def sejong_chatbot():
    session_state['selected_character'] = "kingsejong"
    initial_content = character_messages["kingsejong"]
    return render_template('kingsejong_chatbot.html', initial_content=initial_content)

# 챗봇 라우팅 - 안중근
@app.route('/ahnjunggeun')
def ahn_chatbot():
    session_state['selected_character'] = "ahnjunggeun"
    initial_content = character_messages["ahnjunggeun"]
    return render_template('ahnjunggeun_chatbot.html', initial_content=initial_content)

# 챗봇 라우팅 - 선덕여왕
@app.route('/sunduk')
def sunduk_chatbot():
    session_state['selected_character'] = "queensunduk"
    initial_content = character_messages["queensunduk"]
    return render_template('queensunduk_chatbot.html', initial_content=initial_content)

# 캐릭터 선택 페이지 라우팅
@app.route('/select')
def select_character():
    return render_template('select_character.html')

# 스토리텔링 페이지 라우팅
@app.route('/story')
def storytelling():
    return render_template('storytelling.html')

# 채팅 처리 라우팅
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    selected_character = session_state.get('selected_character', "kingsejong")  # 기본값은 세종대왕

    initial_content = character_messages.get(selected_character, '''
        저는 초등학생이고 당신은 조선 시대의 세종대왕입니다. 
        당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
        앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
    ''')

    full_prompt = user_input + '\n' + initial_content
    session_state['messages'].append(("User", user_input))
    response = get_response(full_prompt)
    session_state['messages'].append(("GPT", response))

    return jsonify({'message': response})

# ChatGPT를 통해 답변을 받는 함수
def get_response(prompt):
    response = qa_document_chain.run(
        input_document=raw_texts[session_state['selected_character']],
        question=prompt
    )
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
