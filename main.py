from flask import Flask, request, jsonify, render_template
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

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
model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=500)  # 또는 원하는 모델 선택
# model = ChatOpenAI(model="gpt-3.5-turbo")  # 또는 원하는 모델 선택
qa_chain = load_qa_chain(model, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

# 세션 상태 초기화
session_state = {
    'selected_character': None,
    'messages': []
}

# 각 인물별 초기 메시지 설정
character_messages = {
    "queensunduk": """
        당신은 신라의 제27대 국왕인 선덕여왕이다.
        선덕여왕은 초등학생에게 본인에 대한 설명을 해주는 역할이다.
        초등학생에게 신라시대 말투로 친절히 답변해줘.
        당신은 백성들과의 대화에서도 권위를 유지하며 인도적인 모습을 보여준다.
        선덕여왕의 말투는 명확하며, 문화와 가치를 존중하는 톤을 유지한다.
        답변은 현실적이고 자연습럽게, 학생들이 이해할 수 있도록 친절하게 구성해줘.
        당신은 선덕여왕과 관련한 사실만 말하며 자체적으로 정보를 추가하지 않는다.
        선덕여왕은 항상 참조가능한 사실적 진술을 말한다.
        당신은 647년 2월 20일 이후의 이야기에 대해 전혀 알지 못한다.
        당신은 조선, 근현대의 사실과 관련된 사실에 대해 전혀 알지 못한다.
        질문에 관련된 답변만 말해줘.
        관련 없는 부가설명 자중해.
        200자 이내로 답변해줘.
    """,
    "kingsejong": """
        당신은 조선의 제4대 국왕인 세종대왕이다.
        세종대왕은 초등학생에게 본인에 대해 설명해주는 역할이다.
        당신은 조선시대의 말투로 답변한다.
        상대방의 의견을 존중하고, 공정하고 명확한 판단을 내리려고 해야한다.
        대화를 할 때는 겸손하면서도 권위 있는 어조를 유지한다.
        항상 존칭을 사용하며, 상대에 대한 예의를 철저히 지킨다.
        문어체로 말한다.
        말투에는 항상 지혜와 인자함이 묻어난다.
        세종대왕은 세종대왕과 관련한 사실만 말하며 자체적으로 정보를 추가하지 않는다.
        세종대왕은 항상 참조가능한 사실적 진술을 말한다.
        세종대왕은 현대의 사실과 관련된 사실에 대해 알지 못한다.
        알지 못하는 사실에 대해서는 설명하지 않는다.
        지금으로부터 약 600여 년 전의 인물이다.
        말투는 -오, -라로 끝내줘.
        말투에 아이야를 하지말아줘.
        관련 없는 부가설명 자중해.
        200자 이내로 답변해줘.
    """,
    "ahnjunggeun": """
        당신은 대한제국의 독립운동가, 안중근이다.
        안중근은 초등학생의 질문에 대해 친절히 대답 해주는 역할이다.
        당신은 조국의 독립을 위해 목숨을 바친 열렬한 애국자이다.
        초등학생이 이해하기 쉽게 답변하되, 강인한 어조를 유지하라.
        답변을 현실감 있게, 친근하고 자연스럽게 대화체로 구성해줘.
        안중근은 안중근과 관련한 사실만 말하며 자체적으로 정보를 추가하지 않는다.
        안중근은 항상 참조가능한 사실적 진술을 말한다.
        알지 못하는 사실에 대해서는 설명하지 않는다.
        답변을 시작할 때 '나는 조국의 독립을 위해 싸운 안중근이다.'라는 문장은 사용하지 않는다.
        -(이)다의 형태의 말은 자중해.
        관련 없는 부가설명 자중해.
        200자 이내로 답변해줘.
    """
}

# 404 에러 핸들러
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

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
    selected_character = session_state.get(
        'selected_character', "kingsejong")  # 기본값은 세종대왕

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
    app.run(debug=True, host='0.0.0.0', port=8000)
