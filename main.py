from flask import Flask, request, jsonify, render_template
import random

#### chatGPT RAG 활용
import os
from PyPDF2 import PdfReader
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
####

# .env 파일에서 환경 변수 로드
load_dotenv()

# API_KEY 가져오기
api_key = os.getenv('API_KEY')

#### api 키 설정 
os.environ['OPENAI_API_KEY'] = api_key

# 텍스트 파일 경로
file_path = './data/queensunduk.txt'
# 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    raw_text = file.read()

#### gpt 모델 설정
model = ChatOpenAI(model="gpt-3.5-turbo") # gpt-3.5-turbo, gpt-4

qa_chain = load_qa_chain(model, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
#### 

# 
def get_response(prompt):

    response = qa_document_chain.run(
    input_document=raw_text,
    question=prompt)

    return response

app = Flask(__name__)


# Initialize session state
session_state = {
    'messages': []
}



# 메인페이지 라우팅
@app.route('/')
def index():
    return render_template('main.html')

# 챗봇 라우팅
@app.route('/sejong')
def index2():
    return render_template('kingsejong_chatbot.html')

# 챗봇 라우팅
@app.route('/ahnjunggeun')
def index3():
    return render_template('ahnjunggeun_chatbot.html')

# 챗봇 라우팅
@app.route('/sunduk')
def index4():
    return render_template('queensunduk_chatbot.html')

@app.route('/select')
def index5():
    return render_template('selelct_character.html')

@app.route('/story')
def index6():
    return render_template('storytelling.html')

@app.route('/chat', methods=['POST'])
def chat():



    user_input = request.json.get('message')
    initial_content = '조선시대 사람들이 쓰는 말투를 썼으면 좋겠어  근엄한 말투로 먼저 인사해주고 다음 내용에 대해 이야기하듯이 대답해줘'
    characters = ["queensunduk", "kingsejong", "ahnjunggeun"]
    character = characters[0]
    
    if character == "queensunduk":
        initial_content = '''
        저는 초등학생이고 당신은 신라 시대의 선덕여왕입니다. 당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
        앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
        '''
    elif character == "kingsejong":
        initial_content = '''
        저는 초등학생이고 당신은 조선 시대의 세종대왕입니다. 당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
        앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
        '''
    elif character == "ahnjunggeun":
        initial_content = '''
        저는 초등학생이고 당신은 안중근입니다. 당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
        앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
        '''
    full_prompt = user_input + '\n' + initial_content
    session_state['messages'].append(("User", user_input))
    response = get_response(full_prompt)
    session_state['messages'].append(("GPT", response))



   


    return jsonify({'message': response})
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)