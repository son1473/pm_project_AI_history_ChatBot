from flask import Flask, request, jsonify, render_template
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain # 질문 답변 체인
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage # 페르소나 설정에 필요,

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader # 문서 호출기
from langchain.text_splitter import RecursiveCharacterTextSplitter # 반복적 문자 split

from langchain_openai import OpenAIEmbeddings # 임베딩이 벡터에 넣는거
# from langchain_community.vectorstores import Chroma #vectorstorese 임베딩

loader = TextLoader('./data/kingsejong.txt', encoding='utf-8')
documents = loader.load()


# .env 파일에서 환경 변수 로드
load_dotenv()

# API_KEY 가져오기
api_key = os.getenv('API_KEY')
# print(api_key)
# OpenAI API 키 설정
os.environ['OPENAI_API_KEY'] = api_key

# Flask 애플리케이션 인스턴스 초기화 및 생성
# app = Flask(__name__)


# ChatGPT 모델 설정
model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=500)  # 또는 원하는 모델 선택
# model = ChatOpenAI(model="gpt-3.5-turbo")  # 또는 원하는 모델 선택
# model = ChatOpenAI(model="gpt-4o")  # 또는 원하는 모델 선택

messages = [
    SystemMessage(
        content="""
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
        말투는 -오로 끝내줘.
        말투에 아이야를 하지말아줘.
        300자 이내로 답변해줘.
        """
    ),
    HumanMessage(
        content="똥"
    ),
]


                        # 당신은 조국의 독립과 정의를 위해 싸우는 대의를 강조한다.
                                # 대화할 때는 당당하면서도 진지한 어조를 유지한다.
response = model.invoke(messages)


print(response)
# response = model.invoke("넌 지금부터 세종대왕이야, 세종대왕의 말투로 초등학생에게 친절히 답변해줘. 세종대왕님 지금 몇살이에요? 이 질문에 답변해줘")
# print(response)


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200)
# splited_docs = text_splitter.split_documents(documents)
# # print(len(splited_docs))
# print(splited_docs[10].page_content)

# # vectorstore = Chroma.from_documents(
# #     documents=splited_docs, embedding=OpenAIEmbeddings())
# # docs = vectorstore.similarity_search("세종대왕이 언제 즉위했는지 알려달라")
# # print(len(docs))
# # print(docs[0].page_content)

# qa_chain = load_qa_chain(model, chain_type="map_reduce")
# qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

# raw_text = """
# 세종대왕

# 기본 정보

# 아주 오랜 옛날, 조선이라는 나라가 있었어요. 그 나라에는 세종대왕이라는 훌륭한 왕이 있었어요. 세종대왕은 1397년 4월 10일에 태어났어요. 그의 진짜 이름은 '이도'였고, 세종은 그의 왕 이름이에요. 세종대왕의 아버지는 태종이라는 왕이었고, 어머니는 원경왕후 민씨였어요.

# 세종대왕은 어릴 때부터 똑똑하고 착한 아이였어요. 그래서 1408년에 충녕군이라는 높은 자리에 오르게 되었어요. 1418년에는 형인 양녕대군이 왕이 되기에 적합하지 않다고 판단되어, 세종대왕이 왕의 자리에 오르게 되었어요. 이때 나이는 겨우 22살이었어요.

# 세종대왕(이도, 자는 원정)은 조선의 태종의 셋째 아들로, 원래 태종의 뒤를 이을 왕세자는 양녕대군이었습니다. 그러나 양녕대군이 여러 사건으로 인해 세자로서의 품위를 손상시키자, 태종은 그를 폐위하고 학문에 능하고 정치에 유능한 충녕대군(세종)을 세자로 책봉했습니다. 1418년 6월 세자가 된 세종은 두 달 뒤인 8월에 왕위에 올라 조선의 네 번째 왕이 되었습니다.

# 세종대왕은 왕이 된 초반에 장인이 처형되고 처가가 풍비박산되는 비극을 겪었어요. 또한, 사랑하는 자식들과 아내가 먼저 세상을 떠나서 많은 슬픔을 겪었어요. 세종대왕 자신도 여러 가지 질병으로 고통받았어요.

# 이러한 어려움 속에서도 세종대왕은 나라를 잘 다스리고, 백성들을 위해 많은 일을 했어요. 세종대왕 덕분에 우리는 한글을 사용할 수 있고, 다양한 문화와 과학의 발전을 누릴 수 있게 되었어요. 세종대왕은 정말 위대한 왕이었어요. 그는 1450년에 54세의 나이로 세상을 떠났지만, 그의 업적은 오늘날까지도 기억되고 있어요.

# 많은 신하들의 거센 반대에도 불구하고 새로운 글자를 만든 이 사람은 누구일까요? 그는 왜 새로운 글자를 만들었을까요?

# 세종대왕은 1420년, 왕이 된 지 얼마 되지 않아 학문과 나라의 정책을 연구하는 집현전을 설치하였어요. 집현전 학사들은 젊고 유능한 사람들 가운데 선발되었어요. 세종대왕은 집현전 학사들이 일정 기간 동안 집현전에서만 근무하도록 하여 전문성을 쌓을 수 있게 했어요. 집현전을 통한 인재 양성과 더불어 세종대왕은 농업을 장려하고 다양한 천문 과학 기기를 개발하여 백성들에게 도움이 되도록 했어요.

# 세종대왕은 백성들이 쉽게 읽고 쓸 수 있도록 한글을 만들었어요. 그 결과 우리는 오늘날 한글을 사용할 수 있게 되었어요. 세종대왕의 업적은 한글뿐만 아니라 다양한 문화와 과학의 발전에도 큰 영향을 미쳤어요. 세종대왕은 정말로 위대한 왕이었답니다.



# """

# res = qa_document_chain.invoke(
    
#     input=raw_text,
#     question="세종대왕은 언제 즉위했나요?")


# print(res)

# # 세션 상태 초기화
# session_state = {
#     'selected_character': None,
#     'messages': []
# }

# # 각 인물별 초기 메시지 설정
# character_messages = {
#     "queensunduk": '''
#         저는 초등학생이고 당신은 신라 시대의 선덕여왕입니다. 
#         당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
#         앞 내용에 대해 선덕여왕으로서 학생에게 부드럽게 대화해주세요.
#     ''',
#     "kingsejong": '''
#         저는 초등학생이고 당신은 조선 시대의 세종대왕입니다. 
#         당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
#         앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
#     ''',
#     "ahnjunggeun": '''
#         저는 초등학생이고 당신은 일제 강점기 시대의 안중근입니다. 
#         당신은 비장하며 근엄한 군인의 말투로 대화합니다.
#         앞 내용에 대해 안중근으로서 학생에게 딱딱하게 반말로 대화해주세요. 퉁명스럽게 반말로 말해주세요. 
#     '''
# }

# # 404 에러 핸들러


# @app.errorhandler(404)
# def page_not_found(e):
#     return render_template('404.html'), 404

# # 메인페이지 라우팅


# @app.route('/')
# def index():
#     return render_template('main.html')

# # 챗봇 라우팅 - 세종대왕


# @app.route('/sejong')
# def sejong_chatbot():
#     session_state['selected_character'] = "kingsejong"
#     initial_content = character_messages["kingsejong"]
#     return render_template('kingsejong_chatbot.html', initial_content=initial_content)

# # 챗봇 라우팅 - 안중근


# @app.route('/ahnjunggeun')
# def ahn_chatbot():
#     session_state['selected_character'] = "ahnjunggeun"
#     initial_content = character_messages["ahnjunggeun"]
#     return render_template('ahnjunggeun_chatbot.html', initial_content=initial_content)

# # 챗봇 라우팅 - 선덕여왕


# @app.route('/sunduk')
# def sunduk_chatbot():
#     session_state['selected_character'] = "queensunduk"
#     initial_content = character_messages["queensunduk"]
#     return render_template('queensunduk_chatbot.html', initial_content=initial_content)

# # 캐릭터 선택 페이지 라우팅


# @app.route('/select')
# def select_character():
#     return render_template('select_character.html')

# # 스토리텔링 페이지 라우팅


# @app.route('/story')
# def storytelling():
#     return render_template('storytelling.html')

# # 채팅 처리 라우팅


# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('message')
#     selected_character = session_state.get(
#         'selected_character', "kingsejong")  # 기본값은 세종대왕

#     initial_content = character_messages.get(selected_character, '''
#         저는 초등학생이고 당신은 조선 시대의 세종대왕입니다. 
#         당신은 지혜롭고 친절하며 근엄한 조선시대 말투로 대화합니다.
#         앞 내용에 대해 세종대왕으로서 학생에게 친절히 대화해주세요.
#     ''')

#     full_prompt = user_input + '\n' + initial_content
#     session_state['messages'].append(("User", user_input))
#     response = get_response(full_prompt)
#     session_state['messages'].append(("GPT", response))

#     return jsonify({'message': response})

# # ChatGPT를 통해 답변을 받는 함수


# def get_response(prompt):
#     response = qa_document_chain.run(
#         input_document=raw_texts[session_state['selected_character']],
#         question=prompt
#     )
#     return response


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8000)
