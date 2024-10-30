import re
import boto3
import streamlit as st
import pandas as pd
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import plotly.graph_objects as go
import base64
import json
import os
import random

# Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Image generation settings
def generate_image(prompt):
    seed = random.randint(0, 2147483647)
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 7.5,
            "height": 512,
            "width": 512,
            "seed": seed,
        },
    })
    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-image-generator-v1",
        )
        base64_image_data = json.loads(response["body"].read())["images"][0]
        return base64.b64decode(base64_image_data)
    except Exception as e:
        print(f"Error generating image: {e}")
        raise

def save_image(base64_image_data, prompt):
    """Save base64 image data to a file."""
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)
    # 'prompt'를 파일명으로 사용할 때, 안전한 형식으로 변환
    safe_prompt = re.sub(r'[<>:"/\\|?*]', '', prompt)  # 파일명에 사용할 수 없는 문자 제거
    file_path = os.path.join(output_folder, f"{safe_prompt}.png")
    with open(file_path, "wb") as file:
        file.write(base64_image_data)
    return file_path

# Claude 3.5 파라미터 설정
model_kwargs = {
    "max_tokens": 1000,
    "temperature": 0.01,
    "top_p": 0.01,
}

# Bedrock LLM 설정
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs=model_kwargs,
    streaming=True
)

st.set_page_config(
    page_title="Titleturtle",
    page_icon=':memo:',
    initial_sidebar_state="expanded",
    layout="wide",
)

# Streamlit 채팅 메시지 히스토리 설정
message_history = StreamlitChatMessageHistory(key="chat_messages")

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="message_history"),
        ("human", "{query}"),
    ]
)

# 대화 체인 설정
chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: message_history,  # 항상 이전 대화를 리턴
    input_messages_key="query",
    history_messages_key="message_history",
)

# CSV 파일 로드 (인코딩 형식은 필요에 따라 조정)
try:
    df = pd.read_csv('university_major.csv', encoding='utf-8')  # utf-8 또는 'euc-kr' 사용
except Exception as e:
    st.write(f"Error loading CSV file: {e}")
    df = pd.DataFrame()  # 빈 데이터프레임 생성

# 사이드바 설정
with st.sidebar:
    # 학교 선택
    school = st.selectbox('학교 선택:', ('상명대학교서울', '상명대학교천안'))

    # 학과 리스트 초기화
    majors = []

    # 첫 번째 selectbox의 결과에 따라 두 번째 selectbox의 옵션 결정
    if school == '상명대학교서울':
        df_seoul = df[df['시도명'] == '서울특별시']
        df_seoul = df_seoul[df_seoul['시군구명'] == '종로구']
        majors = df_seoul['학과명'].unique()
    elif school == '상명대학교천안':
        df_cheonan = df[df['시도명'] == '충청남도']
        df_cheonan = df_cheonan[df_cheonan['시군구명'] == '천안시']
        majors = df_cheonan['학과명'].unique()

    # 학과 선택
    major = st.selectbox('학과 선택:', sorted(majors))

    # Form to input details
    with st.form(key='sidebar_form'):
        st.write('')
        # 과제 문제 입력
        assignment_view = st.text_area('과제 관련 내용 입력:', height=200)

        # 과제 문제 입력
        assignment_q = st.text_area('과제 문제 입력:', height=200)

        # 과제 내용 입력
        assignment_content = st.text_area('과제 내용 입력:', height=200)

        # 과제 종류 선택
        catecate = st.selectbox('과제 종류 선택:', ('보고서', '문제풀이', '코드작성'))

        st.write('')

        # 관대함과 창의성 중심 슬라이더
        lenient_weight = st.slider('관대함', 0, 100, 50, 1)
        creativity_weight = st.slider('창의성 중심', 0, 100, 50, 1)

        st.write('')

        # 체크박스들
        checkbox_options = {
            "표절률 검사 가능여부": False
        }

        checkbox_selections = {}
        for label in checkbox_options.keys():
            checkbox_selections[label] = st.checkbox(label)

        st.write('')  # 추가적인 여백

        # 제출 버튼
        submit_button = st.form_submit_button(label='적용')

        # 제출 버튼이 눌렸을 때만 세션 스테이트에 값을 저장
        if submit_button:
            st.session_state['school'] = school
            st.session_state['major'] = major
            st.session_state['catecate'] = catecate
            st.session_state['lenient_weight'] = lenient_weight
            st.session_state['creativity_weight'] = creativity_weight
            st.session_state['checkbox_selections'] = checkbox_selections
            st.session_state['assignment_content'] = assignment_content
            st.session_state['assignment_q'] = assignment_q
            st.session_state['assignment_view'] = assignment_view

            # 선택된 학과의 주요 교과목명 저장
            if major and school in ['상명대학교서울', '상명대학교천안']:
                if school == '상명대학교서울':
                    df_filtered = df_seoul[df_seoul['학과명'] == major]
                else:
                    df_filtered = df_cheonan[df_cheonan['학과명'] == major]

                # 주요 교과목명 추출 및 분리
                if not df_filtered.empty:
                    major_subjects = df_filtered['주요교과목명'].values
                    if len(major_subjects) > 0 and not pd.isna(major_subjects[0]):
                        subject_list = major_subjects[0].split('+')
                        st.session_state['major_subjects'] = subject_list
                    else:
                        st.session_state['major_subjects'] = ["해당 학과에 대한 주요 교과목 정보가 없습니다."]
                else:
                    st.session_state['major_subjects'] = ["해당 학과에 대한 정보가 없습니다."]
            else:
                st.session_state['major_subjects'] = ["해당 학과에 대한 정보가 없습니다."]

            # Generate and save image
            image_prompt = (
                f"a professor"
                f"a professor whose major is a {major}"
                f"A cartoon professor whose anger level is {100 - lenient_weight}, visually depicted with matching intensity."
                f"a professor as cartoon"
            )

            # 이미지 생성 시도 및 에러 처리
            try:
                image_data = generate_image(image_prompt)
                image_path = save_image(image_data, image_prompt)
                st.session_state['image_path'] = image_path
            except Exception as e:
                st.write(f"Error generating or saving image: {e}")

            # 평가 요소 제공을 위한 프롬프트 생성
            evaluation_prompt = (
                f"사용자는 {st.session_state.get('school', '학교')} {st.session_state.get('major', '학과')}에서 다음 과목들을 학습한 학생입니다. "
                f"참고 자료: {st.session_state.get('assignment_view', '없음')}. "
                f"피드백 받을 자료의 종류: {st.session_state.get('catecate', '없음')}. "
                f"피드백 요청 자료: '{st.session_state.get('assignment_q', '없음')}'. "
                "이 자료를 평가하기 위해 고려해야 할 요소를 6가지 제시해 주세요. 딱 요소만 표시해줘, 다음과 같습니다 그런거 빼고 무조건 1. 뭐뭐 2. 뭐뭐 이렇게 6개만"
            )

            # 평가 요소 제공
            response_stream = chain_with_history.stream