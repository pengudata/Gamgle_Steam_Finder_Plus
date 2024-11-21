import streamlit as st
import pickle
import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from google.cloud import bigquery
import ast
import os
from openai import OpenAI

# 페이지 기본 설정
st.set_page_config(page_title="스팀 게임 추천 시스템", layout="wide")

# 웹페이지 스타일 정의
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .main-title {
        font-size: 2.5em;
        color: #2c3e50;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.2em;
        color: #5a5a5a;
        margin-bottom: 20px;
    }
    
    /* 태그 스타일 수정 */
    .tag {
        background: #f0f2f5;
        padding: 4px 8px;
        border-radius: 12px;
        margin: 0 4px;
        display: inline-block;
        color: #555;
        font-size: 0.9em;
    }
    
    /* 제목 컨테이너 스타일 */
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 28px;  /* 게임 제목의 크기 */
        font-weight: bold;
        color: #2c3e50;
    }
    
    /* 점수 뱃지 스타일링 */
    .score-badge {
        background: linear-gradient(45deg, #4a90e2, #67b26f);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin-left: 10px;
        font-size: 14px;  /* 추천 점수의 크기를 직접 픽셀로 지정 */
        font-weight: 500;
    }
    
    /* 이미지 링크 스타일링 */
    a img {
        transition: transform 0.3s ease;
        width: 100%;
        max-width: 500px;
    }
    
    a img:hover {
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_game_mappings():
    """
    BigQuery에서 게임 데이터를 가져와서 필요한 매핑 정보를 생성합니다.
    반환값:
    - 게임 정보가 담긴 데이터프레임
    - 게임 이름에서 ID로의 매핑 딕셔너리
    - 게임 ID에서 이름으로의 매핑 딕셔너리
    """
    try:
        credentials_dict = st.secrets["big_query_account"]
        client = bigquery.Client.from_service_account_info(credentials_dict)
        query = """
            SELECT 
                appid,
                name as game_name,
                popularity_score,
                positive_reviews,
                negative_reviews,
                Genre,
                Theme,
                Mechanism,
                Feature,
                description_kor,
                image_url
            FROM `gamgle.steam.newbie_game10`
            WHERE popularity_score IS NOT NULL
        """
        df = client.query(query).to_dataframe()
        
        # 리스트 형태의 문자열을 실제 리스트로 변환
        for col in ['Genre', 'Theme', 'Mechanism', 'Feature']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '' else [])
        
        # 게임 이름과 ID 간의 매핑 생성
        name_to_id = dict(zip(df['game_name'], df['appid']))
        id_to_name = dict(zip(df['appid'], df['game_name']))
        
        return df, name_to_id, id_to_name
    except Exception as e:
        st.error(f"게임 데이터 로딩 중 오류 발생: {str(e)}")
        return pd.DataFrame(), {}, {}

@st.cache_resource
def load_model_and_mappings():
    """저장된 추천 모델과 매핑 정보를 불러옵니다."""
    try:
        with open('./model/final.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./model/final_item_map.pkl', 'rb') as f:
            item_map = pickle.load(f)
        return model, item_map
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {str(e)}")
        return None, None

def get_recommendations_for_new_user(model, selected_appids, item_map, df, n_items=20):
    """새로운 사용자를 위한 게임 추천을 생성합니다."""
    try:
        n_factors = model.item_factors.shape[1]
        
        game_factors = []
        for appid in selected_appids:
            try:
                model_idx = [idx for idx, id_ in item_map.items() if id_ == appid][0]
                game_factors.append(model.item_factors[model_idx])
            except IndexError:
                continue

        if not game_factors:
            return [], []

        user_profile = np.mean(game_factors, axis=0)
        similarities = model.item_factors.dot(user_profile)

        for appid in selected_appids:
            try:
                model_idx = [idx for idx, id_ in item_map.items() if id_ == appid][0]
                similarities[model_idx] = -float('inf')
            except IndexError:
                continue

        top_indices = np.argsort(similarities)[-n_items:][::-1]
        scores = similarities[top_indices]
        recommended_appids = [item_map[idx] for idx in top_indices]

        return recommended_appids, scores
    except Exception as e:
        st.error(f"추천 생성 중 오류: {str(e)}")
        return [], []

# OpenAI API 설정
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

def generate_recommendation_reason(game_name):
    prompt = (
        "당신은 10년 경력의 열정적인 게임 판매원입니다. "
        "게임에 대한 해박한 지식을 바탕으로 고객들에게 친근하고 전문적으로 게임을 추천해주세요. "
        "존댓말을 사용하여 설명해주시되, 게임의 이름은 직접적으로 언급하지 말아주세요.\n\n"
        f"'{game_name}' 게임에 대해 다음 구조로 설명해주세요:\n\n"
        "[게임의 핵심을 한 문장으로]\n"
        "[주요 게임플레이 특징]\n"
        "[이 게임을 추천하는 이유]\n\n"
        "각 항목을 명확히 구분하여 작성해주시고, 전체 설명은 200자 내외로 작성해주세요."
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 게임에 대한 전문 지식이 풍부하고 열정적인 게임 판매원입니다. 항상 친절하고 전문적인 어조로 고객과 대화합니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# 웹페이지 UI 구성
st.markdown("<div class='main-title'>🎮 스팀 게임 추천 시스템</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>좋아하는 게임을 선택하면 새로운 게임을 추천해드립니다!</div>", unsafe_allow_html=True)

# 데이터와 모델 로드
all_games_df, name_to_id, id_to_name = load_game_mappings()
model, item_map = load_model_and_mappings()

if not name_to_id or model is None:
    st.error("필요한 데이터를 로드할 수 없습니다. 설정을 확인해주세요.")
else:
    # 사이드바 게임 선택 UI
    with st.sidebar:
        st.subheader("🕹️ 게임 선택")
        selected_games = st.multiselect(
            "즐겨했던 게임을 선택해주세요:",
            options=sorted(name_to_id.keys()),
            help="최소 1개 이상의 게임을 선택해주세요."
        )
        st.markdown("선택한 게임을 바탕으로 새로운 게임을 추천해드립니다.")
    
    # 추천 시작 버튼 및 결과 표시
    if st.sidebar.button("🎯 게임 추천받기"):
        if not selected_games:
            st.warning("게임을 하나 이상 선택해주세요!")
        else:
            with st.spinner("추천 게임을 찾고 있습니다... 잠시만 기다려주세요!"):
                selected_ids = [name_to_id[game] for game in selected_games]
                recommended_ids, scores = get_recommendations_for_new_user(
                    model,
                    selected_ids,
                    item_map,
                    all_games_df
                )
            
            if recommended_ids:
                st.subheader("🎉 추천 게임 목록")
                recommended_games_info = all_games_df[all_games_df['appid'].isin(recommended_ids)]

                # 추천된 각 게임에 대한 정보 표시
                for i, (game_id, score) in enumerate(zip(recommended_ids, scores), 1):
                    game_info = recommended_games_info[recommended_games_info['appid'] == game_id]
                    if not game_info.empty:
                        image_url = game_info['image_url'].iloc[0]
                        game_name = game_info['game_name'].iloc[0]
                        description = game_info['description_kor'].iloc[0]
                        genre = ', '.join(game_info['Genre'].iloc[0])
                        theme = ', '.join(game_info['Theme'].iloc[0])
                        mechanism = ', '.join(game_info['Mechanism'].iloc[0])
                        feature = ', '.join(game_info['Feature'].iloc[0])
                        recommendation = generate_recommendation_reason(game_name)

                        # 2열 레이아웃으로 게임 정보 표시
                        col1, col2 = st.columns([1.5, 3])
                        with col1:
                            # 이미지를 링크로 감싸기
                            st.markdown(f"""
                                <a href="https://store.steampowered.com/app/{game_id}" target="_blank">
                                    <img src="{image_url}" width="500" style="cursor: pointer; border-radius: 10px;">
                                </a>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                                <div class='title-container'>
                                <span class='game-title'>{i}. {game_name}</span>
                                <span class='score-badge'>추천 점수: {score:.4f}</span>
                                </div>
                            """, unsafe_allow_html=True)
                            # 추천 이유를 여러 줄로 나누어 표시
                            st.markdown("**추천 이유:**")
                            paragraphs = recommendation.split('\n')  # 줄바꿈을 기준으로 분리
                            for p in paragraphs:
                                if p.strip():  # 빈 줄 제외
                                    st.markdown(f"{p}", unsafe_allow_html=True)
                            
                            # 항목 레이아웃 수정
                            col2_left, col2_right = st.columns([1, 4])  # 1:4 비율로 열 분할
                            
                            # 장르
                            col2_left.write("**장르:**")
                            col2_right.markdown(' '.join([f"<span class='tag'>{g}</span>" for g in game_info['Genre'].iloc[0]]), unsafe_allow_html=True)
                            
                            # 테마
                            col2_left.write("**테마:**")
                            col2_right.markdown(' '.join([f"<span class='tag'>{t}</span>" for t in game_info['Theme'].iloc[0]]), unsafe_allow_html=True)
                            
                            # 게임 방식
                            col2_left.write("**게임 방식:**")
                            col2_right.markdown(' '.join([f"<span class='tag'>{m}</span>" for m in game_info['Mechanism'].iloc[0]]), unsafe_allow_html=True)
                            
                            # 주요 특징
                            col2_left.write("**주요 특징:**")
                            col2_right.markdown(' '.join([f"<span class='tag'>{f}</span>" for f in game_info['Feature'].iloc[0]]), unsafe_allow_html=True)
                        st.write("---")  # 구분선 추가
            else:
                st.warning("추천 게임을 생성할 수 없습니다. 다른 게임을 선택해보세요.")

