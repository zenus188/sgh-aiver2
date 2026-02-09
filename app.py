# app.py
import json
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI


# -----------------------------
# Helpers
# -----------------------------
def build_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def safe_json_loads(s: str) -> Dict[str, Any]:
    """
    Responses API의 output_text는 보통 JSON 텍스트로 오지만,
    혹시 모를 공백/코드펜스 등을 대비해 최대한 안전하게 파싱.
    """
    s = s.strip()
    # 코드펜스 제거(방어)
    if s.startswith("```"):
        s = s.strip("`")
        # "json\n{...}" 형태 방어
        if "\n" in s:
            s = s.split("\n", 1)[1].strip()
    return json.loads(s)


def join_nonempty(items: List[str]) -> str:
    items = [x.strip() for x in items if x and x.strip()]
    return ", ".join(items)


def build_profile_text(
    preferred_genres: List[str],
    disliked_genres: List[str],
    emotions: List[str],
    played_games: str,
    platforms: List[str],
    hours_per_day: float,
) -> str:
    return f"""
[사용자 선호 프로필]
- 선호 장르: {join_nonempty(preferred_genres) if preferred_genres else "없음/미선택"}
- 비선호 장르: {join_nonempty(disliked_genres) if disliked_genres else "없음/미선택"}
- 원하는 감정(플레이 경험): {join_nonempty(emotions) if emotions else "없음/미선택"}
- 재미있게 플레이한 게임(참고): {played_games.strip() if played_games.strip() else "미입력"}
- 선호 플랫폼/기기: {join_nonempty(platforms) if platforms else "없음/미선택"}
- 하루 예상 플레이시간: {hours_per_day}시간
""".strip()


def recommendations_schema() -> Dict[str, Any]:
    # Structured Outputs (json_schema) 스키마
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "playmate_game_recommendations",
            "description": "User preferences-based game recommendations with brief platform/price info.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": "string"},
                                "genre": {"type": "string"},
                                "platforms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "price_range_krw": {
                                    "type": "string",
                                    "description": "Approximate KRW price range (varies by store/region/sale).",
                                },
                                "store_hint": {
                                    "type": "string",
                                    "description": "Where to check price/platform (e.g., Steam/PS Store/eShop/Google Play).",
                                },
                                "why_recommended": {"type": "string"},
                                "fit_emotions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "time_fit": {
                                    "type": "string",
                                    "description": "How it fits the user's daily playtime.",
                                },
                                "caution_or_note": {
                                    "type": "string",
                                    "description": "Any caution: difficulty, motion sickness, horror intensity, etc.",
                                },
                            },
                            "required": [
                                "title",
                                "genre",
                                "platforms",
                                "price_range_krw",
                                "store_hint",
                                "why_recommended",
                                "fit_emotions",
                                "time_fit",
                                "caution_or_note",
                            ],
                        },
                    },
                    "summary": {
                        "type": "string",
                        "description": "One short paragraph summarizing the overall recommendation logic.",
                    },
                    "price_disclaimer": {
                        "type": "string",
                        "description": "A clear disclaimer that prices vary by store/region/sales and should be verified.",
                    },
                },
                "required": ["recommendations", "summary", "price_disclaimer"],
            },
        },
    }


def call_openai_chat(
    client: OpenAI,
    model: str,
    system_instructions: str,
    messages: List[Dict[str, str]],
) -> str:
    # messages를 단일 input으로 합쳐서 전달(단순/견고)
    convo = []
    for m in messages[-20:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        convo.append(f"{role.upper()}: {content}")
    input_text = "\n".join(convo)

    resp = client.responses.create(
        model=model,
        instructions=system_instructions,
        input=input_text,
    )
    return resp.output_text


def call_openai_recommendations(
    client: OpenAI,
    model: str,
    system_instructions: str,
    profile_text: str,
) -> Dict[str, Any]:
    prompt = f"""
너는 게임 추천 전문가다.
아래 [사용자 선호 프로필]을 기반으로, 사용자가 좋아할 가능성이 높은 게임 5개를 추천하라.

- 반드시 5개만.
- 사용자의 '비선호 장르'는 최대한 피하라.
- 사용자의 '플랫폼/기기'에서 플레이 가능한 타이틀을 우선하라.
- '가격'은 정확한 실시간 조회가 아니라 "대략적인 가격대(원)"로 제시하고, 어떤 스토어에서 확인하면 되는지(store_hint)를 적어라.
- 출력은 지정된 JSON 스키마를 엄격히 따른다.

{profile_text}
""".strip()

    resp = client.responses.create(
        model=model,
        instructions=system_instructions,
        input=prompt,
        response_format=recommendations_schema(),
    )
    return safe_json_loads(resp.output_text)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="플레이메이트", layout="wide")

# Sidebar (API key must be at top-left => put it first)
with st.sidebar:
    st.markdown("### 🔑 API 키")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-... 또는 프로젝트 키",
        help="키는 로컬에서만 사용되도록 구성하세요. (배포 시 st.secrets 권장)",
    )
    st.divider()

    st.markdown("### 🎮 취향 설정")

    GENRES = ["액션 게임", "슈팅 게임", "어드벤쳐 게임", "전략 게임", "롤플레잉 게임", "퍼즐 게임", "음악게임"]
    EMOTIONS = ["힐링", "성장", "경쟁", "공포", "수집", "몰입 스토리"]
    PLATFORMS = ["PC", "PS", "Xbox", "Switch", "모바일"]

    preferred_genres = st.multiselect("선호 장르", GENRES, default=[])
    disliked_genres = st.multiselect("비선호 장르", GENRES, default=[])
    emotions = st.multiselect("게임에서 원하는 감정", EMOTIONS, default=[])

    played_games = st.text_area(
        "재미있게 플레이한 게임 (자유 입력)",
        placeholder="예: 젤다 야숨, 엘든 링, 하데스 ...",
        height=90,
    )

    platforms = st.multiselect("플랫폼/기기", PLATFORMS, default=[])

    hours_per_day = st.number_input(
        "하루 예상 플레이시간 (시간)",
        min_value=0.0,
        max_value=24.0,
        value=1.5,
        step=0.5,
    )

    st.divider()

    model = st.selectbox(
        "모델",
        options=["gpt-5.2", "gpt-5", "gpt-4.1"],
        index=0,
        help="가용 모델은 계정/프로젝트 설정에 따라 다를 수 있어요.",
    )

    get_recs = st.button("✨ 추천 받기", use_container_width=True)


st.title("플레이메이트")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 저는 플레이메이트 🎮\n사이드바에서 취향을 고르고, 채팅으로 원하는 게임 느낌을 말해줘요. (예: '협동으로 30분씩 하기 좋은 거')",
        }
    ]
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

profile_text = build_profile_text(
    preferred_genres=preferred_genres,
    disliked_genres=disliked_genres,
    emotions=emotions,
    played_games=played_games,
    platforms=platforms,
    hours_per_day=float(hours_per_day),
)

system_instructions = f"""
너는 '플레이메이트'라는 이름의 게임 추천 챗봇이다.
- 한국어로 답한다.
- 사용자의 선호/비선호 장르, 원하는 감정, 플레이한 게임, 플랫폼, 하루 플레이시간을 최우선 반영한다.
- 사실을 지어내지 않는다. (특히 가격/플랫폼의 정확한 실시간 정보는 단정하지 말 것)
- 사용자가 원하는 경우에만 길게 설명하고, 기본은 짧고 명확하게.
- 추천을 할 때는 사용자가 왜 좋아할지 2~3줄로 핵심만 말한다.

{profile_text}
""".strip()

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle "추천 받기"
if get_recs:
    if not api_key:
        st.error("사이드바 왼쪽 위에 OpenAI API 키를 먼저 입력해줘.")
    else:
        try:
            client = build_client(api_key)
            with st.spinner("취향 분석 중..."):
                recs_obj = call_openai_recommendations(
                    client=client,
                    model=model,
                    system_instructions=system_instructions,
                    profile_text=profile_text,
                )
            st.session_state.recommendations = recs_obj
        except Exception as e:
            st.session_state.recommendations = None
            st.error(f"추천 생성 실패: {e}")

# Show recommendations (if any)
recs_obj = st.session_state.recommendations
if recs_obj:
    st.subheader("추천 게임 5선")
    st.caption(recs_obj.get("price_disclaimer", ""))

    cols = st.columns(2)
    recs = recs_obj.get("recommendations", [])[:5]
    for i, r in enumerate(recs):
        col = cols[i % 2]
        with col:
            st.markdown(f"### {i+1}. {r['title']}")
            st.markdown(f"- **장르:** {r['genre']}")
            st.markdown(f"- **플랫폼:** {', '.join(r['platforms'])}")
            st.markdown(f"- **가격대(원):** {r['price_range_krw']}")
            st.markdown(f"- **가격/구매 확인:** {r['store_hint']}")
            st.markdown(f"- **추천 이유:** {r['why_recommended']}")
            st.markdown(f"- **맞는 감정:** {', '.join(r['fit_emotions'])}")
            st.markdown(f"- **시간 적합:** {r['time_fit']}")
            st.markdown(f"- **주의/메모:** {r['caution_or_note']}")
            st.divider()

    st.info(recs_obj.get("summary", ""))

    # Let user quickly ask follow-up about a specific game
    st.markdown("원하면 채팅에 이렇게 물어봐도 돼요: `2번 게임 비슷한 거 더 추천해줘`, `공포 강도 어느 정도야?`")

# Chat input
user_text = st.chat_input("원하는 게임 느낌을 말해줘 (예: '힐링 + 수집, 스위치로 1시간씩')")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    if not api_key:
        assistant_text = "API 키가 아직 없어요. 사이드바 왼쪽 위에 OpenAI API 키를 먼저 입력해줘."
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
    else:
        try:
            client = build_client(api_key)
            with st.spinner("답변 생성 중..."):
                assistant_text = call_openai_chat(
                    client=client,
                    model=model,
                    system_instructions=system_instructions,
                    messages=st.session_state.messages,
                )
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
        except Exception as e:
            err = f"오류가 났어: {e}"
            st.session_state.messages.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.markdown(err)
