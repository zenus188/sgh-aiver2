# streamlit_app.py
import os
import json
import random
import calendar
from datetime import date, timedelta

import requests
import pandas as pd
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("🗓️ AI 습관 캘린더")
st.caption("캘린더처럼 한 달을 훑어보고, 오늘의 체크인과 리포트를 한 번에!")

# ----------------------------
# Sidebar: API Keys
# ----------------------------
with st.sidebar:
    st.header("🔑 API 키 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", ""))
    st.divider()
    st.caption("키는 로컬/세션에만 사용되도록 구성하세요. (배포 시 Secrets 권장)")

# ----------------------------
# Helpers: APIs
# ----------------------------
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap 현재 날씨
    - 한국어(lang=kr)
    - 섭씨(units=metric)
    - 실패 시 None
    """
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "lang": "kr",
        "units": "metric",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "city": city,
            "temp_c": float(data["main"]["temp"]),
            "feels_like_c": float(data["main"]["feels_like"]),
            "humidity": int(data["main"]["humidity"]),
            "desc": str(data["weather"][0]["description"]),
            "icon": str(data["weather"][0].get("icon", "")),
        }
    except Exception:
        return None


def _parse_dog_breed_from_url(img_url: str) -> str:
    """
    Dog CEO 이미지 URL에서 품종 추출 시도.
    예: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
        -> hound (afghan)
    """
    try:
        # .../breeds/{breed}/...
        parts = img_url.split("/breeds/", 1)[1].split("/", 1)[0]
        # parts: "hound-afghan" or "retriever-golden" or "akita"
        if "-" in parts:
            base, sub = parts.split("-", 1)
            return f"{base} ({sub})"
        return parts
    except Exception:
        return "unknown"


def get_dog_image():
    """
    Dog CEO 랜덤 강아지 이미지 URL + 품종
    실패 시 None
    """
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
        if not img_url:
            return None
        breed = _parse_dog_breed_from_url(img_url)
        return {"image_url": img_url, "breed": breed}
    except Exception:
        return None


def _style_system_prompt(style: str) -> str:
    if style == "스파르타 코치":
        return (
            "너는 엄격하고 직설적인 습관 코치다. 변명은 허용하지 않는다. "
            "하지만 모욕은 금지한다. 짧고 강하게, 실행 가능한 지시만 내린다."
        )
    if style == "따뜻한 멘토":
        return (
            "너는 따뜻하고 현실적인 멘토다. 판단하지 않고, 사용자가 내일 바로 실천할 수 있는 "
            "작은 행동을 제안한다. 과한 감정 과잉은 금지, 담백하게 격려한다."
        )
    # 게임 마스터
    return (
        "너는 RPG 게임 마스터다. 사용자의 습관을 퀘스트/스탯/보상으로 비유해 재미있게 동기부여한다. "
        "유치하게 늘어지지 말고, 짧고 임팩트 있게 구성한다."
    )


def generate_report(
    openai_key: str,
    coach_style: str,
    habits: dict,
    mood: int,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI 호출
    - 모델: gpt-5-mini
    - 실패 시 None
    """
    if not openai_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return "OpenAI SDK가 설치되어 있지 않습니다. `pip install openai` 후 다시 실행하세요."

    client = OpenAI(api_key=openai_key)

    payload = {
        "date": str(date.today()),
        "mood": mood,
        "habits": habits,
        "weather": weather,
        "dog_breed": None if dog is None else dog.get("breed"),
    }

    system = _style_system_prompt(coach_style)
    user = (
        "아래 JSON을 바탕으로 'AI 습관 코치 리포트'를 작성해.\n"
        "반드시 다음 출력 형식을 지켜:\n\n"
        "1) 컨디션 등급: (S/A/B/C/D 중 하나)\n"
        "2) 습관 분석: (체크된 것/빠진 것, 핵심 3줄)\n"
        "3) 날씨 코멘트: (날씨가 없으면 '날씨 데이터 없음' 한 줄)\n"
        "4) 내일 미션: (딱 3개, 체크박스 형태로)\n"
        "5) 오늘의 한마디: (한 문장)\n\n"
        "문장은 한국어로. 군더더기 없이.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ----------------------------
# Session state: history (6일 샘플 + 오늘)
# ----------------------------
HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏋️", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Sejong",
    "Jeju",
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]


def _seed_demo_history():
    # 최근 6일 샘플 데이터(데모)
    rng = random.Random(20260209)  # 고정 시드(재현 가능)
    today = date.today()
    rows = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked = rng.randint(1, 5)
        mood = rng.randint(4, 9)
        rows.append(
            {
                "date": d,
                "checked": checked,
                "mood": mood,
            }
        )
    return rows


if "history" not in st.session_state:
    st.session_state.history = _seed_demo_history()

if "last_saved_date" not in st.session_state:
    st.session_state.last_saved_date = None

if "reports" not in st.session_state:
    st.session_state.reports = {}

# ----------------------------
# Check-in UI
# ----------------------------
st.subheader("✅ 오늘의 체크인")

left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("**습관 체크**")
    c1, c2 = st.columns(2)

    habit_state = {}
    for idx, (key, emoji, label) in enumerate(HABITS):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            habit_state[key] = st.checkbox(f"{emoji} {label}", value=False, key=f"habit_{key}")

    st.markdown("---")
    mood = st.slider("🙂 오늘 기분 점수", min_value=1, max_value=10, value=7)

with right:
    st.markdown("**환경 설정**")
    city = st.selectbox("📍 도시 선택", CITIES, index=0)
    coach_style = st.radio("🎭 코치 스타일", COACH_STYLES, horizontal=False)

# ----------------------------
# Metrics + Save today record
# ----------------------------
checked_count = sum(1 for k, _, _ in HABITS if habit_state.get(k))
achievement = round((checked_count / len(HABITS)) * 100)

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

# 오늘 데이터 기록 저장(세션 기준) - 같은 날짜면 업데이트
today = date.today()
today_row = {"date": today, "checked": checked_count, "mood": mood}

# history에 오늘이 이미 있으면 교체, 없으면 추가
hist = st.session_state.history
if len(hist) == 0 or hist[-1]["date"] != today:
    # 마지막이 오늘이 아니면 추가
    hist.append(today_row)
else:
    # 오늘이면 업데이트
    hist[-1] = today_row
st.session_state.history = hist

# ----------------------------
# 7일 바 차트
# ----------------------------
st.subheader("🗓️ 월간 캘린더")

df = pd.DataFrame(st.session_state.history).copy()
if not df.empty:
    df["date"] = pd.to_datetime(df["date"]).dt.date

today = date.today()
month_start = today.replace(day=1)
month_last_day = calendar.monthrange(today.year, today.month)[1]
month_days = [month_start + timedelta(days=i) for i in range(month_last_day)]
month_df = pd.DataFrame({"date": month_days})
month_df = month_df.merge(df, on="date", how="left")
month_df["checked"] = month_df["checked"].fillna(0).astype(int)
month_df["mood"] = month_df["mood"].fillna(0).astype(int)
month_df["report"] = month_df["date"].map(st.session_state.reports).fillna("")

calendar_rows = calendar.Calendar(firstweekday=6).monthdatescalendar(today.year, today.month)
weekday_labels = ["일", "월", "화", "수", "목", "금", "토"]

header_cols = st.columns(7)
for idx, label in enumerate(weekday_labels):
    header_cols[idx].markdown(f"**{label}**")

for week in calendar_rows:
    week_cols = st.columns(7)
    for idx, day in enumerate(week):
        day_data = month_df.loc[month_df["date"] == day]
        in_month = day.month == today.month
        checked = int(day_data["checked"].iloc[0]) if not day_data.empty else 0
        mood_value = int(day_data["mood"].iloc[0]) if not day_data.empty else 0
        report_text = str(day_data["report"].iloc[0]) if not day_data.empty else ""
        report_line = report_text.splitlines()[0] if report_text else ""
        status = "●" * checked + "○" * (len(HABITS) - checked)
        mood_label = f"🙂 {mood_value}" if mood_value > 0 else "🙂 -"
        report_label = f"🧾 {report_line}" if report_line else "🧾 -"
        with week_cols[idx]:
            st.markdown(
                f"""
<div style="padding:10px;border:1px solid #E6E6E6;border-radius:10px;min-height:120px;">
  <div style="font-size:14px;font-weight:600;opacity:{1 if in_month else 0.35};">
    {day.day}
  </div>
  <div style="margin-top:6px;font-size:12px;opacity:{1 if in_month else 0.35};">
    {status}
  </div>
  <div style="margin-top:6px;font-size:12px;opacity:{1 if in_month else 0.35};">
    {mood_label}
  </div>
  <div style="margin-top:6px;font-size:11px;opacity:{1 if in_month else 0.35};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
    {report_label}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

st.markdown("#### 📊 이번 달 요약")
summary_cols = st.columns(3)
month_checked_sum = int(month_df["checked"].sum())
month_days_logged = int((month_df["checked"] > 0).sum())
avg_mood = round(month_df.loc[month_df["mood"] > 0, "mood"].mean() or 0, 1)

summary_cols[0].metric("누적 달성", f"{month_checked_sum}개")
summary_cols[1].metric("체크인 일수", f"{month_days_logged}일")
summary_cols[2].metric("평균 기분", f"{avg_mood}/10")

# ----------------------------
# Weather + Dog + AI Report
# ----------------------------
st.subheader("🧠 컨디션 리포트")

btn = st.button("🚀 컨디션 리포트 생성", type="primary", use_container_width=True)

weather_data = None
dog_data = None
report = None

if btn:
    with st.spinner("날씨/강아지/AI 코치 리포트를 준비 중..."):
        weather_data = get_weather(city, owm_api_key)
        dog_data = get_dog_image()
        habits_for_ai = {label: bool(habit_state[key]) for key, _, label in HABITS}
        report = generate_report(
            openai_key=openai_api_key,
            coach_style=coach_style,
            habits=habits_for_ai,
            mood=mood,
            weather=weather_data,
            dog=dog_data,
        )
        if report:
            st.session_state.reports[str(today)] = report

    # 결과 표시 (2열 카드 + 리포트)
    card1, card2 = st.columns(2)

    with card1:
        st.markdown("### ☁️ 오늘의 날씨")
        if weather_data is None:
            st.warning("날씨 데이터를 가져오지 못했어요. (API Key/도시/네트워크 확인)")
        else:
            st.write(f"**도시:** {weather_data['city']}")
            st.write(f"**날씨:** {weather_data['desc']}")
            st.write(f"**기온:** {weather_data['temp_c']:.1f}°C (체감 {weather_data['feels_like_c']:.1f}°C)")
            st.write(f"**습도:** {weather_data['humidity']}%")

    with card2:
        st.markdown("### 🐶 오늘의 강아지")
        if dog_data is None:
            st.warning("강아지 이미지를 가져오지 못했어요. (Dog CEO API/네트워크 확인)")
        else:
            st.write(f"**품종:** {dog_data['breed']}")
            st.image(dog_data["image_url"], use_container_width=True)

    st.markdown("### 🧾 AI 코치 리포트")
    if report is None:
        st.error("리포트를 생성하지 못했어요. (OpenAI Key/SDK/네트워크 확인)")
    else:
        st.markdown(report)

    # 공유용 텍스트
    st.markdown("### 📌 공유용 텍스트")
    weather_line = (
        "날씨 데이터 없음"
        if weather_data is None
        else f"{weather_data['city']} / {weather_data['desc']} / {weather_data['temp_c']:.1f}°C"
    )
    dog_line = "강아지 데이터 없음" if dog_data is None else f"오늘의 강아지: {dog_data['breed']}"
    habits_line = ", ".join([f"{emoji}{label}" for key, emoji, label in HABITS if habit_state.get(key)]) or "달성 습관 없음"

    share_text = (
        f"📊 AI 습관 트래커 ({today})\n"
        f"- 달성률: {achievement}% ({checked_count}/{len(HABITS)})\n"
        f"- 달성: {habits_line}\n"
        f"- 기분: {mood}/10\n"
        f"- 날씨: {weather_line}\n"
        f"- {dog_line}\n\n"
        f"{report if report else ''}"
    )
    st.code(share_text)
else:
    saved_report = st.session_state.reports.get(str(today))
    if saved_report:
        st.markdown("### 🧾 오늘의 저장된 리포트")
        st.markdown(saved_report)

# ----------------------------
# API 안내
# ----------------------------
with st.expander("ℹ️ API 안내 / 트러블슈팅"):
    st.markdown(
        """
**1) OpenWeatherMap**
- 현재 날씨 API 사용: `https://api.openweathermap.org/data/2.5/weather`
- 파라미터: `q=도시`, `appid=키`, `lang=kr`, `units=metric`
- 흔한 실패 원인:
  - API Key 미입력 / 만료
  - 도시명 오타 (예: `Seoul`, `Busan` 등)
  - 무료 플랜 호출 제한

**2) Dog CEO**
- 랜덤 이미지: `https://dog.ceo/api/breeds/image/random`
- 품종은 이미지 URL에서 추출(완벽하지 않을 수 있음)

**3) OpenAI**
- 리포트 모델: `gpt-5-mini`
- 필요 패키지: `openai` (설치: `pip install openai`)
- 흔한 실패 원인:
  - API Key 오류
  - 네트워크/프록시 문제
  - 사용량 제한/과금 이슈
"""
    )
