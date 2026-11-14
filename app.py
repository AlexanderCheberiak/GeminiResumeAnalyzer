import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
import streamlit as st

load_dotenv()


# 1. Схема структурованого звіту (Structured Outputs)
class TechnicalQuestion(BaseModel):
    question: str = Field(
        description="Конкретне технічне запитання для перевірки прогалини кандидата"
    )
    skill_gap: str = Field(
        description="Вимога або технологія з вакансії, якої бракує в резюме"
    )
    expected_answer_hint: str = Field(
        description="Ключові тези, на які рекрутер або техлід має звернути увагу у відповіді"
    )


class ATSReport(BaseModel):
    candidate_name: str = Field(
        description="Ім'я та прізвище кандидата з резюме або 'Не вказано'"
    )
    match_score: int = Field(
        ge=0,
        le=100,
        description="Об'єктивна оцінка відповідності кандидата вимогам вакансії від 0 до 100",
    )
    verdict: str = Field(
        description="Стислий аналітичний висновок (2-3 речення) щодо сильних сторін і ризиків"
    )
    matched_skills: list[str] = Field(
        description="Навички та вимоги з вакансії, які підтверджені досвідом у резюме"
    )
    missing_skills: list[str] = Field(
        description="Обов'язкові або бажані навички з вакансії, відсутні в резюме"
    )
    interview_questions: list[TechnicalQuestion] = Field(
        min_length=5,
        max_length=5,
        description="Рівно 5 технічних запитань для перевірки виявлених прогалин на співбесіді",
    )


# 2. Системна інструкція для налаштування ролі моделі
SYSTEM_INSTRUCTION = """
Ти — старший технічний рекрутер та ATS-експерт (Applicant Tracking System).
Твоє завдання — неупереджено оцінити PDF-резюме кандидата відносно тексту вакансії.

Правила оцінювання:
1. Аналізуй документ напряму: враховуй структуру, реальний комерційний досвід, стек технологій та досягнення, а не лише формальну наявність ключових слів.
2. Оцінку match_score (0-100) обчислюй за формулою: 60% ваги — збіг обов'язкових технічних вимог (hard skills), 25% — релевантність комерційного досвіду та домену, 15% — додаткові переваги (nice-to-have).
3. У missing_skills винось лише ті вимоги, які прямо зазначені у вакансії, але не підтверджені в резюме.
4. Згенеруй рівно 5 практичних технічних запитань (interview_questions), спрямованих на перевірку саме тих прогалин, які потрапили до missing_skills або викликають сумніви.
5. Увесь текст звіту формуй українською мовою, зберігаючи оригінальні англійські назви технологій (наприклад, Docker, FastAPI, PostgreSQL).
""".strip()


def analyze_resume(pdf_bytes: bytes, job_description: str, api_key: str) -> ATSReport:
    client = genai.Client(api_key=api_key)

    # Пряме передавання PDF-файлу без стороннього парсингу тексту
    pdf_part = types.Part.from_bytes(
        data=pdf_bytes,
        mime_type="application/pdf",
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            pdf_part,
            f"Текст опису вакансії:\n\n{job_description}",
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ATSReport,
            temperature=0.2,
        ),
    )

    if response.parsed is not None:
        return response.parsed
    return ATSReport.model_validate_json(response.text)


# 3. Вебінтерфейс на Streamlit
st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")
st.title("ATS-аналізатор резюме під вакансію")
st.caption("Прямий аналіз PDF-документів через Google Gemini API зі структурованим виводом")

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("GEMINI_API_KEY", type="password")

col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("Вхідні дані")
    uploaded_pdf = st.file_uploader(
        "Завантажте резюме кандидата (формат PDF)",
        type=["pdf"],
    )
    job_text = st.text_area(
        "Опис вакансії та вимоги роботодавця",
        height=320,
        placeholder="Вставте текст вакансії: стек технологій, вимоги до досвіду, обов'язки...",
    )
    run_btn = st.button("Проаналізувати відповідність", type="primary", use_container_width=True)

with col_result:
    st.subheader("Картка результатів скринінгу")

    if run_btn:
        if not api_key:
            st.error("Вкажіть GEMINI_API_KEY у файлі .env або в бічній панелі.")
        elif uploaded_pdf is None:
            st.warning("Завантажте PDF-файл із резюме кандидата.")
        elif not job_text.strip():
            st.warning("Додайте текст опису вакансії.")
        else:
            with st.spinner("Модель зчитує PDF та зіставляє вимоги..."):
                try:
                    report = analyze_resume(
                        pdf_bytes=uploaded_pdf.getvalue(),
                        job_description=job_text.strip(),
                        api_key=api_key,
                    )

                    with st.container(border=True):
                        st.markdown(f"### Кандидат: {report.candidate_name}")
                        st.metric(label="Індекс відповідності (ATS Score)", value=f"{report.match_score} / 100")
                        st.progress(report.match_score)
                        st.write(report.verdict)

                        st.divider()
                        skills_col1, skills_col2 = st.columns(2)

                        with skills_col1:
                            st.markdown("**Підтверджені навички**")
                            for skill in report.matched_skills:
                                st.markdown(f"- :green[{skill}]")

                        with skills_col2:
                            st.markdown("**Відсутні навички (прогалини)**")
                            if report.missing_skills:
                                for skill in report.missing_skills:
                                    st.markdown(f"- :red[{skill}]")
                            else:
                                st.markdown("- Критичних прогалин не виявлено")

                        st.divider()
                        st.markdown("**5 технічних запитань для перевірки прогалин**")
                        for idx, item in enumerate(report.interview_questions, start=1):
                            with st.expander(f"{idx}. {item.question}", expanded=(idx == 1)):
                                st.markdown(f"**Перевіряє навичку:** `{item.skill_gap}`")
                                st.markdown(f"**Очікувана відповідь:** {item.expected_answer_hint}")

                except errors.APIError as e:
                    st.error(f"Помилка Gemini API ({e.code}): {e.message}")
                except Exception as e:
                    st.error(f"Помилка під час обробки запиту: {e}")
    else:
        st.info("Завантажте PDF-резюме та вставте опис вакансії ліворуч, щоб отримати звіт.")