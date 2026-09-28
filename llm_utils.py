import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


def get_llm_reasoning(prompt):
    if not client:
        return (
            "Penjelasan AI tidak tersedia karena API key OpenAI belum diatur "
            "atau belum aktif. Deteksi utama masih berjalan dengan model dan rule-based."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content

    except OpenAIError as e:
        return (
            "Penjelasan AI tidak tersedia saat ini karena token tidak valid, "
            "quota habis, atau koneksi API bermasalah. Deteksi utama tetap bisa digunakan."
        )
    except Exception:
        return (
            "Penjelasan AI gagal saat ini. Sistem tetap menjalankan deteksi utama "
            "tanpa penjelasan tambahan."
        )