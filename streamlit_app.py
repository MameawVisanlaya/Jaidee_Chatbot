import os
import json
import requests
import streamlit as st
from google.cloud import storage
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# โหลดค่า Environment Variables จากไฟล์ .env
load_dotenv()

# ดึง API Key จากไฟล์ .env
API_KEY = os.getenv("API_KEY")

# ตั้งค่า Google Cloud Service Account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspaces/Jaidee_Chatbot/service-account-key.json"

# Google Cloud Storage Bucket
BUCKET_NAME = "chat_bot_file"
FILE_PATH = "Chatbot/Chunks/ocr_results_chunks.jsonl"

# ฟังก์ชันโหลดไฟล์จาก Google Cloud Storage
def load_chunks_from_gcs(bucket_name, file_path):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        content = blob.download_as_text()
        chunks = [json.loads(line) for line in content.splitlines()]
        return chunks
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์จาก Google Cloud Storage ได้: {e}")
        return []

# ฟังก์ชันสร้างเวกเตอร์ด้วย Text Embedding 004
def get_embedding(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/textembedding-gecko-004:embedText"
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}
    try:
        response = requests.post(f"{url}?key={API_KEY}", json=payload, headers=headers)
        response_data = response.json()
        if "embedding" in response_data:
            return response_data["embedding"]["value"]
        else:
            return None
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# ฟังก์ชันค้นหาบริบทที่เกี่ยวข้อง
def get_relevant_context(question, chunks):
    question_embedding = get_embedding(question)
    if question_embedding is None:
        return "ไม่สามารถสร้าง embedding จากคำถามได้ค่ะ"

    context_embeddings = []
    context_texts = []

    for chunk in chunks:
        text = chunk.get("text", "")
        embedding = get_embedding(text)
        if embedding is not None:
            context_embeddings.append(embedding)
            context_texts.append(text)

    # คำนวณความคล้ายคลึง (Cosine Similarity)
    similarities = cosine_similarity([question_embedding], context_embeddings)[0]

    # เลือกข้อความที่มีความคล้ายมากที่สุด 3 ข้อความ
    top_indices = np.argsort(similarities)[::-1][:3]
    top_contexts = [context_texts[i] for i in top_indices]

    # ปรับบริบทให้เน้นการนัดปรึกษาแพทย์
    return " ".join(top_contexts) + " หากคุณมีความกังวล ควรนัดหมายแพทย์เพื่อรับคำแนะนำเพิ่มเติมค่ะ"

# ฟังก์ชันสร้างคำตอบด้วย Generative AI API
def generate_answer_with_gemini(question, context="", chat_history=[]):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    
    # รวมประวัติการสนทนาใน Prompt
    history_text = "\n".join(
        [f"ผู้ใช้: {entry[1]}" if entry[0] == "user" else f"น้องใจดี: {entry[1]}" 
         for entry in chat_history]
    )

    prompt = f"""
        บทบาท: คุณคือน้องพยาบาลฝึกหัดที่น่ารัก ใจดี เป็นผู้ฟังที่ดี เข้าใจความรู้สึก และคอยให้กำลังใจ  และเป็นผู้เชี่ยวชาญที่ให้คำแนะนำด้านสุขภาพและโภชนาการ พร้อมให้คำแนะนำที่ช่วยลดความกังวล

        คำแนะนำ:
        - ตอบคำถามด้วยความกระชับ เข้าใจง่าย
        - ให้คำตอบประมาณ 3-4 ประโยค
        - ใช้ภาษาที่สุภาพและให้กำลังใจ
        - แนะนำให้ผู้ใช้นัดพบแพทย์ในกรณีที่มีข้อสงสัย

        บริบท:
        {context}

        ประวัติการสนทนา:
        {history_text}

        คำถามใหม่: {question}

        กรุณาตอบด้วยน้ำเสียงที่สุภาพ เช่น ลงท้ายด้วยคำว่า "ค่ะ" หรือ "นะคะ"
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generation_config": {
            "max_output_tokens": 350,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40
        }
    }
    try:
        response = requests.post(f"{url}?key={API_KEY}", json=payload, headers=headers)
        response_data = response.json()
        if "candidates" in response_data:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "ไม่พบคำตอบจากโมเดลค่ะ"
    except Exception as e:
        return f"Error: {e}"

# เริ่มต้น Streamlit UI
st.title("😊น้องใจดี")
st.markdown("🩺👩🏻‍⚕️💬 น้องพยาบาลฝึกหัดที่พร้อมรับฟังปัญหาสุขภาพใจ  และให้คำแนะนำด้านโภชนาการ")

# ตรวจสอบและกำหนดค่าเริ่มต้นของ session_state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# โหลดไฟล์ Chunk จาก Google Cloud Storage
st.write("กำลังโหลดข้อมูลจาก Google Cloud Storage...")
chunks = load_chunks_from_gcs(BUCKET_NAME, FILE_PATH)
if chunks:
    st.success("โหลดข้อมูลสำเร็จ!")

# รับคำถามจากผู้ใช้
if user_input := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):

    # ดึงบริบทที่เกี่ยวข้อง 
    combined_context = get_relevant_context(user_input, chunks)  

    # อ่านข้อมูลเดิมจากไฟล์ (ถ้ามี)
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            try:
                chat_history = json.load(f)
            except json.JSONDecodeError:  # กรณีไฟล์ว่างหรือมีข้อมูลไม่ถูกต้อง
                chat_history = []
    else:
        chat_history = []

    # เพิ่มข้อมูลใหม่เข้าไปใน list
    chat_history.append(("user", user_input))

    # สร้างคำตอบ
    bot_response = generate_answer_with_gemini(  # เรียกใช้ครั้งเดียว
        user_input, 
        context=combined_context, 
        chat_history=st.session_state["chat_history"]
    )

    chat_history.append(("assistant", bot_response))  # เพิ่มคำตอบของ bot 

    # บันทึก chat_history ลงไฟล์ JSON
    with open('chat_history.json', 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

    st.session_state["chat_history"].append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    st.chat_message("assistant").markdown(bot_response)


if st.button("🔄 ล้างประวัติคำถามล่าสุด"):
    if st.session_state["chat_history"]:
        st.session_state["chat_history"].pop()  # ลบคำถาม/คำตอบล่าสุด
