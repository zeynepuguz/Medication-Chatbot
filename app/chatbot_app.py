import pickle
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

# DistilBERT Retriever yükleme
distilbert_model = SentenceTransformer("models/distilbert_retriever/model")

with open("models/distilbert_retriever/questions.pkl", "rb") as f:
    questions = pickle.load(f)

with open("models/distilbert_retriever/answers.pkl", "rb") as f:
    answers = pickle.load(f)

with open("models/distilbert_retriever/question_embeddings.pkl", "rb") as f:
    question_embeddings = pickle.load(f)

# T5 yükleme
t5_model_path = "models/t5"

t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

# DistilBERT cevap fonksiyonu
def distilbert_answer(user_question):
    user_embedding = distilbert_model.encode([user_question])

    similarities = cosine_similarity(
        user_embedding,
        question_embeddings
    )[0]

    top_indices = similarities.argsort()[-3:][::-1]
    best_score = similarities[top_indices[0]]

    if best_score < 0.70:
        return f"""
Bu soru için yeterince güvenilir bir eşleşme bulunamadı.

En yüksek benzerlik skoru: {best_score:.4f}
"""

    response = ""

    for rank, idx in enumerate(top_indices, 1):
        response += f"""
SONUÇ {rank}

Benzerlik skoru: {similarities[idx]:.4f}

Eşleşen soru:
{questions[idx]}

Cevap:
{answers[idx]}

{'-' * 50}

"""

    return response

# T5 cevap fonksiyonu
def t5_answer(user_question):
    input_text = "question: " + user_question

    inputs = t5_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = t5_model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    generated_answer = t5_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    generated_embedding = distilbert_model.encode([generated_answer])

    answer_embeddings = distilbert_model.encode(
        answers,
        convert_to_tensor=False,
        show_progress_bar=False
    )

    similarities = cosine_similarity(
        generated_embedding,
        answer_embeddings
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    reference_question = questions[best_index]
    reference_answer = answers[best_index]

    response = f"""
MODEL: T5 Generative

Üretilen Cevap:
{generated_answer}

Benzerlik skoru: {best_score:.4f}

En yakın veri seti sorusu:
{reference_question}

En yakın veri seti cevabı:
{reference_answer}
"""

    return response

# Ortak chatbot fonksiyonu
def chatbot(model_choice, user_question):
    if not user_question.strip():
        return "Lütfen bir soru giriniz."

    if model_choice == "DistilBERT Retriever":
        return distilbert_answer(user_question)

    elif model_choice == "T5 Generative":
        return t5_answer(user_question)

    else:
        return "Lütfen bir model seçiniz."

# Gradio arayüz
interface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.Dropdown(
            choices=["DistilBERT Retriever", "T5 Generative"],
            value="DistilBERT Retriever",
            label="Model Seçiniz"
        ),
        gr.Textbox(
            lines=2,
            placeholder="İlaçla ilgili sorunuzu İngilizce yazın...",
            label="Kullanıcı Sorusu"
        )
    ],
    outputs=gr.Textbox(
        lines=25,
        label="Model Cevabı"
    ),
    title="İlaç Bilgilendirme Soru-Cevap Chatbot",
    description="Bu arayüzde DistilBERT Retriever ve T5 Generative modelleri karşılaştırmalı olarak test edilebilir."
)

interface.launch()