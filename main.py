import json
import fitz  # PyMuPDF untuk ekstraksi teks dari PDF
from transformers import pipeline
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def load_context_from_pdf(pdf_path):
    """Membaca teks dari file PDF sebagai konteks chatbot."""
    try:
        doc = fitz.open(pdf_path)  # Buka PDF
        text = ""
        for page in doc:  # Iterasi setiap halaman
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error membaca PDF: {e}")
        return ""

def start_chat():
    """Memulai chatbot dengan konteks dari PDF."""
    device = 0 if torch.cuda.is_available() else -1  # Gunakan GPU jika tersedia

    chatbot = pipeline(
        "question-answering",
        model="Rifky/Indobert-QA",
        tokenizer="Rifky/Indobert-QA",
        device=device
    )

    print("Hallo Edo! Disini Chatbot yang akan membantu kamu!\nKetik 'exit' untuk keluar.")

    # Load konteks dari file PDF
    context = load_context_from_pdf("data.pdf")

    if not context:
        print("Konteks tidak ditemukan. Pastikan file 'data.pdf' tersedia.")
        return

    while True:
        user_input = input("\nYou: ").strip()  # Hapus whitespace berlebih

        if user_input.lower() in ["exit", "keluar", "quit"]:
            print("Goodbye Edo! Sampai jumpa lagi!")
            break

        if len(user_input) < 3:  # Hindari input terlalu pendek
            print("ChatBot: Pertanyaan terlalu pendek, bisa diperjelas?")
            continue

        response = chatbot({
            "question": user_input,
            "context": context
        }, top_k=3, max_answer_len=50)  # Tuning hyperparameter

        best_answer = response[0]["answer"] if response else "Maaf, saya tidak dapat menemukan jawaban."

        print("ChatBot:", best_answer)

if __name__ == "__main__":
    start_chat()
