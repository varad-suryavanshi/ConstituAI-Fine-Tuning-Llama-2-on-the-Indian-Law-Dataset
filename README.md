# 🧠 ConstituAI: Fine-Tuning LLaMA 2 on Indian Legal Texts 🇮🇳⚖️

ConstituAI is a legal-domain language model fine-tuned on Indian constitutional and legal texts. Built using Meta’s LLaMA 2–7B Chat architecture, the model is trained to generate high-quality, instruction-following responses relevant to the Indian legal system. The project explores how powerful open-source large language models can be adapted to legal tasks such as constitutional Q&A, legal document summarization, and explanatory dialogue.

🔗 **Model on Hugging Face:** [varad-suryavanshi12/Llama-2-7b-chat-finetune](https://huggingface.co/varad-suryavanshi12/Llama-2-7b-chat-finetune)  
📊 **Dataset on Kaggle:** [LLM Fine-Tuning Dataset of Indian Legal Texts](https://www.kaggle.com/datasets/akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts)  
🖥️ **Demo Included:** Streamlit UI for interactive inference

---

## 📌 About the Project

India's legal system is complex and largely underrepresented in most commercial LLMs. ConstituAI bridges this gap by fine-tuning a high-performance open-source model (LLaMA 2) on a rich dataset of Indian legal texts, primarily focused on the Constitution of India.

This project demonstrates:
- How open LLMs like LLaMA 2 can be specialized for niche domains (like Indian law)
- The process of preparing structured legal Q&A datasets for instruction tuning
- Deployment of a fully functional legal chatbot using a lightweight interface (Streamlit)

Applications include legal education, citizen support, law student assistance, and more.

---

## 🧠 Key Features

- Fine-tuned LLaMA 2–7B Chat on Indian constitutional data
- Instruction-style Q&A with contextual understanding of legal terms
- Interactive Streamlit interface for real-time response generation
- 4-bit quantized model loading for efficient local inference

---

## 📁 Project Structure

| File                                  | Description                                 |
|---------------------------------------|---------------------------------------------|
| `Fine_tune_Llama_2_Indian_Law_Dataset.ipynb` | Notebook to process dataset and fine-tune model |
| `streamlitapp.py`                     | Streamlit UI for inference/demo             |
| `constitution_qa*.json`               | (Ignored) Large dataset files from Kaggle   |
| `llama2_Finetuned_Indian_Law/`        | (Ignored) Saved model checkpoint directory  |

---

## 📊 Dataset

Sourced from Kaggle:

📦 [LLM Fine-Tuning Dataset of Indian Legal Texts](https://www.kaggle.com/datasets/akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts)

- Based on Indian constitutional law
- Converted into instruction-style prompts (`<s>[INST] question [/INST] answer`)
- Dataset files are excluded from this repo due to size

---

## 🤗 Model on Hugging Face

A fine-tuned version of `llama-2-7b-chat`, hosted here:

🔗 [varad-suryavanshi12/Llama-2-7b-chat-finetune](https://huggingface.co/varad-suryavanshi12/Llama-2-7b-chat-finetune)

- Chat-optimized
- Trained for legal domain text generation
- Loadable using `transformers` with 4-bit quantization via `BitsAndBytes`

---

## 🚀 Run the Streamlit App Locally

1. Clone this repository
2. Install the required packages (consider using a virtual environment)
3. Run the app:

```bash
streamlit run streamlitapp.py
