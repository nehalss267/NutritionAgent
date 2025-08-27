# Nutrition Agent
This project aims to develop “The Smartest AI Nutrition Assistant” using generative AI models. By integrating health data, food databases, and LLM-powered reasoning, this solution will bridge the gap between one-size-fits-all diet apps &amp; in-person nutrition counselling—delivering an AI that thinks, learns, &amp; cares like a real nutrition expert.

# 🥗 Personalized Diet & Lifestyle Assistant

This project is an AI-powered nutrition assistant built with **LangChain**, **Milvus**, **Replicate**, and **Gradio**.  
It provides personalized, practical, and culturally relevant nutrition advice based on user input such as age, food preferences, medical history, lifestyle, and health goals.

---

## 🚀 Features
- 📖 Loads and splits a nutrition knowledge base (`nutrition_literacy.txt`)
- 🧠 Embeds text into **Milvus vector database** using **HuggingFace embeddings**
- 🔍 Retrieves relevant context for queries (RAG – Retrieval Augmented Generation)
- 🤖 Uses **Replicate** LLM (`ibm-granite/granite-3.3-8b-instruct`)
- 🎯 Provides:
  - Tailored diet plans (daily/weekly)
  - Portion size & nutrient breakdowns
  - Local & affordable food options
  - Do’s & Don’ts for medical conditions
  - Lifestyle & hydration tips
- 🌐 Simple **Gradio UI** for interaction

---

## 📂 Project Structure

```
nutrition-agent/
│── nutrition_literacy.txt     # Base nutrition knowledge
│── app.py                     # Main application
│── requirements.txt           # Dependencies
│── README.md                  # Documentation

```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/NutritionAgent.git
cd NutritionAgent
````

Install Python libraries:

```bash
pip install -r requirements.txt
```


## 🔑 Replicate API Setup

1. Sign up on [Replicate](https://replicate.com/)
2. Generate an API token from your [Replicate Account Settings](https://replicate.com/account)
3. Set the token in your environment:

```bash
export REPLICATE_API_TOKEN="your_token_here"
```

Or directly inside the code:

```python
os.environ["REPLICATE_API_TOKEN"] = "your_token_here"
```

---

## ▶️ Run the App

Start the Gradio interface:

```bash
python app.py
```

You’ll get a local URL and a public sharing link from Gradio.

---

## 🧪 Example Queries

* *“I am a 25-year-old vegetarian with low BP. Suggest a 2-day meal plan for healthy weight gain.”*
* *“What are the benefits of eating fruits daily?”*
* *“Suggest a high-protein diet for muscle gain in India.”*

---

## 📚 Tech Stack

* **[LangChain](https://www.langchain.com/)** – LLM orchestration
* **[Milvus](https://milvus.io/)** – Vector database for retrieval
* **[HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** – Text embeddings
* **[Replicate](https://replicate.com/)** – Model inference
* **[Gradio](https://gradio.app/)** – Web UI

---

## 📦 Model Info

- **Model**: [`ibm-granite/granite-3.3-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
- **Library**: Hugging Face `transformers`
- **Device**: Uses GPU if available, else CPU

---
## Demo

[Nutrition Agent Demo](assets/demo.png)
   
## 📝 Future Improvements

* ✅ Add support for multi-turn conversations
* ✅ Integrate user profiles for persistent personalization
* ✅ Add regional diet recommendations with seasonal foods

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is for research and educational use. Ensure usage of the IBM Granite model complies with its [license terms](https://huggingface.co/ibm-granite/granite)



