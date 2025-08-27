!pip install gradio langchain langchain_community langchain_milvus transformers pymilvus

!pip install replicate

!pip install sentence-transformers

import os
import gradio as gr
import tempfile
from langchain_milvus import Milvus
from langchain_community.llms import Replicate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.embeddings import HuggingFaceEmbeddings

#Steps of generating replicating API Token in README.md
os.environ['REPLICATE_API_TOKEN'] = "<YOUR_REPLICATE_API_TOKEN>"
model_path = "ibm-granite/granite-3.3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Replicate(model=model_path, replicate_api_token=os.environ['REPLICATE_API_TOKEN'])

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)

filename = "nutrition_literacy.txt"
with open(filename, "w") as f:
    f.write("""
Balanced Diet: A balanced diet includes carbohydrates, proteins, fats, vitamins, minerals, and water in the right amounts for proper growth and health.

Macronutrients: Carbohydrates provide energy, proteins build and repair tissues, and fats store energy and support cell function.

Micronutrients: Vitamins and minerals help boost immunity, support growth, and maintain overall body functions. Examples include Vitamin C, Iron, and Calcium.

Hydration: Drinking enough water keeps the body hydrated, improves digestion, and regulates body temperature.

Reading Food Labels: Check nutritional information on packaged foods. Look for calories, sugar, fat, protein, and fiber content before consuming.

Healthy Eating Habits: Eat more fruits, vegetables, whole grains, and lean proteins. Avoid excess sugar, junk food, and deep-fried items.

Meal Planning: Plan meals in advance to ensure balanced nutrition. Include variety in food groups for better health.

Portion Control: Eat in moderate portions to avoid overeating. Smaller plates can help manage food intake.

Physical Activity: Combine good nutrition with at least 30 minutes of daily exercise for a healthy lifestyle.

Food Safety: Wash fruits and vegetables, cook food properly, and store leftovers safely to avoid food-borne illness.

Explain about Importance of Breakfast. Explain Benefits of Eating Fruits Daily also.
""")

loader = TextLoader(filename)
documents = loader.load()
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=tokenizer.model_max_length // 2,
    chunk_overlap=0,
)
texts = splitter.split_documents(documents)
for i, doc in enumerate(texts):
    doc.metadata["doc_id"] = i + 1
vector_db.add_documents(texts)

template = """
You are a personalized nutrition assistant. User Prompt :{question} 
Collect and consider the user’s details, including:
Age & gender
Food preferences (veg/non-veg/vegan, likes/dislikes)
Medical history & conditions (diabetes, hypertension, allergies, etc.)
Lifestyle (activity level, work type, sleep patterns)
City/region (for local food availability & climate)
Goals (weight loss, muscle gain, balanced diet, disease management, etc.)
Based on this, generate specific, practical, and tailored nutrition advice such as:
Daily/weekly meal plans
Portion sizes & nutrient breakdowns
Local, affordable food options
Do’s and Don’ts aligned with their medical history
Hydration & lifestyle tips

Always keep responses personalized, culturally relevant, and realistic to follow.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(llm=model, prompt=prompt)
combine_chain = StuffDocumentsChain(llm_chain=llm_chain)

rag_chain = RetrievalQA(
    retriever=vector_db.as_retriever(),
    combine_documents_chain=combine_chain,
    return_source_documents=False
)

def ask_nutrition_agent(query):
    try:
        response = rag_chain.run(query)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

iface = gr.Interface(
    fn=ask_nutrition_agent,
    inputs=gr.Textbox(
        label="Ask a nutrition question",
        placeholder="e.g. What’s a good diet plan for weight loss?"
    ),
    outputs=gr.Textbox(label="Answer"),
    title="Nutrition Agent",
    description="Ask about healthy diets, meal plans, food preferences, lifestyle tips, and medical condition-based nutrition advice.",
    theme="default"
)

iface.launch()


