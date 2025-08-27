# NutritionAgent
This project aims to develop “The Smartest AI Nutrition Assistant” using generative AI models. By integrating health data, food databases, and LLM-powered reasoning, this solution will bridge the gap between one-size-fits-all diet apps &amp; in-person nutrition counselling—delivering an AI that thinks, learns, &amp; cares like a real nutrition expert.

# 🥗 Virtual Nutritionist AI (IBM Granite 3.3 8B)

This project demonstrates a simple AI-powered virtual nutritionist using the IBM Granite-3.3-8B-Instruct model. It processes user input and generates context-aware meal plans tailored to health goals, dietary preferences, and recent meals.

## 🔍 Features

- Loads and runs the `ibm-granite/granite-3.3-8b-instruct` language model locally using PyTorch.
- Acts as a virtual nutritionist that:
  - Understands user profile and food input.
  - Provides personalized meal suggestions.
  - Explains recommendations in natural language.
- Includes a mock food image analyzer to simulate nutrition detection.

---

## 🚀 Setup Instructions

### 1. Install Dependencies

Make sure Python 3.8+ is installed, then run:

```bash
pip install torch transformers
```

### 2. Run the Script

```bash
python nutritionist.py
```

It will:
- Simulate a user profile and meal input.
- Analyze the food (mocked).
- Generate a customized meal plan using the Granite model.

---

## 📁 File Overview

### `nutritionist.py`

Main script that:
- Loads model and tokenizer (`granite-3.3-8b-instruct`).
- Defines system prompt and dialogue logic.
- Simulates a user and analyzes a meal.
- Generates and prints a dinner recommendation.

---

-3.3-8b-instruct).
