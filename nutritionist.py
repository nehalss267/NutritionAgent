from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import traceback

# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load model/tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
    model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
    model.to(device)
    model.eval()
    print("[INFO] Model and tokenizer loaded successfully.")
except Exception:
    print("[ERROR] Failed to load model/tokenizer.")
    traceback.print_exc()
    raise

SYSTEM_PROMPT = (
    "You are a highly experienced virtual nutritionist. "
    "You take user inputs (text, photos, voice), infer health goals, preferences, allergies, cultural cuisine. "
    "You propose meal plans, explain choices, and adapt strategies based on feedback.\n"
)

def generate_response(user_input, history=None, max_new_tokens=150):
    if history is None:
        history = []
    try:
        # Build a simple conversational prompt manually
        prompt = SYSTEM_PROMPT
        for msg in history:
            prompt += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
        prompt += f"user: {user_input}\nassistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                # you can add early_stopping=True if desired
            )
        # Extract generated part
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return reply
    except Exception:
        print("[ERROR] Exception during generation.")
        traceback.print_exc()
        return "Error during generation."

# Mocked food analysis
def analyze_food_image(image_bytes):
    food_label = "Idli with sambar"
    nutrition = {
        "calories": 300,
        "protein": "6g",
        "carbs": "45g",
        "fat": "8g"
    }
    return food_label, nutrition

def plan_meal(user_profile, nutrition_data):
    prompt = (
        f"User Profile: {user_profile}\n"
        f"They just ate a meal with the following nutrients: {nutrition_data}\n"
        "Suggest a balanced dinner plan that aligns with their weight loss goal. Explain the choices."
    )
    return generate_response(prompt)

# === Run the flow ===
if __name__ == "__main__":
    image_bytes = None  # placeholder
    user_profile = {
        "age": 35,
        "goals": "weight loss",
        "allergies": ["nuts"],
        "preferences": "South Indian vegetarian"
    }

    label, nutrition = analyze_food_image(image_bytes)
    print(f"\nDetected Food: {label}\nNutrition Info: {nutrition}")

    response = plan_meal(user_profile, nutrition)
    print("\nMeal Recommendation:\n")
    print(response)
