import re
import json
import pandas as pd
import nltk
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
from langdetect import detect, LangDetectException
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def configure_vertex_ai(project_id, location, model_name):
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)
        logger.info(f"Vertex AI Gemini model initialized: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
        raise

def preprocess_message(message):
    logger.debug(f"Preprocessing message: {message}")
    message = re.sub(r'^~[^:]+:\s*', '', message)
    message = re.sub(r'\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\]\s*', '', message)
    message = re.sub(r'\+\d{1,3}\s*\d{5}\s*\d{5}', '', message)
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
    message = re.sub(r'[\U0001F600-\U0001F6FF\U0001F900-\U0001F9FF]', '', message)
    message = re.sub(r'[^\w\s\']', '', message)
    message = re.sub(r'\s+', ' ', message).strip().lower()
    tokens = nltk.word_tokenize(message)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words or token in ['what', 'how', 'when']]
    cleaned_message = ' '.join(cleaned_tokens)
    try:
        language = detect(cleaned_message) if cleaned_message else 'unknown'
    except LangDetectException:
        language = 'unknown'
    logger.debug(f"Cleaned message: {cleaned_message}, Language: {language}")
    return cleaned_message, language


def classify_and_extract(message, model):
    """Classify intent and extract entities using Vertex AI Gemini model."""
    prompt = f"""
You are a smart assistant for understanding and extracting structured information from informal WhatsApp messages.

### Your Task:
Analyze the message below and:
1. **Classify the user's intent** by inferring the most appropriate intent based on the message content.
2. **Extract structured entities** as key-value pairs relevant to the intent, based on the context.
3. Handle informal language, spelling mistakes, emojis, abbreviations, or missing words.
4. If some entity values are missing, set them to `null`.
5. If the message is irrelevant or purely conversational, assign it to `Casual Chat` with `entities` set to `null`.

### Expected Output Format:
Respond only with a JSON block wrapped in triple backticks like this:

```json
{{
  "intent": "<Intent Name>",
  "entities": {{
    ...key-value pairs OR null...
  }}
}}
```

### Examples:
- Message: "What’s your refund policy?"
  Output: ```json{{"intent": "Policy Inquiry", "entities": {{"Policy": "refund"}}}}```
- Message: "anyone referral please mail vivekhumansofcodecom case need detail reach referral gift well"
  Output: ```json{{"intent": "Referral Inquiry", "entities": {{"Program": "referral program", "Aspect": "rewards"}}}}```
- Message: "2BHK required around Bellandur by Nov end. Ping me 1:1 if you have any leads."
  Output: ```json{{"intent": "Housing Inquiry", "entities": {{"Service": "housing", "Location": "Bellandur", "Time": "November"}}}}```
- Message: "Hey, how’s it going?"
  Output: ```json{{"intent": "Casual Chat", "entities": null}}```

Message: {message}
Output:
"""
    try:
        logger.debug(f"Sending request to Vertex AI for message: {message}")
        response = model.generate_content(prompt)
        content = response.text.strip()
        logger.debug(f"Raw response content: {content}")
        json_content = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_content:
            json_content = json_content[0].strip()
            logger.debug(f"Extracted JSON: {json_content}")
            return json.loads(json_content)
        logger.error("No JSON block found in response")
        return {
            "intent": "Error",
            "entities": "null",
            "error": "No JSON block in response"
        }
    except Exception as e:
        logger.error(f"Model error: {str(e)}")
        return {
            "intent": "Error",
            "entities": "null",
            "error": str(e)
        }


def process_message(message, model, results_list):
    cleaned_message, language = preprocess_message(message)
    result = classify_and_extract(cleaned_message, model)
    has_intent = result.get('intent') != 'Casual Chat' and result.get('intent') != 'Error'
    results_list.append({
        'original_message': message,
        'cleaned_message': cleaned_message,
        'language': language,
        'has_intent': has_intent,
        'intent': result.get('intent', 'Error'),
        'entities': json.dumps(result.get('entities', None)),
        'error': result.get('error', None)
    })
    time.sleep(2)
    return result

if __name__ == "__main__":
    project_id = "sandbox-451305"
    location = "us-central1"
    model_name = "gemini-2.5-flash"
    try:
        model = configure_vertex_ai(project_id, location, model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
        print(f"Failed to initialize Vertex AI client: {str(e)}")
        exit()


    file_path = "C:/Gangadhar/Strawhat/Chat_Classification/updated_chat.txt"

    if not Path(file_path).is_file():
        logger.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")
        exit()

    results_list = []
    print("Processing messages...\n")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        print(f"Failed to read file {file_path}: {str(e)}")
        exit()

    for line in lines:
        if not line.strip() or "omitted" in line.lower() or "joined using this group's invite link" in line or "This message was deleted" in line.lower():
            logger.debug(f"Skipping line: {line.strip()}")
            continue

        match = re.search(r"\]\s*(.*)", line) or re.search(r"~[^:]+:\s*(.*)", line)
        message = match.group(1) if match else line.strip()


        result = process_message(message, model, results_list)

        print(f"Original: {message}")
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 60)

    try:
        results_df = pd.DataFrame(results_list)
        print("\nDataset Preview:")
        print(results_df.head())
        results_df.to_csv("processed_whatsapp_intents.csv", index=False)
        print("\nSaved to processed_whatsapp_intents.csv")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        print(f"Failed to save CSV: {str(e)}")
        exit()