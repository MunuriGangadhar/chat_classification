# import re
# import json
# import pandas as pd
# import nltk
# from google.cloud import aiplatform
# from google.cloud.aiplatform.gapic import PredictionServiceClient
# from google.cloud.aiplatform_v1.types import Value
# from google.protobuf.json_format import MessageToDict
# from langdetect import detect, LangDetectException
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import time
# from pathlib import Path
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Download NLTK resources
# nltk.download('wordnet', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt_tab', quiet=True)

# # Initialize lemmatizer and stopwords
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# # Configure Vertex AI client
# def configure_vertex_ai(project_id, location, model_name):
#     """Initialize Vertex AI client using Application Default Credentials."""
#     try:
#         aiplatform.init(project=project_id, location=location)
#         client = PredictionServiceClient()
#         endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_name}"
#         logger.info(f"Vertex AI client initialized with endpoint: {endpoint}")
#         return client, endpoint
#     except Exception as e:
#         logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
#         raise

# # Enhanced preprocessing
# def preprocess_message(message):
#     """Preprocess WhatsApp message with advanced cleaning."""
#     logger.debug(f"Preprocessing message: {message}")
#     # Remove timestamps (e.g., [12/12/24, 10:00 AM])
#     message = re.sub(r'\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\]\s*', '', message)
#     # Remove phone numbers (e.g., +91 12345 67890)
#     message = re.sub(r'\+\d{1,3}\s*\d{5}\s*\d{5}', '', message)
#     # Remove URLs
#     message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
#     # Remove emojis (basic Unicode range)
#     message = re.sub(r'[\U0001F600-\U0001F6FF\U0001F900-\U0001F9FF]', '', message)
#     # Remove punctuation and normalize spaces
#     message = re.sub(r'[^\w\s]', '', message)
#     message = re.sub(r'\s+', ' ', message).strip().lower()
#     # Tokenize, remove stopwords, and lemmatize
#     tokens = nltk.word_tokenize(message)
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
#     cleaned_message = ' '.join(tokens)
#     # Detect language (optional, for multilingual handling)
#     try:
#         language = detect(cleaned_message) if cleaned_message else 'unknown'
#     except LangDetectException:
#         language = 'unknown'
#     logger.debug(f"Cleaned message: {cleaned_message}, Language: {language}")
#     return cleaned_message, language

# # Intent classification and entity extraction
# def classify_and_extract(message, client, endpoint):
#     """Classify intent and extract entities using Vertex AI."""
#     prompt = f"""
# You are a smart assistant for understanding and extracting structured information from informal WhatsApp messages.

# ### Your Task:
# Analyze the message below and:
# 1. **Classify the user's intent**.
# 2. **Extract structured entities** related to that intent.
# 3. Handle messages with informal language, spelling mistakes, emojis, abbreviations, or missing words.
# 4. If the intent is clear but some entity values are missing, set them to `null`.
# 5. If the message is irrelevant or doesn't match any listed intent, assign it to `Casual Chat` and set `entities` to `null`.

# ### Expected Output Format:
# Respond only with a JSON block wrapped in triple backticks like this:

# ```json
# {{
#   "intent": "<Intent Name>",
#   "entities": {{
#     ...key-value pairs OR null...
#   }}
# }}
# ```

# Intent Definitions and Entity Formats:
# 1. Contact Request: {{"Location": "...", "Company": "..."}} 
#    Example: "Contact Facebook in Bangalore" → ```json{{"intent": "Contact Request", "entities": {{"Location": "Bangalore", "Company": "Facebook"}}}}```
# 2. Information Query: {{"Product": "..."}} 
#    Example: "What are the features of the Apple Watch?" → ```json{{"intent": "Information Query", "entities": {{"Product": "Apple Watch"}}}}```
# 3. Technical Support: {{"Issue": "..."}} 
#    Example: "My app crashes" → ```json{{"intent": "Technical Support", "entities": {{"Issue": "app crash"}}}}```
# 4. Feedback: {{"Sentiment": "positive/negative/mixed", "Topic": "..."}} 
#    Example: "Great service!" → ```json{{"intent": "Feedback", "entities": {{"Sentiment": "positive", "Topic": "service"}}}}```
# 5. Purchase Inquiry: {{"Product": "...", "Aspect": "price/availability/..."}} 
#    Example: "How much is the premium plan?" → ```json{{"intent": "Purchase Inquiry", "entities": {{"Product": "premium plan", "Aspect": "price"}}}}```
# 6. Booking Request: {{"Time": "...", "Quantity": "...", "Service": "..."}} 
#    Example: "Book a table for two at 7 PM" → ```json{{"intent": "Booking Request", "entities": {{"Time": "7 PM", "Quantity": "two", "Service": "table"}}}}```
# 7. Order Status: {{"OrderID": "..."}} 
#    Example: "Where is my package?" → ```json{{"intent": "Order Status", "entities": {{"OrderID": null}}}}```
# 8. Complaint: {{"Issue": "..."}} 
#    Example: "The product is defective" → ```json{{"intent": "Complaint", "entities": {{"Issue": "defective product"}}}}```
# 9. Casual Chat: None 
#    Example: "Hey, how’s it going?" → ```json{{"intent": "Casual Chat", "entities": null}}```
# 10. Account Management: {{"Task": "..."}} 
#     Example: "I forgot my password" → ```json{{"intent": "Account Management", "entities": {{"Task": "password reset"}}}}```
# 11. Event Inquiry: {{"Event": "...", "Aspect": "date/location/..."}} 
#     Example: "When is the next webinar?" → ```json{{"intent": "Event Inquiry", "entities": {{"Event": "webinar", "Aspect": "date"}}}}```
# 12. Subscription Inquiry: {{"Action": "...", "Service": "..."}} 
#     Example: "How do I cancel my subscription?" → ```json{{"intent": "Subscription Inquiry", "entities": {{"Action": "cancel", "Service": "subscription"}}}}```
# 13. Location Inquiry: {{"Location": "...", "Place": "..."}} 
#     Example: "Where is your store in Mumbai?" → ```json{{"intent": "Location Inquiry", "entities": {{"Location": "Mumbai", "Place": "store"}}}}```
# 14. Recommendation Request: {{"Category": "...", "Location": "..."}} 
#     Example: "Recommend a restaurant in Delhi" → ```json{{"intent": "Recommendation Request", "entities": {{"Category": "restaurant", "Location": "Delhi"}}}}```
# 15. Policy Inquiry: {{"Policy": "..."}} 
#     Example: "What’s your refund policy?" → ```json{{"intent": "Policy Inquiry", "entities": {{"Policy": "refund"}}}}```
# 16. Appointment Cancellation: {{"Service": "...", "Time": "..."}} 
#     Example: "Cancel my 3 PM appointment" → ```json{{"intent": "Appointment Cancellation", "entities": {{"Service": "appointment", "Time": "3 PM"}}}}```
# 17. Product Comparison: {{"Product1": "...", "Product2": "..."}} 
#     Example: "Compare iPhone 13 and Samsung Galaxy S21" → ```json{{"intent": "Product Comparison", "entities": {{"Product1": "iPhone 13", "Product2": "Samsung Galaxy S21"}}}}```
# 18. Delivery Inquiry: {{"OrderID": "...", "Aspect": "time/method/..."}} 
#     Example: "When will my order #123 arrive?" → ```json{{"intent": "Delivery Inquiry", "entities": {{"OrderID": "123", "Aspect": "time"}}}}```
# 19. Payment Issue: {{"Issue": "..."}} 
#     Example: "My payment failed" → ```json{{"intent": "Payment Issue", "entities": {{"Issue": "payment failed"}}}}```
# 20. Service Inquiry: {{"Service": "...", "Aspect": "availability/details/..."}} 
#     Example: "Do you offer home delivery?" → ```json{{"intent": "Service Inquiry", "entities": {{"Service": "home delivery", "Aspect": "availability"}}}}```
# 21. Refund Request: {{"OrderID": "...", "Reason": "..."}} 
#     Example: "I want a refund for order #456" → ```json{{"intent": "Refund Request", "entities": {{"OrderID": "456", "Reason": null}}}}```
# 22. Warranty Inquiry: {{"Product": "...", "Aspect": "coverage/claim/..."}} 
#     Example: "Is my laptop under warranty?" → ```json{{"intent": "Warranty Inquiry", "entities": {{"Product": "laptop", "Aspect": "coverage"}}}}```
# 23. Promotion Inquiry: {{"Promotion": "...", "Aspect": "details/eligibility/..."}} 
#     Example: "Tell me about your Black Friday sale" → ```json{{"intent": "Promotion Inquiry", "entities": {{"Promotion": "Black Friday sale", "Aspect": "details"}}}}```
# 24. Complaint Escalation: {{"Issue": "...", "Department": "..."}} 
#     Example: "Escalate my billing issue to a manager" → ```json{{"intent": "Complaint Escalation", "entities": {{"Issue": "billing issue", "Department": "manager"}}}}```
# 25. Product Return: {{"Product": "...", "Reason": "..."}} 
#     Example: "I want to return my headphones" → ```json{{"intent": "Product Return", "entities": {{"Product": "headphones", "Reason": null}}}}```
# 26. Customer Support Request: {{"Issue": "...", "Urgency": "low/medium/high"}} 
#     Example: "Need urgent help with my account" → ```json{{"intent": "Customer Support Request", "entities": {{"Issue": "account", "Urgency": "high"}}}}```
# 27. Feature Request: {{"Feature": "...", "Product": "..."}} 
#     Example: "Can you add dark mode to the app?" → ```json{{"intent": "Feature Request", "entities": {{"Feature": "dark mode", "Product": "app"}}}}```
# 28. Billing Inquiry: {{"BillID": "...", "Aspect": "amount/due date/..."}} 
#     Example: "Why is my bill so high?" → ```json{{"intent": "Billing Inquiry", "entities": {{"BillID": null, "Aspect": "amount"}}}}```
# 29. Account Verification: {{"Task": "...", "Method": "email/phone/..."}} 
#     Example: "Verify my account via email" → ```json{{"intent": "Account Verification", "entities": {{"Task": "verify account", "Method": "email"}}}}```
# 30. Feedback Request: {{"Topic": "..."}} 
#     Example: "Can you send me a feedback form?" → ```json{{"intent": "Feedback Request", "entities": {{"Topic": "feedback form"}}}}```
# 31. Order Cancellation: {{"OrderID": "..."}} 
#     Example: "Cancel my order #789" → ```json{{"intent": "Order Cancellation", "entities": {{"OrderID": "789"}}}}```
# 32. Loyalty Program Inquiry: {{"Program": "...", "Aspect": "points/benefits/..."}} 
#     Example: "How many loyalty points do I have?" → ```json{{"intent": "Loyalty Program Inquiry", "entities": {{"Program": "loyalty program", "Aspect": "points"}}}}```
# 33. Event Registration: {{"Event": "...", "Quantity": "..."}} 
#     Example: "Register me for the workshop" → ```json{{"intent": "Event Registration", "entities": {{"Event": "workshop", "Quantity": "one"}}}}```
# 34. Technical Inquiry: {{"Topic": "...", "Product": "..."}} 
#     Example: "How does the cloud storage work?" → ```json{{"intent": "Technical Inquiry", "entities": {{"Topic": "cloud storage", "Product": null}}}}```
# 35. Shipping Address Update: {{"OrderID": "...", "NewAddress": "..."}} 
#     Example: "Change my shipping address for order #101" → ```json{{"intent": "Shipping Address Update", "entities": {{"OrderID": "101", "NewAddress": null}}}}```
# 36. Payment Method Update: {{"Method": "...", "Action": "add/remove/update"}} 
#     Example: "Add a new credit card" → ```json{{"intent": "Payment Method Update", "entities": {{"Method": "credit card", "Action": "add"}}}}```
# 37. Subscription Upgrade: {{"Service": "...", "Plan": "..."}} 
#     Example: "Upgrade my plan to premium" → ```json{{"intent": "Subscription Upgrade", "entities": {{"Service": "subscription", "Plan": "premium"}}}}```
# 38. Product Availability: {{"Product": "...", "Location": "..."}} 
#     Example: "Is the iPhone 14 available in Delhi?" → ```json{{"intent": "Product Availability", "entities": {{"Product": "iPhone 14", "Location": "Delhi"}}}}```
# 39. Installation Request: {{"Product": "...", "Time": "..."}} 
#     Example: "Install my AC tomorrow" → ```json{{"intent": "Installation Request", "entities": {{"Product": "AC", "Time": "tomorrow"}}}}```
# 40. Service Cancellation: {{"Service": "..."}} 
#     Example: "Cancel my internet service" → ```json{{"intent": "Service Cancellation", "entities": {{"Service": "internet service"}}}}```
# 41. Account Deletion: {{"AccountType": "..."}} 
#     Example: "Delete my profile" → ```json{{"intent": "Account Deletion", "entities": {{"AccountType": "profile"}}}}```
# 42. Coupon Inquiry: {{"CouponCode": "...", "Aspect": "validity/discount/..."}} 
#     Example: "Is coupon XYZ123 valid?" → ```json{{"intent": "Coupon Inquiry", "entities": {{"CouponCode": "XYZ123", "Aspect": "validity"}}}}```
# 43. Product Review Request: {{"Product": "..."}} 
#     Example: "Can I review the headphones?" → ```json{{"intent": "Product Review Request", "entities": {{"Product": "headphones"}}}}```
# 44. Referral Inquiry: {{"Program": "...", "Aspect": "details/rewards/..."}} 
#     Example: "Tell me about your referral program" → ```json{{"intent": "Referral Inquiry", "entities": {{"Program": "referral program", "Aspect": "details"}}}}```
# 45. Gift Card Inquiry: {{"GiftCardID": "...", "Aspect": "balance/validity/..."}} 
#     Example: "Check my gift card balance" → ```json{{"intent": "Gift Card Inquiry", "entities": {{"GiftCardID": null, "Aspect": "balance"}}}}```
# 46. Survey Participation: {{"Survey": "..."}} 
#     Example: "I want to join the customer survey" → ```json{{"intent": "Survey Participation", "entities": {{"Survey": "customer survey"}}}}```
# 47. Job Inquiry: {{"Role": "...", "Location": "..."}} 
#     Example: "Any job openings in Mumbai?" → ```json{{"intent": "Job Inquiry", "entities": {{"Role": null, "Location": "Mumbai"}}}}```
# 48. Partnership Inquiry: {{"Type": "...", "Company": "..."}} 
#     Example: "Interested in a business partnership" → ```json{{"intent": "Partnership Inquiry", "entities": {{"Type": "business", "Company": null}}}}```
# 49. Donation Request: {{"Cause": "...", "Amount": "..."}} 
#     Example: "How can I donate to charity?" → ```json{{"intent": "Donation Request", "entities": {{"Cause": "charity", "Amount": null}}}}```
# 50. Social Media Inquiry: {{"Platform": "...", "Action": "follow/share/..."}} 
#     Example: "Follow your Instagram page" → ```json{{"intent": "Social Media Inquiry", "entities": {{"Platform": "Instagram", "Action": "follow"}}}}```

# Message: {message}
# Output:
# """
#     try:
#         # Prepare the content as a Value proto message
#         content = Value()
#         content.string_value = prompt
#         instance = {"content": content}
        
#         logger.debug(f"Sending request to Vertex AI for message: {message}")
#         # Make the prediction request
#         response = client.predict(
#             endpoint=endpoint,
#             instances=[instance]
#         )
        
#         # Extract the response content
#         predictions = response.predictions
#         if predictions:
#             content = MessageToDict(predictions[0]).get('content', '')
#             logger.debug(f"Raw response content: {content}")
#             json_content = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
#             if json_content:
#                 json_content = json_content[0].strip()
#                 logger.debug(f"Extracted JSON: {json_content}")
#                 return json.loads(json_content)
#             logger.error("No JSON block found in response")
#             return {"error": "No JSON block found", "raw_response": content}
#         logger.error("No predictions returned from Vertex AI")
#         return {"error": "No predictions returned", "raw_response": None}
#     except Exception as e:
#         logger.error(f"Model error: {str(e)}")
#         return {"error": f"Model error: {str(e)}", "raw_response": None}

# # Process message and add to results
# def process_message(message, client, endpoint, results_list):
#     """Process a single message and append results to the list."""
#     cleaned_message, language = preprocess_message(message)
#     result = classify_and_extract(cleaned_message, client, endpoint)
#     has_intent = result.get('intent') != 'Casual Chat'
#     results_list.append({
#         'original_message': message,
#         'cleaned_message': cleaned_message,
#         'language': language,
#         'has_intent': has_intent,
#         'intent': result.get('intent', 'Error'),
#         'entities': json.dumps(result.get('entities', None)),
#         'error': result.get('error', None)
#     })
#     # Add delay to avoid API rate limiting
#     time.sleep(2)
#     return result

# # Main execution
# if __name__ == "__main__":
#     # Vertex AI configuration
#     project_id = "sandbox-451305"
#     location = "us-central1"  # Use a supported region for Vertex AI
#     model_name = "gemini-2.5-flash"

#     # Initialize Vertex AI client
#     try:
#         client, endpoint = configure_vertex_ai(project_id, location, model_name)
#     except Exception as e:
#         logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
#         print(f"Failed to initialize Vertex AI client: {str(e)}")
#         exit()

#     # Read and process WhatsApp chat file
#     file_path = "C:/Gangadhar/Strawhat/Chat_Classification/updated_chat.txt"

#     if not Path(file_path).is_file():
#         logger.error(f"File not found: {file_path}")
#         print(f"File not found: {file_path}")
#         exit()

#     # Read and process WhatsApp chat line by line
#     results_list = []
#     with open(file_path, "r", encoding="utf-8") as file:
#         lines = file.readlines()

#     print("Processing messages...\n")
#     for line in lines:
#         # Skip empty lines, metadata, or media omitted
#         if not line.strip() or "omitted" in line.lower() or line.startswith("Near By Tech Parks"):
#             logger.debug(f"Skipping line: {line.strip()}")
#             continue

#         # Extract message content after timestamp if it exists
#         match = re.search(r"\]\s*(.*)", line)
#         message = match.group(1) if match else line.strip()

#         # Process each message
#         result = process_message(message, client, endpoint, results_list)

#         print(f"Original: {message}")
#         print(f"Result: {json.dumps(result, indent=2)}")
#         print("-" * 60)

#     # Create DataFrame and save to CSV locally
#     results_df = pd.DataFrame(results_list)
#     print("\nDataset Preview:")
#     print(results_df.head())
#     results_df.to_csv("processed_whatsapp_intents.csv", index=False)
#     print("\nSaved to processed_whatsapp_intents.csv")


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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Configure Vertex AI client
def configure_vertex_ai(project_id, location, model_name):
    """Initialize Vertex AI client for Gemini model."""
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)
        logger.info(f"Vertex AI Gemini model initialized: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
        raise

# Enhanced preprocessing
def preprocess_message(message):
    """Preprocess WhatsApp message with balanced cleaning."""
    logger.debug(f"Preprocessing message: {message}")
    # Remove sender prefixes (e.g., ~ Mayank Lalaiya:)
    message = re.sub(r'^~[^:]+:\s*', '', message)
    # Remove timestamps (e.g., [12/12/24, 10:00 AM])
    message = re.sub(r'\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\]\s*', '', message)
    # Remove phone numbers (e.g., +91 12345 67890)
    message = re.sub(r'\+\d{1,3}\s*\d{5}\s*\d{5}', '', message)
    # Remove URLs
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
    # Remove emojis (basic Unicode range)
    message = re.sub(r'[\U0001F600-\U0001F6FF\U0001F900-\U0001F9FF]', '', message)
    # Remove punctuation except apostrophes to preserve contractions like "what's"
    message = re.sub(r'[^\w\s\']', '', message)
    message = re.sub(r'\s+', ' ', message).strip().lower()
    # Tokenize, remove stopwords, and lemmatize, but preserve key phrases
    tokens = nltk.word_tokenize(message)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words or token in ['what', 'how', 'when']]
    cleaned_message = ' '.join(cleaned_tokens)
    # Detect language
    try:
        language = detect(cleaned_message) if cleaned_message else 'unknown'
    except LangDetectException:
        language = 'unknown'
    logger.debug(f"Cleaned message: {cleaned_message}, Language: {language}")
    return cleaned_message, language

# Intent classification and entity extraction
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
        # Generate content using the Gemini model
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

# Process message and add to results
def process_message(message, model, results_list):
    """Process a single message and append results to the list."""
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
    # Add delay to avoid API rate limiting
    time.sleep(2)
    return result

# Main execution
if __name__ == "__main__":
    # Vertex AI configuration
    project_id = "sandbox-451305"
    location = "us-central1"
    model_name = "gemini-2.5-flash"

    # Initialize Vertex AI client
    try:
        model = configure_vertex_ai(project_id, location, model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
        print(f"Failed to initialize Vertex AI client: {str(e)}")
        exit()

    # Read and process WhatsApp chat file
    file_path = "C:/Gangadhar/Strawhat/Chat_Classification/updated_chat.txt"

    if not Path(file_path).is_file():
        logger.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")
        exit()

    # Read and process WhatsApp chat line by line
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
        # Skip empty lines, metadata, group join messages, or deleted messages
        if not line.strip() or "omitted" in line.lower() or "joined using this group's invite link" in line or "This message was deleted" in line.lower():
            logger.debug(f"Skipping line: {line.strip()}")
            continue

        # Extract message content after timestamp or sender prefix if it exists
        match = re.search(r"\]\s*(.*)", line) or re.search(r"~[^:]+:\s*(.*)", line)
        message = match.group(1) if match else line.strip()

        # Process each message
        result = process_message(message, model, results_list)

        print(f"Original: {message}")
        print(f"Result: {json.dumps(result, indent=2)}")
        print("-" * 60)

    # Create DataFrame and save to CSV locally
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