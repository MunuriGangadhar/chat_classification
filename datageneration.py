# import json
# import pandas as pd
# from google.cloud import aiplatform
# import vertexai
# from vertexai.generative_models import GenerativeModel
# import time
# import logging
# import re

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def configure_vertex_ai(project_id, location, model_name):
#     try:
#         vertexai.init(project=project_id, location=location)
#         model = GenerativeModel(model_name)
#         logger.info(f"Vertex AI Gemini model initialized: {model_name}")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
#         raise

# def generate_whatsapp_messages(num_rows, model):
#     prompt = f"""
# You are a creative assistant tasked with generating realistic WhatsApp group chat messages in English, simulating informal, lively, and diverse conversations typical of group chats in India. The messages should feel natural, casual, and varied in tone, style, and content, resembling real-world queries. Each message must fall into one of three intent categories: `Person`, `Product`, or `Place`. You will also extract structured entities relevant to the intent.

# ### Your Task:
# 1. Generate **{num_rows} unique** WhatsApp-style messages in English, ensuring no duplicates.
# 2. Classify each message's intent as `Person`, `Product`, or `Place`.
# 3. Extract structured entities as key-value pairs relevant to the intent, focusing on details like location, product type, price, roles, skills, or purposes.
# 4. Ensure messages are informal, varied, and mimic real-world group chat scenarios, including colloquial language, abbreviations, slang, emojis, or casual phrasing (e.g., "guys", "pls", "any leads?", "what’s good?").
# 5. If some entity values are missing, set them to `null`.
# 6. Exclude purely conversational messages (e.g., "Hey, how’s it going?") or intents outside `Person`, `Product`, or `Place`.

# ### Guidelines for Diversity:
# - **Person**: Messages seeking professionals (e.g., tutors, designers, developers) with specific skills, locations, or contexts (e.g., events, projects). Use varied phrasing like "any leads on", "need someone ASAP", "who’s good at", or "know anyone who".
# - **Product**: Messages inquiring about products (e.g., phones, gadgets, appliances) with details like price, brand, features, or use cases. Include queries like "what’s the best deal", "any recos", "worth it?", or "where to buy".
# - **Place**: Messages asking for recommendations about locations (e.g., restaurants, cafes, parks) with specific purposes or characteristics. Use phrases like "where to chill", "good spot for", "any hidden gems", or "best place for".
# - Vary locations (e.g., Indian cities like Hyderabad, Mumbai, Delhi, or areas like Indranagar, Bandra, Koramangala), contexts, and details to ensure wide coverage.
# - Use different sentence structures, tones (e.g., urgent, curious, chill)
# - Reflect Indian group chat culture: include local references (e.g., "near MG Road", "for Diwali", "Navratri event"), casual abbreviations (e.g., "pls", "k"), and diverse scenarios (e.g., events, work, leisure, festivals).

# ### Expected Output Format:
# Return a JSON array of objects, each containing:
# ```json
# {{
# "original_message": "<Generated Message>",
# "intent": "<Person|Product|Place>",
# "entities": {{ ...key-value pairs or null... }}
# }}
# ```

# ### Examples:
# - Message: "Anyone know a good web developer in Hyderabad for a startup project?"
# Output: {{"original_message": "Anyone know a good web developer in Hyderabad for a startup project?", "intent": "Person", "entities": {{"Skill": "Web Development", "Location": "Hyderabad", "Tags": ["Startup"]}}}}
# - Message: "Need a dance choreographer in Chennai for a wedding sangeet next month!"
# Output: {{"original_message": "Need a dance choreographer in Chennai for a wedding sangeet next month!", "intent": "Person", "entities": {{"Skill": "Dance Choreographer", "Location": "Chennai", "Event": "Wedding Sangeet", "Timeframe": "Next Month"}}}}
# - Message: "Guys, any math tutor in Mumbai for 12th CBSE? Prefer online classes"
# Output: {{"original_message": "Guys, any math tutor in Mumbai for 12th CBSE? Prefer online classes", "intent": "Person", "entities": {{"Role": "Math Tutor", "Location": "Mumbai", "TargetAudience": "12th-grade CBSE", "Mode": "Online"}}}}
# - Message: "Need a freelance graphic designer in Bangalore for a logo ASAP!"
# Output: {{"original_message": "Need a freelance graphic designer in Bangalore for a logo ASAP!", "intent": "Person", "entities": {{"Skill": "Graphic Designer", "Location": "Bangalore", "ProjectType": "Logo", "EmploymentType": "Freelance"}}}}
# - Message: "Any leads on a yoga instructor in Delhi for morning sessions? "
# Output: {{"original_message": "Any leads on a yoga instructor in Delhi for morning sessions?", "intent": "Person", "entities": {{"Role": "Yoga Instructor", "Location": "Delhi", "Purpose": "Morning Sessions"}}}}
# - Message: "What’s the best Samsung phone under 10k? Need good battery life"
# Output: {{"original_message": "What’s the best Samsung phone under 10k? Need good battery life", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 10000", "Brand": "Samsung", "Feature": "Battery Life"}}}}
# - Message: "Any recos for a gaming phone under 15k with high refresh rate?"
# Output: {{"original_message": "Any recos for a gaming phone under 15k with high refresh rate?", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Features": ["High Refresh Rate", "Gaming"]}}}}
# - Message: "Guys, where to buy a budget smartwatch under 5k with fitness tracking?"
# Output: {{"original_message": "Guys, where to buy a budget smartwatch under 5k with fitness tracking?", "intent": "Product", "entities": {{"Product": "Smartwatch", "Price": "Rs. 5000", "Feature": "Fitness Tracking"}}}}
# - Message: "Need a good water purifier for home in Pune, any suggestions?"
# Output: {{"original_message": "Need a good water purifier for home in Pune, any suggestions?", "intent": "Product", "entities": {{"Product": "Water Purifier", "UseCase": "Home Use", "Location": "Pune"}}}}
# - Message: "Any drone under 20k with a decent camera for vlogging?"
# Output: {{"original_message": "Any drone under 20k with a decent camera for vlogging? 📸", "intent": "Product", "entities": {{"Product": "Drone", "Price": "Rs. 20000", "Feature": "Camera", "UseCase": "Vlogging"}}}}
# - Message: "Yo, where’s the best biryani spot in Indranagar? Craving some"
# Output: {{"original_message": "Yo, where’s the best biryani spot in Indranagar? Craving some", "intent": "Place", "entities": {{"Location": "Indranagar, Bangalore", "Type": "Restaurant", "Tags": ["Biryani"]}}}}
# - Message: "Any chill cafes in Bandra West for working on my laptop?"
# Output: {{"original_message": "Any chill cafes in Bandra West for working on my laptop?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Bandra West", "Purpose": "Work"}}}}
# - Message: "Guys, know any quiet parks in Delhi for evening walks?"
# Output: {{"original_message": "Guys, know any quiet parks in Delhi for evening walks?", "intent": "Place", "entities": {{"Type": "Park", "Location": "Delhi", "Purpose": "Evening Walks"}}}}
# - Message: "Any good rooftop restaurants in Koramangala for a date night?"
# Output: {{"original_message": "Any good rooftop restaurants in Koramangala for a date night?", "intent": "Place", "entities": {{"Type": "Restaurant", "Location": "Koramangala, Bangalore", "Purpose": "Date Night"}}}}
# - Message: "Need a karaoke spot in Hyderabad for a bday party, budget 3k!"
# Output: {{"original_message": "Need a karaoke spot in Hyderabad for a bday party, budget 3k! ", "intent": "Place", "entities": {{"Type": "Karaoke Lounge", "Location": "Hyderabad", "Purpose": "Birthday Party", "Price": "Rs. 3000"}}}}

# ### Output:
# Generate exactly {num_rows} unique messages in a single JSON array:
# ```json
# [
# {{"original_message": "...", "intent": "...", "entities": {{...}}}},
# ...
# ]
# ```
# """

#     results_list = []
#     generated_messages = set()

#     while len(results_list) < num_rows:
#         try:
#             logger.debug("Sending request to Vertex AI for message generation")
#             response = model.generate_content(prompt)
#             content = response.text.strip()
#             logger.debug(f"Raw response content: {content}")

#             # Extract JSON from response
#             json_content = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
#             if not json_content:
#                 logger.error("No JSON block found in response")
#                 continue

#             json_content = json_content[0].strip()
#             try:
#                 generated_data = json.loads(json_content)
#             except json.JSONDecodeError as e:
#                 logger.error(f"Failed to parse JSON response: {str(e)}")
#                 continue

#             # Filter unique and valid messages
#             for item in generated_data:
#                 message = item.get("original_message")
#                 if not message or message in generated_messages:
#                     continue
#                 if item.get("intent") not in ["Person", "Product", "Place"]:
#                     continue
#                 generated_messages.add(message)
#                 results_list.append({
#                     "original_message": message,
#                     "intent": item.get("intent"),
#                     "entities": json.dumps(item.get("entities", None))
#                 })
#                 if len(results_list) >= num_rows:
#                     break

#             # Avoid overwhelming the API
#             time.sleep(2)

#         except Exception as e:
#             logger.error(f"Error generating messages: {str(e)}")
#             time.sleep(2)

#     return results_list[:num_rows]

# if __name__ == "__main__":
#     # Vertex AI configuration
#     project_id = "sandbox-451305"
#     location = "us-central1"
#     model_name = "gemini-2.5-flash"

#     # Initialize Vertex AI client
#     try:
#         model = configure_vertex_ai(project_id, location, model_name)
#     except Exception as e:
#         logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
#         print(f"Failed to initialize Vertex AI client: {str(e)}")
#         exit()

#     # Specify the number of rows to generate
#     num_rows = 30  # Adjust based on your requirement

#     # Generate unique messages using Vertex AI
#     results_list = generate_whatsapp_messages(num_rows, model)

#     # Create DataFrame and save to CSV
#     try:
#         results_df = pd.DataFrame(results_list)
#         print("\nGenerated Messages Preview:")
#         print(results_df.head())
#         results_df.to_csv("generated_whatsapp_messages_vertex.csv", index=False)
#         print("\nSaved to generated_whatsapp_messages_vertex.csv")
#     except Exception as e:
#         logger.error(f"Failed to save CSV: {str(e)}")
#         print(f"Failed to save CSV: {str(e)}")
#         exit()


import json
import pandas as pd
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import time
import logging
import re
import random
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_prompt(batch_size, batch_num, seed):
    """
    Build a dynamic prompt with randomness, variety hints, and all provided examples to reduce duplicates.
    """
    return f"""
You are a creative assistant tasked with generating realistic WhatsApp group chat messages in English, simulating informal, lively, and diverse conversations typical of group chats in India. The messages should feel natural, casual, and varied in tone, style, and content, resembling real-world queries. Each message must fall into one of three intent categories: `Person`, `Product`, or `Place`. You will also extract structured entities relevant to the intent.

### Your Task:
1. Generate **{batch_size} unique** WhatsApp-style messages in English, ensuring no duplicates.
2. Classify each message's intent as `Person`, `Product`, or `Place`.
3. Extract structured entities as key-value pairs relevant to the intent, focusing on details like location, product type, price, roles, skills, or purposes.
4. Ensure messages are informal, varied, and mimic real-world group chat scenarios, including colloquial language, abbreviations, slang, emojis, or casual phrasing (e.g., "guys", "pls", "any leads?", "what’s good?").
5. If some entity values are missing, set them to `null`.
6. Exclude purely conversational messages (e.g., "Hey, how’s it going?") or intents outside `Person`, `Product`, or `Place`.

### Guidelines for Diversity:
- **Person**: Messages seeking professionals (e.g., tutors, designers, developers) with specific skills, locations, or contexts (e.g., events, projects). Use varied phrasing like "any leads on", "need someone ASAP", "who’s good at", or "know anyone who".
- **Product**: Messages inquiring about products (e.g., phones, gadgets, appliances) with details like price, brand, features, or use cases. Include queries like "what’s the best deal", "any recos", "worth it?", or "where to buy".
- **Place**: Messages asking for recommendations about locations (e.g., restaurants, cafes, parks) with specific purposes or characteristics. Use phrases like "where to chill", "good spot for", "any hidden gems", or "best place for".
- Vary locations (e.g., Indian cities like Hyderabad, Mumbai, Delhi, or areas like Indranagar, Bandra, Koramangala), contexts, and details to ensure wide coverage.
- Use different sentence structures, tones (e.g., urgent, curious, chill).
- Reflect Indian group chat culture: include local references (e.g., "near MG Road", "for Diwali", "Navratri event"), casual abbreviations (e.g., "pls", "k"), and diverse scenarios (e.g., events, work, leisure, festivals).
- **Randomization**: Use seed {seed} to make this generation different from previous ones. Avoid repeating messages or styles from earlier batches. Include fresh topics, styles, and contexts for batch {batch_num}.

### Expected Output Format:
Return a JSON array of objects, each containing:
```json
{{
"original_message": "<Generated Message>",
"intent": "<Person|Product|Place>",
"entities": {{ ...key-value pairs or null... }}
}}
```

### Examples:
- {{"original_message": "Anyone know a good web developer in Hyderabad for a startup project?", "intent": "Person", "entities": {{"Skill": "Web Development", "Location": "Hyderabad", "Tags": ["Startup"]}}}}
- {{"original_message": "Need a dance choreographer in Chennai for a wedding event next month", "intent": "Person", "entities": {{"Skill": "Dance Choreographer", "Location": "Chennai", "Event": "Wedding", "Timeframe": "Next Month"}}}}
- {{"original_message": "Looking for a math tutor in Mumbai who can teach 12th-grade CBSE online", "intent": "Person", "entities": {{"Role": "Math Tutor", "Location": "Mumbai", "TargetAudience": "12th-grade CBSE", "Mode": "Online"}}}}
- {{"original_message": "Need a UX researcher in Bangalore with experience in mobile apps", "intent": "Person", "entities": {{"Skill": "UX Researcher", "Location": "Bangalore", "Expertise": "Mobile Apps", "Experience": "Any"}}}}
- {{"original_message": "Any experienced interior designer in Delhi for home renovations?", "intent": "Person", "entities": {{"Skill": "Interior Design", "Location": "Delhi", "Tags": ["Home Renovations"]}}}}
- {{"original_message": "Know a machine learning expert in Patna for a 6-month contract?", "intent": "Person", "entities": {{"Skill": "Machine Learning", "Location": "Patna", "Tags": ["Contract", "6-month"]}}}}
- {{"original_message": "Seeking a digital marketing specialist in Vizag with SEO expertise", "intent": "Person", "entities": {{"Role": "Digital Marketing Specialist", "Location": "Vizag", "Expertise": "SEO", "Availability": null}}}}
- {{"original_message": "Searching for a physics tutor in Gurgaon for high school IB students", "intent": "Person", "entities": {{"Role": "Physics Tutor", "Location": "Gurgaon", "TargetAudience": "High School IB", "Mode": "Hybrid"}}}}
- {{"original_message": "Hunting for a freelance animator in Chennai for short ad films", "intent": "Person", "entities": {{"Role": "Animator", "Location": "Chennai", "ProjectType": "Ad Films", "EmploymentType": "Freelance"}}}}
- {{"original_message": "Find me a history tutor in Mumbai for IGCSE curriculum", "intent": "Person", "entities": {{"Role": "History Tutor", "Location": "Mumbai", "Tags": ["IGCSE Students", "Online"]}}}}
- {{"original_message": "What’s the best Samsung mobile under 10K?", "intent": "Product", "entities": {{"Product": "Mobile", "Price": "Rs. 10000", "Brand": "Samsung", "Details": null}}}}
- {{"original_message": "Suggest a phone under 15k with a good camera and battery life.", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Tags": ["Camera", "Battery Life"]}}}}
- {{"original_message": "Which Samsung phone has the best camera for vlogging?", "intent": "Product", "entities": {{"Product": "Phone", "Brand": "Samsung", "Feature": "Camera", "UseCase": "Vlogging"}}}}
- {{"original_message": "Can anyone suggest a gaming phone under 15k with a high refresh rate?", "intent": "Product", "entities": {{"Product": "Phone", "Price": "Rs. 15000", "Features": ["High Refresh Rate", "Gaming"]}}}}
- {{"original_message": "Searching for a projector for classroom presentations?", "intent": "Product", "entities": {{"Product": "Projector", "UseCase": "Classroom Presentations", "Location": null}}}}
- {{"original_message": "What’s a good fitness tracker under 4k for running in Pune?", "intent": "Product", "entities": {{"Product": "Fitness Tracker", "Price": "Rs. 4000", "Purpose": "Running", "Location": "Pune"}}}}
- {{"original_message": "Can anyone recommend a smart doorbell under 8k with Wi-Fi connectivity?", "intent": "Product", "entities": {{"Product": "Smart Doorbell", "Price": "Rs. 8000", "Connectivity": "Wi-Fi"}}}}
- {{"original_message": "Searching for a Xiaomi phone with 5G in Hyderabad?", "intent": "Product", "entities": {{"Product": "Phone", "Brand": "Xiaomi", "Connectivity": "5G", "Location": "Hyderabad"}}}}
- {{"original_message": "Guys, what’s a good smartwatch under 6k with heart rate monitoring?", "intent": "Product", "entities": {{"Product": "Smartwatch", "Price": "Rs. 6000", "Feature": "Heart Rate Monitoring"}}}}
- {{"original_message": "Searching for a water purifier for home use in Delhi?", "intent": "Product", "entities": {{"Product": "Water Purifier", "UseCase": "Home Use", "Location": "Delhi"}}}}
- {{"original_message": "Anyone got ideas for a drone under 30k with 4K camera?", "intent": "Product", "entities": {{"Product": "Drone", "Price": "Rs. 30000", "Feature": "4K Camera"}}}}
- {{"original_message": "Got any leads on a coffee grinder under 8k for espresso?", "intent": "Product", "entities": {{"Product": "Coffee Grinder", "Price": "Rs. 8000", "Tags": ["Espresso"]}}}}
- {{"original_message": "What’s the best Biryani place in Indranagar?", "intent": "Place", "entities": {{"Location": "Indranagar, Bangalore", "Type": "Restaurant", "Tags": ["Biryani"]}}}}
- {{"original_message": "Any nice cafes in Bandra West where I can work?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Bandra West", "Purpose": "Work"}}}}
- {{"original_message": "Can anyone share a peaceful park in Chennai for morning walks?", "intent": "Place", "entities": {{"Type": "Park", "Location": "Chennai", "Purpose": "Morning Walks"}}}}
- {{"original_message": "Searching for a quiet tea house in Pune for afternoon chats?", "intent": "Place", "entities": {{"Type": "Tea House", "Location": "Pune", "Purpose": "Afternoon Chats", "Ambiance": "Quiet"}}}}
- {{"original_message": "Can you guys point to a karaoke lounge in Hyderabad under 2k for parties?", "intent": "Place", "entities": {{"Type": "Karaoke Lounge", "Location": "Hyderabad", "Price": "Rs. 2000", "Purpose": "Parties"}}}}
- {{"original_message": "Know a nice juice bar in Delhi for fresh drinks?", "intent": "Place", "entities": {{"Type": "Juice Bar", "Location": "Delhi", "Tags": ["Fresh Drinks"]}}}}
- {{"original_message": "Working from Andheri tomorrow, any laptop-friendly coffee spots?", "intent": "Place", "entities": {{"Type": "Cafe", "Location": "Andheri", "Purpose": "Remote Work"}}}}
- {{"original_message": "Where do you guys buy plants in Mumbai? Any aesthetic nursery?", "intent": "Place", "entities": {{"Type": "Plant Nursery", "Location": "Mumbai", "Tags": ["Aesthetic"]}}}}
- {{"original_message": "Need midnight maggi places near MG Road", "intent": "Place", "entities": {{"Type": "Late Night Eatery", "Location": "MG Road", "Tags": ["Maggi", "Midnight Food"]}}}}
- {{"original_message": "Best salon for men’s haircut in Rajajinagar?", "intent": "Place", "entities": {{"Type": "Salon", "Location": "Rajajinagar", "Tags": ["Men", "Haircut"]}}}}

### Output:
Generate exactly {batch_size} unique messages in a single JSON array:
```json
[
{{"original_message": "...", "intent": "...", "entities": {{...}}}},
...
]
```
"""

def configure_vertex_ai(project_id, location, model_name):
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel(model_name)
        logger.info(f"Vertex AI Gemini model initialized: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
        raise

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_whatsapp_messages_batch(batch_size, batch_num, model):
    """
    Generate a batch of WhatsApp messages with retry logic.
    """
    seed = random.randint(1, 999999)
    prompt = build_prompt(batch_size, batch_num, seed)
    generated_messages = set()
    results_list = []

    try:
        logger.debug(f"Sending request to Vertex AI for batch {batch_num} with seed {seed}")
        response = model.generate_content(prompt)
        content = response.text.strip()
        logger.debug(f"Raw response content for batch {batch_num}: {content}")

        # Extract JSON from response
        json_content = re.findall(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if not json_content:
            logger.error(f"No JSON block found in response for batch {batch_num}")
            return []

        json_content = json_content[0].strip()
        try:
            generated_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for batch {batch_num}: {str(e)}")
            return []

        # Filter unique and valid messages
        for item in generated_data:
            message = item.get("original_message")
            if not message or message in generated_messages:
                continue
            if item.get("intent") not in ["Person", "Product", "Place"]:
                continue
            generated_messages.add(message)
            results_list.append({
                "original_message": message,
                "intent": item.get("intent"),
                "entities": json.dumps(item.get("entities", None))
            })

        logger.info(f"Generated {len(results_list)} unique messages in batch {batch_num}")
        return results_list

    except Exception as e:
        logger.error(f"Error generating messages for batch {batch_num}: {str(e)}")
        return []

def generate_whatsapp_messages(total_rows, batch_size, model):
    all_results = []
    all_messages = set()
    batch_num = 0

    while len(all_results) < total_rows:
        batch_num += 1
        remaining_rows = total_rows - len(all_results)
        current_batch_size = min(batch_size, remaining_rows)

        batch_results = generate_whatsapp_messages_batch(current_batch_size, batch_num, model)
        for result in batch_results:
            message = result["original_message"]
            if message not in all_messages:
                all_messages.add(message)
                all_results.append(result)

        logger.info(f"Total unique messages so far: {len(all_results)}")
        time.sleep(2)  
        if len(all_results) % 1000 == 0 and all_results:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(f"temp_whatsapp_messages_{len(all_results)}.csv", index=False)
            logger.info(f"Saved intermediate results to temp_whatsapp_messages_{len(all_results)}.csv")

    return all_results[:total_rows]

if __name__ == "__main__":
    project_id = "sandbox-451305"
    location = "us-central1"
    model_name = "gemini-2.5-flash"
    total_rows = 10000
    batch_size = 25 
    try:
        model = configure_vertex_ai(project_id, location, model_name)
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
        print(f"Failed to initialize Vertex AI client: {str(e)}")
        exit()
    try:
        results_list = generate_whatsapp_messages(total_rows, batch_size, model)
    except Exception as e:
        logger.error(f"Failed to generate messages: {str(e)}")
        print(f"Failed to generate messages: {str(e)}")
        exit()

    # Create DataFrame and save to CSV
    try:
        results_df = pd.DataFrame(results_list)
        print("\nGenerated Messages Preview:")
        print(results_df.head())
        results_df.to_csv("generated_whatsapp_messages_vertex_10000.csv", index=False)
        print(f"\nSaved {len(results_df)} messages to generated_whatsapp_messages_vertex.csv")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        print(f"Failed to save CSV: {str(e)}")
        exit()




