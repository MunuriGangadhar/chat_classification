import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Imports
import re
import json
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import flax.linen as nn
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
import optax
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pathlib import Path
import pickle

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_message(message):
    if pd.isna(message) or not message:
        return "", "unknown"
    message = str(message)
    message = re.sub(r'\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\]\s*', '', message)
    message = re.sub(r'\+\d{1,3}\s*\d{5}\s*\d{5}', '', message)
    message = re.sub(r'http[s]?://\S+', '', message)
    message = re.sub(r'[\U0001F600-\U0001F6FF\U0001F900-\U0001F9FF]', '', message)
    message = re.sub(r'[^\w\s]', '', message)
    message = re.sub(r'\s+', ' ', message).strip().lower()
    tokens = nltk.word_tokenize(message)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    cleaned_message = ' '.join(tokens)
    try:
        language = detect(cleaned_message) if cleaned_message else 'unknown'
    except LangDetectException:
        language = 'unknown'
    return cleaned_message, language

# Flax model definition
class IntentClassifier(nn.Module):
    num_classes: int
    hidden_dim: int = 256

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_dim)
        self.dense2 = nn.Dense(self.hidden_dim // 2)
        self.dense_out = nn.Dense(self.num_classes)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = self.dense_out(x)
        return x

# Create train state
def create_train_state(rng, model, input_shape, learning_rate):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

# Compute accuracy
def compute_accuracy(logits, labels):
    pred_class = jnp.argmax(logits, axis=1)
    true_class = jnp.argmax(labels, axis=1)
    return jnp.mean(pred_class == true_class)

# Training step
def train_step(state, inputs, labels):
    apply_fn = state.apply_fn

    @jit
    def loss_fn(params):
        logits = apply_fn({'params': params}, inputs)
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = compute_accuracy(logits, labels)
    return state, loss, acc

# Prediction function
def predict(model, state, message, vectorizer, intent_to_index):
    cleaned_message, language = preprocess_message(message)
    if not cleaned_message:
        return {"intent": "Casual Chat", "entities": None}
    features = vectorizer.transform([cleaned_message]).toarray()
    features = jnp.array(features, dtype=jnp.float32)
    logits = model.apply({'params': state.params}, features)
    predicted_idx = int(jnp.argmax(logits, axis=1)[0])
    index_to_intent = {v: k for k, v in intent_to_index.items()}
    predicted_intent = index_to_intent[predicted_idx]
    return {"intent": predicted_intent, "entities": None}

# Save model
def save_model(state, vectorizer, intent_to_index, model_path="intent_model.bin", vectorizer_path="vectorizer.pkl"):
    with open(model_path, "wb") as f:
        f.write(to_bytes(state.params))
    with open(vectorizer_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "intent_to_index": intent_to_index}, f)
    print(f"Model saved to {model_path}, Vectorizer saved to {vectorizer_path}")

# Load model
def load_model(model, model_path="intent_model.bin", vectorizer_path="vectorizer.pkl"):
    with open(model_path, "rb") as f:
        params = from_bytes(model.init(random.PRNGKey(0), jnp.ones((1, 5000)))['params'], f.read())
    with open(vectorizer_path, "rb") as f:
        saved_data = pickle.load(f)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(0.0005))
    return state, saved_data["vectorizer"], saved_data["intent_to_index"]

# Train model
def train_model(messages, intents, epochs=100, batch_size=32, learning_rate=0.0005):
    valid_data = [(msg, intent) for msg, intent in zip(messages, intents) if pd.notna(msg) and msg.strip()]
    if not valid_data:
        raise ValueError("No valid messages found.")
    messages, intents = zip(*valid_data)

    unique_intents = sorted(set(intents))
    intent_to_index = {intent: idx for idx, intent in enumerate(unique_intents)}
    num_classes = len(unique_intents)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(messages).toarray()
    X = jnp.array(X, dtype=jnp.float32)

    y = jnp.array([intent_to_index[intent] for intent in intents])
    y_one_hot = jax.nn.one_hot(y, num_classes)

    X_train, _, y_train, _ = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    rng = random.PRNGKey(0)
    model = IntentClassifier(num_classes=num_classes, hidden_dim=256)
    input_shape = (1, X_train.shape[1])
    state = create_train_state(rng, model, input_shape, learning_rate)

    for epoch in range(epochs):
        epoch_losses = []
        epoch_accs = []
        for i in range(0, len(X_train), batch_size):
            batch_inputs = X_train[i:i+batch_size]
            batch_labels = y_train[i:i+batch_size]
            state, loss, acc = train_step(state, batch_inputs, batch_labels)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
        avg_acc = float(jnp.mean(jnp.array(epoch_accs)))
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    return model, state, vectorizer, intent_to_index

# Process test message
def process_message(message, model, state, vectorizer, intent_to_index, results_list):
    cleaned_message, language = preprocess_message(message)
    result = predict(model, state, cleaned_message, vectorizer, intent_to_index)
    has_intent = result['intent'] != 'Casual Chat'
    results_list.append({
        'original_message': message,
        'cleaned_message': cleaned_message,
        'language': language,
        'has_intent': has_intent,
        'intent': result['intent'],
        'entities': json.dumps(result['entities']),
        'error': None
    })
    return result

# Main execution
dataset_path = "/content/processed_whatsapp_intents_first.csv"
if not Path(dataset_path).is_file():
    print(f"File not found: {dataset_path}. Please upload it in the Files tab.")
else:
    df = pd.read_csv(dataset_path)
    if 'cleaned_message' not in df.columns or 'intent' not in df.columns:
        raise ValueError("Dataset must contain 'cleaned_message' and 'intent' columns.")
    df = df[df['error'].isna()]
    df = df[df['cleaned_message'].notna() & (df['cleaned_message'].str.strip() != '')]
    if df.empty:
        raise ValueError("No valid rows after filtering.")

    messages = df['cleaned_message'].tolist()
    intents = df['intent'].tolist()

    print("Training model...")
    model, state, vectorizer, intent_to_index = train_model(messages, intents, epochs=100, learning_rate=0.0005)

    save_model(state, vectorizer, intent_to_index)

    # Test messages
    test_messages = [
        "Looking for a 2BHK flat in Bellandur, available from November.",
        "Anyone hiring for a data scientist role in Bangalore?",
        "Need a good dermatologist in HSR Layout, any recommendations?",
        "Selling 2 tickets for a comedy show in Whitefield on October 15th.",
        "Hi, does anyone work at Microsoft? Looking to connect for a referral.",
        "Urgent: Need blood donors for a patient in Manipal Hospital.",
        "Hey all, just saying hi!",
        "Looking for a female roommate for a 3BHK in Koramangala."
    ]

    results_list = []
    print("\nTesting model...")
    for message in test_messages:
        result = process_message(message, model, state, vectorizer, intent_to_index, results_list)
        print(f"Message: {message}\nResult: {json.dumps(result, indent=2)}\n{'-'*50}")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("test_whatsapp_intents_output.csv", index=False)
    print("\nTest results saved to test_whatsapp_intents_output.csv")
