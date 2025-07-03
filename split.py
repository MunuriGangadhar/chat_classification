import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(file_path: str, train_path: str, val_path: str, test_path: str, 
                 test_size: float = 0.15, val_size: float = 0.15, random_seed: int = 42):
    """
    Split the dataset into training, validation, and test sets with stratified sampling.
    
    Args:
        file_path (str): Path to the input CSV file.
        train_path (str): Path to save the training set CSV.
        val_path (str): Path to save the validation set CSV.
        test_path (str): Path to save the test set CSV.
        test_size (float): Proportion of the dataset to use for testing.
        val_size (float): Proportion of the dataset to use for validation.
        random_seed (int): Random seed for reproducibility.
    """
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {file_path} not found.")
    
    # Verify required columns
    if 'cleaned_message' not in df.columns or 'intent' not in df.columns:
        raise ValueError("CSV must contain 'cleaned_message' and 'intent' columns")
    
    # Handle any missing or invalid entries
    df = df.dropna(subset=['cleaned_message', 'intent'])
    df = df[df['cleaned_message'].str.strip() != '']

    # Remove classes with fewer than 2 samples
    df = df.groupby('intent').filter(lambda x: len(x) >= 2)

    # Print remaining class distribution (optional)
    print("Remaining class distribution after filtering:")
    print(df['intent'].value_counts())

    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=df['intent']
    )
    
    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size_adjusted, 
        random_state=random_seed, 
        stratify=train_val_df['intent']
    )
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train set ({len(train_df)} samples) to {train_path}")
    print(f"Saved validation set ({len(val_df)} samples) to {val_path}")
    print(f"Saved test set ({len(test_df)} samples) to {test_path}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    try:
        split_dataset(
            file_path="C:/Gangadhar/Strawhat/Chat_Classification/processed_whatsapp_intents.csv",
            train_path="train_whatsapp_intents.csv",
            val_path="val_whatsapp_intents.csv",
            test_path="test_whatsapp_intents.csv"
        )
    except Exception as e:
        print(f"Error: {str(e)}")
