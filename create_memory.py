import pandas as pd
import glob
import os

DATA_PATH = r"C:\Users\Muhmmad shaban\Desktop\DATA"


def load_csv_files(data_path):
    # Create a list to hold all DataFrames
    all_dataframes = []
    
    # Use glob to find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    # Loop through the list of CSV files and read them into DataFrames
    for file in csv_files:
        # Check if the file is empty
        if os.path.getsize(file) > 0:
            try:
                # Read the CSV file and specify the columns to use
                df = pd.read_csv(file, usecols=["Question", "Answer"])
                all_dataframes.append(df)
            except pd.errors.EmptyDataError:
                print(f"Warning: The file {file} is empty and will be skipped.")
            except ValueError as ve:
                print(f"Warning: {ve} in file {file}. Ensure it has 'question' and 'answer' columns.")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"Warning: The file {file} is empty and will be skipped.")
    
    # Concatenate all DataFrames into a single DataFrame (optional)
    if all_dataframes:
        combined_dataframe = pd.concat(all_dataframes, ignore_index=True)
    else:
        combined_dataframe = pd.DataFrame(columns=["Question", "Answer"])  # Return an empty DataFrame with specified columns
    
    return combined_dataframe

# Example usage
documents = load_csv_files(DATA_PATH)
print(f"Loaded {len(documents)} documents.")
print(documents.head())  # Display the first few rows of the DataFrame
print(documents.isnull().sum())  # Display the columns of the DataFrame