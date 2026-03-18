import pandas as pd
import os
from utils import clean_stimulus

def load_participant_file(filepath: str) -> pd.DataFrame:
    """Load and clean a single participant CSV"""
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df.dropna(subset=['Stimulus', 'Transcription Rater 1'])
    df = df[df['Stimulus'].astype(str).str.strip() != '']
    df['Participant'] = os.path.basename(filepath).replace('.csv', '')
    df['Stimulus'] = df['Stimulus'].apply(clean_stimulus)
    return df

def load_all_files(filepaths: list) -> list:
    """Load all participant files"""
    return [load_participant_file(fp) for fp in filepaths]