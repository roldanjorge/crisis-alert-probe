"""
Clean Dataset Module

This module provides functionality to load datasets with pandas and remove duplicated utterances.
Specifically designed for text datasets containing conversational or message data.
"""

import pandas as pd
import os
from typing import Optional, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetCleaner:
    """
    A class to handle dataset cleaning operations, specifically focused on removing duplicated utterances.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DatasetCleaner with a path to the dataset.
        
        Args:
            data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.original_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        
    def load_dataset(self, file_format: str = 'auto') -> pd.DataFrame:
        """
        Load the dataset from the specified path.
        
        Args:
            file_format (str): Format of the file ('csv', 'pkl', 'json', or 'auto' for auto-detection)
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        # Auto-detect file format if not specified
        if file_format == 'auto':
            file_format = self.data_path.split('.')[-1].lower()
        
        logger.info(f"Loading dataset from {self.data_path} (format: {file_format})")
        
        try:
            if file_format == 'csv':
                self.original_df = pd.read_csv(self.data_path)
            elif file_format == 'pkl':
                self.original_df = pd.read_pickle(self.data_path)
            elif file_format == 'json':
                self.original_df = pd.read_json(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            logger.info(f"Successfully loaded dataset with {len(self.original_df)} rows and {len(self.original_df.columns)} columns")
            return self.original_df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def analyze_duplicates(self, text_column: str = 'message') -> Dict[str, Any]:
        """
        Analyze duplicate utterances in the dataset.
        
        Args:
            text_column (str): Name of the column containing the text/utterances
            
        Returns:
            Dict[str, Any]: Dictionary containing duplicate analysis results
        """
        if self.original_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if text_column not in self.original_df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {list(self.original_df.columns)}")
        
        total_rows = len(self.original_df)
        unique_utterances = self.original_df[text_column].nunique()
        duplicate_count = self.original_df[text_column].duplicated().sum()
        
        # Find the most duplicated utterances
        utterance_counts = self.original_df[text_column].value_counts()
        most_duplicated = utterance_counts.head(10).to_dict()
        
        analysis = {
            'total_rows': total_rows,
            'unique_utterances': unique_utterances,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': (duplicate_count / total_rows) * 100,
            'most_duplicated_utterances': most_duplicated,
            'text_column': text_column
        }
        
        return analysis
    
    def remove_duplicates(self, 
                         text_column: str = 'message', 
                         keep: str = 'first',
                         subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicated utterances from the dataset.
        
        Args:
            text_column (str): Name of the column containing the text/utterances
            keep (str): Which duplicate to keep ('first', 'last', or False to drop all duplicates)
            subset (Optional[list]): List of column names to consider for duplicate detection
            
        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        if self.original_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if text_column not in self.original_df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {list(self.original_df.columns)}")
        
        logger.info(f"Removing duplicates from column '{text_column}' (keep='{keep}')")
        
        # Use the specified subset or default to the text column
        if subset is None:
            subset = [text_column]
        
        # Remove duplicates
        self.cleaned_df = self.original_df.drop_duplicates(subset=subset, keep=keep)
        
        removed_count = len(self.original_df) - len(self.cleaned_df)
        logger.info(f"Removed {removed_count} duplicate rows. Dataset size: {len(self.original_df)} -> {len(self.cleaned_df)}")
        
        return self.cleaned_df
    
    def get_cleaned_dataset(self) -> pd.DataFrame:
        """
        Get the cleaned dataset.
        
        Returns:
            pd.DataFrame: The cleaned dataset
            
        Raises:
            ValueError: If the dataset hasn't been cleaned yet
        """
        if self.cleaned_df is None:
            raise ValueError("Dataset not cleaned yet. Call remove_duplicates() first.")
        return self.cleaned_df
    
    def save_cleaned_dataset(self, output_path: str, file_format: str = 'auto') -> None:
        """
        Save the cleaned dataset to a new file.
        
        Args:
            output_path (str): Path where to save the cleaned dataset
            file_format (str): Format of the output file ('csv', 'pkl', 'json', or 'auto' for auto-detection)
        """
        if self.cleaned_df is None:
            raise ValueError("Dataset not cleaned yet. Call remove_duplicates() first.")
        
        # Auto-detect file format if not specified
        if file_format == 'auto':
            file_format = output_path.split('.')[-1].lower()
        
        logger.info(f"Saving cleaned dataset to {output_path} (format: {file_format})")
        
        try:
            if file_format == 'csv':
                self.cleaned_df.to_csv(output_path, index=False)
            elif file_format == 'pkl':
                self.cleaned_df.to_pickle(output_path)
            elif file_format == 'json':
                self.cleaned_df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            logger.info(f"Successfully saved cleaned dataset with {len(self.cleaned_df)} rows")
            
        except Exception as e:
            logger.error(f"Error saving cleaned dataset: {e}")
            raise
    
    def print_summary(self, text_column: str = 'message') -> None:
        """
        Print a summary of the cleaning process.
        
        Args:
            text_column (str): Name of the column containing the text/utterances
        """
        if self.original_df is None:
            print("No dataset loaded.")
            return
        
        analysis = self.analyze_duplicates(text_column)
        
        print("\n" + "="*60)
        print("DATASET CLEANING SUMMARY")
        print("="*60)
        print(f"Original dataset: {analysis['total_rows']:,} rows")
        print(f"Unique utterances: {analysis['unique_utterances']:,}")
        print(f"Duplicate utterances: {analysis['duplicate_count']:,} ({analysis['duplicate_percentage']:.1f}%)")
        
        if self.cleaned_df is not None:
            print(f"Cleaned dataset: {len(self.cleaned_df):,} rows")
            print(f"Rows removed: {analysis['total_rows'] - len(self.cleaned_df):,}")
            print(f"Reduction: {((analysis['total_rows'] - len(self.cleaned_df)) / analysis['total_rows'] * 100):.1f}%")
        
        print(f"\nMost duplicated utterances:")
        for utterance, count in list(analysis['most_duplicated_utterances'].items())[:5]:
            print(f"  '{utterance[:50]}{'...' if len(utterance) > 50 else ''}' - {count} times")
        
        print("="*60)


def clean_dataset(data_path: str, 
                 output_path: Optional[str] = None,
                 text_column: str = 'message',
                 keep: str = 'first') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to clean a dataset in one step.
    
    Args:
        data_path (str): Path to the input dataset
        output_path (Optional[str]): Path to save the cleaned dataset (optional)
        text_column (str): Name of the column containing the text/utterances
        keep (str): Which duplicate to keep ('first', 'last', or False to drop all duplicates)
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataset and analysis results
    """
    cleaner = DatasetCleaner(data_path)
    
    # Load dataset
    cleaner.load_dataset()
    
    # Analyze duplicates
    analysis = cleaner.analyze_duplicates(text_column)
    
    # Remove duplicates
    cleaned_df = cleaner.remove_duplicates(text_column, keep)
    
    # Save if output path provided
    if output_path:
        cleaner.save_cleaned_dataset(output_path)
    
    # Print summary
    cleaner.print_summary(text_column)
    
    return cleaned_df, analysis


if __name__ == "__main__":
    # Example usage
    data_path = "/teamspace/studios/this_studio/mech_interp_exploration/data/genai_test_dataset_100000.csv"
    
    if os.path.exists(data_path):
        print("Running dataset cleaning example...")
        cleaned_df, analysis = clean_dataset(
            data_path=data_path,
            output_path="/teamspace/studios/this_studio/mech_interp_exploration/data/genai_test_dataset_cleaned.csv",
            text_column='message',
            keep='first'
        )
        
        print(f"\nCleaning completed!")
        print(f"Original size: {analysis['total_rows']:,} rows")
        print(f"Cleaned size: {len(cleaned_df):,} rows")
        print(f"Duplicates removed: {analysis['duplicate_count']:,}")
    else:
        print(f"Dataset file not found: {data_path}")
        print("Please update the data_path variable with the correct path to your dataset.")
