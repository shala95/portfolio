import pandas as pd
import logging
from pathlib import Path
from typing import Union

def load_file(file_path: Union[str, Path], default_sep=";") -> pd.DataFrame:
    """
    Universal file loader that handles CSV, XLS, and XLSX formats.
    
    Args:
        file_path: Path to the file
        default_sep: Default separator for CSV files
        
    Returns:
        Loaded DataFrame or empty DataFrame if file doesn't exist
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not path.exists():
        logging.warning(f"File not found: {path}")
        return pd.DataFrame()
    
    try:
        suffix = path.suffix.lower()
        if suffix in ['.xlsx', '.xls']:
            # Excel file
            return pd.read_excel(path)
        elif suffix == '.csv':
            # CSV file - try with specified separator first
            try:
                return pd.read_csv(path, sep=default_sep)
            except:
                # If that fails, try with auto-detection
                return pd.read_csv(path, sep=None, engine='python')
        else:
            # Unknown format - try CSV first, then Excel
            try:
                return pd.read_csv(path, sep=default_sep)
            except:
                try:
                    return pd.read_excel(path)
                except:
                    logging.error(f"Could not determine format for file: {path}")
                    return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading file {path}: {str(e)}")
        return pd.DataFrame()