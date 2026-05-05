#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

def mask_account_ids(text):
    """Replace 12-digit numbers (AWS account IDs) with XXXXXXXXXXXX"""
    return re.sub(r'\b\d{12}\b', 'XXXXXXXXXXXX', str(text))

def clean_notebook(notebook_path):
    """Clean account IDs from notebook outputs"""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping {notebook_path} - invalid JSON")
        return
    
    for cell in notebook.get('cells', []):
        # Clean outputs
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    if isinstance(output['text'], list):
                        output['text'] = [mask_account_ids(line) for line in output['text']]
                    else:
                        output['text'] = mask_account_ids(output['text'])
                
                if 'data' in output:
                    for key, value in output['data'].items():
                        if isinstance(value, list):
                            output['data'][key] = [mask_account_ids(line) for line in value]
                        else:
                            output['data'][key] = mask_account_ids(value)
    
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for notebook_path in sys.argv[1:]:
            clean_notebook(notebook_path)
    else:
        # Clean all notebooks in current directory
        for notebook_path in Path('.').glob('**/*.ipynb'):
            clean_notebook(notebook_path)