import json
import sys

def extract_source_nb(path):
    # Extract code cells from a Jupyter notebook and return as a list of strings and output to a .py file in comments
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown':
            code = ''.join(cell['source'])
            code_cells.append(code)
            outputs = cell.get('outputs', [])
            for output in outputs:
                if output['output_type'] == 'stream':
                    output_text = ''.join(output['text'])
                    code_cells.append(f'# Output:\n# {output_text.replace("\n", "\n# ")}')
                elif output['output_type'] == 'execute_result':
                    output_text = ''.join(output['data'].get('text/plain', ''))
                    code_cells.append(f'# Result:\n# {output_text.replace("\n", "\n# ")}')
                elif output['output_type'] == 'error':
                    error_msg = ''.join(output['traceback'])
                    code_cells.append(f'# Error:\n# {error_msg.replace("\n", "\n# ")}')

    return code_cells
    
if __name__ == "__main__":
    notebook_path = sys.argv[1]
    code_cells = extract_source_nb(notebook_path)
    with open('extracted_notebook_code.py', 'w', encoding='utf-8') as f:
        for cell in code_cells:
            f.write(cell + '\n\n')    
