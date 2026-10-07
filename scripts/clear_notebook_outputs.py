"""Clear outputs before committing; run from the repository root."""
import json
from pathlib import Path

for folder in ['notebooks', 'experiments']:
    for path in Path(folder).rglob('*.ipynb'):
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
            cell['metadata'].pop('execution', None)
        notebook['metadata'].pop('widgets', None)
        path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n')
        print(path)
