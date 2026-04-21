#!/usr/bin/env python3
"""
Add the missing forest plot calling code to delta_analysis.ipynb
"""

import json

def add_forest_plot_calls(notebook_path):
    """Add the forest plot calling code after the function definition"""
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find the cell with create_forest_plot function and add a new cell after it
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Find the cell with the forest plot function
            if 'def create_forest_plot' in source and 'return fig' in source:
                print(f"Found create_forest_plot function at cell {i}")
                
                # Create new code cell with the calling code
                new_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "forest_plot_calls",
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Create forest plots\n",
                        "print(\"\\n=== CREATING FOREST PLOTS ===\")\n",
                        "for task_name, res in results.items():\n",
                        "    print(f\"\\n{task_name}:\")\n",
                        "    create_forest_plot(\n",
                        "        task_name,\n",
                        "        res['coefficients'],\n",
                        "        res['std'],\n",
                        "        res['p_values_fdr'],\n",
                        "        res['features'],\n",
                        "        res['significant_fdr'],\n",
                        "        top_n=20,\n",
                        "        output_dir=output_dir\n",
                        "    )\n"
                    ]
                }
                
                # Insert the new cell after the function definition
                nb['cells'].insert(i + 1, new_cell)
                print(f"Inserted forest plot calling code at cell {i + 1}")
                break
    
    # Write fixed notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\nFixed notebook saved to: {notebook_path}")
    print("\nAdded:")
    print("- Forest plot calling code after function definition")
    print("- Will generate: forest_plot_ranking.png, forest_plot_classification.png, forest_plot_link_prediction.png")
    
    return notebook_path

if __name__ == '__main__':
    notebook_path = 'delta_analysis.ipynb'
    add_forest_plot_calls(notebook_path)

# Made with Bob
