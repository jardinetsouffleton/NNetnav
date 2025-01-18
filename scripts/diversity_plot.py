# Install plotly and kaleido before running this script

import json
import os
import plotly.graph_objects as go
import plotly.io as pio
# Make sure kaleido is installed for PDF export
try:
    import kaleido
except ImportError:
    raise ImportError("Please install kaleido: pip install -U kaleido")

def process_json_files(directory):
    data = []
    
    # Process each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.startswith('clusters_') and filename.endswith('.json'):
            # Extract website name from filename
            website_name = filename.replace('clusters_https:_', '').replace('clusters_', '').replace('_.json', '').replace('.json', '')
            
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
                # Process each category in the JSON file
                for category, intents in file_data.items():
                    # Count the number of intents in this category
                    intent_count = len(intents)
                    
                    # Add data for the sunburst chart
                    data.append({
                        'website': website_name,
                        'category': category,
                        'count': intent_count
                    })
    
    return data

def create_sunburst_chart(data):
    # Prepare data for the sunburst chart
    labels = []
    parents = []
    values = []
    
    # Add websites as first level
    websites = set(item['website'] for item in data)
    for website in websites:
        labels.append(website)
        parents.append("")
        # Sum up all intent counts for this website
        website_total = sum(item['count'] for item in data if item['website'] == website)
        values.append(website_total)
    
    # Add categories as second level
    for item in data:
        labels.append(f"{item['category']}")
        parents.append(item['website'])
        values.append(item['count'])
    
    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
    ))
    
    # Update layout
    fig.update_layout(
        title="Website Intent Categories Distribution",
        width=1000,
        height=1000,
        title_x=0.5,
    )
    
    return fig

def main():
    # Get the current directory
    current_dir = "clusters_webarena"
    
    # Process the JSON files
    data = process_json_files(current_dir)
    
    # Create the visualization
    fig = create_sunburst_chart(data)
    
    # Save the visualization as PDF
    fig.write_image("intent_distribution_webarena.pdf")
    print("Visualization has been saved as 'intent_distribution.pdf'")

if __name__ == "__main__":
    main()