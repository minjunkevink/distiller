import graphviz
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from loguru import logger

def create_pipeline_diagram():
    """Create a diagram of the RAG pipeline."""
    dot = graphviz.Digraph(comment='RAG Pipeline')
    dot.attr(rankdir='LR')
    
    # Set global styling
    dot.attr('node', 
             shape='box',
             style='rounded,filled',
             fillcolor='#E8F4F8',
             fontname='Helvetica',
             fontsize='12',
             margin='0.3')
    dot.attr('edge',
             color='#2C3E50',
             penwidth='1.5',
             fontname='Helvetica',
             fontsize='10')
    
    # Add nodes with gradient colors
    dot.node('pdf', 'PDF Input', fillcolor='#3498DB')
    dot.node('extract', 'Text Extraction', fillcolor='#2ECC71')
    dot.node('chunk', 'Text Chunking', fillcolor='#E74C3C')
    dot.node('summarize', 'Summarization', fillcolor='#F1C40F')
    dot.node('embed', 'Embedding', fillcolor='#9B59B6')
    dot.node('faiss', 'FAISS Index', fillcolor='#1ABC9C')
    dot.node('query', 'Query Interface', fillcolor='#E67E22')
    
    # Add edges with labels
    dot.edge('pdf', 'extract', 'Extract')
    dot.edge('extract', 'chunk', 'Chunk')
    dot.edge('chunk', 'summarize', 'Summarize')
    dot.edge('summarize', 'embed', 'Embed')
    dot.edge('embed', 'faiss', 'Index')
    dot.edge('faiss', 'query', 'Query')
    
    # Save the diagram
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    dot.render(output_dir / 'pipeline', format='png', cleanup=True)
    logger.info("Pipeline diagram generated successfully")

def create_architecture_diagram():
    """Create a system architecture diagram with model information."""
    dot = graphviz.Digraph(comment='System Architecture')
    dot.attr(rankdir='TB')
    
    # Set global styling
    dot.attr('node', 
             shape='box',
             style='rounded,filled',
             fontname='Helvetica',
             fontsize='10',
             margin='0.3')
    dot.attr('edge',
             color='#2C3E50',
             penwidth='1.5',
             fontname='Helvetica',
             fontsize='9')
    
    # Add nodes with model information and colors
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Layer', style='rounded,filled', fillcolor='#E8F4F8', fontname='Helvetica', fontsize='12')
        c.node('pdf', 'PDF Files\n(PyMuPDF)', fillcolor='#3498DB')
        c.node('query', 'User Queries', fillcolor='#3498DB')
    
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Processing Layer', style='rounded,filled', fillcolor='#E8F4F8', fontname='Helvetica', fontsize='12')
        c.node('extract', 'Text Extraction\n(PyMuPDF)', fillcolor='#2ECC71')
        c.node('chunk', 'Text Chunking\n(Chunk Size: 500)', fillcolor='#E74C3C')
        c.node('summarize', 'Summarization\n(GPT-4)\nMax Tokens: 1000\nTemp: 0.7', fillcolor='#F1C40F')
        c.node('embed', 'Embedding\n(text-embedding-ada-002)\n1536 dim', fillcolor='#9B59B6')
    
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Storage Layer', style='rounded,filled', fillcolor='#E8F4F8', fontname='Helvetica', fontsize='12')
        c.node('faiss', 'FAISS Index\n(FlatL2)\nCosine Similarity', fillcolor='#1ABC9C')
        c.node('summaries', 'Summaries\n(Text Storage)', fillcolor='#1ABC9C')
    
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Output Layer', style='rounded,filled', fillcolor='#E8F4F8', fontname='Helvetica', fontsize='12')
        c.node('results', 'Search Results\n(Top-k: 3)', fillcolor='#E67E22')
    
    # Add edges with additional information
    dot.edge('pdf', 'extract', 'Raw Text')
    dot.edge('extract', 'chunk', 'Structured Text')
    dot.edge('chunk', 'summarize', 'Chunks')
    dot.edge('summarize', 'embed', 'Summaries')
    dot.edge('summarize', 'summaries', 'Store')
    dot.edge('embed', 'faiss', 'Vectors')
    dot.edge('query', 'faiss', 'Query Vector')
    dot.edge('faiss', 'results', 'Ranked Results')
    
    # Save the diagram
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    dot.render(output_dir / 'architecture', format='png', cleanup=True)
    logger.info("Architecture diagram generated successfully")

def create_benchmark_graph():
    """Create a benchmark graph showing processing times."""
    # Example data (replace with actual benchmarks)
    stages = ['Extraction', 'Chunking', 'Summarization', 'Embedding']
    times = [2.5, 1.2, 15.3, 8.7]  # Example times in seconds
    
    # Set style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Create figure with custom style
    plt.figure(figsize=(10, 6))
    
    # Create color palette
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F1C40F']
    
    # Create bar plot
    bars = plt.bar(stages, times, color=colors, alpha=0.8)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#2C3E50')
    
    # Customize plot
    plt.title('Processing Time per Pipeline Stage', 
              fontsize=14, 
              fontweight='bold',
              pad=20,
              color='#2C3E50')
    plt.xlabel('Pipeline Stage', fontsize=12, color='#2C3E50')
    plt.ylabel('Time (seconds)', fontsize=12, color='#2C3E50')
    plt.xticks(rotation=45, fontsize=10, color='#2C3E50')
    plt.yticks(fontsize=10, color='#2C3E50')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3, color='#2C3E50')
    
    # Set background color
    plt.gca().set_facecolor('#F8F9FA')
    plt.gcf().set_facecolor('#FFFFFF')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Benchmark graph generated successfully")

if __name__ == "__main__":
    create_pipeline_diagram()
    create_architecture_diagram()
    create_benchmark_graph() 