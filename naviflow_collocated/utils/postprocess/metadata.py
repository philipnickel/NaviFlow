import os
import yaml
import pandas as pd
import subprocess
import tempfile
from naviflow_collocated.utils.postprocess.utils import flatten_dict

def yaml_to_latex_pdf(yaml_path, output_pdf_path):
    """Convert a YAML file to a PDF using LaTeX formatting."""
    # Load YAML and flatten
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    flat_data = flatten_dict(data)

    # Create DataFrame and convert to markdown
    df = pd.DataFrame(flat_data.items(), columns=["Parameter", "Value/Setting"])
    md_content = df.to_markdown(index=False)
    
    # Create temporary markdown file
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as tmp:
        tmp.write(md_content)
        tmp_path = tmp.name

    # Create temporary PDF path for initial generation
    temp_pdf = output_pdf_path + ".temp.pdf"

    # Run pandoc with options to:
    # - Make table fill page width
    # - Remove page numbers
    # - Use full page width
    subprocess.run([
        "pandoc",
        tmp_path,
        "-o", temp_pdf,
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=0.2in",
        "-V", "pagenumbers=false",
        "-V", "tables:width=1.0\\textwidth"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Crop the PDF to remove excess whitespace but keep some margin
    subprocess.run([
        "pdfcrop",
        "--margins", "25 25 25 25",  # left top right bottom margins in points
        temp_pdf,
        output_pdf_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Clean up temporary files
    os.unlink(tmp_path)
    os.unlink(temp_pdf)
    
    print(f"PDF saved: {output_pdf_path}") 