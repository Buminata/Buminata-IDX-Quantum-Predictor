import pdfplumber
import pandas as pd

def extract_investor_data(pdf_path):
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    clean_row = [str(cell).strip().replace('\n', ' ') if cell is not None else "" for cell in row]
                    extracted_data.append(clean_row)
    
    df = pd.DataFrame(extracted_data)
    # The actual columns based on PDF structure
    # Based on IDX ownership PDF structure
    if not df.empty:
        # Just save it raw to see if we can find right columns
        df.to_csv("investor_data.csv", index=False)

extract_investor_data(r"C:\Kuliah Arya\porto\stocks\investor oewnership.pdf")
