import streamlit as st
import requests
import pandas as pd
import time
import logging

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/extract"
PAGE_TITLE = "Furniture NER Extractor"
PAGE_ICON = "🪑"

logging.basicConfig(level=logging.INFO)

# --- PAGE SETUP ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# Custom CSS for better look
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("""
    This tool extracts furniture product names from e-commerce websites 
    using a **Fine-Tuned DistilBERT** model served via **FastAPI**.
""")

# --- INPUT SECTION ---
url_input = st.text_input("Enter Product URL:", placeholder="https://www.example.com/products/sofa")

if st.button("Extract Products"):
    logging.info(f"User requested extraction for: {url_input}")
    if not url_input:
        st.warning("Please enter a URL first.")
    else:
        # Start timer and spinner
        start_time = time.time()
        with st.spinner("Scraping and analyzing... (Model is warming up)"):
            try:
                # Call the API
                response = requests.post(API_URL, json={"url": url_input})
                
                # Handling Responses
                if response.status_code == 200:
                    data = response.json()
                    products = data.get("products", [])
                    proc_time = data.get("processing_time", 0)
                    
                    st.success("Extraction Complete!")
                    
                    # --- METRICS ROW ---
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Products Found", value=len(products))
                    with col2:
                        st.metric(label="Processing Time", value=f"{proc_time}s")
                    
                    # --- RESULTS TABLE ---
                    if products:
                        st.subheader("Found Entities:")
                        # Convert to DataFrame for a nice interactive table
                        df = pd.DataFrame(products, columns=["Product Name"])
                        st.dataframe(df, use_container_width=True)
                        
                        # --- DOWNLOAD BUTTON ---
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name='extracted_furniture.csv',
                            mime='text/csv',
                        )
                    else:
                        st.info("No products found on this page. Try another URL.")

                else:
                    # Show API errors nicely
                    try:
                        err_msg = response.json().get("detail", "Unknown Error")
                    except:
                        err_msg = response.text
                    st.error(f"API Error ({response.status_code}): {err_msg}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the Backend API.")
                st.info("Make sure the FastAPI server is running: `uv run python -m src.api.main`")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- FOOTER ---
st.markdown("---")
with st.expander("ℹ️ How it works"):
    st.markdown("""
    1. **Scraper (BS4)**: Downloads the HTML and cleans tags/scripts.
    2. **NER Model (DistilBERT)**: Scans the text using a sliding window approach.
    3. **Heuristics**: Filters out non-product text (prices, menu items).
    4. **API (FastAPI)**: Orchestrates the process and returns JSON.
    """)