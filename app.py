import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from databricks import sql
from pinecone import Pinecone
import torch
import time
from transformers import AutoTokenizer, AutoModel

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Enterprise QBR Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Retrieve credentials from environment variables
DATABRICKS_HOST = os.environ["DATABRICKS_HOST"]
DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
DATABRICKS_SERVING_ENDPOINT_URL = os.environ["DATABRICKS_SERVING_ENDPOINT_URL"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Define Unity Catalog from environment variables
UC_CATALOG = os.environ["UC_CATALOG"]
UC_SCHEMA = os.environ["UC_SCHEMA"]
UC_TABLE = os.environ["UC_TABLE"]
DBX_ENDPOINT = os.environ.get("DBX_ENDPOINT", "databricks-meta-llama-3-1-405b-instruct")

# Initialize Pinecone using environment variables
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# Load embedding model once
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    """Convert text into an embedding vector using a pre-trained model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

def build_prompt(company_data, similar_contexts, template_type, view_type):
    """Builds a prompt with RAG context using Pinecone, template modifications, and view-specific emphasis."""

    # Define unique instructions for each QBR type
    template_instructions = {
        "Standard QBR": """
        This is a full Quarterly Business Review (QBR) covering all key aspects, including health score analysis, adoption metrics, customer satisfaction, and strategic recommendations.
        """,
        "Executive Summary Only": """
        This QBR should be concise and high-level, focusing only on key insights, major wins, critical challenges, and high-level recommendations.
        Exclude deep technical details, adoption trends, and granular product feature analysis.
        """,
        "Technical Deep Dive": """
        This QBR should focus on technical aspects such as system architecture, integrations, API usage, performance metrics, and technical challenges.
        Prioritize technical success metrics, potential optimizations, and engineering recommendations.
        Minimize business-level overviews and executive summaries.
        """,
        "Customer Success Focus": """
        This QBR should emphasize customer engagement, product adoption, support trends, and user satisfaction.
        Focus on training needs, adoption blockers, support ticket patterns, and customer success strategies.
        Minimize in-depth technical or executive-level details.
        """
    }

    # Define unique instructions for each View Type
    view_type_instructions = {
        "Sales View": """
        This QBR should focus on revenue impact, upsell opportunities, contract value, expansion potential, and risk mitigation.
        Prioritize key financial metrics, deal health, and strategic recommendations for account growth.
        Minimize highly technical discussions unless relevant for deal positioning.
        """,
        "Executive View": """
        This QBR should provide a high-level strategic overview, emphasizing business outcomes, financial impact, and alignment with company goals.
        Keep details concise, use bullet points, and focus on key wins, challenges, and high-level recommendations.
        Minimize operational or highly technical details.
        """,
        "Technical View": """
        This QBR should provide a deep dive into system performance, architecture, integrations, and product adoption from a technical perspective.
        Prioritize API usage, reliability metrics, infrastructure considerations, and upcoming technical improvements.
        Minimize business-oriented insights unless relevant to product engineering.
        """,
        "Customer Success View": """
        This QBR should focus on customer satisfaction, adoption trends, support tickets, training needs, and customer engagement.
        Prioritize recommendations for improving retention, reducing churn, and addressing adoption blockers.
        Minimize purely financial or highly technical content unless relevant for success strategy.
        """
    }

    # Define dynamic section structures per View Type
    view_based_sections = {
        "Sales View": """
        1. Account Health Summary  
        2. Revenue & Expansion Opportunities  
        3. Usage Trends & Adoption Insights  
        4. Competitive Positioning  
        5. Strategic Sales Recommendations  
        """,
        
        "Executive View": """
        1. Key Business Outcomes  
        2. ROI & Financial Impact  
        3. Adoption & Customer Engagement  
        4. Strategic Roadmap Alignment  
        5. High-Level Recommendations  
        """,
        
        "Technical View": """
        1. System Performance & API Usage  
        2. Infrastructure & Security Considerations  
        3. Feature Adoption & Implementation Status  
        4. Engineering Challenges & Optimization Strategies  
        5. Technical Roadmap & Upcoming Enhancements  
        """,
        
        "Customer Success View": """
        1. Customer Engagement & Satisfaction Metrics  
        2. Product Adoption & User Retention  
        3. Support Trends & Resolution Efficiency  
        4. Training & Enablement Opportunities  
        5. Customer Success Strategy & Next Steps  
        """
    }

    # Get the instructions and section structure based on selection
    qbr_type_instructions = template_instructions.get(template_type, "")
    view_specific_instructions = view_type_instructions.get(view_type, "")
    dynamic_sections = view_based_sections.get(view_type, "1. Executive Summary\n2. Business Impact\n3. Strategic Recommendations")

    # Construct the final prompt with the selected QBR and View Type instructions
    prompt = f"""
    You are an expert business analyst creating a Quarterly Business Review (QBR). 
    Generate a {template_type} QBR using the following data and format:

    {qbr_type_instructions}

    {view_specific_instructions}

    Company Data:
    {company_data}

    Historical Context:
    {similar_contexts if similar_contexts else 'No historical context available'}

    Structure the QBR based on {view_type}, prioritizing the most relevant insights.
    
    Use the following section structure:
    {dynamic_sections}

    Format the QBR professionally with clear section headers and bullet points for key insights.
    Prioritize the most relevant information for {view_type} and {template_type}.
    """

    return prompt

def call_serving_endpoint(prompt):
    """Call Databricks Serving Endpoint"""
    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert business analyst specializing in creating detailed, data-driven Quarterly Business Reviews."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        headers = {
            'Authorization': f'Bearer {DATABRICKS_TOKEN}',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            DATABRICKS_SERVING_ENDPOINT_URL,
            json=payload,
            headers=headers,
            timeout=90
        )

        if response.status_code != 200:
            st.error(f"❌ Error Response: {response.text}")
            return None

        return response.json()

    except Exception as e:
        st.error(f"❌ Error calling Serving Endpoint: {str(e)}")
        return None

def query_unity_catalog(company_name=None):
    """Query Unity Catalog with optional company filter"""
    try:
        connection = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN
        )

        if company_name:
            QUERY = f"""
                SELECT * 
                FROM `{UC_CATALOG}`.`{UC_SCHEMA}`.`{UC_TABLE}`
                WHERE company_name = '{company_name}'
            """
        else:
            QUERY = f"""
                SELECT DISTINCT company_name 
                FROM `{UC_CATALOG}`.`{UC_SCHEMA}`.`{UC_TABLE}`
                ORDER BY company_name
            """

        with connection.cursor() as cursor:
            cursor.execute(QUERY)
            result = cursor.fetchall()

        connection.close()  # Explicitly close the connection after query execution

        if company_name:
            df = pd.DataFrame(result, columns=['company_name', 'company_id', 'qbr_information', 'metadata'])
        else:
            df = pd.DataFrame(result, columns=['company_name'])
        return df
    except Exception as e:
        st.error(f"❌ Error querying Databricks: {str(e)}")
        return pd.DataFrame()

def display_metrics_dashboard(metrics_data):
    """Display key metrics dashboard"""
    # Extract metrics from the QBR information string
    import re
    
    def extract_metric(text, pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0
    
    health_score = extract_metric(metrics_data, r"health score is (\d+\.?\d*)")
    contract_value = extract_metric(metrics_data, r"contract value is \$(\d+\.?\d*)")
    csat_score = extract_metric(metrics_data, r"CSAT score is (\d+\.?\d*)")
    active_users = extract_metric(metrics_data, r"active users is (\d+\.?\d*)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Health Score", 
            f"{health_score:.1f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Contract Value", 
            f"${contract_value:,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "CSAT Score", 
            f"{csat_score:.1f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Active Users", 
            int(active_users),
            delta=None
        )

def search_similar_companies(query, top_k=3):
    """Search for similar companies using semantic search"""
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            if match.metadata and 'qbr_info' in match.metadata:
                formatted_results.append(match.metadata['qbr_info'])
        
        return '\n\n---\n\n'.join(formatted_results) if formatted_results else None
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None

# Initialize session state
if 'qbr_history' not in st.session_state:
    st.session_state.qbr_history = []
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

# Title and Description
st.title("🎯 Enterprise QBR Generator")
st.write("""
Generate comprehensive, data-driven Quarterly Business Reviews using Fivetran, Pinecone, and Databricks.
This Databricks Gen AI Data App combines sales data, support data, product data, current metrics, and predictive
insights to create instant, standardardized and actionable QBRs.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("QBR Preferences")

    # Business Settings
    st.subheader("Settings")

    # Company Selection
    companies_df = query_unity_catalog()
    selected_company = st.selectbox(
        "Select Company",
        options=[""] + companies_df['company_name'].tolist(),
        help="Type to search for a specific company"
    )

    # Template Selection
    template_type = st.selectbox(
        "QBR Template",
        ["Standard QBR", "Executive Summary Only", "Technical Deep Dive", "Customer Success Focus"]
    )

    view_type = st.selectbox(
        "View Type",
        ["Sales View", "Executive View", "Technical View", "Customer Success View"]
    )

    # Advanced Options
    with st.expander("Advanced Options"):
        use_historical = st.checkbox(
            "Include Historical Context",
            value=True,
            help="Use similar QBRs for enhanced insights"
        )

        num_contexts = st.slider(
            "Number of similar QBRs to include",
            min_value=1,
            max_value=10,
            value=5,
            help="Select how many similar QBRs to use for context"
        )

    # Add spacing before branding text
    for _ in range(10):
        st.sidebar.write("")

    # Branding Text (Above the logo)
    st.sidebar.markdown(
        "<h4 style='text-align: center; font-weight: normal;'>Fivetran | Pinecone | Databricks</h4>", 
        unsafe_allow_html=True
    )

    # Add spacing before logo
    for _ in range(1):
        st.sidebar.write("")

    # Correct logo URL from your app.py
    logo_url = "https://i.imgur.com/ioN9AJ3.png"

    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="{logo_url}" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Content Area
tabs = st.tabs(["QBR Generation", "QBR History", "Settings"])

with tabs[0]:
    if selected_company:
        # Get company data
        company_data = query_unity_catalog(selected_company)
        
        if not company_data.empty:
            # Display metrics dashboard
            display_metrics_dashboard(company_data['qbr_information'].iloc[0])
            
            # QBR Generation Button
            if st.button("Generate QBR"):
                with st.spinner("Preparing QBR..."):
                    # Get similar contexts if enabled
                    similar_contexts = None
                    if use_historical:
                        query_embedding = get_embedding(company_data['qbr_information'].iloc[0])
                        results = index.query(
                            vector=query_embedding,
                            top_k=num_contexts,
                            include_metadata=True
                        )
                        
                        similar_contexts = []
                        for match in results.matches:
                            if match.metadata and 'qbr_info' in match.metadata:
                                similar_contexts.append(match.metadata['qbr_info'])
                        similar_contexts = '\n\n'.join(similar_contexts)
                    
                    # Build prompt and generate content
                    prompt = build_prompt(
                        company_data['qbr_information'].iloc[0],
                        similar_contexts,
                        template_type,
                        view_type
                    )
                    
                    response_data = call_serving_endpoint(prompt)
                    
                    if response_data:
                        qbr_content = response_data['choices'][0]['message']['content']
                        token_metrics = response_data.get('usage', {})
                        
                        # Display QBR
                        st.header(f"Quarterly Business Review: {selected_company}")
                        st.write(qbr_content)

                        # Display metrics
                        st.write("### Metrics")
                        st.write(f"Total Tokens: {token_metrics.get('total_tokens', 'N/A')}")
                        
                        # Add download button
                        st.download_button(
                            label="Download QBR",
                            data=qbr_content,
                            file_name=f"QBR_{selected_company}_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )
                        
                        # Save to history
                        st.session_state.qbr_history.append({
                            'company': selected_company,
                            'date': pd.Timestamp.now(),
                            'content': qbr_content,
                            'template': template_type,
                            'view_type': view_type
                        })

with tabs[1]:
    if st.session_state.qbr_history:
        for qbr in reversed(st.session_state.qbr_history):
            with st.expander(f"{qbr['company']} - {qbr['date'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(qbr['content'])
    else:
        st.info("No QBR history available")

with tabs[2]:
    st.write("QBR Generation Settings")
    
    st.subheader("Databricks Settings")
    st.write(f"**Catalog:** {UC_CATALOG}")
    st.write(f"**Schema:** {UC_SCHEMA}")
    st.write(f"**Model:** {DBX_ENDPOINT}")
        
    st.subheader("Pinecone Settings")
    st.write(f"**Index:** {os.environ.get('PINECONE_INDEX_NAME', 'po-embeddings')}")
    st.write(f"**Model:** {MODEL_NAME}")
        
    # Add test search box
    st.subheader("Test Semantic Search")
    test_query = st.text_input("Enter a search query to test Pinecone", 
                            placeholder="E.g., Assembly Tech")
    if test_query and st.button("Search"):
        with st.spinner("Searching similar companies..."):
            similar_companies = search_similar_companies(test_query, top_k=3)
            if similar_companies:
                st.success("Similar companies found!")
                for i, company in enumerate(similar_companies.split('\n\n---\n\n')):
                    with st.expander(f"Similar Company #{i+1}"):
                        st.write(company)
            else:
                st.warning("No similar companies found.")
