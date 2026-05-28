import streamlit as st

APPS = [
    ("hotnews4u", "HotNews4U", "Personalized news recommender"),
    ("rag", "Multi-media RAG", "Search, NER, Q&A"),
    ("papie", "PAPIE", "Personal assistant chatbot"),
    ("mywardrobe", "MyWardrobe", "ShopTheLook outfit recommender"),
    ("cooldrinks", "CoolDrinks", "Beer SKU recommender"),
    ("mynexthome", "MyNextHome", "Real-estate recommender"),
    ("mymedicine", "MyMedicine", "Travel medicine lookup"),
    ("ebooks", "E-book / Audiobook RecSys", "Content recommender"),
    ("scitubbies", "SciTubbies", "YouTube content RecSys"),
    ("jobpromis", "JobPromis", "Job recommender"),
    ("drift-monitor", "Drift Monitor", "Model & data drift"),
    ("datalab-aas", "DataLab-as-a-Service", "Jupyter for DS teams"),
    ("segmentation", "Customer Segmentation", "Persona creation"),
    ("jobminder", "JobMinder", "Job chatbot"),
    ("sku-forecast", "SKU Forecaster", "Demand forecasting"),
    ("declarative-search", "Declarative Search", "Multi-agent scrape"),
    ("socialtraces", "SocialTraces", "Social fuzzy search"),
    ("assetmanager", "AssetManager", "Article summarizer + NER"),
    ("mylocalradar", "MyLocalRadar", "Location disambiguation"),
    ("aifluent", "AIFluent", "Skills acquisition"),
    ("chap", "CHAP", "Hybrid agent platform"),
    ("auctionlab", "AuctionLab", "Auction simulator"),
    ("emagazzine", "EMagazzine", "Price comparator"),
    ("skillsplan", "SkillsPlan", "Curriculum optimizer"),
    ("mysmartdiet", "MySmartDiet", "Diet recommender"),
    ("cloud-ml-estimator", "Cloud ML Estimator", "Pricing estimator"),
]

st.set_page_config(page_title="MyDataSciencePortfolio - Streamlit", layout="wide")
st.title("MyDataSciencePortfolio")
st.caption("Each app is also runnable as `streamlit run apps/<slug>/streamlit_app.py`.")

cols = st.columns(3)
for i, (slug, name, desc) in enumerate(APPS):
    with cols[i % 3]:
        with st.container(border=True):
            st.subheader(name)
            st.write(desc)
            st.code(f"streamlit run apps/{slug}/streamlit_app.py", language="bash")
            st.code(f"python apps/{slug}/cli.py --help", language="bash")
