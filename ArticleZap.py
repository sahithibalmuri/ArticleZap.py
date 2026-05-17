import streamlit as st
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline
from newspaper import Article
import time

# Ensure nltk punkt tokenizer is downloaded
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Page configuration
st.set_page_config(page_title="ArticleLens - NLP Summarizer", page_icon="🔍", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4B4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8C8C8C;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .summary-box {
        background-color: #f9f9f9;
        border-left: 5px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
    }
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .main-header { color: #f0f2f6; }
        .sub-header { color: #b0b3b8; }
        .summary-box {
            background-color: #262730;
            color: #f0f2f6;
            border-left: 5px solid #4CAF50;
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_hf_pipeline():
    # Load a lightweight model for abstractive summarization
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        return None, f"Error extracting text: {e}"

def extractive_summary(text, sentences_count):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary_sentences])

def abstractive_summary(text, max_length, min_length):
    summarizer = load_hf_pipeline()
    # Chunk text if it's too long (distilbart has a max token limit usually around 1024)
    # For a simple demo, we truncate roughly to 800 words
    text_words = text.split()
    if len(text_words) > 800:
        text = " ".join(text_words[:800])
    
    # max_length and min_length must be within model's limits and smaller than text length
    input_length = len(text.split())
    adj_max = min(max_length, input_length - 1) if input_length > max_length else input_length
    adj_min = min(min_length, adj_max - 1) if adj_max > min_length else max(1, adj_max // 2)

    try:
        result = summarizer(text, max_length=adj_max, min_length=adj_min, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error during abstractive summarization: {e}"

# UI Elements
st.markdown('<div class="main-header">ArticleLens 🔍</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Lengthy Article Summarizer</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.title("⚙️ Settings")
summary_type = st.sidebar.radio(
    "Choose Summarization Type:",
    ("Extractive (Fast)", "Abstractive (Advanced)")
)

if summary_type == "Extractive (Fast)":
    st.sidebar.info("Extracts the most important existing sentences from the text.")
    sentences_count = st.sidebar.slider("Number of Sentences", min_value=1, max_value=20, value=5)
else:
    st.sidebar.info("Generates new sentences to form a cohesive summary. (May take longer)")
    st.sidebar.warning("Note: Input is truncated to ~800 words for the abstractive model.")
    max_len = st.sidebar.slider("Max Summary Words (Approx)", min_value=50, max_value=300, value=130)
    min_len = st.sidebar.slider("Min Summary Words (Approx)", min_value=10, max_value=100, value=30)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Article")
    input_method = st.radio("Provide text via:", ("Paste Text", "URL"))
    
    article_text = ""
    article_title = ""
    
    if input_method == "Paste Text":
        article_text = st.text_area("Paste your lengthy article here...", height=300)
    else:
        url = st.text_input("Enter Article URL:")
        if url:
            with st.spinner("Extracting text from URL..."):
                title, extracted = extract_text_from_url(url)
                if title:
                    st.success(f"Extracted: **{title}**")
                    article_title = title
                    article_text = extracted
                    with st.expander("View Extracted Text"):
                        st.write(article_text)
                else:
                    st.error(extracted)

with col2:
    st.subheader("✨ Summary")
    if st.button("Summarize", use_container_width=True):
        if not article_text or len(article_text.strip()) < 50:
            st.warning("Please provide a longer article text to summarize. (At least 50 characters)")
        else:
            start_time = time.time()
            with st.spinner("Generating summary..."):
                if summary_type == "Extractive (Fast)":
                    summary_result = extractive_summary(article_text, sentences_count)
                else:
                    summary_result = abstractive_summary(article_text, max_len, min_len)
            
            end_time = time.time()
            
            # Display result
            if summary_result.startswith("Error"):
                st.error(summary_result)
            else:
                st.markdown(f'<div class="summary-box">{summary_result}</div>', unsafe_allow_html=True)
                
                # Metrics
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**📊 Metrics**")
                m1, m2, m3 = st.columns(3)
                original_words = len(article_text.split())
                summary_words = len(summary_result.split())
                reduction = 100 - (summary_words / original_words * 100) if original_words > 0 else 0
                
                m1.metric("Original Words", original_words)
                m2.metric("Summary Words", summary_words)
                m3.metric("Reduction", f"{reduction:.1f}%")
                
                st.caption(f"Processed in {end_time - start_time:.2f} seconds")
