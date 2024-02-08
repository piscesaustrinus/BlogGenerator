import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
# Import necessary libraries
from transformers import AutoModel, AutoTokenizer, TextGenerationPipeline

# Load the LLama-2-13B-chat-GGML model
model_name = "TheBloke/Llama-2-13B-chat-GGML"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate blog content
def generate_blog(topic, num_words, target_audience):
    # Craft a clear and informative prompt focusing on blog-specific features:
    prompt = f"""Write a blog for {target_audience} about {topic} in {num_words} words. Include insights, challenges, and opportunities relevant to {topic}. Ensure the tone is engaging and informative, addressing a wider audience. Focus on clarity, readability, and providing valuable information."""

    # Generate tokens from the prompt using the tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt")

    # Initialize the text generation pipeline with specific parameters
    text_generator = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        do_sample=True,  # Enable sampling for more creativity
        top_p=0.9,  # Focus on high-probability, grammatically correct text
        num_beams=2,  # Use beam search for improved coherence
        max_length=num_words,  # Control output length
        no_repeat_ngram_size=2,  # Avoid repetitive phrases
    )

    # Generate text using the pipeline
    generated_text = text_generator(input_ids)[0]["generated_text"]

    # Decode the generated tokens back to text
    decoded_text = tokenizer.decode(generated_text, skip_special_tokens=True)

    return decoded_text

# Streamlit app setup
st.set_page_config(
    page_title="Generate Blogs with Hugging Face API",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs")

# User input fields
input_text = st.text_input("Enter Blog Topic (e.g., 'Data Science Careers')", placeholder="Type your topic here...")
num_words = st.number_input("Number of Words (50-500)", min_value=50, max_value=500, value=250)
blog_style = st.selectbox("Target Audience", ["Researchers", "Data Scientists", "General Public"])

submit = st.button("Generate Blog")

if submit:
    try:
        blog_content = generate_blog(input_text, num_words, blog_style)
        st.write(blog_content)
    except Exception as e:
        st.error("Error: An unexpected error occurred. Please check the console for details.")
        print(e)  # Log the error for debugging
