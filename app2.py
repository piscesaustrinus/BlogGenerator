import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import os

# Function to get response from LLama 2 model
def get_llama_response(input_text, num_words, blog_style):
    try:
        # Check if LLama 2 model path is specified
        model_path = os.environ.get("llama-2-13b-chat.ggmlv3.q8_0.bin")
        if not model_path:
            raise ValueError("/Users/sans22/Documents/BlogGeneration environment variable not set.")

        # Load LLama 2 model
        llm = CTransformers(model=model_path, model_type='llama',
                            config={'max_new_tokens': 256, 'temperature': 0.01})

        # Simplified and focused prompt template
        template = """\
        Write a blog for {blog_style} about {input_text} in {num_words} words.
        Include insights, challenges, and opportunities specific to this role.
        """

        prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_words"],
                              template=template)

        # Generate response
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, num_words=num_words))
        return response

    except Exception as e:
        st.error("Error: Could not generate blog. Please ensure the LLama 2 model path is set correctly and try again.")
        print(e)  # Log the exception for debugging

# Streamlit app setup
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs")

# Input with placeholder, improved instructions, and word count validation
input_text = st.text_input("Enter the Blog Topic (e.g., 'Data Science Careers')", placeholder="Type your topic here...")
num_words = st.number_input("Number of Words (50-500)", min_value=50, max_value=500, value=250)
if num_words < 50 or num_words > 500:
    st.error("Please enter a word count between 50 and 500.")
    # Prevent further execution if word count is invalid

# Target audience options
blog_style = st.selectbox("Target Audience", ["Researchers", "Data Scientists", "General Public"])

submit = st.button("Generate Blog")

if submit:
    try:
        response = get_llama_response(input_text, num_words, blog_style)
        st.write(response)
    except Exception as e:
        st.error("Error: An unexpected error occurred. Please check the console for details.")
        print(e)  # Log the exception for debugging

