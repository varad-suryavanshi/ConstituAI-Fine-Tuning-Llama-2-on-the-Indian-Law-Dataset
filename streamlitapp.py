import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

@st.cache_resource
def load_text_generation_pipeline():
    model_name = "varad-suryavanshi12/Llama-2-7b-chat-finetune"  # Your model repository
    
    # Configure BitsAndBytes for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # Use nf4 quantization format
        bnb_4bit_compute_dtype="float16",    # Use float16 for computations
        bnb_4bit_use_double_quant=False      # Disable nested quantization
    )
    
    # Load the model with 4-bit quantization enabled
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Create the text-generation pipeline
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=200
    )
    return pipe

# Load the pipeline
pipe = load_text_generation_pipeline()

# Streamlit UI
st.title("ConstituAI: Llama‑2 Fine‑Tuned on Indian Law Dataset")
st.write("Enter your prompt below (in plain text) to receive a response in the Llama input/output format.")

user_prompt = st.text_input("Enter your prompt:", value="What is a large language model?")
if st.button("Generate Response"):
    formatted_prompt = f"<s>[INST] {user_prompt} [/INST]"
    with st.spinner("Generating response..."):
        result = pipe(formatted_prompt)
    generated_text = result[0]['generated_text']
    st.markdown("### Response:")
    st.write(generated_text)
