import os
import time
import gradio as gr
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

TITLE = "<h1><center>CodeAssist AI</center></h1>"
PLACEHOLDER = """<center><p>Hi, I'm your coding assistant. Ask me anything about programming!</p></center>"""

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3, p, h1 {
    text-align: center;
    color: white;
}
footer {
    text-align: center;
    padding: 10px;
    width: 100%;
    background-color: rgba(240, 240, 240, 0.8);
    z-index: 1000;
    position: relative;
    margin-top: 10px;
    color: black;
}
"""

FOOTER_TEXT = """<footer> <p>If you enjoyed the functionality of the app, please leave a like!<br> Check out more on <a href="https://www.linkedin.com/in/girish-wangikar/" target="_blank">LinkedIn</a> | <a href="https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/" target="_blank">Portfolio</a></p></footer>"""

def generate_response(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 512
):
    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])
    conversation.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="llama-3.1-8B-Instant",
        messages=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

def clear_conversation():
    return [], None

chatbot = gr.Chatbot(height=600, placeholder=PLACEHOLDER)

with gr.Blocks(css=CSS, theme="Nymbo/Nymbo_Theme") as demo:
    gr.HTML(TITLE)

    with gr.Accordion("⚙️ Parameters", open=False):
        system_prompt = gr.Textbox(
            value="You are a helpful coding assistant, specialized in code completion, debugging, and analysis. Provide concise and accurate responses.",
            label="System Prompt",
        )
        temperature = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            value=0.5,
            label="Temperature",
        )
        max_tokens = gr.Slider(
            minimum=50,
            maximum=1024,
            step=1,
            value=512,
            label="Max tokens",
        )

    chat_interface = gr.ChatInterface(
        fn=generate_response,
        chatbot=chatbot,
        additional_inputs=[system_prompt, temperature, max_tokens],
        examples=[
            ["What are Python generators, and how do I use them effectively in my code?"],
            ["Can you explain the different types of SQL joins with examples?"],
            ["How do I calculate the time complexity of a function? Explain with example"],
            ["Explain this code to me: [paste your code here]"],
        ],
        cache_examples=False,
    )

    clear_btn = gr.Button("Clear Conversation")
    clear_btn.click(clear_conversation, outputs=[chatbot, chatbot])

    gr.HTML(FOOTER_TEXT)

if __name__ == "__main__":
    demo.launch()
