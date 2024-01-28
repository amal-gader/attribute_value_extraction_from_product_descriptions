import gradio as gr
from api import inference

context = gr.inputs.Textbox(lines=5, placeholder="Enter paragraph/context here...")
question = gr.inputs.Textbox(lines=3, placeholder="Enter Question/keyword here...")
prefix = gr.inputs.Textbox(lines=3, placeholder="Enter prefix here...")
answer = gr.outputs.Textbox(type="auto", label="Answer")

iface = gr.Interface(
    fn=inference,
    inputs=[prefix, prefix, context],
    outputs=answer)

iface.launch(debug=False, share=True)
