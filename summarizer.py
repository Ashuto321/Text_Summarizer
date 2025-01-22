import torch
import gradio as gr
# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = ("C:\\Users\\Ashutosh Pandey\\Desktop\\text-summarizer\\model\\models--sshleifer--distilbart-cnn-12-3\\snapshots")

text_summary = pipeline("summarization", 
                model="sshleifer/distilbart-cnn-12-3",
                  torch_dtype=torch.bfloat16)

#give you text here

# text= """Space is a three-dimensional continuum 
# containing positions and directions.[1] 
# In classical physics, physical space is 
# often conceived in three linear dimensions. 
# Modern physicists usually consider it, with time, 
# to be part of a boundless four-dimensional continuum 
# known as spacetime.[2] The concept of space is considered
#  to be of fundamental importance to an understanding of 
#  the physical universe. However, disagreement continues 
#  between philosophers over whether it is itself an entity,
#    a relationship between entities,
# or part of a conceptual framework."""
# print(text_summary(text))


def summary(input):
       output=text_summary(input)
       return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary,inputs="text",outputs="text")
demo= gr.Interface(fn=summary,
inputs=[gr.Textbox(label="Input text summarizer",lines=6)],
outputs=[gr.Textbox(label="Summarized text", lines=4)],
title="Gen AI project1: Text Summarizer",
description="This summarizer is used to summarize text in one go")
demo.launch(share=True)

