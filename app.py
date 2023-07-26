
import gradio as gr

import transformers as pipeline

from transformers import AutoTokenizer,AutoModelForSequenceClassification

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

 

model_name = "Ausbel/Vaccine-tweet-sentiments-analysis-model-2" 

# Replace with the name of the pre-trained model you want to use

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
def get_sentiment(input_text):

        return sentiment(input_text)



 

iface = gr.Interface(fn=get_sentiment,title="Sentimental Analysis", inputs="text",outputs="text")

iface.launch(inline=True)
