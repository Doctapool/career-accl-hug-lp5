#Create the Gradio app interface
import gradio as gr
from transformers import pipeline
#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained('Sonny4Sonnix/twitter-roberta-base-sentimental-analysis-of-covid-tweets')
#from transformers import RobertaForSequenceClassification
#from transformers import AutoModelForSequenceClassification

#sentiment = pipeline("/content/drive/MyDrive/Colab Notebooks/Twitter_roBERTa_base_for_Sentiment_Analysis.ipynb")

sentiment = pipeline("sentiment-analysis",model ="Ausbel/Vaccine-tweet-sentiments-analysis-model-2")
def get_sentiment(iput_text):
    return sentiment(input_text)

results = get_sentiment("Covid is very dangerous")
results

iface = gr.Interface(fn=get_sentiment, input=gr.inputs.Textbox(),outputs="text")
iface.launch()