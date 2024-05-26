import streamlit as st
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.title('tweet_analyze')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = 'bert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



# Initialize df as None
if 'df' not in st.session_state:
    st.session_state.df = None


@st.cache
def load_dataframe(data_file):
    return pd.read_csv(data_file)


data_file = st.file_uploader("Upload a CSV file file that contains two columns 'Comments' and 'Labels' ", type=['csv'])
if st.button("Process"):
    if data_file is not None:
        file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
        st.write(file_details)
        st.session_state.df = load_dataframe(data_file)
        st.dataframe(st.session_state.df)


def preprocessing(str):
    #lowercase string
    str = str.lower()
    #remove rt, mention and link
    str = re.sub('rt |@[a-z]*|http([a-z]|[0-9]|/|:|.)*|pic.twitter.com/([a-z]|[0-9])*', '', str)
    #remove punctuation and emoticon
    str = re.sub('[^a-z0-9]+', ' ', str)
    #remove extra white spaces
    str = ' '.join(str.split())
    #tokenization
    # str = str.split()
    # if str == []:
    # return float('NaN')
    return str


# if st.button("Filter"):
#     if st.session_state.df is not None:
#         df1 = st.session_state.df.loc[st.session_state.df['Label'] == 0]
#         st.dataframe(df1)
#     else:
#         st.warning("Please upload and process a file first.")

id2label = {0: "non-racism", 1: "racism", 2: "xenophobia", 3: 'non_xenophobia'}


# function to tokenize and predict the label for a list of comments
@st.cache
def load_model():
    device = torch.device("cpu")
    return torch.load('./best_model.pth', map_location=device)

def predict_comments_df(model, tokenizer, df, max_len):


    # Check if CUDA (GPU) is available, and move the model accordingly
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     model = model.to(device)
    #     state_dict = torch.load('./best_model.pth')
    # else:
    #     # Use CPU
    #     #device = torch.device("cpu")
    #
    #
    #     #state_dict = torch.load('./best_model.pth', map_location=device)
    #     state_dic = load_model()

    # Load the model state_dict
    model.load_state_dict(load_model())
    predictions = []
    for index, row in df.iterrows():
        comment = row['Comments']
        comment = preprocessing(comment)
        encoding = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        predictions.append(prediction)

    return predictions

if st.button("predict_label"):
    if st.session_state.df is not None:
        predicted_df = st.session_state.df.copy()
        max_len = 256  # or any other max length that suits your needs

        predictions = predict_comments_df(model, tokenizer, st.session_state.df, max_len)

        # map predicted label indices to their corresponding labels using id2label
        predicted_labels = [id2label[predict] for predict in predictions]

        # add the predicted labels as a new column to the DataFrame
        predicted_df['predicted_label'] = predicted_labels

        st.dataframe(predicted_df)




