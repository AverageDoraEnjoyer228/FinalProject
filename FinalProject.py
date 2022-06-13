import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import requests
from bs4 import BeautifulSoup

import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

import torch
import transformers
import tokenizers

from wordcloud import WordCloud, STOPWORDS


with st.echo(code_location='below'):
    def print_header():
        st.title("ArXiv - изучение и определение тем статей")
        st.markdown("<h1 style='text-align: center;'><img width=300px src='https://media.wired.com/photos/592700e3cfe0d93c474320f1/191:100/w_1200,h_630,c_limit/faces-icon.jpg'>",
                    unsafe_allow_html=True)
        st.markdown("""
            В данном проекте я исследую сайт с научными статьями на технические темы - arXiv. Сначала я
            провожу визуализацию и изучение датасета, который содержит информацию о 75k статьях с arXiv,
            а потом строю модель, которая по названию и абстракту статьи предсказывает,
            по какой научной области написана статья.\n
            Исходный датасет на 1.7М+ статей взят с
            https://www.kaggle.com/datasets/Cornell-University/arxiv?select=arxiv-metadata-oai-snapshot.json,
            который в процессе был прилично обработан и урезан. Работу с датасетом,
            а также обучение самой модели (за основу взята модель типа трансформер)
            можно посмотреть в ipynb-файле, приложенном вместе с ссылкой на данную страницу.
        """)


    @st.cache(suppress_st_warning=True)
    def load_and_process_data():
        data_title_abstract = pd.read_csv('dataset_full.csv')
        return data_title_abstract


    @st.cache(suppress_st_warning=True, hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def plot_histograms(data):
        title_words = [len(data['title'][i].split()) for i in range(len(data['title']))]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        hist_title = ax1.hist(title_words, bins=30)
        ax1.set_xlabel('Число слов в названии')
        ax1.set_ylabel('Количество статей')

        title_abstracts = [len(data['abstract'][i].split()) for i in range(len(data['abstract']))]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        hist_abstract = ax2.hist(title_abstracts, bins=30, color='orange')
        ax2.set_xlabel('Число слов в абстракте')
        ax2.set_ylabel('Количество статей')

        return fig1, fig2


    def plot_wordcloud(data):
        st.write('Выберите одну из 8 тем:')
        category = st.selectbox("", label_to_theme.values())
        text = " ".join(
            title for title in data[data['categories'] == {v: k for k, v in label_to_theme.items()}[category]]['title']
        )
        stopwords = set(STOPWORDS)
        stopwords.update(['based', 'using'])
        wordcloud = WordCloud(stopwords=stopwords).generate(text)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


    def plot_graph():
        file = open("graph.html", 'r', encoding='utf-8')
        source_code = file.read()
        components.html(source_code, height=600, width=600)


    def get_latest_article():
        st.write('Выберите одну из 8 тем:')
        category = st.selectbox("", sorted(code_to_theme.values()))
        category = str({v: k for k, v in code_to_theme.items()}[category]) + '/new'
        entrypoint = 'https://arxiv.org/list/'
        r = requests.get(entrypoint + category)
        soup = BeautifulSoup(r.text)
        pdf = soup.find('dt').find_all('a')[2]['href']
        link = 'https://arxiv.org' + pdf
        title = soup.find('dd').find(class_="list-title mathjax").text.strip()[7:]
        abstract = soup.find('dd').find('p').text.strip()
        st.write('**Title**: ' + title)
        st.write('**Abstract**: ' + abstract)
        st.write('**Link**: ' + link)


    ##### MODEL #####


    @st.cache(suppress_st_warning=True, hash_funcs={tokenizers.Tokenizer: lambda _: None})
    def load_model():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
        model.load_state_dict(torch.load('model_weights3.pt', map_location=torch.device('cpu')))
        model.eval()
        return tokenizer, model


    def predict(title, summary, tokenizer, model):
        text = title + "\n" + summary
        tokens = tokenizer.encode(text)
        with torch.no_grad():
            logits = model(torch.as_tensor([tokens]))[0]
            probs = torch.softmax(logits[-1, :], dim=-1).data.cpu().numpy()

        classes = np.flip(np.argsort(probs))
        sum_probs = 0
        ind = 0
        prediction = []
        prediction_probs = []
        while sum_probs < 0.95:
            prediction.append(label_to_theme[classes[ind]])
            prediction_probs.append(str("{:.2f}".format(100 * probs[classes[ind]])) + "%")
            sum_probs += probs[classes[ind]]
            ind += 1

        return prediction, prediction_probs


    def get_results(prediction, prediction_probs):
        frame = pd.DataFrame({'Category': prediction, 'Confidence': prediction_probs})
        frame.index = np.arange(1, len(frame) + 1)
        return frame


    label_to_theme = {0: 'Computer science', 1: 'Economics', 2: 'Electrical Engineering and Systems Science', 3: 'Math',
                      4: 'Quantitative biology', 5: 'Quantitative Finance', 6: 'Statistics', 7: 'Physics'}
    code_to_theme = {'cs': 'Computer science', 'econ': 'Economics',
                     'eess': 'Electrical Engineering and Systems Science', 'math': 'Math',
                     'q-bio': 'Quantitative biology', 'q-fin': 'Quantitative Finance',
                     'stat': 'Statistics', 'physics': 'Physics'}


    data_title_abstract = load_and_process_data()

    print_header()

    st.subheader('Число слов в заголовках и абстрактах статей')
    fig1, fig2 = plot_histograms(data_title_abstract)
    st.pyplot(fig1)
    st.pyplot(fig2)

    st.subheader('Ключевые слова в названии статей в зависимости от темы')
    plot_wordcloud(data_title_abstract)

    st.subheader('Темы с общими статьями')
    st.write("""
        Данный граф показывает, есть ли статья, написанная по определенной комбинации основных тем.
        Ребро между двумя темами означает, что есть статья, относящаяся одновременно к этим двум темам.
    """)
    plot_graph()
    st.write('Что ж, видим, что какую бы пар тем мы ни взяли, в мире уже есть статья по этой паре.')


    st.subheader('Самая свежая статья по конкретной теме')
    get_latest_article()

    ##### MODEL #####

    st.subheader('Предсказание темы статьи по названию и абстракту')
    st.write("""
        Здесь вы можете попробовать позапускать обученную мной модель на различных примерах статей.
        Модель работает как с одним лишь названием, так и с названием и абстрактом (ответы чаще 
        получаются правильными, если ввести и то, и то). Строго модель не судите, правильно ее обучить - сложная
        задача. Возможно, в каких-то случаях модель будет очень уверенно выдавать совершенно неправильные
        ответы. Попробуйте позапускать ее на разных статьях из разных тем, как только по названию,
        так и по названию с абстрактом. Примеры статей с названиями и абстрактами вы можете
        найти здесь: https://arxiv.org/.
        \n
        NB: иногда Streamlit шаманит, и, возможно, придется два раза нажать кнопку Run для получения предсказания.
    """)
    tokenizer, model = load_model()
    title = st.text_area(label='Title', height=100)
    summary = st.text_area(label='Abstract (optional)', height=250)
    button = st.button('Run')
    if button:
        prediction, prediction_probs = predict(title, summary, tokenizer, model)
        ans = get_results(prediction, prediction_probs)
        if len(title + "\n" + summary) < 20:
            st.error("Your input is too short. It is probably not a real article, please try again.")
        else:
            st.subheader('Results:')
            st.write(ans)
