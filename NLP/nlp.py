import bs4 as bs
import urllib.request
import nltk
import spacy
from spacy.matcher import PhraseMatcher
from IPython.core.display import HTML
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS

def exemplo1():
    nlp = spacy.load('pt_core_news_sm')

    ex = nlp("O rato roeu a roupa do rei de Roma")

    # POS - Part of Speech, é a classificação gramatical de cada palavra, por exemplo, substantivo, verbo, adjetivo, etc.
    for token in ex:
        print(token.text, token.pos_)

    ex2 = nlp("encontrei encontraram encontrarão encontrar curso cursarão cursando")
    # LEMMA - é a forma base da palavra, por exemplo, o lema de "correr" é "correr", o lema de "correu" é "correr"
    [print(token.text, token.lemma_) for token in ex2]

def load_text(url='https://en.wikipedia.org/wiki/Artificial_intelligence'):
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()

    article = bs.BeautifulSoup(raw_html, 'html.parser')
    paragraphs = article.find_all('p')
    article_text = ''
    for p in paragraphs:
        article_text += p.text
    return remove_stop_words(article_text)

def search_word(text, word):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('AI', None, nlp(word))
    matches = matcher(doc)
    if matches:
        print(f'Found {len(matches)} matches')
        return matches
    return None

def exemplo2():
    nlp = spacy.load('en_core_web_sm')
    text = ''
    word = 'artificial'
    size = 50
    artikel = load_text()
    match = search_word(artikel, word)
    doc = nlp(artikel)
    #display(HTML(f'<h1>{word.toUpperCase()}</h1>'))
    #display(HTML(f"""<p><strong>Results:</strong> {len(match)}</p>"""))
    for i in match:
        start = i[1] - size
        if start < 0:
            start = 0
        text += str(doc[start:i[2] + size]).replace(word, f'<mark>{word}</mark>')
        text += "<br  /><br  />"
    #display(HTML(f"""{text}"""))    

def exemplo3():
    color_map = ListedColormap(['red', 'green', 'blue', 'yellow'])
    cloud = WordCloud(background_color='white', colormap=color_map, max_words=100)
    cloud = cloud.generate(load_text())
    plt.figure(figsize=(15, 15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.savefig('wordcloud.png')
    plt.show()


def remove_stop_words(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return ' '.join([token.text for token in doc if token.text.lower() not in STOP_WORDS])

exemplo3()