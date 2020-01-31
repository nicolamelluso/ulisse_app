# coding: utf-8
"""
Example of a Streamlit app for an interactive spaCy model visualizer. You can
either download the script, or point streamlit run to the raw URL of this
file. For more details, see https://streamlit.io.
Installation:
pip install streamlit
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download de_core_news_sm
Usage:
streamlit run streamlit_spacy.py
"""
from __future__ import unicode_literals

import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

ner_displacy_palette = {
    "CLUE": "#da3650",
    "USER": "#67328b",
    "SOFT SKILL": "#87CEFA",
    "EXTR": "#fcd548",
    "GRIPPER": "#007bac",
    "OPERATION": "#6c63a5",
    "CONCEPT": "#df5a35",
}

ner_displacy_options = {
    "ents": [
        "CLUE",
        "SOFT SKILL",
        "EXTR",
        "USER",
        "BODY",
        "GRIPPER",
        "OPERATION",
        "CONCEPT",
    ],
    "colors": ner_displacy_palette,
}
#SPACY_MODEL_NAMES = ["ergonomy_spacy_model"]

import pandas as pd
df = pd.read_pickle('./data/df_ss_label.pickle')


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""


@st.cache()
def load_model(name):
    return spacy.load(name)


@st.cache()
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)

st.sidebar.title("Soft Skills Extractor")
st.sidebar.markdown(
    """
Process text with [spaCy](https://spacy.io) models and visualize Soft Skills. Uses spaCy's built-in
[displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
"""
)
page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

if page == 'Homepage':
    
    

    st.header("Entities")

    label_set = df.label.unique().tolist()

    labels = st.multiselect("Soft Skill Label", label_set, label_set)




    df_tmp = df[df['soft_skill'].isin(labels)]

    df_tmp = df_tmp.drop_duplicates(subset = ['sent'])


    for _,row in df_tmp.iterrows():
    
        html = displacy.render(row['sent_spacy'], style="ent", options={"ents": ['SOFT SKILL'],"colors": ner_displacy_palette})
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
#
#
#for sent in doc_main.sents:
#    doc = process_text(spacy_model, sent.text)
#    
#    if any(elem in ents for elem in [ent.label_ for ent in doc.ents]) & ('\n' not in doc.text):
#        
#        
#        html = displacy.render(doc, style="ent", options={"ents": labels,"colors": ner_displacy_palette})
#        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
#        
#        data = [
#                [str(getattr(ent, attr)) for attr in attrs]
#                for ent in doc.ents
#                if ent.label_ in labels
#                ]
#        
#        dfs.append(pd.DataFrame(data, columns=attrs))
#
#df = pd.concat(dfs)



#html = displacy.render(ent_df.loc[0,'sent'], style="ent")

if page == 'Exploration':
    
    from PIL import Image
    
    
    
    image = Image.open('./data/plot_labels.png')
    st.image(image, caption='Plot of Soft Skills',use_column_width=True)
    
    image = Image.open('./data/plot.png')
    st.image(image, caption='Sunrise by the mountains',use_column_width=True)


    df = df.groupby(['skill'])['sentSkillId'].count().sort_values(ascending = False).to_frame()
    df = df.reset_index()

    st.table(df)