
import streamlit as st
import pandas as pd
import speech_recognition as sr
import requests
import bs4
from bs4 import BeautifulSoup

def app1 (file_name) :
    r = sr.Recognizer()
    with sr.AudioFile(file_name) as source :
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    if st.button('load Audio') :    
        st.audio(file_name,"wav")
    if st.button('Speech to Text') :
        st.write(text)

def app2 () : 
    r = sr.Recognizer()
    st.write('Please talk')
    with sr.Microphone() as source :
        audio_data = r.record(source, duration=10)
        st.write('Recognizing...')
        text = r.recognize_google(audio_data)
        st.write(text)

code_speech = """
import speech_recognition as sr
import streamlit as st

def app1 (file_name) :
    r = sr.Recognizer()
    with sr.AudioFile(file_name) as source :
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    if st.button('load Audio') :    
        st.audio(file_name,"wav")
    if st.button('Speech to Text') :
        st.write(text)


def app2 () : 
    r = sr.Recognizer()
    st.write('Please talk')
    with sr.Microphone() as source :
        audio_data = r.record(source, duration=10)
        st.write('Recognizing...')
        text = r.recognize_google(audio_data)
        st.write(text)


st.title('SPEECH TO TEXT : SMALL APP') 
st.image('welcom.jpg' , output_format='jpg',width=500,use_column_width=True)       
st.sidebar.title('OPTIONS')
genre = st.sidebar.radio(
"What's your choice?",
('None','Learning', 'Music', 'talk'))
if genre == 'Learning' :
    app1("lesson-1.wav")
if genre == 'Music' :
    app1("speech.wav")
if genre == 'talk' :
    app2()
"""

def presentation_speech ():
    st.title('SPEECH TO TEXT ') 
           
    st.sidebar.title('OPTIONS')
    genre = st.sidebar.radio(
    "What's your choice?",
    ('None','Audio', 'talk','code.py'))
    if genre == 'Audio' :
        app1("lesson-1.wav")
    if genre == 'talk' :
        app2()
    if genre == 'code.py' :
        st.write('Vous avez ici tout le code Python bien detaillés!')
        st.code(code_speech,"python") 

def aff_data():  
  uploaded_file = st.file_uploader("data")
  if uploaded_file is not None:
      d = pd.read_csv(uploaded_file) 
      st.write(d) 

codes = """""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, confusion_matrix, classification_report

    #chargement des datasets
    #client_train = pd.read_csv('/content/drive/MyDrive/client_train.csv')
    #invoice_train = pd.read_csv('/content/drive/MyDrive/invoice_train.csv')

    #nombres de clients different
    len(client_train['client_id'].unique()), len(invoice_train['client_id'].unique())

    # les types de variables 
    client_train.info() , invoice_train.info()

    # les données manquantes 
    client_train.isna().sum() , invoice_train.isna().sum()

    #repartion des classes 
    client_train['target'].value_counts(True).plot.pie() # on peut noter que les données sont trop  desequilibrer

    #les variables de type int dans le client
    for col in client_train.select_dtypes('int64'):
        print(f'{col}----------------------',client_train[col].unique())

    #les variables de type int dans le invoice
    for col in invoice_train.select_dtypes('int64'):
        print(f'{col}----------------------',invoice_train[col].unique())

    ##les categories de type object dans le client
    for col in invoice_train.select_dtypes('object'):
        print(f'{col}----------------------',invoice_train[col].unique())

    #repartition des types de compteurs 
    invoice_train['counter_type'].value_counts(True).plot.pie()

    #fonction Merger le invoice et le client
    def merge_data (invoice, client) : 
      return invoice.merge(client)

    #suppression des valeurs aberantes
    def supp_Vl_aberantes (dataset) : 
        iso = IsolationForest()
        iso.fit(dataset.select_dtypes('int64'))
        iso_array = iso.predict(dataset.select_dtypes('int64'))
        dataset[iso_array==-1]['target'].value_counts()
        dataset = dataset[iso_array==1]
        return dataset

    #undersampling function
    def under_sampling (data) :
      iso = IsolationForest()
      for i in range(5): 
        iso.fit(data.select_dtypes('int64'))
        iso_array = iso.predict(data.select_dtypes('int64'))
        data = data[iso_array==1]
      return data    

    #padding function
    def pad_lot(lot_facts, seuil):

        #si le nombre de factures est superieur au seuil on ecarte le reste 
        if len(lot_facts) >= seuil:
            lot_facts_padded = lot_facts.iloc[:seuil, :]

        # si non on complete par la derniere facture pour atteindre le seuil
        else:
            #On calcul l'ecart entre le seuil et la taille du lot
            ecart = seuil - len(lot_facts)
            ecart_df = pd.DataFrame([lot_facts.iloc[-1]]*ecart)
            #On concat lot_fact et le dataframe contenant uniquement la derniere facture de lot_fact ecart fois 
            lot_facts_padded = pd.concat([lot_facts, ecart_df], ignore_index=True)
        return lot_facts_padded

    #fonction qui met dans une liste tous les lots de factures du dataset
    def lister_dataset_padded (data, nb_factures) :
      depart=0
      list_facts= []
      for nb in range(len(data.groupby('client_id'))) :
        list_facts.append(data.iloc[depart:depart + nb_factures,:].drop('client_id', axis=1))
        depart = depart + nb_factures
      return list_facts
    #preprocessing
    def pre_processing (invoice , client) : 

      #merger le invoice et le client
      dataset = merge_data(invoice, client)

      #suppression des valeurs aberantes
      dataset = supp_Vl_aberantes(dataset)

      # division de la base par classe
      class_0 = dataset[dataset['target']==0].reset_index(drop=True)
      class_1 = dataset[dataset['target']==1].reset_index(drop=True)

      #undersampling sur la classe 0
      class_0 = under_sampling(class_0)

      #concatenons class_0 et class_1
      dataset =pd.concat([class_0,class_1], ignore_index=True)
      
      #rangement des factures par client_id et invoice_date
      dataset = dataset.sort_values(by=['client_id','invoice_date'], ignore_index=True)

      # selection de variables 
      cols = ['client_id','tarif_type','counter_coefficient','consommation_level_1','consommation_level_2',
              'consommation_level_3','consommation_level_4','old_index','new_index','months_number','target']
      data = dataset[cols] 
      
      #padding data
      dataset_padded = data.groupby('client_id').apply(pad_lot, 100).reset_index(drop=True)

      #colonne a standardiser et colonnes à categoriser
      col_to_scaling = ['consommation_level_1','consommation_level_2','consommation_level_3','consommation_level_4','new_index','old_index','months_number']
      col_cat = ['tarif_type','counter_coefficient']

      #rendre categorielle les colonnes et rendre numerique ensuite
      dataset_padded[col_cat] = dataset_padded[col_cat].astype('category')
      dataset_padded = pd.get_dummies(dataset_padded, columns=col_cat, drop_first=True)

      #standardiser les variables continues
      std = StandardScaler()
      dataset_padded[col_to_scaling]=std.fit_transform(dataset_padded[col_to_scaling])

      #recuperation de la target
      client = client.set_index(client['client_id']).drop('client_id',axis=1)
      target = client.loc[list(dataset_padded.client_id.unique())].target
      dataset_padded.drop(['target'],axis=1, inplace=True)
      print(dataset_padded.shape)

      #lister_lot sur dataset_padded
      dataset_padded_list = lister_dataset_padded(dataset_padded,100)
    
      #transformer en numpy array la liste des sequences de 100 factures
      label = np.asarray(target)
      data_clean = np.asarray(dataset_padded_list)
    
      return data_clean, label

    data_clean , label = pre_processing(invoice_train, client_train)

    MODELISATION
    #Import package
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.callbacks import TensorBoard

    #Instaciation du model
    def my_model (input_shape) :
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Input(shape=input_shape))
      model.add(tf.keras.layers.LSTM(100))
      model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2()))
      model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2()))
      model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2()))
      #Compilation du model
      model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['acc',tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives()])
      return model
        
    #fonction de prediction
    def y_predict (model, X_val):
      y_pred = []
      for m in model.predict(X_val) : 
        if  m[0] > .5 :
          y_pred.append(1)
        else : 
            y_pred.append(0)
      return np.asarray(y_pred) 

    # Definir les callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)]

    #Entrainement et Evaluation du model
    def evaluation (model) : 
        
        history = model.fit(X_train,y_train, epochs=100, validation_split=.2, batch_size=128, class_weight={0:1, 1:nb_0s//nb_1s},callbacks=callbacks)
        y_pred = y_predict(model, X_val)
        
        print (confusion_matrix(y_val,y_pred))
        print( classification_report(y_val,y_pred))
        print(history.history.keys())

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for false_negatives
        plt.plot(history.history['false_negatives_2'])
        plt.plot(history.history['val_false_negatives_2'])
        plt.title('model false_negatives')
        plt.ylabel('false_negatives')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # summarize history for false_positives
        plt.plot(history.history['false_positives_2'])
        plt.plot(history.history['val_false_positives_2'])
        plt.title('model false_positives')
        plt.ylabel('false_positives')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

   #APPLICATIONS

    X_train, X_val , y_train, y_val = train_test_split(data_clean, label, test_size=.2, random_state=42)

    # equilibrage des classes
    nb_1s = sum(label)
    nb_0s = len(label) - nb_1s

    model = my_model((data_clean.shape[1], data_clean.shape[2]))
    evaluation(model)
"""""
def presentation_fraude() : 
  st.title('DETECTION DE FRAUDE ')
  st.sidebar.title('OPTIONS') 
  #st.image('download.jpeg' , output_format='jpeg',width=1,use_column_width=True)
  option = st.sidebar.selectbox(
  'Vous souhaitez afficher quelle base?',
  ('None','client_train', 'invoice_train')) 
  if option == 'client_train' :
    st.write('Veuillez selectionner :', option)
    aff_data()
  if option == 'invoice_train' :
    st.write('Veuillez selectionner :', option)
    aff_data()
  option = st.sidebar.selectbox(
  'Vous souhaitez afficher ?',
  ('None','code','diapo', 'notebook','visualisation'))
  if option == 'code' :
    st.write('Vous avez ici tout le code Python bien detaillés!')
    st.code(codes,"python")
  if option == 'diapo' :
    st.text('Je vous présente ici le lien vers le DIAPO : veuillez cliquer dessus!')
    st.write('https://docs.google.com/presentation/d/1s7CZnfWyUTTG07cf1Hn9j9XrFFKRBdp2YxEp2FhcDm0/edit#slide=id.gcb9a0b074_1_0')
  if option == 'notebook' :
    st.text('Je vous présente ici le lien vers le notebook sur collab : veuillez cliquer dessus! ')
    st.write('https://colab.research.google.com/drive/1xmqCUDag04rDbwIFKCNUCj63WgDWwJf_#scrollTo=LqjQuMNvTFhG')
  if option == 'visualisation' :
    st.text('Je vous présente ici le lien vers le notebook de la Data visualisation : veuillez cliquer dessus')
    st.write('https://colab.research.google.com/drive/1RFq2rXi0ZacY-DssomuNvkcKe9uEZy9O')

  option = st.sidebar.selectbox(
  'souhaitez - vous afficher quelle courbe de performance du model ?',
  ('None','accuracy','loss', 'false_positives','false_negatives','all'))

  if option == 'accuracy' :
    st.write("Vous avez ici la courbe d'accuracy du model LSTM enrainné!")
    st.image('acc_curve.png' , output_format='png',width=500,use_column_width=True)
  if option == 'loss' :
    st.write("Vous avez ici la courbe de perte du model LSTM enrainné!")
    st.image('loss_curve.png' , output_format='png',width=500,use_column_width=True)  
  if option == 'false_positives' :
    st.write("Vous avez ici la courbe des FP du model LSTM enrainné!")
    st.image('false_positive_curve.png' , output_format='png',width=500,use_column_width=True)
  if option == 'false_negatives' :
    st.write("Vous avez ici la courbe des FN du model LSTM enrainné!")
    st.image('false_negative_curve.png' , output_format='png',width=500,use_column_width=True)
  if option == 'all' :
    st.write("Vous avez ici toutes les courbes de performance du model LSTM enrainné!")
    st.image('acc_curve.png' , output_format='png',width=500,use_column_width=True)   
    st.write("Vous avez ici la courbe de perte du model LSTM enrainné!")
    st.image('loss_curve.png' , output_format='png',width=500,use_column_width=True)   
    st.write("Vous avez ici la courbe des FP du model LSTM enrainné!")
    st.image('false_positive_curve.png' , output_format='png',width=500,use_column_width=True)
    st.write("Vous avez ici la courbe des FN du model LSTM enrainné!")
    st.image('false_negative_curve.png' , output_format='png',width=500,use_column_width=True)


codes_web ="""
import requests
import bs4
from bs4 import BeautifulSoup
import streamlit as st

def recuper_text_bakhtech() :
    text = []
    # get web site
    site = requests.get('https://bakhtech.com/')
    
    # extract html code of website
    site_html = BeautifulSoup(site.content, 'html.parser')
    
    #les sous titres
    contenu = site_html.find_all("div", class_ ="block votreref-bloc")
    
    #liste de de sous titres
    for i in range(len(contenu)): 
        text.append(contenu[i].text)
    return text

def recuper_text_dit() :
    text = []
    # get web site
    site = requests.get('https://dit.sn/')
    
    # extract html code of website
    site_html = BeautifulSoup(site.content, 'html.parser')
    
    #les sous titres
    contenu = site_html.find_all("div", class_ ="et_pb_blurb_container")
    
    #liste de de sous titres
    for i in range(len(contenu)): 
        text.append(contenu[i].text)
    return text  

def app_bakhtech(): 
    st.image('bakhtech.png' , output_format='png',width=None,use_column_width=True,clamp=False)   
    text = recuper_text_bakhtech()
    for i in range(len(text)) :
        st.write(text[i])

def app_dit(): 
    st.image('dit.png' , output_format='png',width=None,use_column_width=True,clamp=False)   
    text = recuper_text_dit()
    for i in range(len(text)) :
        st.write(text[i])

st.sidebar.title('OPTIONS')
option = st.sidebar.selectbox(
  'Choisit votre site svp!',
  ('None','DIT','BAKHTECH'))
if option == 'DIT' :
    app_dit() 
if option == 'BAKHTECH' :
    app_bakhtech() 
"""

def recuper_text_bakhtech() :
    text = []
    # get web site
    site = requests.get('https://bakhtech.com/')
    
    # extract html code of website
    site_html = BeautifulSoup(site.content, 'html.parser')
    
    #les sous titres
    contenu = site_html.find_all("div", class_ ="block votreref-bloc")
    
    #liste de de sous titres
    for i in range(len(contenu)): 
        text.append(contenu[i].text)
    return text

def recuper_text_dit() :
    text = []
    # get web site
    site = requests.get('https://dit.sn/')
    
    # extract html code of website
    site_html = BeautifulSoup(site.content, 'html.parser')
    
    #les sous titres
    contenu = site_html.find_all("div", class_ ="et_pb_blurb_container")
    
    #liste de de sous titres
    for i in range(len(contenu)): 
        text.append(contenu[i].text)
    return text  

def app_bakhtech(): 
    st.image('bakhtech.png' , output_format='png',width=None,use_column_width=True,clamp=False)   
    text = recuper_text_bakhtech()
    for i in range(len(text)) :
        st.write(text[i])

def app_dit(): 
    st.image('dit.png' , output_format='png',width=None,use_column_width=True,clamp=False)   
    text = recuper_text_dit()
    for i in range(len(text)) :
        st.write(text[i])

def app_web () :
    st.sidebar.title('OPTIONS')
    option = st.sidebar.selectbox(
    'Choisit votre site svp!',
    ('None','DIT','BAKHTECH','CODE.py'))
    if option == 'None' :
        st.image('scraping.jpeg' , output_format='jpeg',width=None,use_column_width=True,clamp=False)  
    if option == 'DIT' :
        app_dit() 
    if option == 'BAKHTECH' :
        app_bakhtech()          
    if option == 'CODE.py' :
        st.write('Voici le code détaillé !!!')
        st.code(codes_web,'python') 

def remerciements () :
    if st.sidebar.button('DEDICACES') :
        st.write('CES QUELQUES PAROLES ME VIENNENT DU COEUR!!! ')
        st.image('dedicace.jpg' , output_format='jpg',width=None,use_column_width=True,clamp=False)
    if st.sidebar.button('REMERCIEMNETS') :
        st.image('remercier.jpg' , output_format='jpg',width=None,use_column_width=True,clamp=False)    


st.image('ROUMNA.png' , output_format='png',width=None,use_column_width=True,clamp=False)
genre = st.radio(
"What's your Option?",
('Welcom','FRAUD DETECTION', 'SPEECH TO TEXT','WEB SCRAPING','Bye'))
if genre == 'Welcom' :
    st.image('welcom.jpg' , output_format='jpg',width=500,use_column_width=True)
    remerciements()
if genre == 'FRAUD DETECTION' :
    presentation_fraude()
if genre == 'SPEECH TO TEXT' :
    presentation_speech()  
if genre == 'WEB SCRAPING' :
    app_web()      
if genre == 'Bye' :
    st.image('love_datascience.jpeg' , output_format='jpeg',width=500,use_column_width=True)
