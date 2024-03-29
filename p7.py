import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from lime import lime_tabular

st.title('Décision d\'octroi de crédit')
#getting our trained model from a file we created earlier
model = pickle.load(open("modele.pkl","rb")) 
df_clients = pd.read_csv('df_clients.csv')
df_voisins = pd.read_csv('df_voisins.csv')
df_target = pd.read_csv('df_target.csv')




st.subheader('Liste des clients')
st.write(df_clients)




def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}



def explainer_lime(sk_id):
    if sk_id in df_clients['SK_ID_CURR'].values:

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(df_clients.iloc[:,1:]),
            feature_names=df_clients.iloc[:,1:].columns,
            class_names=['good', 'bad'],
            mode='classification'
        )
        
        test_1 = df_clients.iloc[df_clients[df_clients['SK_ID_CURR']==sk_id].index.values[0],1:]
        
        lime_exp = lime_explainer.explain_instance(
            data_row=test_1,
            predict_fn=model.predict_proba,
            num_features=20
        )
        

        
        html = lime_exp.as_html()
        components.html(html, width=1000,height=600)




def indicateur(sk_id,data):
    seuil=data[0]['seuil']
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = 100*data[1],
        mode = "gauge+number",
        title = {'text': "Probabilité de remboursement (%)"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "black"},
                 'steps' : [
                     {'range': [0, 100*(1-seuil)], 'color': "red"},
                     {'range': [100*(1-seuil), 100], 'color': "green"}]}))

    st.plotly_chart(fig)




def predict(sk_id,data):        
    st.write('La probabilité que ce client rembourse est de ',data[1])        
    if data[0]['predictions']:
        st.write('Le crédit est refusé')
    else:
        st.write('Le crédit est accordé')   
  



def voisins(sk_id,a,b): 
    df_clients['TARGET']=df_target['TARGET']
    
    fig = px.scatter(df_clients[df_clients['SK_ID_CURR']==sk_id], 
                                 x=a, 
                                 y=b,
                                 text='SK_ID_CURR')
    
    fig2 = px.scatter(df_clients[df_clients['SK_ID_CURR'].isin(
        df_voisins[df_voisins['SK_ID_CURR']==sk_id].values[0])], 
        x=a, 
        y=b,
        color='TARGET',
        hover_name='SK_ID_CURR')
    
    

    fig3 = go.Figure(data=fig.data + fig2.data)
    
    fig3.update_layout(coloraxis = dict(
        cauto=False, 
        cmin=0, 
        cmax=1))
    
    fig3.update_coloraxes(colorscale=[[0, "green"],[1,"red"]]) 
    
    fig3.update_layout(
        xaxis_title=a,
        yaxis_title=b,
        coloraxis_colorbar=dict(title="TARGET"))
  
    st.write(fig3)
    
 
    
session = requests.Session()
index = st.number_input("Entrez l'identifiant du client",min_value =0)
 

 
 
def main():
    st.write("Résultat :")
    
    data = fetch(session,f"https://p7-oc-ql.herokuapp.com/predict?sk_id={index}")
    if len(data)==2: #data[0]:prediction, data[1]=proba
        indicateur(index,data)
        predict(index,data)    
        
        with st.expander("Explications"):
            explainer_lime(index)        
        
        with st.expander("Comparaison avec les clients ayant des caractéristiques similaires"):
            a=st.selectbox('variable x', options=df_clients.columns)
            b=st.selectbox('variable y', options=df_clients.columns)
            voisins(index,a,b)
    else :
        st.write('identifiant incorrect')


main()        
            

   
