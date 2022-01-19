import streamlit as st

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.title('Décision d\'octroi de crédit')
#getting our trained model from a file we created earlier
model = pickle.load(open("modele.pkl","rb")) 
df_clients = pd.read_csv('df_clients.csv')
st.subheader('Raw data')
st.write(df_clients)




from lime import lime_tabular

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}



def lime(sk_id):
    if sk_id in df_clients['SK_ID_CURR'].values:

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(df_clients.iloc[:,1:]),
            feature_names=df_clients.iloc[:,1:].columns,
            class_names=['bad', 'good'],
            mode='classification'
        )
        
        test_1 = df_clients.iloc[df_clients[df_clients['SK_ID_CURR']==sk_id].index.values[0],1:]
        
        lime_exp = lime_explainer.explain_instance(
            data_row=test_1,
            predict_fn=model.predict_proba
        )
        
        st.write(df_clients.iloc[df_clients[df_clients['SK_ID_CURR']==sk_id].index.values[0],:])
        # lime_exp.as_pyplot_figure()
        # # fig=plt.tight_layout()
        # st.pyplot()

        lime_exp.as_pyplot_figure()
        st.pyplot()
        lime_exp.html(lime_exp.as_html(), height=800)







def predict(sk_id):    
    data = fetch(session,f"https://p7-oc-ql.herokuapp.com/predict?sk_id={index}")
    if len(data)==2: #data[0]:prediction, data[1]=proba
        st.write('La probabilité que ce client rembourse est de ',data[1])        
        if data[0]['predictions']:
            st.write('Le crédit est refusé')
        else:
            st.write('Le crédit est accordé')   
    else :
        st.write('identifiant incorrect')



session = requests.Session()
with st.form("my_form"):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    index = st.number_input("Entrez l'identifiant du client",min_value =0)
    submitted = st.form_submit_button("Valider")

    if submitted:
        st.write("Résultat :")
        predict(index)
        lime(index)
# if __name__ == "__main__":
#     app.run(debug=True)