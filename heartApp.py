import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
knn = pickle.load(open('KNN.pkl','rb'))

#load dataset
data = pd.read_csv('Heart_Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Penyakit Jantung')

html_layout1 = """
<br>
<div style="background-color:black ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Penyakit Jantung Checkup</b></h2>
</div>
<br>
<br>
"""

background_image = """
<style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://images.unsplash.com/photo-1628348070889-cb656235b4eb?q=80&w=870&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: 100vw 100vh;
        bacground-position: center;
        background-repeat: no-repeat;
        background-color: rgba(255, 255, 255, 0.5);
    }
"""

st.markdown(background_image, unsafe_allow_html=True)
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['KNN','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)

# Center the sidebar
col1, col2, col3 = st.columns(3)

st.sidebar.header('Data Pasien')

sns.set_style('darkgrid')

with col1:
    age = st.slider('Usia',10,100,67)

with col2:
    sex = st.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
    if sex == 'Laki-laki':
        sex = 1
    else:
        sex = 0

with col1:
    cp = st.selectbox('Jenis Nyeri Dada', ['angina typikal (nyeri dada khas)', 'angina atipikal (nyeri dada tidak khas)', 'non-anginal (Nyeri dada non-anginal)', 'normal'])
    if cp == 'angina typikal (nyeri dada khas)':
        cp = 0
    elif cp == 'angina atipikal (nyeri dada tidak khas)':
        cp = 1
    elif cp == 'non-anginal (Nyeri dada non-anginal)':
        cp = 2
    else:
        cp = 3

with col2:
    trtbps = st.slider('Tekanan Darah Istirahat', 90, 200, 120)

with col1:
    chol = st.slider('Kolestrol', 126, 564, 233)

with col2:
    fbs = st.selectbox('Gula Darah > 120 mg/dl', ['Tidak', 'Ya'])
    if fbs == 'Tidak':
        fbs = 0
    else:
        fbs = 1

with col1:
    restecg = st.selectbox('Kondisi Jantung Saat Istirahat ', ['Normal', 'memiliki kelainan gelombang ST-T', 'hipertrofi ventrikel kiri'])
    if restecg == 'Normal':
        restecg = 0
    elif restecg == 'memiliki kelainan gelombang ST-T':
        restecg = 1
    else:
        restecg = 2

with col2:
    thalachh = st.slider('Denyut Jantung Maksimal', 71, 202, 150)

with col1:
    exng = st.selectbox('Indikator Angina pada Saat Berolahraga ', ['Tidak', 'Ya'])
    if exng == 'Tidak':
        exng = 0
    else:
        exng = 1

with col2:
    oldpeak = st.slider('Indeks Depresi ST pada Saat Berolahraga ', 0.0, 6.2, 2.0)

with col1:
    slp = st.radio('Kondisi Slope ST pada Saat Berolahraga ', ['Naik', 'Datar', 'Menurun'])
    if slp == 'Naik':
        slp = 0
    elif slp == 'Datar':
        slp = 1
    else:
        slp = 2

with col2:
    caa = st.slider('Intensitas Pewarnaan Fluoroskopi pada Pembuluh Utama', 0, 3, 2)

with col1:
    thall = st.selectbox('Status Thalassemia ', ['Normal', 'Cacat Tetap', 'Cacat Reversibel'])
    if thall == 'Normal':
        thall = 0
    elif thall == 'Cacat Tetap':
        thall = 1
    else:
        thall = 2
    

#train test split
X = data.drop('output',axis=1)
y = data['output']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

def user_report():

    if st.sidebar.checkbox("Tentang Dataset"):
        html_layout2 ="""
        <br>
        <p>Ini adalah dataset UCI</p>
        """
        st.markdown(html_layout2,unsafe_allow_html=True)
        st.subheader('Dataset')
        st.write(data.head(10))
        st.subheader('Describe dataset')
        st.write(data.describe())

    if st.sidebar.checkbox('EDA'):
        pr =ProfileReport(data,explorative=True)
        st.header('*Input Dataframe*')
        st.write(data)
        st.write('---')
        st.header('*Profiling Report*')
        st_profile_report(pr)

    #Training Data
    if st.sidebar.checkbox('Train-Test Dataset'):
        st.subheader('X_train')
        st.write(X_train.head())
        st.write(X_train.shape)
        st.subheader("y_train")
        st.write(y_train.head())
        st.write(y_train.shape)
        st.subheader('X_test')
        st.write(X_test.shape)
        st.subheader('y_test')
        st.write(y_test.head())
        st.write(y_test.shape)
            
    user_report_data = {
        'age':age,
        'sex':sex,
        'cp':cp,
        'trtbps':trtbps,
        'chol':chol,
        'fbs':fbs,
        'restecg':restecg,
        'thalachh':thalachh,
        'exng':exng,
        'oldpeak':oldpeak,
        'slp':slp,
        'caa':caa,
        'thall':thall
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = knn.predict(user_data)
knn_score = accuracy_score(y_test,knn.predict(X_test))

#output
with col3:
    st.subheader('Hasilnya adalah : ')
    output=''
    if user_result[0]==0:
        output='Kamu Aman'
    else:
        output ='Kamu terkena penyakit jantung'
    st.success(output)
    st.subheader('Model yang digunakan : \n'+option)
    st.subheader('Accuracy : ')
    st.write(str(knn_score*100)+'%')