import streamlit as st
import pickle
import pandas as pd
 
df = pd.read_csv('cwurData.csv')
 
def load_model():
    with open('wur_prediction.pkl', 'rb') as f:
        return pickle.load(f)
 
st.header("University Score Prediction App")
 
with st.sidebar:
    st.image('parami.jpg')
    st.text("Student Name: Chue Myat Noe")
    st.text("Student ID: PIUS20240082")
    st.text("Class of 2028")
 
col1, col2 = st.columns(2)
 
with col1:
    var1 = st.slider("Quality of education (rank)",
                     int(df['quality_of_education'].min()),
                     int(df['quality_of_education'].max()))
    var2 = st.slider("Alumni employment (rank)",
                     int(df['alumni_employment'].min()),
                     int(df['alumni_employment'].max()))
    var3 = st.slider("Quality of faculty (rank)",
                     int(df['quality_of_faculty'].min()),
                     int(df['quality_of_faculty'].max()))
    var4 = st.slider("Publications (rank)",
                     int(df['publications'].min()),
                     int(df['publications'].max()))
 
with col2:
    var5 = st.slider("Influence (rank)",
                     int(df['influence'].min()),
                     int(df['influence'].max()))
    var6 = st.slider("Citations (rank)",
                     int(df['citations'].min()),
                     int(df['citations'].max()))
    var7 = st.slider("Patents (rank)",
                     int(df['patents'].min()),
                     int(df['patents'].max()))
    var8 = st.selectbox("Country", df['country'].unique())
 
predict_button = st.button("Predict University Score")
 
if predict_button:
    input_data = pd.DataFrame([{
        "quality_of_education": var1,
        "alumni_employment": var2,
        "quality_of_faculty": var3,
        "publications": var4,
        "influence": var5,
        "citations": var6,
        "patents": var7,
        "country": var8
    }])
 
    model = load_model()
    result = model.predict(input_data)[0]
    st.write(f"Predicted CWUR Score: {result:.2f}")

    same_country = df[df['country'] == var8].copy()
    same_country['difference'] = (same_country['score'] - result).abs()
    similar = same_country.sort_values('difference').head(5)
    st.write(f"Universities in {var8} with a similar score:")
    st.dataframe(similar[['institution', 'country', 'year', 'score']])