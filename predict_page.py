import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly


def encode_data(dataframe_series):
        if dataframe_series.dtype=='object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
        return dataframe_series

def show_predict_page():
    st.title("Churn Predict")
    st.write("""### We need some data for predict ###""")
    

    #import
    dataset_file = st.file_uploader("Upload dataset",
                                    type=['xlsx'])
    if dataset_file is not None:
        dataset = pd.read_excel(dataset_file)
        dataset = dataset.drop(['Churn Label'],axis = 1)
        #change nama column
        dataset.rename(columns={'Customer ID':'customer_id','Tenure Months':'tenure_months','Location':'location',
                        'Device Class':'device_class', 'Games Product':'games_product', 'Music Product' : 'music_product',
                        'Education Product' : 'education_product', 'Call Center': 'call_center', 'Video Product': 'video_product',
                        'Use MyApp' : 'use_myapp', 'Payment Method': 'payment_method', 'Monthly Purchase (Thou. IDR)':'monthly_purchase', 
                        'Longitude': 'longitude', 'Latitude': 'latitude','CLTV (Predicted Thou. IDR)':'cltv'},inplace=True)
        

        #drop column
        data_procces = dataset.drop(['customer_id','latitude','longitude','location','cltv'], axis = 1)

    
        data_procces = data_procces.apply(lambda x: encode_data(x))

        #scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        data_procces[['monthly_purchase','tenure_months']] = scaler.fit_transform(dataset[['monthly_purchase','tenure_months']])

        x = data_procces.values
    
        with open('model','rb') as m :
            mod = pickle.load(m)

        data_predict = mod.predict(x)
        df_predict = pd.DataFrame(data_predict,columns=['churn_predict'])

        value_churn = (df_predict['churn_predict'] == 1).sum()
        value_nchurn = (df_predict['churn_predict'] == 0).sum()

        percent_churn = (value_churn / (value_churn + value_nchurn)) * 100
        percent_nchurn = (value_nchurn / (value_churn + value_nchurn)) * 100

        #Simple EDA

        data_eda = pd.concat([dataset,df_predict],axis='columns')
        data_eda2 = data_eda.drop(['customer_id','latitude','longitude','location','cltv'], axis = 1)
        eda_dummies = pd.get_dummies(data_eda[['churn_predict','games_product', 'music_product', 'education_product' ,
                                            'call_center', 'video_product', 'use_myapp','device_class','payment_method']], dtype=int)

        st.write("""## EDA ##""")
        #summary stat
        st.write("""Summary Stat""")
        st.write(data_eda2.describe())
        #Piechart churn label 
        st.write("""Piechart churn label""")
        fig = px.pie(data_eda.groupby('churn_predict')['customer_id'].nunique().reset_index(), 
             values='customer_id', 
             names='churn_predict')
        st.write(fig)

        #Location 
        st.write("""Piechart Location""")
        fig_loc = px.pie(data_eda.groupby('location')['customer_id'].nunique().reset_index(), 
             values='customer_id', 
             names='location')
        st.write(fig_loc)

        #Churn bedasarkan loc
        #Piechart churn label == yes bedasarkan lokasi
        filtered_data = data_eda[data_eda['churn_predict'] == 1]

        #Piechart churn label 
        st.write("""Piechart Churn Based on Location""")
        fig_churn_loc = px.pie(filtered_data.groupby('location')['customer_id'].count().reset_index(), 
             values= 'customer_id', 
             names= 'location')
        st.write(fig_churn_loc)

        #Customer lifetime
        st.write("""Bar Chart Customer Lifetime""")
        fig_lt = px.histogram(data_eda, x="tenure_months", color="churn_predict",marginal="box" )
        st.write(fig_lt)

        #Monthly purchase Customer
        st.write("""Bar Chart Monthly Purchase Customer""")
        fig_mp = px.histogram(data_eda, x="monthly_purchase", color="churn_predict",
                   marginal="box"
                  )
        st.write(fig_mp)

        #Bar Chart Payment Method Churn
        st.write("""Bar Chart Payment Method Churn""")
        fig_pmc = px.bar(data_eda.groupby(['payment_method',
                                                'churn_predict'])['customer_id'].count().reset_index(),
             x="customer_id",
             y="payment_method", 
             color="churn_predict", 
             text = 'customer_id'
            )
        st.write(fig_pmc)

        #Chart factor churn
        st.write("""Bar Chart Factor Churn""")
        fig_fc = px.bar(eda_dummies.corr()['churn_predict'].sort_values(ascending = False), 
             color = 'value')
        st.write(fig_fc)



        #Solution
        st.write("""## Solution ##""")
        data_sol = dataset.drop(['customer_id','latitude','longitude','location','cltv'], axis = 1)
        data_sol = pd.concat([data_sol,df_predict],axis="columns")
        data_sol = pd.get_dummies(data_sol, dtype=int)

        factor = data_sol.corr()['churn_predict'].sort_values(ascending = False)
        factor = pd.DataFrame(factor,columns=['churn_predict'])
        factor = factor.iloc[1:4].reset_index()
        factor.columns = ['factor','rate_churn']

        #factor
        internet_factor = ['games_product_No','games_product_Yes','video_product_No',
                           'video_product_Yes','music_product_No','music_product_Yes',
                           'education_product_Yes','education_product_No',
                           'education_product_No internet service',
                            'video_product_No internet service',
                            'music_product_No internet service',
                            'games_product_No internet service']
        
        internet_no = ['games_product_No','video_product_No','music_product_No','education_product_No',
                       'education_product_No internet service',
                       'video_product_No internet service',
                       'music_product_No internet service',
                       'games_product_No internet service']
        
        internet_yes = ['games_product_Yes','video_product_Yes','music_product_Yes',
                        'education_product_Yes']


        payment_method = ['payment_method_Pulsa','payment_method_Digital Wallet','payment_method_Debit',
                          'payment_method_Credit']
        
        pulsa = ['payment_method_Pulsa']

        nonpulsa = ['payment_method_Digital Wallet','payment_method_Debit',
                          'payment_method_Credit']
        
        device= ['device_class_High End','device_class_Mid End','device_class_Low End']

        device_low = ['device_class_Low End']



        #get solution
        for i in factor['factor'] :
            if i in internet_factor:
                if i in internet_yes:
                    st.write("Factor Churn : ", i)
                    st.write("Perbaikan terhadap layanan !")
                elif i in internet_no:
                    st.write("Factor Churn : ", i)
                    st.write("Butuh promosi !")
            elif i in payment_method:
                if i in pulsa:
                    st.write("Factor Churn : ", i)
                    st.write("Butuh Promosi !")
                elif i in nonpulsa:
                    st.write("Factor Churn : ", i)
                    st.write("Perbaikan terhadap layanan !")
            elif i in device:
                if i in device_low:
                    st.write("Factor Churn : ", i)
                    st.write("Penyebab tidak diketahui!")
                else:
                    st.write("Factor Churn : ", i)
                    st.write("Perbaikan terhadap layanan !")