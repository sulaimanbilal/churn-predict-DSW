import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly
import openpyxl  


def encode_data(dataframe_series):
        if dataframe_series.dtype=='object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
        return dataframe_series

def show_predict_page():
    colT1,colT2 = st.columns([1,8])
    with colT2:
        st.title("Churn Predict DSW by Aruci")

    st.write(" Get a insight and solution by uploading your dataset! - Aruci ")
    
    #import
    dataset_file = st.file_uploader("Upload dataset",
                                    type=['xlsx'])
    if dataset_file is not None:
        dataset = pd.read_excel(dataset_file)
        dataset = dataset.drop(['Churn Label'],axis = 1)
        dataset_cpy = dataset.copy()
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
    
        with open('modelv2','rb') as m :
            mod = pickle.load(m)

        data_predict = mod.predict(x)
        df_predict = pd.DataFrame(data_predict,columns=['churn_predict'])
        df_predict_1 = df_predict.copy()
        df_predict_loc = df_predict.copy()
        mapping = {0: 'No', 1: 'Yes'}
        df_predict_1['churn_predict'] = df_predict_1['churn_predict'].replace(mapping)
        #Simple EDA

        data_eda = pd.concat([dataset,df_predict_1],axis='columns')
        data_eda_loc = data_eda.copy()
        data_eda2 = data_eda.drop(['customer_id','latitude','longitude','location','cltv'], axis = 1)
        data_dummies= pd.concat([dataset,df_predict],axis='columns')
        eda_dummies = pd.get_dummies(data_dummies[['churn_predict','games_product', 'music_product', 'education_product' ,
                                            'call_center', 'video_product', 'use_myapp','device_class','payment_method']], dtype=int)
        eda_dummies_loc = eda_dummies.copy()
        st.subheader(""" GENERAL EDA """)
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
        filtered_data = data_eda[data_eda['churn_predict'] == 'Yes']

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

        #LOCATION
        #change nama column
        dataset_cpy.rename(columns={'Customer ID':'customer_id','Tenure Months':'tenure_months','Location':'location',
                   'Device Class':'device_class', 'Games Product':'games_product', 'Music Product' : 'music_product',
                   'Education Product' : 'education_product', 'Call Center': 'call_center', 'Video Product': 'video_product',
                   'Use MyApp' : 'use_myapp', 'Payment Method': 'payment_method', 'Monthly Purchase (Thou. IDR)':'monthly_purchase', 
                   'Longitude': 'longitude', 'Latitude': 'latitude','CLTV (Predicted Thou. IDR)':'cltv'},inplace=True)

        list_customer_loc = dataset_cpy['customer_id']
        id_customer_loc = pd.DataFrame(list_customer_loc)


        data_location = dataset_cpy['location'].unique()
        for i in data_location:
            data_process_loc = data_eda_loc[data_eda_loc["location"] == i]
            
            value_churn = (data_process_loc['churn_predict'] == 'Yes').sum()
            value_nchurn = (data_process_loc['churn_predict'] == 'No').sum()

            percent_churn = (value_churn / (value_churn + value_nchurn)) * 100
            percent_nchurn = (value_nchurn / (value_churn + value_nchurn)) * 100

            
            if percent_churn >= 0.25 :
                data_eda_loc_2 = data_process_loc.drop(['customer_id','latitude','longitude','location','cltv'], axis = 1)
                data_dummies_loc= pd.concat([dataset[dataset["location"] == i],df_predict],axis='columns')
                eda_dummies_loc = pd.get_dummies(data_dummies_loc[['churn_predict','games_product', 'music_product', 'education_product' ,
                                            'call_center', 'video_product', 'use_myapp','device_class','payment_method']], dtype=int)
                
                colT1,colT2 = st.columns([4,6])
                with colT2:
                    st.subheader(" Region : " + i)
                    
                st.write("""## EDA ##""")
                #summary stat
                st.write("""Summary Stat""")
                st.write(data_eda_loc_2.describe())
                #Piechart churn label 
                st.write("""Piechart churn label""")
                fig = px.pie(data_process_loc.groupby('churn_predict')['customer_id'].nunique().reset_index(), 
                    values='customer_id', 
                    names='churn_predict')
                st.write(fig)

                #Monthly purchase Customer
                st.write("""Bar Chart Monthly Purchase Customer""")
                fig_mp = px.histogram(data_process_loc, x="monthly_purchase", color="churn_predict",
                    marginal="box"
                    )
                st.write(fig_mp)

                data_mp_churn = (data_process_loc[data_process_loc['churn_predict'] == 'Yes']  )     
                mp_churn = data_mp_churn['monthly_purchase'].median() 

                data_mp_nchurn = (data_process_loc[data_process_loc['churn_predict'] == 'No']  )     
                mp_nchurn =data_mp_nchurn['monthly_purchase'].median()

                #Chart factor churn
                st.write("""Bar Chart Factor Churn""")
                fig_fc = px.bar(eda_dummies_loc.corr()['churn_predict'].sort_values(ascending = False), 
                    color = 'value')
                st.write(fig_fc)

                #Solution
                st.write("""## Solution ##""")
                st.write("The following are factors that cause customer churn: ")
                data_sol = dataset.drop(['customer_id','latitude','longitude','location','cltv'], axis = 1)
                data_sol = pd.concat([data_sol,df_predict_loc],axis="columns")
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

                service_factor = ['call_center_No','call_center_Yes', 'use_myapp_Yes','use_myapp_No']

                
                #get solution
                for f in factor['factor'] :
                    if f in internet_factor:
                        if f in internet_yes:
                            if data_mp_churn > mp_nchurn :
                                st.write("Factor Churn : ", f)
                                st.write("Solution : ")
                                st.write("- There is a problem with the internet service, you are expected to fix the internet service. Do not forget to communicate customers to use the call center service if there is a problem. ")
                                st.write("- Since the median purchase of customers who churn is greater than the median purchase of customers who do not churn, perhaps the price of the service is too high and not commensurate with the service provided. Offering a discount or lowering the price might solve this. ")
                            else:
                                st.write("Factor Churn : ", f)
                                st.write("Solution : ")
                                st.write("- There is a problem with the internet service, you are expected to fix the internet service. Do not forget to communicate customers to use the call center service if there is a problem. ")
                        elif f in internet_no:
                            if mp_churn > mp_nchurn :
                                st.write("Factor Churn : ", f)
                                st.write("Solution : ")
                                st.write("- The company may need to promote the use of internet services, this may reduce customers will churn. Do not forget to communicate customers to use the call center service if there is a problem. ")
                                st.write("- Since the median purchase of customers who churn is greater than the median purchase of customers who do not churn, perhaps the price of the service is too high and not commensurate with the service provided. Offering a discount or lowering the price might solve this. ")
                            else:
                                st.write("Factor Churn : ", f)
                                st.write("Solution : ")
                                st.write("- The company may need to promote the use of internet services, this may reduce customers will churn. Do not forget to communicate customers to use the call center service if there is a problem. ")
                    elif f in payment_method:
                        if f in pulsa:
                            st.write("Factor Churn : ", f)
                            st.write("Solution : Payment using credit is less efficient, the company may need to promote the use of electronic payment, this may reduce customers will churn. Do not forget to communicate customers to use the call center service if there is a problem. ")
                        elif f in nonpulsa:
                            st.write("Factor Churn : ", f)
                            st.write("Solution : There are problems in using electronic payments, the company is advised to make technical improvements to electronic payments.  Do not forget to communicate customers to use the call center service if there is a problem. ")
                    elif f in device:
                        if f in device_low:
                            st.write("Factor Churn : ", f)
                            st.write("Solution : There is no information that can be known, this is due to the device used by the customer or something else. Do not forget to communicate customers to use the call center service if there is a problem.")
                        else:
                            st.write("Factor Churn : ", f)
                            st.write("Solution : Perbaikan terhadap layanan !")
                    elif f in service_factor:
                        st.write("Solution : Perbaikan terhadap layanan !")
            else:
                st.write(percent_churn)
                st.write(percent_nchurn)
        return()
