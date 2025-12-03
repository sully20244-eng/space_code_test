# # importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import pickle


# # configure page
st.set_page_config(page_title="Facebook Ads Analytics", layout='wide') 

# # loading data
df = pd.read_csv('facebook_ads.csv', encoding="ISO-8859-1")

# # sidebar
option = st.sidebar.selectbox("Pick a choice:", ['Home','EDA','ML'])

# # -------------------------------- HOME ------------------------------------
if option == 'Home':
     st.title("📊 Facebook Ads Analytics App")
     st.markdown("### 👨‍💻 Author: **Mohamed Sulaiman**")
     st.write("This dashboard visualizes user behavior and predicts whether a user will click on a Facebook ad.")
     st.dataframe(df.head())

# # -------------------------------- EDA -------------------------------------
elif option == 'EDA':
     st.title("📈 Exploratory Data Analysis (EDA)")
     st.markdown("Gain insights into how users interact with ads based on **Time Spent on Site**, **Salary**, and whether they **Clicked**.")

#     # --- Layout ---
     col1, col2 = st.columns(2)

#     # ---------------- Scatter Plot ----------------
     st.subheader("1️⃣ Time on Site vs Salary (Colored by Clicked)")
     fig = px.scatter(
         df,
         x='Time Spent on Site',
         y='Salary',
         color='Clicked',
         size='Salary',
         trendline="ols",
         title="Relationship Between Time on Site & Salary",
         color_discrete_map={0: "gray", 1: "red"}
     )
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("#### 🔍 Insight:")
     st.info("""
     • Users who **click ads** tend to spend **more time on the website**.  
     • Higher salary does **not strongly influence** clicking behavior.  
     """)

#     # ---------------- Distribution of Time ----------------
     with col1:
        st.subheader("2️⃣ Time Spent on Site Distribution")
        fig = px.histogram(
             df,
             x='Time Spent on Site',
             nbins=25,
             color='Clicked',
             marginal='box',
             title="Distribution of Time Spent on Website"
         )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 Insight:")
        st.info("""
         • The majority spend **lower to mid-range** time on site.  
         • Users who spend **very high time** are more likely to click.
         """)

#     # ---------------- Clicked Count ----------------
     with col2:
         st.subheader("3️⃣ Clicked vs Not Clicked")
         fig = px.pie(
             df,
             names='Clicked',
             title="Percentage of Users Who Clicked Ads",
             color='Clicked',
             color_discrete_map={0: "skyblue", 1: "red"},
             hole=0.4
         )
         st.plotly_chart(fig, use_container_width=True)

         st.markdown("#### 🔍 Insight:")
         st.info("""
         • A **small percentage** of users actually click ads.  
         • Most users scroll without interacting.
         """)

#     # ---------------- Salary Distribution ----------------
     st.subheader("4️⃣ Salary Distribution by Clicked Status")
     fig = px.violin(
         df,
         y='Salary',
         x='Clicked',
         box=True,
         color='Clicked',
         title="Salary Levels for Clicking vs Not Clicking Users"
     )
     st.plotly_chart(fig, use_container_width=True)

     st.markdown("#### 🔍 Insight:")
     st.info("""
     • Salary does **not significantly differentiate** who clicks ads.  
     • Ad behavior seems driven more by **time on website** than income.
     """)

# # -------------------------------- ML -------------------------------------
elif option == "ML":
     st.title("🤖 Ads Click Prediction Model")
     st.write("Enter the user inputs below to predict if they would click an ad:")

     time = st.number_input("⏱ Time Spent on Website (minutes)")
     salary = st.number_input("💰 User Salary ($)")

     btn = st.button("Predict")

     ms = MinMaxScaler()
     clf = pickle.load(open('my_model.pkl','rb'))
     result = clf.predict(ms.fit_transform([[time, salary]]))

     if btn:
         if result == 1:
             st.success("✔ The user is **LIKELY to click the ad**.")
         else:
             st.error("✖ The user is **NOT likely to click the ad**.")
