import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import base64

st.set_page_config(page_title="Hotel Booking Analytics Dashboard", layout="wide")
st.title("Hotel Booking Analytics Dashboard")
st.markdown("""
Explore hotel bookings with powerful analytics:  
Visualize trends, segment customers, predict outcomes, and uncover business opportunities.
""")

@st.cache_data
def load_data():
    return pd.read_excel("hotel_bookings.xlsx")

# Data loading
try:
    df = load_data()
except FileNotFoundError:
    st.warning("hotel_bookings.xlsx not found! Please upload via the sidebar.")
    df = None

# Sidebar upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a hotel bookings Excel or CSV file", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")

def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

if df is not None:
    tabs = st.tabs([
        "Data Visualization",
        "Classification",
        "Clustering",
        "Association Rule Mining",
        "Regression"
    ])

    # ============== DATA VISUALIZATION TAB (Plotly only) ==============
    with tabs[0]:
        st.header("Data Visualization")
        st.markdown("**Explore and filter descriptive insights from the hotel bookings dataset.**")
        col1, col2, col3 = st.columns(3)
        with col1:
            hotel_type = st.selectbox("Select Hotel Type", options=["All"] + list(df['hotel'].unique()))
        with col2:
            market_segment = st.selectbox("Market Segment", options=["All"] + list(df['market_segment'].unique()))
        with col3:
            year = st.selectbox("Arrival Year", options=["All"] + sorted(df['arrival_date_year'].unique()))

        filtered_df = df.copy()
        if hotel_type != "All":
            filtered_df = filtered_df[filtered_df['hotel'] == hotel_type]
        if market_segment != "All":
            filtered_df = filtered_df[filtered_df['market_segment'] == market_segment]
        if year != "All":
            filtered_df = filtered_df[filtered_df['arrival_date_year'] == year]

        st.markdown("### Key Descriptive Insights")
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

        # 1. Booking counts by month
        temp1 = filtered_df.groupby('arrival_date_month').size().reindex(month_order)
        fig1 = px.bar(temp1.reset_index(), x='arrival_date_month', y=0, labels={'0': 'Booking Count', 'arrival_date_month': 'Month'}, title='Bookings by Month')
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Shows seasonality in bookings.")

        # 2. Booking cancellation rate
        cancel_rate = filtered_df['is_canceled'].mean()
        st.metric("Cancellation Rate (%)", f"{cancel_rate*100:.2f}")
        st.caption("Percent of bookings canceled.")

        # 3. Most common market segments (Pie)
        market_seg_counts = filtered_df['market_segment'].value_counts().reset_index()
        fig2 = px.pie(market_seg_counts, values='market_segment', names='index', title="Market Segment Distribution")
        st.plotly_chart(fig2, use_container_width=True)

        # 4. Average lead time
        st.metric("Average Lead Time (days)", f"{filtered_df['lead_time'].mean():.1f}")
        st.caption("Average days between booking and arrival.")

        # 5. ADR over time (Line)
        adr_month = filtered_df.groupby('arrival_date_month')['adr'].mean().reindex(month_order).reset_index()
        fig3 = px.line(adr_month, x='arrival_date_month', y='adr', title='Average Daily Rate (ADR) by Month')
        st.plotly_chart(fig3, use_container_width=True)

        # 6. Room type demand (Bar)
        room_counts = filtered_df['assigned_room_type'].value_counts().reset_index()
        fig4 = px.bar(room_counts, x='index', y='assigned_room_type', labels={'index': 'Room Type', 'assigned_room_type': 'Count'}, title='Assigned Room Type Distribution')
        st.plotly_chart(fig4, use_container_width=True)

        # 7. Special requests (Bar)
        special_req = filtered_df['total_of_special_requests'].value_counts().sort_index().reset_index()
        fig5 = px.bar(special_req, x='index', y='total_of_special_requests', labels={'index': 'Special Requests', 'total_of_special_requests': 'Count'}, title='Special Requests Count')
        st.plotly_chart(fig5, use_container_width=True)

        # 8. Country-wise bookings (Top 10 Bar)
        country_counts = filtered_df['country'].value_counts().head(10).reset_index()
        fig6 = px.bar(country_counts, x='index', y='country', labels={'index': 'Country', 'country': 'Bookings'}, title='Top 10 Countries by Booking Count')
        st.plotly_chart(fig6, use_container_width=True)

        # 9. Stay duration (weekend vs. week)
        st.metric("Avg. Weekend Nights", f"{filtered_df['stays_in_weekend_nights'].mean():.2f}")
        st.metric("Avg. Week Nights", f"{filtered_df['stays_in_week_nights'].mean():.2f}")

        # 10. Booking changes (Bar)
        booking_chg = filtered_df['booking_changes'].value_counts().sort_index().reset_index()
        fig7 = px.bar(booking_chg, x='index', y='booking_changes', labels={'index': 'Booking Changes', 'booking_changes': 'Count'}, title='Booking Changes')
        st.plotly_chart(fig7, use_container_width=True)
        st.caption("How often bookings are modified.")

        # 11. Customer type breakdown (Pie)
        customer_type_counts = filtered_df['customer_type'].value_counts().reset_index()
        fig8 = px.pie(customer_type_counts, values='customer_type', names='index', title="Customer Types")
        st.plotly_chart(fig8, use_container_width=True)

        st.markdown("**Download filtered data:**")
        st.markdown(get_table_download_link(filtered_df, "filtered_hotel_bookings.csv"), unsafe_allow_html=True)

    # ============= CLASSIFICATION TAB (Confusion, ROC Plotly) =============
    with tabs[1]:
        st.header("Booking Cancellation Prediction (Classification)")
        st.markdown("""
        This module predicts booking cancellation (`is_canceled`) using:
        - K-Nearest Neighbors (KNN)
        - Decision Tree (DT)
        - Random Forest (RF)
        - Gradient Boosting (GBRT)
        """)

        features = ['lead_time', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 
                    'booking_changes', 'deposit_type', 'customer_type', 'adr', 'total_of_special_requests']
        cat_features = ['deposit_type', 'customer_type']

        clf_df = df[features + ['is_canceled']].dropna().copy()
        for col in cat_features:
            clf_df[col] = LabelEncoder().fit_transform(clf_df[col])

        X = clf_df[features]
        y = clf_df['is_canceled']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        metrics_table = []
        y_preds = {}
        y_scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_preds[name] = y_pred
            if hasattr(model, "predict_proba"):
                y_scores[name] = model.predict_proba(X_test)[:,1]
            else:
                y_scores[name] = model.decision_function(X_test)
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            metrics_table.append([name, acc, pre, rec, f1])

        metric_df = pd.DataFrame(metrics_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
        st.dataframe(metric_df.style.highlight_max(axis=0), use_container_width=True)

        st.markdown("#### Confusion Matrix")
        selected_model = st.selectbox("Choose model for confusion matrix", options=list(models.keys()))
        cm = confusion_matrix(y_test, y_preds[selected_model])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Canceled', 'Canceled'],
            y=['Not Canceled', 'Canceled'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}"
        ))
        fig_cm.update_layout(title=f'Confusion Matrix: {selected_model}')
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### ROC Curves")
        fig_roc = go.Figure()
        for name in models:
            fpr, tpr, _ = roc_curve(y_test, y_scores[name])
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Predict Cancellation on New Data")
        uploaded_predict = st.file_uploader("Upload new hotel booking data (same columns as model features)", type=["csv", "xlsx"], key="clf_pred")
        if uploaded_predict is not None:
            if uploaded_predict.name.endswith(".csv"):
                new_X = pd.read_csv(uploaded_predict)
            else:
                new_X = pd.read_excel(uploaded_predict)
            for col in cat_features:
                if col in new_X:
                    new_X[col] = LabelEncoder().fit_transform(new_X[col])
            selected_predict_model = st.selectbox("Select model for prediction", list(models.keys()), key="pred_model2")
            preds = models[selected_predict_model].predict(new_X[features])
            new_X['is_canceled_prediction'] = preds
            st.dataframe(new_X.head())
            st.markdown(get_table_download_link(new_X, "predicted_cancellation.csv"), unsafe_allow_html=True)

    # ============= CLUSTERING TAB (KMeans, Plotly Elbow/Persona) =============
    with tabs[2]:
        st.header("Customer Segmentation (Clustering)")
        st.markdown("Segment customers using KMeans clustering. Adjust the number of clusters and download labeled data.")

        cluster_features = ['lead_time', 'adults', 'children', 'babies', 'previous_cancellations', 
                            'previous_bookings_not_canceled', 'booking_changes', 'adr', 'total_of_special_requests']
        cluster_df = df[cluster_features].dropna().copy()
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(cluster_df)

        k = st.slider("Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster)
        cluster_df['cluster'] = cluster_labels

        inertia = []
        for i in range(2, 11):
            kmeans_i = KMeans(n_clusters=i, random_state=42).fit(X_cluster)
            inertia.append(kmeans_i.inertia_)
        fig_elbow = px.line(x=list(range(2,11)), y=inertia, markers=True, title="Elbow Method For Optimal K")
        fig_elbow.update_xaxes(title="Number of Clusters")
        fig_elbow.update_yaxes(title="Inertia")
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.markdown("#### Cluster Personas")
        persona = cluster_df.groupby('cluster').mean().reset_index()
        st.dataframe(persona)

        # Download
        full_df = df.copy()
        full_df = full_df.iloc[cluster_df.index]
        full_df['cluster'] = cluster_labels
        st.markdown(get_table_download_link(full_df, "hotel_bookings_with_clusters.csv"), unsafe_allow_html=True)

    # ============= ASSOCIATION RULE MINING TAB (Apriori) =============
    with tabs[3]:
        st.header("Association Rule Mining (Apriori)")
        st.markdown("Discover frequent itemsets and associations in hotel bookings.")

        apriori_cols = st.multiselect("Select at least 2 categorical columns for association mining:",
                                      options=['meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                                               'assigned_room_type', 'deposit_type', 'customer_type'],
                                      default=['meal', 'market_segment'])
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)

        if len(apriori_cols) >= 2:
            assoc_df = df[apriori_cols].dropna().astype(str)
            onehot = pd.get_dummies(assoc_df)
            freq_items = apriori(onehot, min_support=min_support, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
            rules = rules.sort_values("confidence", ascending=False).head(10)
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            st.caption("Top 10 associations by confidence")
        else:
            st.info("Please select at least 2 columns.")

    # ============= REGRESSION TAB (Plotly Actual vs Predicted) =============
    with tabs[4]:
        st.header("Regression Analysis")
        st.markdown("Apply regression models to extract business insights.")

        reg_features = ['lead_time', 'adults', 'children', 'babies', 'previous_cancellations',
                        'previous_bookings_not_canceled', 'booking_changes', 'total_of_special_requests']
        reg_target = 'adr'
        reg_df = df[reg_features + [reg_target]].dropna().copy()

        Xr = reg_df[reg_features]
        yr = reg_df[reg_target]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

        regressors = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree Regression": DecisionTreeRegressor(random_state=42)
        }

        reg_results = []
        reg_preds = {}
        for name, reg in regressors.items():
            reg.fit(Xr_train, yr_train)
            pred = reg.predict(Xr_test)
            reg_preds[name] = pred
            mse = np.mean((pred - yr_test)**2)
            r2 = reg.score(Xr_test, yr_test)
            reg_results.append([name, mse, r2])

        reg_table = pd.DataFrame(reg_results, columns=["Model", "MSE", "R2"])
        st.dataframe(reg_table.style.highlight_max(axis=0), use_container_width=True)

        best_idx = reg_table["R2"].idxmax()
        best_name = reg_table.iloc[best_idx]["Model"]
        best_pred = reg_preds[best_name]
        fig_pred = px.scatter(x=yr_test, y=best_pred, labels={'x': 'Actual ADR', 'y': 'Predicted ADR'}, title=f"Actual vs. Predicted ADR ({best_name})")
        fig_pred.add_shape(type="line", x0=yr_test.min(), y0=yr_test.min(), x1=yr_test.max(), y1=yr_test.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("### Quick Insights from Regression")
        st.write("- **Higher lead times tend to be associated with higher ADRs (advance bookings pay more).**")
        st.write("- **Special requests often correlate with higher revenue per room.**")
        st.write("- **Previous cancellations negatively impact ADR.**")
        st.write("- **Family size (adults/children) has a mild impact on ADR.**")
        st.write("- **Booking changes slightly decrease ADR.**")
        st.write("- **Ridge/Lasso can be used to regularize and avoid overfitting.**")
        st.write("- **Decision Trees can identify non-linear patterns in pricing.**")

else:
    st.info("No data loaded. Please upload your hotel_bookings.xlsx or CSV file.")
