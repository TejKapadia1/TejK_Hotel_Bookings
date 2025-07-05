# ================= DATA VISUALIZATION TAB (SUPER ROBUST) ===================
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
    try:
        if not temp1.empty and temp1.sum() > 0:
            fig1 = px.bar(temp1.reset_index(), x='arrival_date_month', y=0, labels={'0': 'Booking Count', 'arrival_date_month': 'Month'}, title='Bookings by Month')
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Shows seasonality in bookings.")
        else:
            st.info("No data for Bookings by Month with current filter.")
    except Exception as e:
        st.warning(f"Could not plot Bookings by Month: {e}")

    # 2. Booking cancellation rate
    if not filtered_df.empty:
        cancel_rate = filtered_df['is_canceled'].mean()
        st.metric("Cancellation Rate (%)", f"{cancel_rate*100:.2f}")
        st.caption("Percent of bookings canceled.")
    else:
        st.info("No data for Cancellation Rate with current filter.")

    # 3. Most common market segments (Pie)
    market_seg_counts = filtered_df['market_segment'].value_counts().reset_index()
    market_seg_counts.columns = ['market_segment', 'count']
    try:
        if not market_seg_counts.empty and (market_seg_counts['count'] > 0).any():
            fig2 = px.pie(market_seg_counts, values='count', names='market_segment', title="Market Segment Distribution")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data to plot Market Segment Distribution for current filter selection.")
    except Exception as e:
        st.warning(f"Could not plot Market Segment Distribution: {e}")

    # 4. Average lead time
    if not filtered_df.empty:
        st.metric("Average Lead Time (days)", f"{filtered_df['lead_time'].mean():.1f}")
        st.caption("Average days between booking and arrival.")
    else:
        st.info("No data for Average Lead Time with current filter.")

    # 5. ADR over time (Line)
    adr_month = filtered_df.groupby('arrival_date_month')['adr'].mean().reindex(month_order).reset_index()
    try:
        if not adr_month.empty and adr_month['adr'].notna().any():
            fig3 = px.line(adr_month, x='arrival_date_month', y='adr', title='Average Daily Rate (ADR) by Month')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data for ADR by Month with current filter.")
    except Exception as e:
        st.warning(f"Could not plot ADR by Month: {e}")

    # 6. Room type demand (Bar)
    room_counts = filtered_df['assigned_room_type'].value_counts().reset_index()
    room_counts.columns = ['assigned_room_type', 'count']
    try:
        if not room_counts.empty and (room_counts['count'] > 0).any():
            fig4 = px.bar(room_counts, x='assigned_room_type', y='count', title='Assigned Room Type Distribution')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No data for Assigned Room Type Distribution with current filter.")
    except Exception as e:
        st.warning(f"Could not plot Room Type Distribution: {e}")

    # 7. Special requests (Bar)
    special_req = filtered_df['total_of_special_requests'].value_counts().sort_index().reset_index()
    special_req.columns = ['total_of_special_requests', 'count']
    try:
        if not special_req.empty and (special_req['count'] > 0).any():
            fig5 = px.bar(special_req, x='total_of_special_requests', y='count', title='Special Requests Count')
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No data for Special Requests Count with current filter.")
    except Exception as e:
        st.warning(f"Could not plot Special Requests: {e}")

    # 8. Country-wise bookings (Top 10 Bar)
    country_counts = filtered_df['country'].value_counts().head(10).reset_index()
    country_counts.columns = ['country', 'count']
    try:
        if not country_counts.empty and (country_counts['count'] > 0).any():
            fig6 = px.bar(country_counts, x='country', y='count', title='Top 10 Countries by Booking Count')
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No data for Top 10 Countries by Booking Count with current filter.")
    except Exception as e:
        st.warning(f"Could not plot Country-wise bookings: {e}")

    # 9. Stay duration (weekend vs. week)
    if not filtered_df.empty:
        st.metric("Avg. Weekend Nights", f"{filtered_df['stays_in_weekend_nights'].mean():.2f}")
        st.metric("Avg. Week Nights", f"{filtered_df['stays_in_week_nights'].mean():.2f}")
    else:
        st.info("No data for Stay Duration with current filter.")

    # 10. Booking changes (Bar)
    booking_chg = filtered_df['booking_changes'].value_counts().sort_index().reset_index()
    booking_chg.columns = ['booking_changes', 'count']
    try:
        if not booking_chg.empty and (booking_chg['count'] > 0).any():
            fig7 = px.bar(booking_chg, x='booking_changes', y='count', title='Booking Changes')
            st.plotly_chart(fig7, use_container_width=True)
            st.caption("How often bookings are modified.")
        else:
            st.info("No data for Booking Changes with current filter.")
    except Exception as e:
        st.warning(f"Could not plot Booking Changes: {e}")

    # 11. Customer type breakdown (Pie)
    customer_type_counts = filtered_df['customer_type'].value_counts().reset_index()
    customer_type_counts.columns = ['customer_type', 'count']
    try:
        if not customer_type_counts.empty and (customer_type_counts['count'] > 0).any():
            fig8 = px.pie(customer_type_counts, values='count', names='customer_type', title="Customer Types")
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.info("No data for Customer Types with current filter.")
    except Exception as e:
        st.warning(f"Could not plot Customer Types: {e}")

    st.markdown("**Download filtered data:**")
    if not filtered_df.empty:
        st.markdown(get_table_download_link(filtered_df, "filtered_hotel_bookings.csv"), unsafe_allow_html=True)
    else:
        st.info("No data to download with current filter.")
