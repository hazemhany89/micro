"""
app.py

Streamlit app to analyze demand vs supply for a keyword using Google Trends and YouTube.
"""

import streamlit as st
import pandas as pd
from core.fetch_google_trends import fetch_google_trends
from core.fetch_youtube_videos import fetch_youtube_videos
from core.fetch_google_news import fetch_google_news
from analyzer import analyze_opportunity
import plotly.express as px # For charting
from datetime import datetime, timedelta
import re # For parsing published_time in helper function
from core.generate_ai_recommendations import generate_ai_recommendations

st.set_page_config(page_title="Demand vs Supply Analyzer", layout="wide") # Changed layout to wide

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("🔎 Demand vs Supply Analyzer for Digital Content Opportunities")
    st.markdown("Analyze **demand** (Google Trends) vs **supply** (YouTube videos) for any topic. Discover content niches with high demand and low supply!")

    # --- User Input Filters ---
    st.sidebar.header("🔍 Analysis Settings")
    keyword = st.sidebar.text_input("Enter a keyword to analyze", placeholder="e.g. تعلم بايثون للمبتدئين")

    # Country Selector
    countries = {
        "Worldwide": "", "United States": "US", "Egypt": "EG", "Saudi Arabia": "SA",
        "India": "IN", "United Kingdom": "GB", "Canada": "CA", "Australia": "AU",
        "Germany": "DE", "France": "FR", "Japan": "JP", "Brazil": "BR", "United Arab Emirates": "AE",
        "Morocco": "MA", "Algeria": "DZ", "Tunisia": "TN", "Qatar": "QA", "Kuwait": "KW", "Oman": "OM" # More Arab countries
    }
    selected_country_name = st.sidebar.selectbox("Select Country for Google Trends", list(countries.keys()))
    geo_code = countries[selected_country_name]

    # Timeframe Selector
    timeframes = {
        "Last 1 Month": "today 1-m",
        "Last 3 Months": "today 3-m",
        "Last 12 Months": "today 12-m",
        "Past 5 Years": "today 5-y"
    }
    selected_timeframe_name = st.sidebar.selectbox("Select Timeframe for Google Trends", list(timeframes.keys()))
    timeframe_code = timeframes[selected_timeframe_name]

    # Number of YouTube videos to fetch
    youtube_limit = st.sidebar.slider("Number of YouTube Videos to Fetch", min_value=5, max_value=50, value=30, step=5) # Max 50, default 30
    
    # Google News Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("📰 Google News Settings")
    news_enabled = st.sidebar.checkbox("Include Google News Articles", value=True)
    
    # Only show these settings if Google News is enabled
    news_limit = 20
    news_start_date = None
    news_end_date = None
    news_source = None
    
    if news_enabled:
        news_limit = st.sidebar.slider("Number of News Articles to Fetch", min_value=5, max_value=50, value=20, step=5)
        
        # Date range picker for news articles
        col_date1, col_date2 = st.sidebar.columns(2)
        with col_date1:
            thirty_days_ago = datetime.now() - timedelta(days=30)
            news_start_date = st.date_input("Start Date", thirty_days_ago)
        with col_date2:
            news_end_date = st.date_input("End Date", datetime.now())
            
        # Convert dates to string format for GoogleNews
        news_start_date = news_start_date.strftime('%m/%d/%Y')
        news_end_date = news_end_date.strftime('%m/%d/%Y')
        
        # Optional source filter
        news_source = st.sidebar.text_input("Filter by News Source (Optional)", "", 
                                          help="Enter a specific news source domain (e.g., 'bbc.com', 'cnn.com')")
        if news_source and not news_source.strip():
            news_source = None

    # Advanced Analysis Checkbox
    st.sidebar.markdown("---")
    advanced_analysis_enabled = st.sidebar.checkbox("Enable Advanced Analysis", value=True)
    
    # AI Recommendations Options
    enable_ai_recommendations = st.sidebar.checkbox("Enable AI Content Recommendations", value=False,
                                              help="Generate smart content recommendations using OpenAI API based on analysis data.")
    
    openai_api_key = ""
    ai_language = "Arabic"
    if enable_ai_recommendations:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                      help="Enter your OpenAI API key. The key is not stored and is only used for this session.")
        ai_language = st.sidebar.radio("Recommendation Language", ["Arabic", "English"], index=0,
                              help="Select the language for AI recommendations.")


    if not keyword:
        st.info("⬆️ Please enter a keyword in the sidebar to get started.")
        st.stop()

    demand_score = 0.0
    videos_data = []
    news_data = []

    # --- Pre-flight Data Validation & Fetching ---
    with st.spinner(f"Fetching data for '{keyword}' in {selected_country_name} over {selected_timeframe_name.lower()}..."):
        # Fetch demand data (Google Trends)
        demand_score = fetch_google_trends(keyword, geo=geo_code, timeframe=timeframe_code)
        
        # Fetch YouTube videos
        videos_data = fetch_youtube_videos(keyword, limit=youtube_limit)
        
        # Fetch Google News articles if enabled
        if news_enabled:
            news_data = fetch_google_news(keyword, limit=news_limit, 
                                         start_date=news_start_date, 
                                         end_date=news_end_date, 
                                         source=news_source)

    # Validate fetched data
    if demand_score == 0:
        st.error(f"❌ Could not retrieve Google Trends data for '{keyword}' in {selected_country_name} ({selected_timeframe_name}). Please try a different keyword or check your settings.")
        st.stop()
    
    if not videos_data or len(videos_data) < 5: # Minimum 5 videos for meaningful analysis
        st.error(f"❌ Not enough YouTube videos found for '{keyword}' (found {len(videos_data)}). Need at least 5 videos for analysis. Please try a different keyword or increase the 'Number of YouTube Videos to Fetch'.")
        st.stop()
        
    # Warning for Google News if enabled but no results
    if news_enabled and not news_data:
        st.warning(f"⚠️ No Google News articles found for '{keyword}' with the current filters. Analysis will proceed with YouTube data only.")
        # Don't stop execution, just continue with YouTube data

    # Prepare data for analyzer
    # YouTube data
    video_titles = [video['title'] for video in videos_data]
    channel_names = [video['channel_name'] for video in videos_data]
    published_times = [video['published_time'] for video in videos_data]
    video_views_list = [video['views'] for video in videos_data]

    # Calculate total views and video count
    df_videos = pd.DataFrame(videos_data)
    total_views = df_videos['views'].sum()
    video_count = len(videos_data)
    
    # Google News data
    article_count = 0
    article_titles = []
    article_sources = []
    article_dates = []
    df_news = pd.DataFrame()
    
    if news_enabled and news_data:
        article_titles = [article['title'] for article in news_data]
        article_sources = [article['source'] for article in news_data]
        article_dates = [article['date'] for article in news_data]
        article_count = len(news_data)
        df_news = pd.DataFrame(news_data)

    # Analyze opportunity with the combined analyzer (YouTube + Google News)
    analysis_results = analyze_opportunity(
        demand_score, video_count, total_views,
        video_titles=video_titles,
        channel_names=channel_names,
        published_times=published_times,
        video_views_list=video_views_list,
        article_count=article_count,
        article_titles=article_titles,
        article_sources=article_sources,
        article_dates=article_dates
    )
    opportunity_message = analysis_results['opportunity_message']
    analysis_insights = analysis_results['insights']

    # --- Streamlit Display ---
    st.subheader("📊 Analysis Overview")
    
    # Metrics row 1 - Demand and Supply Overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Google Trends Demand (Avg)", f"{demand_score:.1f}/100", help=f"Average interest score over {selected_timeframe_name} in {selected_country_name}")
    with col2:
        st.metric(f"YouTube Videos", f"{video_count} videos", help=f"Number of top videos found for '{keyword}'.")
    with col3:
        if news_enabled:
            st.metric("Google News Articles", f"{article_count} articles", help=f"Number of news articles found for '{keyword}'.")
        else:
            st.metric("Total Views (Top Videos)", f"{total_views:,}", help="Total views of the fetched YouTube videos.")
            
    # Metrics row 2 - Additional metrics
    if news_enabled and article_count > 0:
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Total Views (Top Videos)", f"{total_views:,}", help="Total views of the fetched YouTube videos.")
        with col5:
            total_content = video_count + article_count
            st.metric("Total Content Items", f"{total_content}", help="Combined count of videos and news articles.")
        with col6:
            if 'news_sources_count' in analysis_insights:
                st.metric("Unique News Sources", f"{analysis_insights['news_sources_count']}", help="Number of different news sources found.")
            else:
                st.metric("Avg. Views per Video", f"{analysis_insights.get('average_views_per_video', 0):,}", help="Average views per video.")

    st.markdown("---") # Separator for clarity

    st.subheader("⭐ Content Opportunity Assessment")
    st.info(opportunity_message)

    # Add visualizations if news is enabled
    if news_enabled and (video_count > 0 or article_count > 0):
        st.markdown("---") # Separator for clarity
        st.subheader("📊 Content Distribution")
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Content type distribution chart
            content_types = ['YouTube Videos', 'News Articles']
            content_counts = [video_count, article_count]
            
            fig1 = px.bar(
                x=content_types, 
                y=content_counts,
                labels={'x': 'Content Type', 'y': 'Count'},
                color=content_types,
                color_discrete_map={'YouTube Videos': '#FF0000', 'News Articles': '#1E90FF'},
                title="Content Type Distribution"
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            # News sources pie chart (if news data exists)
            if news_enabled and article_count > 0 and 'news_sources_count' in analysis_insights:
                # Count occurrences of each source
                source_counts = {}
                for source in article_sources:
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts[source] = 1
                
                # Create pie chart of news sources
                fig2 = px.pie(
                    values=list(source_counts.values()),
                    names=list(source_counts.keys()),
                    title="News Sources Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # If no news data, show video views distribution
                if video_count > 0:
                    # Get top 5 videos by views for pie chart
                    top_videos = sorted(videos_data, key=lambda x: x['views'], reverse=True)[:5]
                    top_video_titles = [f"{v['title'][:20]}..." for v in top_videos]
                    top_video_views = [v['views'] for v in top_videos]
                    
                    fig2 = px.pie(
                        values=top_video_views,
                        names=top_video_titles,
                        title="Top 5 Videos by Views"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    # --- Raw Data for Debugging ---
    with st.expander("Show Raw Data & Basic Insights (for debugging)", expanded=False):
        st.write("Google Trends Demand Score:", demand_score)
        st.write("YouTube Videos Data:", videos_data)
        st.dataframe(df_videos) # Display full dataframe for raw video data
        if news_enabled and not df_news.empty:
            st.write("Google News Articles Data:")
            st.dataframe(df_news)
        st.write("Analysis Insights Dictionary:", analysis_insights)


    # --- Advanced Analysis Display ---
    if advanced_analysis_enabled:
        st.subheader("🚀 Advanced Content Insights & Visualizations")
        if analysis_insights:
            # Metrics
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            with col_adv1:
                st.metric("Average Views per Video", f"{analysis_insights.get('average_views_per_video', 0):,}",
                          help="Average views each of the top videos received.")
                if analysis_insights.get('low_engagement_warning'):
                    st.warning("Low average views despite demand. Market might not be engaging or content quality is an issue.")
            with col_adv2:
                st.metric("Median Views per Video", f"{analysis_insights.get('median_views', 0):,}",
                          help="Median views (middle value) of the fetched videos, less affected by outliers.")
            with col_adv3:
                if 'num_unique_channels' in analysis_insights:
                    st.metric("Unique Channels Count", analysis_insights['num_unique_channels'],
                              help="Number of distinct channels found among the top videos.")
                    if analysis_insights.get('channel_monopolization_warning'):
                        st.warning("Few channels dominate. Harder to break in unless content is unique.")

            # Recent Activity & Sentiment
            col_adv4, col_adv5 = st.columns(2)
            with col_adv4:
                st.write(f"**📈 Recent Content Activity:**")
                st.info(f"{analysis_insights.get('recent_videos_30_days_count', 0)} videos ({analysis_insights.get('recent_videos_30_days_percentage', 0):.1f}%) published in last 30 days.")
                if analysis_insights.get('high_recent_activity'):
                    st.success("High recent activity indicates growing interest or new competition.")
                elif analysis_insights.get('recent_videos_30_days_count', 0) == 0:
                    st.info("No very recent videos found among the top results.")
            with col_adv5:
                st.write(f"**😊 Overall Sentiment of Titles:**")
                st.info(analysis_insights.get('overall_sentiment', 'N/A'))
                if analysis_insights.get('overall_sentiment') == "Negative":
                    st.info("Consider creating positive/solution-oriented content to stand out.")


            # Visualizations
            st.markdown("#### Visualizations")
            
            # Bar Chart: Views per Video
            if not df_videos.empty:
                st.write("**Views Distribution per Video**")
                fig_views = px.bar(df_videos.sort_values('views', ascending=False), 
                                   x='title', y='views', 
                                   title='Top Videos by Views',
                                   labels={'title': 'Video Title', 'views': 'Views'},
                                   height=400)
                fig_views.update_xaxes(tickangle=45, tickfont=dict(size=10))
                st.plotly_chart(fig_views, use_container_width=True)
            
            # Timeline Chart: Publication Dates
            if 'published_time' in df_videos.columns and not df_videos.empty:
                # Convert published_time to datetime objects for plotting
                df_videos_copy = df_videos.copy()
                df_videos_copy['parsed_date'] = df_videos_copy['published_time'].apply(parse_youtube_published_time)
                df_videos_copy.dropna(subset=['parsed_date'], inplace=True) # Drop rows where parsing failed
                
                if not df_videos_copy.empty:
                    st.write("**Video Publication Timeline**")
                    fig_timeline = px.scatter(df_videos_copy.sort_values('parsed_date'), 
                                              x='parsed_date', y='views', 
                                              size='views', color='channel_name', 
                                              hover_name='title',
                                              title='Video Publication Dates and Views',
                                              labels={'parsed_date': 'Publication Date', 'views': 'Views'},
                                              height=400)
                    fig_timeline.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_timeline, use_container_width=True)

            # Common Keywords in Titles
            if analysis_insights.get('most_frequent_keywords'):
                st.write("**Most Frequent Keywords in Titles**")
                keywords_df = pd.DataFrame(analysis_insights['most_frequent_keywords'], columns=['Keyword', 'Frequency'])
                st.dataframe(keywords_df, use_container_width=True, hide_index=True)
                if analysis_insights.get('title_repetition_detected'):
                    st.warning("High repetition of keywords. Consider diversifying your titles!")


            # Display raw advanced insights dictionary again for completeness
            with st.expander("Raw Detailed Analysis Insights"):
                st.json(analysis_insights)
                
            # AI Content Recommendations Section
            if 'enable_ai_recommendations' in locals() and enable_ai_recommendations:
                st.markdown("---")
                st.markdown("### 🤖 AI Content Recommendations")
                
                if not openai_api_key:
                    st.warning("Please enter your OpenAI API key to generate content recommendations.")
                else:
                    with st.spinner("Generating smart content recommendations..."):
                        # Prepare data for AI recommendations
                        ai_analysis_data = {
                            "keyword": keyword,
                            "country": selected_country_name,
                            "timeframe": selected_timeframe_name,
                            "demand_score": demand_score,
                            "video_count": len(videos_data),
                            "news_count": len(news_data) if news_data else 0,
                            "total_views": total_views,
                            "insights": analysis_insights,
                            "top_videos": videos_data,
                            "news_articles": news_data if news_data else []
                        }
                        
                        # Generate AI recommendations
                        language_code = "ar" if ai_language == "Arabic" else "en"
                        recommendations = generate_ai_recommendations(ai_analysis_data, openai_api_key, language_code)
                        
                        # Display recommendations
                        st.markdown(recommendations)
                        
                        # Add download button for recommendations
                        st.download_button(
                            label="Download Recommendations",
                            data=recommendations,
                            file_name=f"content_recommendations_{keyword}_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
        else:
            st.info("No advanced insights available, likely due to insufficient data.")

    st.markdown("---")
    st.markdown("#### Top YouTube Videos List")
    if not df_videos.empty:
        # Create clickable links and format views
        df_videos['Link'] = df_videos['link'].apply(lambda x: f"[Link]({x})")
        df_videos['Views'] = df_videos['views'].apply(lambda x: f"{x:,}") # Format views for readability
        
        # Select and reorder columns for display
        display_cols = ['title', 'Views', 'published_time', 'channel_name', 'Link']
        df_display = df_videos[display_cols].rename(columns={
            'title': 'Title',
            'published_time': 'Published Date',
            'channel_name': 'Channel'
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info(f"No top YouTube videos found for '{keyword}'.")
        
    # Display Google News Articles if enabled and data exists
    if news_enabled and news_data:
        st.markdown("---")
        st.markdown("#### Recent News Articles")
        
        # Create a DataFrame for news display
        df_news_display = pd.DataFrame({
            'Title': [article['title'] for article in news_data],
            'Source': [article['source'] for article in news_data],
            'Date': [article['date'] for article in news_data],
            'Link': [f"[Read Article]({article['url']})" for article in news_data]
        })
        
        # Display as a dataframe with markdown links
        st.dataframe(df_news_display, use_container_width=True, hide_index=True)

# Helper function to parse YouTube published time strings into datetime objects
def parse_youtube_published_time(pt_str):
    if not pt_str:
        return None
    pt_lower = pt_str.lower()
    num_match = re.search(r'(\d+)', pt_lower)
    if not num_match:
        return None # Cannot parse if no number

    num = int(num_match.group(1))
    current_date = datetime.now()

    if "minute" in pt_lower:
        return current_date - timedelta(minutes=num)
    elif "hour" in pt_lower:
        return current_date - timedelta(hours=num)
    elif "day" in pt_lower:
        return current_date - timedelta(days=num)
    elif "week" in pt_lower:
        return current_date - timedelta(weeks=num)
    elif "month" in pt_lower:
        return current_date - timedelta(days=num * 30) # Approximation
    elif "year" in pt_lower:
        return current_date - timedelta(days=num * 365) # Approximation
    return None


if __name__ == "__main__":
    main()