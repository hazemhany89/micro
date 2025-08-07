"""
analyzer.py

Provides an advanced function to analyze demand vs supply and suggest content opportunities.
Now includes combined analysis of YouTube videos and Google News articles.
"""

import re
from collections import Counter
from datetime import datetime, timedelta
from textblob import TextBlob # For sentiment analysis
import numpy as np # For median calculation
import pandas as pd # For data manipulation

def analyze_opportunity(demand_score: float, video_count: int, total_views: int,
                        video_titles: list = None, channel_names: list = None,
                        published_times: list = None, video_views_list: list = None,
                        article_count: int = 0, article_titles: list = None,
                        article_sources: list = None, article_dates: list = None) -> dict:
    """
    Analyzes demand vs supply and generates an intelligent, human-readable message
    (Arabic preferred, with emojis) to assess content opportunity.
    Now includes combined analysis of YouTube videos and Google News articles.

    Args:
        demand_score (float): Google Trends demand score (0-100).
        video_count (int): Number of top YouTube videos found.
        total_views (int): Total view count of the top YouTube videos.
        video_titles (list, optional): List of strings representing titles of the top videos.
                                       Used to detect repetition. Defaults to None.
        channel_names (list, optional): List of strings representing channel names of the top videos.
                                        Used to detect channel monopolization. Defaults to None.
        published_times (list, optional): List of strings representing published times (e.g., "3 days ago").
                                          Used to detect recent activity. Defaults to None.
        video_views_list (list, optional): List of integers representing individual video views.
                                           Used for median calculation. Defaults to None.
        article_count (int, optional): Number of Google News articles found. Defaults to 0.
        article_titles (list, optional): List of strings representing titles of the news articles.
                                         Used for content analysis. Defaults to None.
        article_sources (list, optional): List of strings representing sources of the news articles.
                                          Used to detect source diversity. Defaults to None.
        article_dates (list, optional): List of strings representing publication dates of the news articles.
                                        Used to detect recent activity. Defaults to None.

    Returns:
        dict: Contains 'opportunity_message' (str) and 'insights' (dict, optional).
    """
    opportunity_message = ""
    insights = {}

    # --- 1. Basic Calculations ---
    # YouTube metrics
    avg_views_per_video = total_views / video_count if video_count > 0 else 0
    insights['average_views_per_video'] = round(avg_views_per_video)
    insights['demand_score'] = demand_score
    insights['video_count'] = video_count
    insights['total_views'] = total_views

    # Median Views
    median_views = np.median(video_views_list) if video_views_list and len(video_views_list) > 0 else 0
    insights['median_views'] = int(median_views)
    
    # Google News metrics
    insights['article_count'] = article_count
    
    # Combined supply metrics
    total_content_count = video_count + article_count
    insights['total_content_count'] = total_content_count
    
    # Calculate content diversity ratio (videos vs articles)
    if total_content_count > 0:
        video_percentage = (video_count / total_content_count) * 100
        article_percentage = (article_count / total_content_count) * 100
        insights['video_percentage'] = round(video_percentage, 1)
        insights['article_percentage'] = round(article_percentage, 1)
        insights['content_diversity'] = 'Balanced' if 40 <= video_percentage <= 60 else ('Video-dominated' if video_percentage > 60 else 'Article-dominated')
    else:
        insights['video_percentage'] = 0
        insights['article_percentage'] = 0
        insights['content_diversity'] = 'No content'


    # --- 2. Core Decision Logic (Combined YouTube + News Analysis) ---

    # If no data at all
    if demand_score == 0 and total_content_count == 0:
        opportunity_message = "🤷‍♀️ لا توجد بيانات كافية: لم يتم العثور على طلب أو محتوى لهذا الكلمة. حاول بكلمة أخرى."
        insights['opportunity_type'] = "No Data"
    # High Opportunity (Golden)
    elif demand_score > 70 and total_content_count <= 15 and total_views < 2_000_000 and avg_views_per_video < 100_000:
        opportunity_message = "💥 فرصة ذهبية: الطلب مرتفع والمحتوى قليل نسبيًا (عدد فيديوهات ومقالات قليل). أنشئ محتوى الآن قبل الزحمة!"
        insights['opportunity_type'] = "Golden Opportunity"
    # Crowded Market (High Demand, High Supply)
    elif demand_score > 70 and (total_content_count > 30 or total_views > 10_000_000 or avg_views_per_video > 500_000):
        opportunity_message = "🧠 سوق مزدحم: الطلب كبير، لكن فيه محتوى كتير (فيديوهات ومقالات كثيرة). عايز تتميز فعلًا!"
        insights['opportunity_type'] = "Crowded Market"
    # No Opportunity (Low Demand)
    elif demand_score < 30:
        opportunity_message = "❌ لا توجد فرصة حاليًا: الطلب منخفض جدًا على هذا الموضوع."
        insights['opportunity_type'] = "No Opportunity"
    # Moderate Opportunity (Default)
    else:
        opportunity_message = "🤔 الفرصة متوسطة: الطلب والعرض متوازنان. لو هتقدم محتوى مختلف وفريد، جرب."
        insights['opportunity_type'] = "Moderate Opportunity"
        
    # Add content diversity insight to the message
    if total_content_count > 0:
        if insights['content_diversity'] == 'Video-dominated' and insights['video_percentage'] > 80:
            opportunity_message += "\n\n📹 ملاحظة: معظم المحتوى الموجود عبارة عن فيديوهات ({}%). قد تكون هناك فرصة لإنشاء مقالات ومحتوى مكتوب.".format(round(insights['video_percentage']))
        elif insights['content_diversity'] == 'Article-dominated' and insights['article_percentage'] > 80:
            opportunity_message += "\n\n📰 ملاحظة: معظم المحتوى الموجود عبارة عن مقالات ({}%). قد تكون هناك فرصة لإنشاء محتوى مرئي/فيديوهات.".format(round(insights['article_percentage']))
        elif insights['content_diversity'] == 'Balanced':
            opportunity_message += "\n\n⚖️ ملاحظة: المحتوى متوازن بين الفيديوهات والمقالات. قد تحتاج إلى التميز في الجودة والمحتوى الفريد."

    # --- 3. Advanced Analysis and Refinements ---

    # Warning for low engagement despite demand (using median views for robustness)
    if demand_score > 50 and median_views > 0 and median_views < 20_000:
        insights['low_engagement_warning'] = True
        if "فرصة ذهبية" in opportunity_message:
            opportunity_message += "\n\nلكن انتبه: متوسط المشاهدات للفيديو الواحد منخفض (المتوسط: {:,})، قد يشير لضعف في جودة المحتوى الحالي أو عدم تفاعل الجمهور 📉.".format(insights['average_views_per_video'])
        else:
            opportunity_message += "\n\n⚠️ ملاحظة: متوسط المشاهدات للفيديو الواحد منخفض (المتوسط: {:,}). قد يشير لضعف في تفاعل الجمهور مع المحتوى الموجود 📉.".format(insights['average_views_per_video'])


    # Title Keyword Analysis & Repetition Detection (Refined)
    if video_titles and video_count > 1:
        all_words = []
        for title in video_titles:
            words = re.findall(r'\b\w+\b', title.lower())
            all_words.extend(words)

        # Expanded stopwords for Arabic and English
        stopwords = {
            "the", "a", "an", "is", "of", "in", "to", "for", "and", "or", "how", "what", "where", "why", "who", "with",
            "على", "من", "في", "إلى", "عن", "مع", "أن", "لا", "بين", "أو", "هو", "هي", "هم", "لكن", "هذا", "هذه", "ذلك", "شرح", "كيف", "ما", "أفضل", "تفسير", "دليل", "كامل",
            "2024", "2025", "tutorial", "guide", "learn", "how to", "best", "top", "review", "new", "free", "vs", "vs.", "vs",
            "شاهد", "حصريا", "الفيديو", "كل", "شيء", "سر", "اسرار", "نصائح", "طرق", "أخبار", "أهم", "آخر", "أول", "خطوات", "سريع", "سهل", "بدون", "للمبتدئين", "احترافي"
        }

        meaningful_words = [word for word in all_words if word not in stopwords and len(word) > 2]
        word_counts = Counter(meaningful_words)

        insights['most_frequent_keywords'] = word_counts.most_common(5) # Top 5 common keywords

        # Heuristic for repetition: if a "meaningful" word appears in a high percentage (e.g., > 60%) of the titles,
        # it suggests a lack of title diversity.
        repetitive_words = [word for word, count in word_counts.items() if count >= (video_count * 0.6)]

        if repetitive_words:
            insights['title_repetition_detected'] = True
            insights['repetitive_words'] = repetitive_words
            repetitive_words_str = ', '.join(repetitive_words[:3])
            opportunity_message += f"\n\n💡 نصيحة إضافية: لاحظنا تكرار بعض الكلمات/المفاهيم في عناوين الفيديو الحالية (مثل: {repetitive_words_str}). فكر في محتوى أكثر ابتكارًا وعناوين مميزة لتبرز في السوق! ✨"
        else:
            insights['title_repetition_detected'] = False
    else:
        insights['most_frequent_keywords'] = []
        insights['title_repetition_detected'] = False

    # Unique Channels Analysis (Detecting monopolization)
    if channel_names:
        unique_channels = set(channel_names)
        num_unique_channels = len(unique_channels)
        insights['num_unique_channels'] = num_unique_channels
        if video_count > 0 and num_unique_channels < (video_count / 2) and num_unique_channels > 1: # If less than half the videos are from unique channels
            insights['channel_monopolization_warning'] = True
            opportunity_message += f"\n\n⚠️ تحذير: عدد قليل من القنوات تهيمن على النتائج ({num_unique_channels} قناة فريدة من أصل {video_count} فيديو). قد يكون السوق يهيمن عليه لاعبون كبار، مما يصعب المنافسة على القنوات الجديدة."
        elif num_unique_channels == 1 and video_count > 1:
            insights['channel_monopolization_warning'] = True
            opportunity_message += f"\n\n🚨 السوق يحتكره قناة واحدة: جميع الفيديوهات من نفس القناة. هذا يمكن أن يكون تحديًا كبيراً للدخول للسوق ما لم تقدم قيمة فريدة جداً."
        else:
            insights['channel_monopolization_warning'] = False
    else:
        insights['num_unique_channels'] = 0
        insights['channel_monopolization_warning'] = False

    # Recent Video Activity (Based on published_times) - Now explicitly for last 30 days
    if published_times and video_count > 0:
        recent_videos_30_days = 0
        current_date = datetime.now()

        def parse_published_time(pt_str):
            pt_lower = pt_str.lower()
            num_match = re.search(r'(\d+)', pt_lower)
            if not num_match:
                return None # Cannot parse if no number

            num = int(num_match.group(1))
            
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
            return None # Cannot parse

        for pt in published_times:
            parsed_date = parse_published_time(pt)
            if parsed_date and (current_date - parsed_date).days <= 30:
                recent_videos_30_days += 1

        insights['recent_videos_30_days_count'] = recent_videos_30_days
        insights['recent_videos_30_days_percentage'] = (recent_videos_30_days / video_count) * 100 if video_count > 0 else 0

        if recent_videos_30_days > (video_count * 0.2): # If more than 20% are recent (last 30 days)
            insights['high_recent_activity'] = True
            opportunity_message += f"\n\n📈 نشاط حديث: هناك {recent_videos_30_days} فيديو ({insights['recent_videos_30_days_percentage']:.1f}%) تم نشرها خلال آخر 30 يومًا. يشير هذا إلى اهتمام متزايد بالموضوع أو منافسة جديدة."
        else:
            insights['high_recent_activity'] = False
    else:
        insights['recent_videos_30_days_count'] = 0
        insights['recent_videos_30_days_percentage'] = 0
        insights['high_recent_activity'] = False

    # Sentiment Analysis on Titles (Combined Video + News)
    all_titles = []
    if video_titles:
        all_titles.extend(video_titles)
    if article_titles:
        all_titles.extend(article_titles)
        
    if all_titles:
        positive_titles = 0
        negative_titles = 0
        neutral_titles = 0
        for title in all_titles:
            analysis = TextBlob(title) # Default English sentiment
            # For Arabic sentiment, might need specific TextBlob models or custom logic
            # For simplicity, using default which might be less accurate for pure Arabic
            if analysis.sentiment.polarity > 0.1: # Positive threshold
                positive_titles += 1
            elif analysis.sentiment.polarity < -0.1: # Negative threshold
                negative_titles += 1
            else:
                neutral_titles += 1
        
        insights['sentiment_analysis'] = {
            'positive_titles': positive_titles,
            'negative_titles': negative_titles,
            'neutral_titles': neutral_titles,
            'total_analyzed': len(all_titles)
        }
        
        if positive_titles > negative_titles and positive_titles > neutral_titles:
            insights['overall_sentiment'] = "Positive"
            opportunity_message += "\n\n😊 توجه إيجابي: أغلب عناوين المحتوى الحالي (فيديوهات ومقالات) تحمل طابعًا إيجابيًا."
        elif negative_titles > positive_titles and negative_titles > neutral_titles:
            insights['overall_sentiment'] = "Negative"
            opportunity_message += "\n\n😔 توجه سلبي: أغلب عناوين المحتوى الحالي تحمل طابعًا سلبيًا. يمكن أن يكون هذا مكانًا للتميز بمحتوى إيجابي."
        else:
            insights['overall_sentiment'] = "Neutral"
            opportunity_message += "\n\n😐 توجه محايد: عناوين المحتوى الحالي تبدو محايدة في المشاعر."
    else:
        insights['sentiment_analysis'] = {}
        insights['overall_sentiment'] = "N/A"
        
    # News Sources Analysis
    if article_sources and len(article_sources) > 0:
        unique_sources = set(article_sources)
        insights['news_sources_count'] = len(unique_sources)
        insights['news_sources'] = list(unique_sources)
        
        # Source diversity analysis
        if article_count > 0:
            source_diversity_ratio = len(unique_sources) / article_count
            insights['source_diversity_ratio'] = round(source_diversity_ratio, 2)
            
            if source_diversity_ratio < 0.3 and article_count > 5:
                opportunity_message += "\n\n📰 ملاحظة: معظم المقالات تأتي من عدد قليل من المصادر. قد تكون هناك فرصة لتقديم وجهة نظر جديدة."
    else:
        insights['news_sources_count'] = 0
        insights['news_sources'] = []
        
    # Recent News Articles Analysis
    if article_dates and article_count > 0:
        try:
            # Convert string dates to datetime objects
            # Assuming dates are in format like '2023-01-15' or similar parseable format
            recent_articles_count = 0
            current_date = datetime.now()
            
            for date_str in article_dates:
                try:
                    # Try to parse the date - format may vary based on GoogleNews output
                    article_date = pd.to_datetime(date_str)
                    if (current_date - article_date).days <= 30:  # Articles in last 30 days
                        recent_articles_count += 1
                except:
                    # Skip dates that can't be parsed
                    continue
            
            insights['recent_articles_30_days'] = recent_articles_count
            insights['recent_articles_percentage'] = round((recent_articles_count / article_count) * 100, 1)
            
            if insights['recent_articles_percentage'] > 50:
                opportunity_message += "\n\n🔥 موضوع ساخن في الأخبار: {}% من المقالات نُشرت في آخر 30 يوم. هذا يشير إلى اهتمام متزايد.".format(insights['recent_articles_percentage'])
        except Exception as e:
            print(f"Error analyzing article dates: {e}")
            insights['recent_articles_30_days'] = 0
            insights['recent_articles_percentage'] = 0

    return {"opportunity_message": opportunity_message, "insights": insights}