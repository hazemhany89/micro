"""generate_ai_recommendations.py

Provides a function to generate intelligent content recommendations using OpenAI API
based on the analysis data from the Demand vs Supply Analyzer.
"""

import openai
import json

def generate_ai_recommendations(analysis_data, api_key=None, language="ar"):
    """
    Generates intelligent content recommendations using OpenAI API based on the analysis data.
    
    Args:
        analysis_data (dict): Dictionary containing all the analysis data including demand score,
                             video data, news data, and insights.
        api_key (str, optional): OpenAI API key. If None, the function will use the key from environment
                                variables or configuration.
        language (str, optional): Language for the recommendations. Default is "ar" for Arabic.
                                 Use "en" for English.
    
    Returns:
        str: AI-generated content recommendations.
    """
    # API key will be used when creating the client
    # No need to set it globally anymore
    
    # Convert numpy int64 to regular Python int to make it JSON serializable
    def convert_to_serializable(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    # Convert data to JSON serializable format and then format as string
    serializable_data = convert_to_serializable(analysis_data)
    formatted_data = json.dumps(serializable_data, ensure_ascii=False, indent=2)
    
    # Select the appropriate prompt based on language
    if language.lower() == "en":
        prompt = f"""You are an expert digital content strategist and data analyst. Your task is to analyze the provided data and deliver practical recommendations for creating impactful and competitive digital content.

### Data Provided for Analysis:
{formatted_data}

### Your Tasks:

1. **Data Summary and Analysis**:
   - Provide a concise summary of the provided data.
   - Identify key trends and patterns in the data.
   - Explain the relationship between demand data (such as search trends) and supply data (such as existing YouTube content and news).

2. **Content Opportunity Identification**:
   - Discover gaps between demand and supply that can be exploited.
   - Identify topics with increasing interest and low competition.
   - Suggest new angles for popular topics.

3. **Appropriate Content Type Recommendations**:
   - Determine the most suitable platforms and content formats (video, article, podcast, etc.) based on the data.
   - Suggest ideal content lengths and formats for each platform.
   - Provide tips on optimal publishing timing if possible.

4. **Innovative Title and Topic Ideas**:
   - Suggest 5-7 engaging and clickable titles for each content opportunity.
   - Provide ideas for sub-content that can be included.
   - Suggest keywords that should be targeted in each content piece.

5. **Content Differentiation Tips**:
   - Provide strategies to make the content stand out from competitors.
   - Suggest unique angles or innovative presentation methods.
   - Identify value-added elements that can be offered to the audience.

6. **Warnings and Potential Challenges**:
   - Point out any risks or challenges in the proposed content strategy.
   - Identify saturated topics that should be avoided.
   - Provide tips for dealing with intense competition in certain areas.

7. **Brief Action Plan**:
   - Provide practical steps to implement the recommendations.
   - Suggest an initial timeline for content production.
   - Identify success indicators that should be monitored.

Present your answers in a clear and organized format, focusing on practical and actionable recommendations. Base your recommendations solely on the provided data, and avoid unsupported assumptions."""
    else:  # Default to Arabic
        prompt = f"""أنت مستشار محتوى رقمي خبير ومحلل استراتيجي. مهمتك تحليل البيانات المقدمة وتقديم توصيات عملية لإنشاء محتوى رقمي مؤثر ومنافس.

### البيانات المقدمة للتحليل:
{formatted_data}

### المطلوب منك:

1. **تلخيص وتحليل البيانات**:
   - قدم ملخصاً موجزاً للبيانات المقدمة.
   - حدد الاتجاهات والأنماط الرئيسية في البيانات.
   - اشرح العلاقة بين بيانات الطلب (مثل اتجاهات البحث) وبيانات العرض (مثل المحتوى الموجود على يوتيوب والأخبار).

2. **تحديد فرص المحتوى**:
   - اكتشف الفجوات بين الطلب والعرض التي يمكن استغلالها.
   - حدد المواضيع ذات الاهتمام المتزايد والمنافسة المنخفضة.
   - اقترح زوايا جديدة للمواضيع الشائعة.

3. **اقتراح أنواع المحتوى المناسبة**:
   - حدد أنسب منصات وأشكال المحتوى (فيديو، مقال، بودكاست، إلخ) بناءً على البيانات.
   - اقترح أطوال وأشكال مثالية للمحتوى لكل منصة.
   - قدم نصائح حول توقيت النشر المثالي إن أمكن.

4. **أفكار عناوين ومواضيع مبتكرة**:
   - اقترح 5-7 عناوين جذابة وقابلة للنقر لكل فرصة محتوى.
   - قدم أفكاراً لمحتوى فرعي يمكن تضمينه.
   - اقترح كلمات مفتاحية يجب استهدافها في كل محتوى.

5. **نصائح لتمييز المحتوى**:
   - قدم استراتيجيات لجعل المحتوى متميزاً عن المنافسين.
   - اقترح زوايا فريدة أو أساليب عرض مبتكرة.
   - حدد عناصر القيمة المضافة التي يمكن تقديمها للجمهور.

6. **تحذيرات وتحديات محتملة**:
   - أشر إلى أي مخاطر أو تحديات في استراتيجية المحتوى المقترحة.
   - حدد المواضيع المشبعة التي يجب تجنبها.
   - قدم نصائح للتعامل مع المنافسة الشديدة في بعض المجالات.

7. **خطة عمل موجزة**:
   - قدم خطوات عملية لتنفيذ التوصيات.
   - اقترح جدولاً زمنياً مبدئياً لإنتاج المحتوى.
   - حدد مؤشرات النجاح التي يجب متابعتها.

قدم إجاباتك بتنسيق واضح ومنظم، مع التركيز على التوصيات العملية والقابلة للتنفيذ. استند في توصياتك إلى البيانات المقدمة فقط، وتجنب الافتراضات غير المدعومة."""
    
    try:
        # Call OpenAI API with the new client format
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using gpt-3.5-turbo which is more widely available
            messages=[
                {"role": "system", "content": "أنت مستشار محتوى رقمي خبير ومحلل استراتيجي." if language.lower() == "ar" else "You are an expert digital content strategist and data analyst."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract and return the recommendations
        recommendations = response.choices[0].message.content
        return recommendations
    
    except Exception as e:
        # Return error message
        return f"Error generating AI recommendations: {str(e)}"