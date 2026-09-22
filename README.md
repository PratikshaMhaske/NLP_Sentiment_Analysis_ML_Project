**📱 Customer Review Sentiment Analysis**

**📌 Project Overview**

This project focuses on Sentiment Analysis of customer reviews using Natural Language Processing (NLP) and traditional Machine Learning techniques.

The objective is to analyze customer reviews and classify them into three sentiment categories:

**😊 Positive**

**😐 Neutral**

**😞 Negative**


The project also includes multilingual review handling, where supported non-English reviews are automatically detected and translated into English before sentiment prediction.

**🎯 Objectives**

Analyze customer reviews using NLP techniques.
Understand patterns and trends in customer feedback.
Classify reviews into Positive, Neutral, and Negative sentiments.
Handle reviews written in multiple languages.
Convert textual data into numerical features using TF-IDF.
Compare different traditional Machine Learning models.
Select a final model based on appropriate evaluation metrics.
Deploy the trained sentiment analysis model using Streamlit.

**📊 Dataset**

The dataset contains 1,440 customer reviews related to mobile phones.

Dataset Features
Column	Description
title	Title of the customer review
rating	Customer rating
body	Detailed review text

The title and body of each review were combined to create the complete review text used for analysis.


**🏷️ Sentiment Labeling**

Sentiment labels were derived from the customer ratings:

**Rating	Sentiment
1–2	Negative
3	Neutral
4–5	Positive**

Note: The sentiment labels are rating-derived rather than manually annotated.


**🔍 Exploratory Data Analysis**

Exploratory Data Analysis (EDA) was performed to understand the dataset and identify important patterns.

**Key Observations**

The dataset contains all three sentiment categories.
Positive reviews form the largest proportion, followed by Negative and Neutral reviews.
Ratings 5 and 1 are relatively prominent, indicating that many customers expressed strong positive or negative opinions.
Most reviews are relatively short, while a small number of reviews are considerably longer.
Frequently occurring terms are related to product features such as:
Phone
Camera
Battery
Screen
Display
Quality
Price
Performance

Customer reviews contain both highly positive and highly negative experiences, resulting in a somewhat polarized rating distribution.


**🌐 Multilingual Review Handling**


The dataset contains primarily English reviews along with a small number of reviews containing other languages and code-mixed text.


A multilingual preprocessing pipeline was implemented:

Review → Language Detection → Translation to English → Text Cleaning → TF-IDF → Sentiment Prediction

Supported non-English languages are translated into English using the NLLB-200 multilingual translation model.

English reviews are processed directly without translation.

This allows the application to handle selected multilingual customer reviews while maintaining a common English-based NLP pipeline.


**🧹 Text Preprocessing**

The following preprocessing steps were applied:

Conversion of text to lowercase
Unicode normalization
Removal of URLs
Removal of HTML content
Removal of punctuation
Removal of extra whitespace
Translation of supported non-English reviews into English

**Stopword Removal**

Stopword removal was not used because words such as "not", "no", and "never" can carry important information in sentiment analysis.


**Lemmatization**

Lemmatization was not applied because the project uses TF-IDF features and the dataset contains multilingual and translated text. Keeping the original word forms also helps preserve potentially useful sentiment-related patterns.


**🔢 Feature Extraction**

**TF-IDF**

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert textual reviews into numerical features suitable for Machine Learning.

Both:

Unigrams
Bigrams

were considered.

After applying frequency-based filtering, 8,464 features were available. To control dimensionality and reduce potential noise, the top 5,000 features were retained for model training.

The TF-IDF vectorizer was fitted only on the training data to avoid data leakage.


**🤖 Machine Learning Models**

Three traditional Machine Learning approaches were initially evaluated:

Logistic Regression
Linear Support Vector Machine (Linear SVM)
Multinomial Naive Bayes

These models are well suited for high-dimensional sparse text representations such as TF-IDF.


**📈 Model Evaluation**

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
Macro F1-score
Weighted F1-score
Confusion Matrix

Since the dataset contains some class imbalance and the Neutral class is more difficult to classify, Macro F1-score was given greater importance during model comparison.


**📊 Model Comparison**

Model	Accuracy	Macro F1	Weighted F1
Logistic Regression – Baseline	78.0%	57.0%	73.0%
Linear SVM – Baseline	78.0%	62.0%	76.0%
Logistic Regression – Tuned	76.7%	61.8%	74.5%
**Linear SVM – Tuned	76.4%	63.0%	74.9%**

**🏆 Final Model**

The Tuned Linear SVM was selected as the final model based on its highest Macro F1-score of approximately 63.0% among the evaluated models.

**Final Model Performance
Accuracy: 76.4%
Macro F1-score: 63.0%
Weighted F1-score: 74.9%**

Although the baseline Linear SVM achieved slightly higher accuracy, the tuned Linear SVM provided the highest Macro F1-score, which was considered more relevant because of the class imbalance and difficulty in identifying Neutral reviews.


**⚠️ Model Limitations**

The Neutral class is comparatively difficult to classify.
The dataset is somewhat imbalanced toward Positive reviews.
Sentiment labels are derived from ratings rather than manually annotated.
Language detection may not always be reliable for very short or heavily code-mixed reviews.
Translation quality may vary for certain multilingual or code-mixed reviews.
The model predicts one overall sentiment for a review and does not perform aspect-level sentiment analysis.
For example, a review containing both positive and negative comments about different product features may still receive only one overall sentiment label.


****🚀 Streamlit Deployment

The trained sentiment analysis system was deployed using Streamlit.

The application allows users to enter a customer review and receive:

Detected language
Translated review when applicable
Processed review text
Predicted sentiment

The multilingual translation model uses lazy loading and caching, so the larger translation model is loaded only when a non-English supported review requires translation.


**🛠️ Technologies Used**

Python
Pandas
NumPy
Scikit-learn
NLTK / NLP techniques
TF-IDF
Logistic Regression
Linear SVM
Multinomial Naive Bayes
Hugging Face Transformers
NLLB-200
LangDetect
PyTorch
Joblib
Streamlit
Matplotlib
WordCloud


**🔮 Future Improvements**

Improve Neutral sentiment classification.
Experiment with class balancing techniques.
Explore advanced transformer-based sentiment models.
Improve handling of code-mixed reviews.
Perform aspect-based sentiment analysis for features such as camera, battery, display, and performance.
Use a larger manually annotated dataset for more reliable sentiment labels.
Add confidence scores to predictions.


**📚 Key Learning Outcomes**

Through this project, I gained practical experience in:

Natural Language Processing
Text preprocessing
Multilingual text handling
Language detection and translation
TF-IDF feature engineering
Traditional Machine Learning for NLP
Model comparison and hyperparameter tuning
Imbalanced classification evaluation
Model serialization
Streamlit deployment
Building an end-to-end NLP application


**✅ Conclusion**

This project demonstrates an end-to-end Customer Review Sentiment Analysis workflow, from exploratory analysis and multilingual text preprocessing to TF-IDF feature extraction, Machine Learning model comparison, model selection, and Streamlit deployment.

The Tuned Linear SVM was selected as the final model based on its Macro F1-score and provides a practical foundation for analyzing customer feedback across Positive, Neutral, and Negative sentiment categories.




**Author: Er. Pratiksha Mhaske**

**LinkedIn: https://www.linkedin.com/in/pratiksha-mhaske**

**GitHub: https://github.com/PratikshaMhaske**
