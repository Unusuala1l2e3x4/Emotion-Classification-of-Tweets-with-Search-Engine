
Project Report: [project report/project_report.pdf](https://github.com/Unusuala1l2e3x4/Emotion-Classification-of-Tweets-with-Search-Engine/blob/main/project%20report/project_report.pdf)


## Resume Entry
Emotion Classification of Tweets with Search Engine\
September 2022 – December 2022\
SI 650 – Information Retrieval
-	Created a specialized search engine in a dual-model setup, utilizing a dataset of 400,000 tweets to classify six emotion labels with SVM and retrieve similar tweets.
-	Integrated a Learning-to-Rank model using PyTerrier to rank tweets by relevance to user queries, employing techniques like BM25, TF-IDF, and Bayesian smoothing.
-	Engineered features such as POS tagging, sentiment measures, and n-grams to optimize the performance of both classifier and retrieval models.
-	Attained high-quality retrieval results, indicated by NDCG scores of 0.84 - 0.93 for the top 5-20 ranked items and an F1 score of 0.85 for emotion classification.



### Data
- download 'merged_training.pkl' from the source denoted in the report, and put it in the folder 'emotion\datasets\Emotion Dataset for Emotion Recognition Tasks'
  - updated: https://www.icloud.com/iclouddrive/084E9TMZ_lykn3QhU-kIX1DDQ
  - original github repo: https://github.com/dair-ai/emotion_dataset

### How to run:
1. **Execute "project_classify.ipynb"**
    - This will train the classification model and vectorizer, and save them both
    - Please ignore the commented-out code used for comparing models and hyperparameter tuning
    - This also saves metric results to a folder
    - The saved model is:
      - LinearSVC(random_state=RANDOM_STATE, max_iter=10000, C=0.085)
2. **Execute "project_ltr_v3.ipynb"**
    - This will create the index (if it doesn't exist) and train the learning-to-rank pipeline
    - This will also save the documents dataframe for the interface, but not the trained learning-to-rank pipeline
    - This also saves metric results to a folder
3. **Execute "project_interface.py"**
    - This is the user interface. It first loads in the model, vectorizer, and documents dataframe. Then it trains the learning-to-rank model (takes a few seconds), then runs the text-based interface
    - Choice 2 is supposed to show a graph in a popup via plt.show()



### Disclaimer
- BM25 and TF-IDF both perform better than my best trained ML model. I am using the ML model for purpose of this assignment



### Known issues:
- **Bug with PyTerrier:** program throws error and exits when no results found
  - https://github.com/terrier-org/pyterrier/issues/352


