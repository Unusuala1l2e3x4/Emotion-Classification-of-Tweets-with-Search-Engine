
# 650 Project - Alexander He


## Resume Entry
Emotion Classification of Tweets with Search Engine\
SI 650: Information Retrieval\
September 2022 – December 2022
-	Compared performance between SVM and Random Forest models for classifying emotions of tweets. Constructed n-grams, POS, and sentiment-related features.
-	Created tweets search engine with emotion filtering using Learning-to-Rank models with relevance features, e.g., Okapi BM25 and TF-IDF with Bayesian smoothing. 
-	Evaluated search result quality with NDCG cutoffs. Identified most important features with ablation study.


### Data
- download 'merged_training.pkl' from the source denoted in the report, and put it in the folder 'emotion\datasets\Emotion Dataset for Emotion Recognition Tasks'
  - https://www.dropbox.com/s/607ptdakxuh5i4s/merged_training.pkl

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


