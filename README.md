# Sentiment Prediction from Tweets

This project applies machine learning and deep learning models to detect signs of depression in social media texts. It includes three distinct approaches:

🌲 Random Forest: Classical ensemble learning

📊 Support Vector Machine (SVM): With GloVe embeddings

🤗 BERT: Transformer-based sentiment analysis

All models are trained and evaluated using a labeled dataset of tweets.

---
## 📄 Project Presentation

You can view the full presentation in PDF format here:

👉 [Click to view the presentation](presentation-depression-prediction.pdf)


## Project Structure

- `sentiment_tweets3.csv`: Dataset of tweets labeled as depressive or not
- `RandomForestModel.ipynb`: The notebook containing the full pipeline from preprocessing to evaluation.
- `SVM.ipynb`: 	Implements SVM with GloVe embeddings
- `BERT.ipynb`: Uses HuggingFace’s BERT for depression prediction
  


## Dataset Overview

The dataset contains two main columns:

- **message**: The tweet text

- **label**: Depression indicator (0 = not depressive, 1 = depressive)

Example entries include casual tweets, tweets with links, and emotional expressions.

| Index | Message | Label |
|-------|---------|-------|
| 106 | just had a real good moment. i missssssssss him so much, | 0 |
| 217 | is reading manga http://plurk.com/p/mzp1e | 0 |
| 288 | @lapcat Need to send 'em to my accountant tomorrow. Oddly, I wasn't even referring to my taxes. Those are supporting evidence, though. | 0 |


## Project Workflow (NB:detailed workflow in below)
### 🌲 Random Forest Model

- **Approach**: Bagging-based ensemble model using decision trees.

- **Features**: TF-IDF vectorization of tweet text.

- **Target**: Binary label indicating depressive vs. non-depressive tweet.

- **Steps**:

  1. **Preprocessing** text cleaning and TF-IDF transformation.
  
  2. **Model training and prediction** with scikit-learn’s Random Forest.
  
  3. **Evaluation** using classification report (accuracy, precision, recall).

---

### 📊 SVM with GloVe Embeddings

- **Approach**: Support Vector Machine (linear and RBF kernels).

- **Features**: Word embeddings from pre-trained GloVe vectors.

- **Steps**:

  1. **Preprocess** text (cleaning, tokenizing, removing stopwords).
  
  2. Use GloVe to generate averaged **vector representations** per tweet.
  
  3. Handle **class imbalance** via oversampling.
  
  4. **Train and evaluate** with K-Fold cross-validation.
  
  5. **Evaluation:** Accuracy, F1-score, confusion matrix.
---
### 🤗 BERT-based Sentiment Classifier

- **Model:** nlptown/bert-base-multilingual-uncased-sentiment from HuggingFace.

- **Approach:** Use pre-trained transformer to classify tweet sentiment.

- **Steps:**

  1. Load **tokenizer and model.**
  
  2. **Tokenize** each tweet and pass through BERT.
  
  3. Apply softmax to obtain **class probabilities.**
  
  4. **Predict** label using a defined probability threshold (e.g., 0.5).
  
  5. **Evaluation:**
  
    - Custom accuracy function.
    
    - Example prediction shown using a randomly selected tweet.
--- 
## Detailed Project Workflow



### 🌲 Random Forest Model
This notebook focuses on detecting depressive sentiment in tweets using Natural Language Processing (NLP) techniques. It includes preprocessing, feature extraction with GloVe embeddings, oversampling, Random Forest classification, and evaluation via K-Fold Cross Validation.
#### 1. Load and Prepare Data
The dataset is loaded, and columns are renamed for simplicity. Missing values are removed to ensure model reliability.

#### 2. Text Preprocessing
The preprocessing step improves the quality of the input for the model. It includes:

Lowercasing: Converts all characters to lowercase for uniformity.

Removing non-alphabetic tokens: Removes numbers, punctuation, and URLs.

Removing stopwords: Common English words (like "the", "is") are removed to reduce noise.

Stemming: Words are reduced to their root form (e.g., "running" becomes "run") to generalize the text representation.

#### 3. K-Fold Cross Validation
To ensure robust model performance, the dataset is split into 3 different train/test combinations. This technique allows the model to be validated on all parts of the dataset without overlap between training and testing data in any fold.

#### 4. Word Embedding with GloVe
To convert text into numerical format:

Pre-trained GloVe embeddings (100-dimensional) are used.

Each tweet is represented as the average of its word vectors.
This helps capture the semantic meaning of the text beyond simple word counts.

#### 5. Handle Imbalanced Classes
To avoid the model favoring the majority class:

Random oversampling is applied to balance the dataset by duplicating examples of the minority class in the training set.

#### 6. Classification
A pipeline is created to standardize the data and then apply a Random Forest classifier. This model is chosen for its robustness, ability to handle non-linear relationships, and feature importance interpretability.

####  7. Model Training and Evaluation
Each fold of the K-Fold Cross Validation involves:

Training the model on the training set

Testing on the validation set

Collecting metrics like accuracy, precision, and recall

Building a confusion matrix to understand classification performance

All metrics are collected across folds and visualized to evaluate the model’s consistency and reliability.

📊 **Evaluation Metrics**
After completing training:

- Accuracy: How often the model correctly predicts.

- Precision: How many predicted positives are actually correct.

- Recall: How many actual positives were correctly predicted.

- Confusion Matrix: Shows the number of true positives, false positives, true negatives, and false negatives.

Plots are generated to visually assess the spread and reliability of these metrics.

---

### 📊 SVM with GloVe Embeddings
This notebook focuses on classifying text messages into two categories (e.g., depressive vs. non-depressive) using Support Vector Machines (SVM) with both linear and RBF kernels. The pipeline includes data cleaning, text preprocessing, word embeddings via GloVe, and model evaluation using K-Fold cross-validation.

####  1. Data Preparation
Loading the dataset: The dataset is imported and two relevant columns are selected: the message and its associated label.

Cleaning the data: Any missing (NaN) values are removed to ensure quality.

Renaming columns: Columns are renamed for better readability: message and label.

####  2. Text Preprocessing
Tokenization: Each message is split into words using NLTK’s tokenizer.

Lowercasing and filtering: All words are converted to lowercase and non-alphabetic characters (like numbers and punctuation) are removed.

Stopword removal: Common English stopwords (e.g., "the", "is", "and") are removed to reduce noise.

Stemming: Words are reduced to their root form using the Porter Stemmer (e.g., "running" → "run").

These steps are essential to normalize the text and reduce vocabulary size for better performance in downstream tasks.

####  3. Handling Imbalanced Data
The class distribution is imbalanced:

Class 0 (e.g., non-depressive): 8000 samples

Class 1 (e.g., depressive): 2314 samples

To address this, Random Oversampling is applied during training to balance the minority class and prevent bias.

####  4. K-Fold Cross-Validation
The dataset is split into 3 folds using KFold with shuffling to ensure robustness and reduce overfitting.

For each fold, the model is trained on two parts and validated on the remaining one.

This ensures that each sample is used for both training and validation across the experiment.

####  5. Text Vectorization using GloVe
The GloVe (Global Vectors for Word Representation) pre-trained embeddings are used to convert words into dense numerical vectors.

Each message is transformed into a fixed-length vector by averaging the embeddings of its words.

This method captures the semantic meaning of text efficiently.

####  6. Model Training and Evaluation (Linear Kernel)
A pipeline is built with a Standard Scaler followed by an SVM with a linear kernel.

For each fold:

The training data is vectorized and oversampled.

The pipeline is trained and evaluated on the validation fold.

Metrics such as accuracy, precision, recall, and confusion matrix are computed and stored.

The results are visualized with boxplots for each metric and a heatmap of the confusion matrix.

####  7. Model Training and Evaluation (RBF Kernel)
The same pipeline is reused but with an SVM using the RBF (Radial Basis Function) kernel.

The process (resampling, scaling, training, validation, and evaluation) is repeated.

Results are again visualized to compare the RBF kernel’s performance to the linear one.

####  8. Evaluation Metrics & Visualizations
For both kernels, the following are generated:

  - Boxplots showing the distribution of:
  
  - Accuracy across folds
  
  - Precision across folds
  
  - Recall across folds
  
  - Confusion Matrix showing how well the model predicted each class

---
### 🤗 BERT-based Sentiment Classifier
This notebook uses a pre-trained BERT model to evaluate the likelihood that a tweet expresses depressive sentiments. Unlike the SVM approach, which uses classical ML with GloVe embeddings, this model applies deep learning and transfer learning from the transformer-based BERT architecture.

####  1. Model Setup
Libraries: The notebook uses the transformers library from Hugging Face, along with PyTorch and scikit-learn.

Pre-trained model: nlptown/bert-base-multilingual-uncased-sentiment is used, which supports multilingual sentiment classification.

Tokenizer: A BERT tokenizer specific to the selected model is loaded to convert raw text into BERT-compatible token IDs.

####  2. Loading the Data
A CSV file containing tweets and their depression labels is loaded into a pandas DataFrame.

The relevant columns are:

"message to examine" – the text of the tweet.

"label (depression result)" – the ground truth label (e.g., 1 = depressive, 0 = non-depressive).

####  3. Depression Probability Prediction
A custom function is implemented to:

Tokenize each tweet.

Pass it through the BERT model to get output logits.

Apply the softmax function to obtain class probabilities.

Return the probability associated with the depressive class (index 1).

####  4. Threshold-Based Classification and Accuracy
A prediction threshold is set (default = 0.5).

Tweets with a predicted probability above the threshold are labeled as depressive (1).

A function calculates classification accuracy by comparing these predictions to the actual labels.

####  5. Example Inference
One tweet is selected randomly (using index 42 in this example).

Its actual label, predicted depression probability, and final predicted label are printed.

Example Output:

**Randomly Selected Tweet:
"@Carly109 love the new song. and the chorus is real nice and catchy. gud job. who did the track and recording jordan?"
Actual Label: 0
Predicted Depression Probability: 0.037
Predicted Label: 0**

This output shows that the BERT model correctly identifies the non-depressive nature of the tweet.


---
👤 Author
Nouha BHY
