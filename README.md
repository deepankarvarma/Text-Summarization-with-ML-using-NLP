# Text Summarizer

## Python code for building a Text Summarizer using Machine Learning - Natural Language Processing.The data is already cleaned hence no pre-processing is required.

### Dataset Link :- https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

Step 1 : Import the necessary libraries for data preprocessing, model training, and evaluation. For example, you can use libraries like pandas, numpy, sklearn, nltk, tensorflow, etc.<br><br>
Step 2 : Load the train.csv and validation.csv files into pandas dataframes using the read_csv() function.<br><br>
Perform any necessary preprocessing on the data, such as removing stopwords, stemming, tokenizing, etc. You can use the NLTK library for these tasks.<br><br>
Step 3 : Split the data into input and target variables. In this case, the input will be the "article" column, and the target will be the "highlight" column.<br><br>
Step 4 : Encode the input and target data using a suitable method. For example, you can use the Tokenizer class in the Keras library to convert the input and target data into sequences of integers.<br><br>
Step 5 : Define and train a neural network model using the training data. You can use a sequence-to-sequence model, such as the Encoder-Decoder model, to generate highlights from the input article.<br><br>
Step 6 : Evaluate the performance of the model on the validation data. You can use metrics such as accuracy, precision, recall, and F1 score to evaluate the performance of the model.<br><br>
Step 7 : Test the accuracy of the model on the test data using the same evaluation metrics.