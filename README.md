Random Forest Classifier in SciKitLearn Using Tf-IDF preprocessing for Text Classification - Base problem category as per Ready Tensor specifications.

* random forest
* sklearn
* python
* pandas
* numpy
* scikit-optimize
* flask
* nginx
* uvicorn
* docker
* text classification

This is a Text Classifier that uses a Random Forest implementation through SciKitLearn.

The classifier starts by creating an ensemble of decision trees and assigns the sample to the class that is predicted by the majority of the decision trees.

The data preprocessing step includes tokenizing the input text, applying a tf-idf vectorizer to the tokenized text, and applying Singular Value Decomposition (SVD) to find the optimal factors coming from the original matrix. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

Hyperparameter Tuning (HPT) is conducted by finding the optimal number of decision trees to use in the forest, number of samples required to split an internal node, and number of samples required to be at a leaf node.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as clickbait, drug, and movie reviews as well as spam texts and tweets from Twitter.

This Text Classifier is written using Python as its programming language. Scikitlearn is used to implement the main algorithm, create the data preprocessing pipeline,  and evaluate the model. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.



