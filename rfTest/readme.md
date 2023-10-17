## My work for THESIS2

This work aims to detect if the URI is an LFI attack.
Our goal is to produce a detection model for LFI using the Random Forest Classifier and evaluate its performance metrics using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Gini Importance (Feature Importances)

The datasets were acquired using Wireshark and several sessions are labeled for normal and attack. Other data were also collected in the ISCX2016URI for benign activities. The machines that we used were Mutillidae II, DVWA, and Metasploitable

For data engineering, since we are only detecting Layer 7, the URIs are our best friend. From the Wireshark session we used our knowledge about LFI and its attack vector. Using that pattern and behavior, we checked and produced a method for labeling if the URI has trait/s of Double Encoding, PHP/Data Wrapper Abuse, Directory Traversal, Null Byte Poisoning, and File Path Manipulation inside. From that we can indicate 0 as none and 1 if it has such patterns shown in the URI.

Another method implemented was to count the number of strings and special characters that occurred in the URI, these were the length of the URI, the Special Character count, the Path Depth, and the Tokenizer that counted how many segments were in the URI. 

After the data was nearly completed, we also implemented SMOTE-ENN method to balance the dataset to ensure the fairness of the playing field. This will help us to reduce the gap, maintain consistency, and not be biased when it comes to evaluation

In tuning the algorithm, we used RandomizedSearchCV to automate finding the best parameter for Random Forest and for detecting Local File Inclusion. We also set the range of values for each attribute and set the cross-validation to 10 with 500 iterations.

In splitting the dataset into 80/20 ratio for training and testing respectfully, the randomized value is to be set at the default of 42.
The model will also produce results on which segments are usually affected the most by LFI.

The comparison with other algorithms was also shown in the notebook and using solely the algorithm at default provides better accuracy compared to other detection algorithms and its efficiency when it comes to time. Deviation was also considered here and remarkably it is close to a value of 0.

## Hypothesis (Expectation)
The hyperparameter tune value will increase the model's accuracy and have the lowest standard deviation amongst the other algorithms to be used

## Output (Reality)
The tuned Random Forest is not efficient when it comes to time since it needs to fit based on the parameters however the Tuned and Default Random Forest has the same accuracy score and standard deviation. We can conclude that even at default, the algorithm performs the best without even tweaking the algorithm expecting it will squeeze more performance to the model. 
