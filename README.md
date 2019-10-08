# News_Headline_Classification
Using a Tesnsorflow LSTM Layer and Classifying News Headlines to their appropriate categories.

## Dataset link https://www.kaggle.com/uciml/news-aggregator-dataset

### Process :
1. Creating a DataFrame from the Dataset.
2. A view at the number of records per each category.
3. Deciding on a number 'n' for taking the maximum chunk of every category.
4. Shuffle the data.
5. seperating each category equally
6. combining all the categories to a DataFrame.
7. Re-Shuffling the dataset
8. making a LABEL column and filling with 0.
9. Creating a numerical map of each categorie in label.
10. Creating a tokenizer.
11. make the tokenizer train on our text i,e News Headlines.
12. Transforming out text into an integer sequence i.e list.
13. Padding the lists to level the playfield.
14. Splitting considered data into Train and Test sets.
15. Creating a LSTM model.
16. Training the model.
17. Evaluating the model against the test data.
18. Visuaizing how accuracy and loss progressed.
