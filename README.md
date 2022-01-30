# IntelligentSystemAssignment
NLP assignment


The repository includes:

Markup: -Intelligent_System_Assignment_Sentiment_Detection_over_labeled_tweets.pdf which is the report of the assignemt
        -AssignmentIntelligentSystems.py which is the code of the assignment
        -AssignmentIntelligentSystems.ipynb which is the code of the assignment but in version of jupyter notebook (RECOMMENDED)
        -train.csv train file
        -test.csv test file

In order to run the code is necessary to place train.csv and test.csv in the same folder as AssignmentIntelligentSystems.ipynb or AssignmentIntelligentSystems.py depending of the chosen way to use the code.

It is recommended to use the file code AssignmentIntelligentSystems.ipynb with jupyter notebook since its more organized and is possible
to see the confusion matrix plot of the report. In order to do that:
-open a terminal and place it inside the folder where train.csv and test.csv are and the code as well
-write on terminal "jupyter notebook"
-open file using jupyter notebook

If the file AssignmentIntelligentSystems.py is chosen, then place a terminal inside the folder where the files are, and on terminal write "python3 AssignmentIntelligentSystems.py "

-NOTE: On the assignment is explored the option  stop_words = 'english' on the vectorization methods , to use it its necessary to include it at the beginning of block 2 in the ipynb (as is shown in the comments) or on the analogue line code 59. The same procedure should be done to change
the vectorization method from CountVectorizer to TFidVectorizer (as shown in the comments)
