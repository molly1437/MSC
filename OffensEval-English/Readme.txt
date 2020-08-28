step1:
Put the data under the data2 file, three files:
task_a_distant.tsv
task_b_distant.tsv
task_b_distant_ann.tsv

step2:
Open the ipynb files of the three tasks and run the code
Process the data before training

method 1:
Traditional machine learning methods are provided in ipynb
tfidf+svc, this method is very easy to use when the amount of data is small, and it will be very slow if there is more data

Method 2:
After the data is processed in ipynb (after the production file under the data file),
the console can use the following commands
python cnn.py A # Run task A
python cnn.py B # run task B

Two solutions were used in Task A and Task B. 
The first method is traditional machine learning (TF-IDF+SVC), 
and the second method is deep learning (embedding+cnn), 
Task C used tf-idf+svc