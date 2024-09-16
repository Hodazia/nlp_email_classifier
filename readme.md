![The new photo](phone2.jpeg 'Title')

# Email/SMS spam classifier

In the given project, basically we have build a SMS/Email classifier system where user by giving their email or sms can predict whether it is a spam or not.

### Dataset
The dataset can be found here [dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Project Steps
Some of the steps performed in the project are 
1. Data cleaning
2. EDA
3. Text Preprocessing
4. Model Building
5. Evaluation
6. Improvement
7. Deploy on streamlit

> Some of the necessary libraries used are in [requirements.txt](https://github.com/Hodazia/nlp_email_classifier/blob/main/requirements.txt)

To use the above code follow the steps below:
1. Create a virtual environment first , open VScode and run the command
``` python -m venv 'virtual_environment_name' ```
   ,you can name virtual_environment_name as 'env1', then to activate it run
```env1\Scripts\activate ```

2. Install all the required libraries
``` pip install -r requirements.txt```
3. Now using streamlit you can create a web app and insert your sms/email, run the code below
``` streamlit run app.py```

Conclusion:
In the above project, we have been given a classification problem where we have tried differet ML algos like DT,RF,Logistic, Bagging,Boosting along with NLP to predict whether a given text can be a spam or not.