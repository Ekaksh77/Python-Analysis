# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
# Import tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
sns.set()
pd.set_option('max_columns', None)

# Load the dataset
df = pd.read_csv('BankChurners.csv')

# Take a first glimpse at the data
df.head()

df.isnull().sum()


df.columns

# Explore the variables
df.describe(include =  'all')

df['Education_Level'].value_counts()

df['Marital_Status'].value_counts()

df['Income_Category'].value_counts()

df['Card_Category'].value_counts()

df['Customer_Age'].plot(kind = 'hist', figsize = (5, 4), color='red')
plt.title("Customer Age", size = 10)
plt.show()


df['Dependent_count'].plot(kind = 'hist', figsize = (5, 4), color = 'blue')
plt.title("Dependent Count", size = 10)
plt.show()


df['Months_on_book'].plot(kind = 'hist', figsize = (5, 4), color = 'purple')
plt.title("Months on book", size = 10)
plt.show()

df['Total_Relationship_Count'].plot(kind = 'hist', figsize = (5, 4), color = 'cyan')
plt.title("Total Relationship Count", size = 10)
plt.show()

df['Months_Inactive_12_mon'].plot(kind = 'hist', figsize = (5, 4), color = 'red')
plt.title("Months Inactive in the last 12 months", size = 10)
plt.show()

df['Contacts_Count_12_mon'].plot(kind = 'hist', figsize = (5, 4), color = 'darkblue')
plt.title("Contacts in the last 12 months", size = 10)
plt.show()

df['Credit_Limit'].plot(kind = 'hist', figsize = (5, 4), color = 'yellow')
plt.title("Credit Limit", size = 10)
plt.show()

df['Total_Revolving_Bal'].plot(kind = 'hist', figsize = (5, 4), color = 'pink')
plt.title("Total Revolving Balance", size = 20)
plt.show()


df['Avg_Open_To_Buy'].plot(kind = 'hist', figsize = (5, 4), color = 'black')
plt.title("Open to Buy Credit (Avg. Last 12 months)", size = 20)
plt.show()


df['Total_Amt_Chng_Q4_Q1'].plot(kind = 'hist', figsize = (5, 4), color = 'red')
plt.title("Change in Transaction Amount (Q4 over Q1))", size = 20)
plt.show()


df['Total_Trans_Amt'].plot(kind = 'hist', figsize = (5, 4), color = 'blue')
plt.title("Total Transaction Amount", size = 20)
plt.show()

df['Total_Trans_Ct'].plot(kind = 'hist', figsize = (5, 4))
plt.title("Total Transaction Count", size = 20)
plt.show()


df['Total_Ct_Chng_Q4_Q1'].plot(kind = 'hist', figsize = (5, 4))
plt.title("Change in Transaction Count (Q4 over Q1)", size = 20)
plt.show()

df['Avg_Utilization_Ratio'].plot(kind = 'hist', figsize = (5, 4))
plt.title("Avg. Card Utilization", size = 20)
plt.show()

#Churn vs. normal 
counts = df.Attrition_Flag.value_counts()
normal = counts[0]
Churn = counts[1]
perc_normal = (normal/(normal+Churn))*100
perc_Churn = (Churn/(normal+Churn))*100
print('There were {} non-Churn ({:.3f}%) and {} Churn ({:.3f}%).'.format(normal, perc_normal, Churn, perc_Churn))


style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,30))
## Plotting heatmap. Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), cmap = "magma", annot=True, mask=mask, center = 0, );
plt.title("Heatmap of all the Features of Train data set", fontsize = 25);


X = df.iloc[:,2:]

y = df.iloc[:,1]

# Split the data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Encode the response variables to 0s and 1s
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
print(y_train)
print(y_test)

# Perform feature scaling to the continuous variables 
sc = StandardScaler()
X_train.iloc[:,[0,2,7,8,9,10,11,12,13,14,15,16,17,18]] = sc.fit_transform(X_train.iloc[:,[0,2,7,8,9,10,11,12,13,14,15,16,17,18]])
X_test.iloc[:,[0,2,7,8,9,10,11,12,13,14,15,16,17,18]] = sc.transform(X_test.iloc[:,[0,2,7,8,9,10,11,12,13,14,15,16,17,18]])

# Turn the categorical variables into dummy variables 

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 3, 4, 5, 6])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))
X_test = np.array(ct.fit_transform(X_test))

# Train the model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Test the model
y_pred_knn = classifier.predict(X_test)
print(np.concatenate((y_pred_knn.reshape(len(y_pred_knn),1), y_test.reshape(len(y_test),1)),1))

# Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
print(accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test,y_pred_knn))
sns.heatmap(cm, annot=True, fmt='d').set_title('knn confusion matrix')

# Train the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Test the model
y_pred_lr = classifier.predict(X_test)
print(np.concatenate((y_pred_lr.reshape(len(y_pred_lr),1), y_test.reshape(len(y_test),1)),1))

# Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
print(accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test,y_pred_lr))
sns.heatmap(cm, annot=True, fmt='d').set_title('Logistic Regression confusion matrix')

# Train the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Test the model
y_pred_svm = classifier.predict(X_test)
print(np.concatenate((y_pred_svm.reshape(len(y_pred_svm),1), y_test.reshape(len(y_test),1)),1))

# Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_svm)
print(cm)
print(accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test,y_pred_svm))
sns.heatmap(cm, annot=True, fmt='d').set_title('SVM confusion matrix')


# Train the model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Test the model
y_pred_dt = classifier.predict(X_test)
print(np.concatenate((y_pred_dt.reshape(len(y_pred_dt),1), y_test.reshape(len(y_test),1)),1))

# Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_dt)
print(cm)
print(accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test,y_pred_dt))
sns.heatmap(cm, annot=True, fmt='d').set_title('Decision Tree confusion matrix')

# Train the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Test the model
y_pred_rf = classifier.predict(X_test)
print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))

# Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test,y_pred_rf))
sns.heatmap(cm, annot=True, fmt='d').set_title('Random Forest confusion matrix')

# Import tensorflow
import tensorflow as tf

# Initialzing the ANN 
ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
ann.fit(X_train, y_train, batch_size = 32, epochs = 50)

# Test the model
y_pred_ann = ann.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5)
print(np.concatenate((y_pred_ann.reshape(len(y_pred_ann),1), y_test.reshape(len(y_test),1)),1))

# Build the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_ann)
print(cm)
print(accuracy_score(y_test, y_pred_ann))
print(classification_report(y_test,y_pred_ann))
sns.heatmap(cm, annot=True, fmt='d').set_title('ANN confusion matrix')





import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from tabulate import tabulate

# Define the model names and corresponding predictions
model_names = ['k-NN', 'Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'ANN']
predictions = [y_pred_knn, y_pred_lr, y_pred_svm, y_pred_dt, y_pred_rf, y_pred_ann]
test_labels = [y_test] * len(model_names)

# Initialize empty lists to store accuracy scores and confusion matrices
accuracy_scores = []
confusion_matrices = []

# Calculate accuracy scores and confusion matrices for each model
for prediction, test_label in zip(predictions, test_labels):
    accuracy = accuracy_score(test_label, prediction)
    matrix = confusion_matrix(test_label, prediction)
    accuracy_scores.append(accuracy)
    confusion_matrices.append(matrix)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_scores,
    'Confusion Matrix': confusion_matrices
})

# Sort the DataFrame by accuracy in ascending order
results_df.sort_values(by='Accuracy', ascending=True, inplace=True)

# Convert the DataFrame to an interactive table with color coding
table = tabulate(results_df, headers='keys', tablefmt='html', 
                 numalign="center", stralign="center",
                 colalign=("center", "center", "center"),
                 floatfmt=".4f", 
                 disable_numparse=True,
                 showindex=True)

# Apply color coding to the table rows based on accuracy
color_table = table.replace('<tr>', '<tr style="background-color: #FFCCCC;">', 1)
color_table = color_table.replace('<tr>', '<tr style="background-color: #CCFFCC;">', -1)

# Display the colorful interactive table
from IPython.display import display, HTML
display(HTML(color_table))
