import pandas as pd

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/heart-disease-cleveland.csv')  

df.replace('?', pd.NA, inplace=True)


df.dropna(inplace=True)

df
#Data Splitting
y=df[' diagnosis']
y

X=df.drop(' diagnosis',axis=1)
X

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

#Linear Regression model
from sklearn.linear_model import LinearRegression 

lr=LinearRegression()
lr.fit(X_train,y_train)
#Applying model to make prediction
y_lr_train_pred=lr.predict(X_train)
y_lr_test_pred=lr.predict(X_test)

#Evaluating model performance
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse=mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2=r2_score(y_test,y_lr_test_pred)

print("Linear Regression: \n")
print("Training MSE: ",lr_train_mse)
print("Training R^2: ", lr_train_r2)
print("Testing MSE: ",lr_test_mse)
print("Testing R^2: ", lr_test_r2)

#Logistic Regression
from sklearn.linear_model import LogisticRegression 

log_r=LogisticRegression(max_iter=2000)
log_r.fit(X_train,y_train)

#Applying model to make prediction
y_log_r_train_pred=log_r.predict(X_train)
y_log_r_test_pred=log_r.predict(X_test)

#Evaluating model performance
from sklearn.metrics import mean_squared_error, r2_score

log_r_train_mse = mean_squared_error(y_train, y_log_r_train_pred)
log_r_train_r2 = r2_score(y_train, y_log_r_train_pred)

log_r_test_mse=mean_squared_error(y_test,y_log_r_test_pred)
log_r_test_r2=r2_score(y_test,y_log_r_test_pred)

print("Logstic Regression: \n")
print("Training MSE: ",log_r_train_mse)
print("Training R^2: ", log_r_train_r2)
print("Testing MSE: ",log_r_test_mse)
print("Testing R^2: ", log_r_test_r2)

#Random Forest
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(X_train,y_train)

#Applying model to make prediction
y_rf_train_pred=rf.predict(X_train)
y_rf_test_pred=rf.predict(X_test)

#Evaluating model performance
from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse=mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2=r2_score(y_test,y_rf_test_pred)

print("Random Forest: \n")
print("Training MSE: ",rf_train_mse)
print("Training R^2: ", rf_train_r2)
print("Testing MSE: ",rf_test_mse)
print("Testing R^2: ", rf_test_r2)