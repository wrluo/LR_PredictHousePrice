import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics  
import numpy as np

data = pd.read_csv('Advertising.csv')

# sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.8, kind='reg')  
# plt.show()

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)
  
linreg = LinearRegression()
linreg.fit(X_train, y_train)  

y_pred = linreg.predict(X_test)  

sum_mean = 0
for i in range(len(y_pred)):  
    sum_mean+=(y_pred[i]-y_test.values[i])**2  
sum_erro=np.sqrt(sum_mean/len(y_pred))

print("RMSE by hand:",sum_erro)

plt.figure()  
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")  
plt.plot(range(len(y_pred)),y_test,'r',label="test")  
plt.legend(loc="upper right")
plt.xlabel("the number of sales")  
plt.ylabel('value of sales')  
plt.show()