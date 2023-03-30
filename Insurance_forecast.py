import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
df = data.copy()

cond_variables = ['age','bmi']
cat_variables = ['sex','children','smoker','region']

df.isnull().sum() # No null values
df.hist(rwidth=0.6)
plt.tight_layout()

plt.subplot(2,1,1)
plt.title("Age vs charges")
plt.scatter(df.age,df.charges,s=2,c='g')

plt.subplot(2,1,2)
plt.title("BMI vs charges")
plt.scatter(df.bmi,df.charges,s=2,c='r')

plt.tight_layout()

for i in cat_variables:
    plt.subplot(2,2,cat_variables.index(i)+1)
    x_cat = df[i].unique()
    y_cat = df.groupby(i).mean()['charges']
    colors = ['r','g','b','c']
    plt.title('charges vs {}'.format(i))
    plt.ylabel("charges")
    plt.xlabel("{}".format(i))
    plt.bar(x_cat,y_cat,color=colors)
    
plt.tight_layout()

correlation = df[['age','bmi','charges']].corr()

plt.acorr(df.charges.astype('float64'),maxlags=12)
# Not much autocorrelation

predicted = df.charges
predicted_log = np.log(predicted)

plt.subplot(2,1,1)
plt.title("Log Normal Distribution of Charges")
predicted.hist(rwidth=0.6,bins=30)

plt.subplot(2,1,2)
plt.title("Normal Distribution of Charges")
predicted_log.hist(rwidth=0.6,bins=30)

plt.tight_layout()

df['charges'] = predicted_log
df['bmilog']=np.log(df.bmi)
df.drop(['bmi'],axis=1,inplace=True)

df[cat_variables] = df[cat_variables].astype('category')
dummy_df = pd.get_dummies(df,drop_first=True)

X = dummy_df.drop(['charges'],axis=1)
Y = dummy_df['charges']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train,y_train)

train_r2 = linreg.score(x_train,y_train)
test_r2 = linreg.score(x_test,y_test)

y_pred = linreg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test,y_pred))

from sklearn.metrics import mean_absolute_error
mse = mean_absolute_error(y_test,y_pred)

