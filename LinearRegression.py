import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Dataframe importieren dont know if I am allowed to publish the URL so i deleted it
df = pd.read_csv("")

print(df.head())
prod_per_year=df.groupby("year").totalprod.mean().reset_index()
X=prod_per_year.year
#print(X)
X=X.values.reshape(-1,1)
#print(X)
y=prod_per_year.totalprod

#Linear Regression Model
regr=linear_model.LinearRegression()
regr.fit(X,y)
print(regr.coef_[0])
print(regr.intercept_)
y_predict=regr.predict(X)


plt.subplot(2,1,1)
plt.plot(X,y_predict,"-r")
plt.scatter(X,y)
plt.show()

X_future=np.array(range(2013,2050))
X_future=X_future.reshape(-1,1)
future_predict=regr.predict(X_future)

plt.subplot(2,1,2)
plt.plot(X_future,future_predict,"o")
plt.show()
