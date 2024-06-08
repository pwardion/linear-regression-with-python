import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pwaregg import find_best_corr_threshold, find_best_pvalue_threshold, find_and_fill, encode, detect_outliers

data = pd.read_csv("C:\\Users\\pward\\OneDrive\\Masaüstü\\cars.csv")
b = find_and_fill(dataframe=data)
y = b.iloc[:, -1:].values
y = pd.DataFrame(data=y, columns=['Price'])
x = b.iloc[:, 0:-1].values
x = pd.DataFrame(data=x, columns=["Car_ID", "Brand", "Model", "Year", "Kilometers_Driven", "Fuel_Type", 
                                  "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Seats"])



x.drop(x.columns[0], axis=1, inplace=True)
x.drop(x.columns[0], axis=1, inplace=True)
x.drop(x.columns[0], axis=1, inplace=True)

x = x.astype({'Year': 'int', 'Kilometers_Driven': 'int', 'Fuel_Type': 'str', 'Transmission': 'str', 
              'Mileage': 'int', 'Engine':'int', 'Power':'int', 'Seats':'int'})


res = encode(x)
df = pd.concat([res, y],axis=1)




x_train, x_test, y_train, y_test = train_test_split(res, y, random_state=42, test_size=0.2)
li = LinearRegression()
li.fit(x_train, y_train)

p1 = li.predict(x_train)
p2 = li.predict(x_test)

mae = mean_absolute_error(y_train, p1)
mae1 = mean_absolute_error(y_test, p2)
r1 = r2_score(y_train, p1)
r2 = r2_score(y_test, p2)
print(int(mae))
print(r1)

print(int(mae1))
print(r2)



newdf = detect_outliers(dataframe=df)
y = newdf.iloc[:, -1:].values
y = pd.DataFrame(data=y)

asd = find_best_corr_threshold(dataframe=newdf, target= 'Price', model='poly', n=0.6)
dff = pd.DataFrame(data=asd)

print(dff)
x_train, x_test, y_train, y_test = train_test_split(dff, y, random_state=42, test_size=0.2)
li = LinearRegression()
li.fit(x_train, y_train)

p1 = li.predict(x_train)
p2 = li.predict(x_test)

mae = mean_absolute_error(y_train, p1)
mae1 = mean_absolute_error(y_test, p2)
r1 = r2_score(y_train, p1)
r2 = r2_score(y_test, p2)
print(int(mae))
print(r1)

print(int(mae1))
print(r2)