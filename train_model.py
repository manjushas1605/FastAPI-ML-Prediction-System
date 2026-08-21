import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

#load dataset
df=pd.read_csv("insurance.csv")

#Feature and Target
x=df.drop("charges",axis=1)
y=df['charges']

#Categorical columns
categorical_columns=["sex","smoker","region"]

#Preprocessor
preprocessor=ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore")
            ,categorical_columns
        )
    ],
    remainder="passthrough"
)

#create pipeline
model=Pipeline(
    steps=[("preprocessor",preprocessor)
        ,("regressor",LinearRegression())])
#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                            random_state=2)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("mean_absolute_error",mae)
print("r2_score",r2)


joblib.dump(model,"model.pkl")

print("model saved sucessfully!")
