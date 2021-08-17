# imports
from TaxiFareModel.utils import haversine_vectorized
from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        # set X and y

        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        return pipe

    def run(self):
        # train the pipelined model

        model = self.fit(X_train, y_train)
        return model

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        def compute_rmse(y_pred, y_true):
            return np.sqrt(((y_pred - y_true)**2).mean())
        # compute y_pred on the test set
        y_pred = self.predict(X_test)
        # call compute_rmse
        score = compute_rmse(y_pred, y_test)
        return score


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    trainer = Trainer(X,y).set_pipeline()
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    model = Trainer.run(trainer)
    pipe = Trainer.evaluate(model,X_val, y_val)

    # evaluate
    print(pipe)
