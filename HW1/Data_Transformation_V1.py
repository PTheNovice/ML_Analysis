"""
Student Name: Perry Francois-Edwards
GT User ID: pdfe3
GT ID: 903010832
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def test_car():
    df = pd.read_csv("data/car.data", delimiter=',')
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    # print(df)
    """
    Preprocessing the dataframe to make it numerical for the algorithms. 
    """
    # print(df)
    label_encoder = LabelEncoder()
    new_df = df
    new_df['buying'] = label_encoder.fit_transform(new_df['buying'])
    new_df['maint'] = label_encoder.fit_transform(new_df['maint'])
    new_df['doors'] = label_encoder.fit_transform(new_df['doors'])
    new_df['persons'] = label_encoder.fit_transform(new_df['persons'])
    new_df['lug_boot'] = label_encoder.fit_transform(new_df['lug_boot'])
    new_df['safety'] = label_encoder.fit_transform(new_df['safety'])
    new_df['class'] = label_encoder.fit_transform(new_df['class'])

    # cat_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    # target_column = ['class']
    # preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns), ], remainder='passthrough')
    # X_encoded = preprocessor.fit_transform(df)
    # # print(X_encoded)
    # new_df = pd.DataFrame(X_encoded)
    # new_df.rename(columns={21: "class"}, inplace=True)

    # print(new_df)
    new_df.to_csv('data/FinalCar.csv', index=False)

    f = new_df['class'].value_counts()[0]
    g = new_df['class'].value_counts()[1]
    h = new_df['class'].value_counts()[2]
    i = new_df['class'].value_counts()[3]
    # print(f, g, h, i)


def test_wine_redonly():
    df_red = pd.read_csv("data/winequality-red.csv", delimiter=';')

    # print(df_red)

    df_red.to_csv('data/FinalWine_Red.csv', index=False)


    d = df_red['quality'].value_counts()[3]
    e = df_red['quality'].value_counts()[4]
    f = df_red['quality'].value_counts()[5]
    g = df_red['quality'].value_counts()[6]
    h = df_red['quality'].value_counts()[7]
    i = df_red['quality'].value_counts()[8]
    # print(d, e, f, g, h, i)




if __name__ == "__main__":
    # test_wine()
    # test_occup()
    test_car()
    test_wine_redonly()