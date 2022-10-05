import numpy as np
import pandas as pd

def insights_of_df(df):
    """Darstellung der Insights eines Dataframes wie Spaltennamen, Anzahl non-null values, Anzahl null values,
    prozentual null values, Anzahl uniquer Werte und Datentyp.
    ---
    Args:
        df: Dataframe

    ---
    Returns:
        output: Dataframe mit den beschriebenen Spalten

    """
    # Erstellung einer Liste
    output  = []

    # for-Schleife für alle Spalten im DataFrame
    for col in df.columns:

        # Berechnung der einzelnen Variablen
        nonNull  = len(df) - np.sum(pd.isna(df[col]))
        NullValues = np.sum(pd.isna(df[col]))
        percentNA = NullValues/(NullValues+nonNull)
        unique = df[col].nunique()
        colType = str(df[col].dtype)

        # hinzufügen der variablen in die output liste
        output.append([col, nonNull, NullValues,percentNA, unique, colType])

    # pandas Dataframe erstellen mit der output liste
    output = pd.DataFrame(output)

    # Spaltennamen des output dataframes ändern
    output.columns = ['colName','non-null values','null values',"percentNA", 'unique', 'dtype']

    return output
