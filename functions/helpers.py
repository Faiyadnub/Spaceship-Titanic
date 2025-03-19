from typing import List, Union

import numpy as np
import seaborn as sns
import pandas as pd


def split_and_insert(df: pd.DataFrame, split_col: str, new_cols: List[str], spliter: str) -> pd.DataFrame:
    """
    Splits a column into new columns and inserts them into the dataframe.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - split_col (str): The name of the column to split.
    - new_cols (list): List of new column names to insert.

    Returns:
    - pd.DataFrame: The modified dataframe with new columns inserted.
    """

    if split_col not in df.columns:
        raise ValueError(f"Column '{split_col}' does not exist in the DataFrame.")
    if len(new_cols) < 2:
        raise ValueError("The list of new columns must be larger than 2 for splitting.")
    if spliter is None:
        raise ValueError("You must provide symbol by which column will be split.")

    # Get column index
    col_index = df.columns.get_loc(split_col)

    # Split the column into new ones
    df[new_cols] = df.loc[:, split_col].str.split(spliter, expand=True)
    list_of_new_col = list(df.columns)

    # Reorder columns
    place = 1
    cols = list_of_new_col.copy()
    for new_col in new_cols:
        cols.insert(col_index + place, cols.pop(cols.index(new_col)))
        place += 1

    df = df[cols]

    return df


def fill_missing_values_using_method(
        df: pd.DataFrame, group_cols: Union[str, List[str]], target_col: str, method: str = "mode") -> pd.DataFrame:
    """
    Fills nan values in the target column of the dataframe by
    using the method of that column within each group specified by multiple columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    group_cols (list): A list of column names to group by.
    target_col (str): The column name where missing values should be filled.
    method (str): The method to use for filling missing values, either "mode" or "median".

    Returns:
    pd.DataFrame: The dataframe with missing values filled.
    """

    # Calculate the mode for the target column within each group
    grouped_df = df.groupby(group_cols)[target_col]

    #  Define a function to calculate the mode or return `pd.NA`
    def calculate_value(series):
        if method == "mode":
            func_series = series.mode()
            if not func_series.empty:
                return func_series.iloc[0]
            else:
                return pd.NA
        elif method == "median":
            return series.median()

    # Apply the function to each group
    mode_series = grouped_df.agg(calculate_value)

    # Reset the index to convert the result back into a dataframe
    mode_df = mode_series.reset_index()

    # Rename the column to avoid errors
    mode_df.rename(columns={target_col: 'MostFrequentValue'}, inplace=True)

    # Merge dataframes on custom column
    df = df.merge(mode_df, on=group_cols, how='left')

    # Fill missing values in target feature
    df[target_col] = df[target_col].fillna(df['MostFrequentValue'])

    # Drop temp column from dataframe
    df.drop(columns=['MostFrequentValue'], inplace=True)

    return df


def cm_matrix(cm: np.ndarray, place: int, model_name: str, axes: np.ndarray) -> None:
    """
    Plots a confusion matrix heatmap.

    Parameters:
        cm: Confusion matrix (2D array-like).
        place: Index of the subplot position.
        model_name: Name of the model for the title.
        axes: Array of axes from plt.subplots.
    """
    # Flatten the axes to handle 2D grid
    flat_axes = axes.flatten()

    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".3g", ax=flat_axes[place], cbar=True)
    flat_axes[place].set_title(f"{model_name}")
    flat_axes[place].set_xlabel("Predicted Label")
    flat_axes[place].set_ylabel("True Label")
