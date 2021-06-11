import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

def load(
    file_name: str = "kc_house_data.csv"
) -> pd.core.frame.DataFrame:
    """ Задание - 1.Загрузка данных """
    data = pd.read_csv(file_name)


    pd.options.display.max_columns = 25

    first_ten = data.head(10)
    last_ten = data.tail(10)
    # Выводим первые 10 элементов
    print(first_ten)
    print(last_ten)

    # Информация о количестве строк и стобцов
    shaped_data = data.shape
    print(f"Число строк: {shaped_data[0]}, число стобцов: {shaped_data[1]}")

    # Вызов функции .info()
    data.info()

    return data


def eazy_statistics(data: pd.core.frame.DataFrame) -> None:
    """ Задание - 2.Простая статистика """

    # Count number of not NaN's in every column
    print(data.count())

    # Count number of unique values in every column
    print(data.nunique())

    # Display basic data statistics
    print(data.describe())

    return None


def complex_statistics(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame.corr:
    """ Задание - 3.Не такая уж простая статистика """

    # Create a correlation matrix
    correlation = data.corr()
    print(correlation)

    # What is the shape of the correlation matrix?
    correlation_shape = correlation.shape
    print(
        f"Число строк: {correlation_shape[0]}, число столбцов: {correlation_shape[1]}"
    )

    # Удаляем колонку, которая сильно коррелирует с колонкой цена (мы ведь будем предсказывать цену)
    highest_corr = correlation.drop("price", axis=0)["price"].idxmax()

    return correlation


def delete_data(data: pd.core.frame.DataFrame) -> None:
    """ Задание - 4.Очистка данных """

    # drop column zipcode
    data.drop("zipcode", axis=1, inplace=True)

    # Set data index
    data.set_index("id", inplace=True)

    print(data.head(10))
    print(f"Число строк: {data.shape[0]}, число стобцов: {data.shape[1]}")

    return None


def basic_date_convertion(data: pd.core.frame.DataFrame) -> None:
    """ Задание - 5.Базовая обработка дат """

    # Convert date to datetime format
    data.date = pd.to_datetime(data.date)

    # Extract and store year and etc
    data["year"] = data.date.dt.year
    data["month"] = data.date.dt.month
    data["day"] = data.date.dt.day
    data["weekday"] = data.date.dt.weekday

    # Drop column date
    data.drop("date", axis=1, inplace=True)

    print(data.head(10))

    print(f"Число строк: {data.shape[0]}, число стобцов: {data.shape[1]}")

    return None


def build_histogram(data: pd.core.frame.DataFrame) -> None:
    """ Задание - 6.Постройте гистограмму """

    plt.hist(data.price, bins=1000)

    plt.title("Распределение цен")
    plt.xlabel("Цена")
    plt.ylabel("Число квартир")

    plt.show()

    return None


def build_point_diagram(correlation: pd.core.frame.DataFrame.corr) -> None:
    """ Задание - 7.Построение точечной диаграммы """

    print(correlation.price.sort_values())
    x = correlation.price
    y = correlation.grade
    colors = np.random.rand(20)
    plt.rcParams["axes.facecolor"] = "#E9EDF5"

    plt.scatter(x, y, c=colors, alpha=0.5, marker="2")
    plt.title("Распределение цен")
    plt.xlabel("Цена")
    plt.ylabel("Оценка")
    plt.show()

    x = correlation.price
    y = correlation.condition
    colors = np.random.rand(20)
    plt.rcParams["axes.facecolor"] = "#E9EDF5"

    plt.scatter(x, y, c=colors, alpha=0.5, marker="1")
    plt.title("Распределение цен")
    plt.xlabel("Цена")
    plt.ylabel("Условие")
    plt.show()


    return None


def data_delete_additional(data: pd.core.frame.DataFrame) -> None:
    """ Задание - 8.Дополнительная очистка данных """

    data.price.quantile(0.99)

    # print number of rows in the data
    data.shape

    # trim the data to price 99th quantile
    data = data[data.price < data.price.quantile(0.99)]

    # Compute number of rows and columns in your data after these operations
    print(f"Число строк: {data.shape[0]}, число стобцов: {data.shape[1]}")

    return None


def build_others_graphics(
    correlation: pd.core.frame.DataFrame.corr, data: pd.core.frame.DataFrame
) -> None:
    """ Задание - 9.Построить дополнительные графики """

    sns.barplot(x="price", y="view", data=data)
    plt.show()

    sns.jointplot(x="price", y="view", data=data, kind="kde")
    plt.show()

    sns.boxplot(x="weekday", y="price", data=data, hue="condition")
    plt.show()

    return None


def prepare_data_and_train(data: pd.core.frame.DataFrame) -> None:
    """ Задание - 10.Подготовьте данные из обучающих моделей машинного обучения """

    y = data["price"]
    x = data.drop("price", axis=1)


    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(
        f"\nX_train  -  число строк: {X_train.shape[0]}, число стобцов: {X_train.shape[1]}"
    )
    print(
        f"X_test  - число строк: {X_test.shape[0]}, число стобцов: {X_test.shape[1]}"
    )

    model_linear = LinearRegression()

    model_linear.fit(X_train, y_train)

    y_pred_train = model_linear.predict(X_train)
    y_pred_test = model_linear.predict(X_test)

    mse_lr_train = mean_squared_error(y_train, y_pred_train)
    print(f"Linear Regression MSE on train data: {mse_lr_train}")

    mse_lr_test = mean_squared_error(y_test, y_pred_test)
    print(f"Linear Regression MSE on test data: {mse_lr_test}")

    model_ridge = Ridge()
    model_ridge.fit(X_train, y_train)

    y_pred_train2 = model_ridge.predict(X_train)
    y_pred_test2 = model_ridge.predict(X_test)

    mse_rig_train = mean_squared_error(y_train, y_pred_train2)
    print(f"Ridge MSE on train data: {mse_rig_train}")

    mse_rig_test = mean_squared_error(y_test, y_pred_test2)
    print(f"Ridge MSE on test data: {mse_rig_test}")

    return None


def main():
    data = load()
    eazy_statistics(data)
    correlation = complex_statistics(data)
    delete_data(data)
    basic_date_convertion(data)
    build_histogram(data)
    build_point_diagram(correlation)
    data_delete_additional(data)
    build_others_graphics(correlation, data)
    prepare_data_and_train(data)


if __name__ == "__main__":
    main()
