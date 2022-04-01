import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# from scipy import optimize


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
    print(f'size of x: {n}', f'size of y: {np.size(y)}')
    assert n == np.size(y), 'x and y must have the same size'

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color='m', marker='o', s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color='g')

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def ols(x, y):
    # assemble matrix A
    A = np.vstack([x, np.ones(len(x))]).T

    # turn y into a column vector
    y = y[:, np.newaxis]
    # Direct least square regression
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), y)
    print(alpha)
    return alpha


def plot_ols(x, y, p):
    # plot the results
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'b.')
    plt.plot(x, function(x, p[0], p[1], p[2], p[3]), 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def function(x, a, b, c, d):
    # return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    print(x)
    array = np.array([])
    for xi in x:
        if xi < a:
            array = np.append(array, b)
            print(array)
        else:
            array = np.append(array, xi * c + d)
    print('array', array)
    return array


def x_as_bool_matrix(x):
    matrix = np.zeros((len(x) + 1, len(x) + 1))
    for i in range(len(x) + 1):
        for xi in x:
            if xi < i:
                matrix[i][xi] = 1
            else:
                matrix[i][xi] = 0
    return matrix


def main():
    # observations / data
    prices_agent = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    )
    prices_comp_reaction = np.array(
        [19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    assert np.size(prices_comp_reaction) == np.size(
        prices_agent
    ), 'x and y must have the same size'
    # b = estimate_coef(prices_agent, prices_comp_reaction)
    # print(
    # 	"Estimated coefficients:\nb_0 = {} \
    # 	\nb_1 = {}".format(
    # 		b[0], b[1]
    # 	)
    # )

    # # plotting regression line
    # plot_regression_line(prices_agent, prices_comp_reaction, b)

    # # OLS
    # alpha = ols(prices_agent, prices_comp_reaction)
    # plot_ols(prices_agent, prices_comp_reaction, alpha)
    # alpha = np.array([11, 19, 1, -1])
    # plot_ols(prices_agent, prices_comp_reaction, alpha)
    # optimal_parameters, cov = optimize.curve_fit(function, prices_agent, prices_comp_reaction)
    # print("cov", cov, "optimal_parameters", optimal_parameters)
    x_matrix = x_as_bool_matrix(prices_agent)
    print(x_matrix)
    result = sm.OLS(prices_comp_reaction, x_matrix).fit()
    # result.model
    print(result.get_prediction(x_as_bool_matrix(np.array([1, 2]))))

    # alpha = np.array([a1, a2])
    # plot_ols(prices_agent, prices_comp_reaction, optimal_parameters)


if __name__ == '__main__':
    main()
