import pulp
import numpy as np
import itertools as it
from dataclasses import dataclass
from collections import Counter


@dataclass
class Data:
    '''Исходные данные для модели.'''
    # Максимальная ширина исходных рулонов.
    RAWS_WIDTH = 10
    # Количество заказов в штуках каждого типа рулонов.
    orders = [9, 79, 90, 27]
    # Ширина заказов каждого типа. Соответствие по индексу в списке.
    # Так необходимо 9 рулонов ширины 3, 79 рулонов ширины 5 и т.д.
    order_sizes = [3, 5, 6, 9]
    # Список паттернов, по каким может быть нарезан исходный рулон.
    # Так, например, паттерн [1, 0, 1, 0] означает, что из исходного рулона ширины 10,
    # может быть вырезан рулон ширины 3 и рулон ширины 6.
    patterns = [[0, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 2, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [3, 0, 0, 0]]
    # Общее количество заказов.
    RAWS_NUMBER = sum(orders)


def base_model(data: Data, relaxation=True):
    '''Функция создает и решает модель целочисленную модель, на выходе выдает две величны
    значение целевой функции и словарь с использованными паттернами.
    '''
    if relaxation:
        vars_type = pulp.LpContinuous
    else:
        vars_type = pulp.LpInteger


    #  Создание модели.
    model = pulp.LpProblem('Kantorovich model', pulp.LpMinimize)

    # Создание необходимых индексов для переменных, для каждого исходного рулона.
    cuts_ids = range(data.RAWS_NUMBER)
    order_types_ids = range(len(data.orders))
    items_ids = list(it.product(order_types_ids, cuts_ids))

    # Создание переменных
    cuts = pulp.LpVariable.dicts("raw_cut", cuts_ids, lowBound=0, upBound=1, cat=vars_type)
    items = pulp.LpVariable.dicts("item", items_ids, lowBound=0, cat=vars_type)

    for t in order_types_ids:
        model += pulp.LpConstraint(pulp.lpSum(items[t, c] for c in cuts_ids) >= data.orders[t],
                                   name="min_demand_{}".format(t), sense=pulp.LpConstraintGE)

    for c in cuts_ids:
        model += pulp.LpConstraint(pulp.lpSum(data.order_sizes[t] * items[t, c]
                                              for t in order_types_ids) <= data.RAWS_WIDTH * cuts[c],
                                   name="max_width_{}".format(c), sense=pulp.LpConstraintLE)

    model += pulp.lpSum([cuts[c] for c in cuts_ids])
    model.solve()


    used_patterns = []
    for c in cuts_ids:
        if cuts[c] >= 0.001:
            used_patterns.append(tuple([int(items[t, c].varValue) for t in order_types_ids]))
    return model.objective.value(), dict(Counter(used_patterns))


def column_model(data: Data, relaxation=True):
    '''Функция создает и решает модель целочисленную модель и выбирает , на выходе выдает две величны
    значение целевой функции и словарь с использованными паттернами.'''
    if relaxation:
        vars_type = pulp.LpContinuous
    else:
        vars_type = pulp.LpInteger


    #  Создание модели.
    model = pulp.LpProblem('Gilmore-Gomory model', pulp.LpMinimize)

    # Создание необходимых индексов для переменных.
    patterns_ids = range(len(data.patterns))
    order_types = range(len(data.orders))

    # Создание переменных модели.
    pattern_vars = pulp.LpVariable.dicts('patterns', patterns_ids, lowBound=0, cat=vars_type)

    # Создание ограничения по обязательности выполнения всех заказов
    for t in order_types:
        model += pulp.LpConstraint(
            pulp.lpSum(np.multiply(pattern_vars[p], data.patterns[p])[t]
                       for p in patterns_ids) >= data.orders[t],
            sense=pulp.LpConstraintGE, name="min_demand_{}".format(t))

    # Добавление целевой функции в модель, необходимой для минимизации количества используемых паттернов.
    model += (pulp.lpSum(pattern_vars[p] for p in patterns_ids))
    model.solve()
    used_patterns = {}
    for p in patterns_ids:
        if pattern_vars[p].varValue >= 0.001:
            used_patterns[tuple(data.patterns[p])] = pattern_vars[p].varValue

    return model.objective.value(), used_patterns


if __name__ == "__main__":
    input_data = Data()
    # Расчет базового целочисленного решения.
    base_obj, base_patterns = base_model(input_data, relaxation=False)
    col_obj, col_patterns = column_model(input_data, relaxation=False)
    # Расчет релаксированного решения, где переменные не обязательно целые числа.
    r_base_obj, r_base_patterns = base_model(input_data, relaxation=True)
    r_col_obj, r_col_patterns = column_model(input_data, relaxation=True)

    print("Значение целевой функции Канторович: {}".format(base_obj))
    print("Значение целевой функции Гилмор-Гомори: {}".format(col_obj))
    print("Значение целевой функции Канторович, релаксация: {}".format(r_base_obj))
    print("Значение целевой функции Гилмор-Гомори, релаксация: {}".format(r_col_obj))
