from random import randint


# функция вывода игрового поля
def iteration(xo):
    print('  0 1 2')
    for i in range(3):
        print(f'{i}|{xo[i][0]}|{xo[i][1]}|{xo[i][2]}|')


# функция перезапуска/выхода игры res=0 ничья, res=1 выиграл "xo"
def rest_exit(res, xo):
    if res == 0:
        cont = input('НИЧЬЯ, еще раз? y/n:')
    else:
        cont = input(f'Выйграл {xo}, еще раз? y/n:')

    while not (cont in 'YyNn'):
        cont = input('Для продолжения (повторной игры) нажмите y, для выхода n:')

    if cont == 'Y' or cont == 'y':
        return True
    elif cont == 'N' or cont == 'n':
        return False


#  функция проверки выигрышной комбинации
def win(field, xo):
    count = [0, 0]
    for i in range(3):
        if (set(field[i]) == set(xo)) or ({field[0][i], field[1][i], field[2][i]} == set(
                xo)):  # проверка выигрышной комбинации по столбцам и строкам
            return True
        elif field[i][i] == xo:  # счетчик для проверки выигрышной комбинации по диагонали \
            count[0] += 1
            if i == 1:
                count[1] += 1  # костылик для учета центрального элемента при пересечении диагоналей \ и /
        elif field[i][2 - i] == xo:
            count[1] += 1  # счетчик для проверки выигрышной комбинации по диагонали /

    if 3 in count:  # проверка наличия выигрышной комбинации по диагоналям
        return True
    else:
        return False


# функция проверки выйигрышной комбинации для на итерации "xo"
def i_win(field_t, coords_t, xo_t):
    # проверка выигрышной комбинации по оставшимся координатам. значение xo выставляется в каждое поле и проверяется
    for j in range(len(coords_t)):
        field_t[coords_t[j][0]][coords_t[j][1]] = xo_t
        if win(field_t, xo_t):
            field_t[coords_t[j][0]][coords_t[j][1]] = '-'
            return coords_t[j]
        else:
            field_t[coords_t[j][0]][coords_t[j][1]] = '-'


# функция игры крестики-нолики между двумя игроками
def hh_ttt():
    # инициализация начальных условий
    field = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
    xo = 'X'
    iteration(field)
    coord_storage = []
    i = 0
    res_table = {'X': 0, '0': 0}  # таблица результатов

    while 1:
        coord_storage.append(list(
            map(int, input(f'Ход {xo}. Введите координаты через пробел (x,y) ').split())))  # ввод текущих координат

        if len(coord_storage[i]) != 2 or (0 < coord_storage[i][0] > 2) or (
                0 < coord_storage[i][1] > 2):  # проверка допустимости координат
            print('Координаты должны быть введены через пробел в диапазоне от 0 до 2')
            iteration(field)
            coord_storage.pop()
            continue

        elif i > 0 and (
                coord_storage[i] in coord_storage[:-1]):  # проверка что на введеных координатах больше ничего нет
            print(f'На указанной позиции уже находится элемент -> {field[coord_storage[i][1]][coord_storage[i][0]]}')
            iteration(field)
            coord_storage.pop()
            continue

        # заполнение поля значением xo по введенным координатам
        field[coord_storage[i][1]][coord_storage[i][0]] = xo

        # проверка наличия выигрышной комбинации или ньчьей. выполняется на 5-ю итерацию т.к.
        # ранее выигрышных вариантов нет
        if (i > 3 and win(field, xo)) or i == 8:
            if win(field, xo):
                iteration(field)
                print(f'Победили {xo}')
                res_table[xo] += 1
                print(res_table)
                res = 1
            else:
                res = 0

            if not (rest_exit(res, xo)):  # запуск перезапуска игры или выхода
                print('Текущий результат: ',res_table)
                return False
            else:
                i = 0
                coord_storage = []
                field = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
                xo = 'X'
                iteration(field)
                continue

        # переход на следующую итерацию
        xo = '0' if xo == 'X' else 'X'
        i += 1
        iteration(field)


# функция игры крестики-нолики между двумя программами
def mm_ttt():
    # инициализация начальных условий
    field = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
    xo = 'X'
    iteration(field)
    # массив координат. в случае "игры" машина-машина на маленьком поле проще использовать заранее известный набор
    coord_storage = [[0, 0], [0, 1], [0, 2],
                     [1, 0], [1, 1], [1, 2],
                     [2, 0], [2, 1], [2, 2]]
    i = 0
    res_table = {'X': 0, '0': 0}  # таблица результатов
    while 1:

        if i < 3:  # первые три итерации ставятся значения произвольно
            coord_now = coord_storage[randint(0, len(coord_storage) - 1)]

            coord_storage.pop(coord_storage.index(coord_now))
            field[coord_now[0]][coord_now[1]] = xo

        elif i > 2:  # начиная с третьей итерации начинаем проверять возможность выигрыша не текущем и следующем ходе

            xo_n = 'X' if xo == '0' else '0'  # переменная для проверки победы опонента на следующей итерации

            if i_win(field.copy(), coord_storage, xo):  # проверка возможности выиграть на текущей итерации
                coord_now = i_win(field, coord_storage, xo)
            elif i_win(field.copy(), coord_storage, xo_n):  # проверка возможности выиграть опоненту на след. итерации
                coord_now = i_win(field, coord_storage, xo_n)  # "перекрытие" выигрышной позиции
            else:
                # в случае отсутствия выигрышной комбинации брать коррдинаты произвольно
                coord_now = coord_storage[randint(0, len(coord_storage) - 1)]

                # исключение выбранной координаты из обработки и установка xo на игровое поле
            coord_storage.pop(coord_storage.index(coord_now))
            field[coord_now[0]][coord_now[1]] = xo

        # проверка выигрыша или ничьей. проверка выигрыша выполняется с 5-й итерации т.к. до этого выигрышных
        # позиций нет
        if (i > 3 and win(field, xo)) or i == 8:
            iteration(field)
            if win(field, xo):  # если выигрыш
                print(f'Победили {xo}')
                res_table[xo] += 1
                print(res_table)
                res = 1
            else:
                res = 0

            if not (rest_exit(res, xo)):  # запуск перезапуска игры или выхода
                print('Текущий результат: ', res_table)
                return False
            else:
                field = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
                xo = 'X'
                iteration(field)
                coord_storage = [[0, 0], [0, 1], [0, 2],
                                 [1, 0], [1, 1], [1, 2],
                                 [2, 0], [2, 1], [2, 2]]
                i = 0
                continue

        xo = '0' if xo == 'X' else 'X'
        i += 1
        iteration(field)


# функция игры крестики-нолики между человеком и компьютером
def hm_ttt():
    # инициализация начальных условий
    field = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

    hm_iter = input('Введите кто начинает ход h(игрок)/m(компьютер): ')  # выбор, кто начинает
    while not (hm_iter in 'hm'):
        hm_iter = input('Введите кто начинает ход h(игрок)/m(компьютер): ')

    xo = 'X'
    iteration(field)
    # массив координат. в случае "игры" проще использовать заранее известный набор
    coord_storage = [[0, 0], [0, 1], [0, 2],
                     [1, 0], [1, 1], [1, 2],
                     [2, 0], [2, 1], [2, 2]]
    i = 0
    res_table = {'X': 0, '0': 0}  # таблица результатов

    while 1:

        if i < 3:  # первые три итерации ставятся значения произвольно или в рчуную, без проверки выигрышных комбинаций

            if hm_iter == 'h':
                coord_now = list(map(int, input(f'Ход {xo}. Введите координаты через пробел (x,y) ').split()))[::-1]

                if len(coord_now) != 2 or (0 < coord_now[0] > 2) or (
                        0 < coord_now[1] > 2):  # проверка допустимости координат
                    print('Координаты должны быть введены через пробел в диапазоне от 0 до 2')
                    iteration(field)
                    continue
                elif not (coord_now in coord_storage):  # проверка что на введеных координатах больше ничего нет
                    print(f'На указанной позиции уже находится элемент -> {field[coord_now[0]][coord_now[1]]}')
                    iteration(field)
                    continue

            else:
                coord_now = coord_storage[randint(0, len(coord_storage) - 1)]

            coord_storage.pop(coord_storage.index(coord_now))
            field[coord_now[0]][coord_now[1]] = xo

        elif i > 2:  # начиная с третьей итерации компьютер дополнительно проверяет возможность выигрыша игрока и своего

            if hm_iter == 'h':
                coord_now = list(map(int, input(f'Ход {xo}. Введите координаты через пробел (x,y) ').split()))[::-1]

                if len(coord_now) != 2 or (0 < coord_now[0] > 2) or (
                        0 < coord_now[1] > 2):  # проверка допустимости координат
                    print('Координаты должны быть введены через пробел в диапазоне от 0 до 2')
                    iteration(field)
                    continue
                elif not (coord_now in coord_storage):  # проверка что на введеных координатах больше ничего нет
                    print(f'На указанной позиции уже находится элемент -> {field[coord_now[0]][coord_now[1]]}')
                    iteration(field)
                    continue

            else:

                xo_n = 'X' if xo == '0' else '0'  # переменная для проверки победы опонента на следующей итерации

                if i_win(field.copy(), coord_storage, xo):  # проверка возможности выиграть на текущей итерации
                    coord_now = i_win(field, coord_storage, xo)
                elif i_win(field.copy(), coord_storage, xo_n):  # проверка возможности выиграть опоненту на
                    # следующей итерации
                    coord_now = i_win(field, coord_storage, xo_n)  # "перекрытие" выигрышной позиции
                else:
                    # в случае отсутствия выигрышной комбинации брать коррдинаты произвольно
                    coord_now = coord_storage[randint(0, len(coord_storage) - 1)]

                    # исключение выбранной координаты из обработки и установка xo на игровое поле
            coord_storage.pop(coord_storage.index(coord_now))
            field[coord_now[0]][coord_now[1]] = xo

        # проверка выигрыша или ничьей. проверка выигрыша выполняется с 5-й итерации т.к. до этого
        # выигрышных позиций нет
        if (i > 3 and win(field, xo)) or i == 8:
            iteration(field)
            if win(field, xo):  # если выигрыш
                print(f'Победили {xo}')
                res_table[xo] += 1
                print(res_table)
                res = 1
            else:
                res = 0

            if not (rest_exit(res, xo)):  # запуск перезапуска игры или выхода
                print('Текущий результат: ', res_table)
                return False
            else:

                field = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
                xo = 'X'
                iteration(field)
                coord_storage = [[0, 0], [0, 1], [0, 2],
                                 [1, 0], [1, 1], [1, 2],
                                 [2, 0], [2, 1], [2, 2]]
                i = 0
                hm_iter = input('Введите кто начинает ход h(игрок)/m(компьютер): ')
                while not (hm_iter in 'hm'):
                    hm_iter = input('Введите кто начинает ход h(игрок)/m(компьютер): ')
                continue

        xo = '0' if xo == 'X' else 'X'
        hm_iter = 'h' if hm_iter == 'm' else 'm'
        i += 1
        iteration(field)


mode = input('Выберите режим игры:\nДва человека (hh)\nДва компьютера\nЧеловек и компьютер(hm)\n---->')

while 1:
    if mode == 'hh':
        hh_ttt()
        mode = input(
            'Выберите режим игры:\nДва человека (hh)\nДва компьютера\nЧеловек и компьютер(hm)\nили выход (exit)---->')
        if mode == 'exit':
            break
    elif mode == 'mm':
        mm_ttt()
        mode = input(
            'Выберите режим игры:\nДва человека (hh)\nДва компьютера\nЧеловек и компьютер(hm)\nили выход (exit)---->')
        if mode == 'exit':
            break
    elif mode == 'hm':
        hm_ttt()
        mode = input(
            'Выберите режим игры:\nДва человека (hh)\nДва компьютера\nЧеловек и компьютер(hm)\nили выход (exit)---->')
        if mode == 'exit':
            break
    else:
        mode = input(
            'Выберите режим игры:\nДва человека (hh)\nДва компьютера\nЧеловек и компьютер(hm)\nили выход (exit)---->')
        if mode == 'exit':
            break
