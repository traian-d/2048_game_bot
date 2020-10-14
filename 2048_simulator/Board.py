class Board:
    def __init__(self):
        self.grid = {'00': 0, '01': 0, '02': 0, '03': 0,
                     '10': 0, '11': 0, '12': 0, '13': 0,
                     '20': 0, '21': 0, '22': 0, '23': 0,
                     '30': 0, '31': 0, '32': 0, '33': 0}
        self.random_spawn()
        self.finished = False
        self.state_changed = False

    def random_spawn(self, prob_of_4=0.1):
        import numpy as np
        zeros = [el for el in self.grid if not self.grid[el]]
        empty_cells = len(zeros)
        if empty_cells == 0:
            self.finished = True
            return
        position = np.random.randint(0, empty_cells - 1) if empty_cells > 1 else 0
        is_4 = np.random.uniform(0, 1)
        new_spawn = 4 if is_4 < prob_of_4 else 2
        self.grid[zeros[position]] = new_spawn

    def push(self, c1, c2, c3, c4):
        reversed_list = []
        if self.grid[c4]:
            reversed_list.append(self.grid[c4])
        if self.grid[c3]:
            reversed_list.append(self.grid[c3])
        if self.grid[c2]:
            reversed_list.append(self.grid[c2])
        if self.grid[c1]:
            reversed_list.append(self.grid[c1])

        collapsed_list = self.sum_up(reversed_list)
        collapsed_list.extend([0] * (4 - len(collapsed_list)))

        if self.has_changed(c1, c2, c3, c4, collapsed_list):
            self.state_changed = True

        self.grid[c1] = collapsed_list[3]
        self.grid[c2] = collapsed_list[2]
        self.grid[c3] = collapsed_list[1]
        self.grid[c4] = collapsed_list[0]

    def has_changed(self, c1, c2, c3, c4, new_vals):
        return self.grid[c1] != new_vals[3] or self.grid[c2] != new_vals[2] or \
                self.grid[c3] != new_vals[1] or self.grid[c4] != new_vals[0]

    def sum_up(self, l):
        output = []
        i = 0
        length = len(l)
        while i < length:
            if i < length - 1 and l[i] == l[i+1]:
                output.append(l[i] + l[i+1])
                i += 1
            else:
                output.append(l[i])
            i += 1
        return output

    def push_up(self):
        self.push('30', '20', '10', '00')
        self.push('31', '21', '11', '01')
        self.push('32', '22', '12', '02')
        self.push('33', '23', '13', '03')

    def push_down(self):
        self.push('00', '10', '20', '30')
        self.push('01', '11', '21', '31')
        self.push('02', '12', '22', '32')
        self.push('03', '13', '23', '33')

    def push_left(self):
        self.push('03', '02', '01', '00')
        self.push('13', '12', '11', '10')
        self.push('23', '22', '21', '20')
        self.push('33', '32', '31', '30')

    def push_right(self):
        self.push('00', '01', '02', '03')
        self.push('10', '11', '12', '13')
        self.push('20', '21', '22', '23')
        self.push('30', '31', '32', '33')

    def print_grid(self):
        print(str(self.grid['00']) + ' ' + str(self.grid['01']) + ' ' + str(self.grid['02']) + ' ' + str(self.grid['03']))
        print(str(self.grid['10']) + ' ' + str(self.grid['11']) + ' ' + str(self.grid['12']) + ' ' + str(self.grid['13']))
        print(str(self.grid['20']) + ' ' + str(self.grid['21']) + ' ' + str(self.grid['22']) + ' ' + str(self.grid['23']))
        print(str(self.grid['30']) + ' ' + str(self.grid['31']) + ' ' + str(self.grid['32']) + ' ' + str(self.grid['33']))
        print('')

    def reached_2048(self):
        maxim = 0
        for cell in self.grid.keys():
            if self.grid[cell] > maxim:
                maxim = self.grid[cell]
        return maxim == 2048

    def is_finished(self):
        return self.finished


def random_input():
    import numpy as np
    return np.random.randint(0, 4)


def keyboard_input():
    k = input()
    if k == 'w':
        return 0
    elif k == 's':
        return 1
    elif k == 'a':
        return 2
    elif k == 'd':
        return 3


def game(delay=0.25, iterations=None, action_func=random_input, show=True):
    import time
    import os
    board = Board()
    if show:
        board.print_grid()
    counter = 0
    while not board.is_finished():
        if show:
            os.system('clear')
        if iterations and counter >= iterations:
            break
        counter += 1
        action = action_func()
        board.state_changed = False
        if action == 0:
            board.push_up()
        if action == 1:
            board.push_down()
        if action == 2:
            board.push_left()
        if action == 3:
            board.push_right()
        if action_func == random_input:
            time.sleep(delay)
        if board.state_changed:
            board.random_spawn()
        if show:
            board.print_grid()
    return counter, board.grid


if __name__ == '__main__':
    game(action_func=keyboard_input, show=True)
    # counters = []
    # states = []
    # for i in range(50000):
    #     counter, final_state = game(iterations=500, delay=0.0, action_func=random_input, show=False)
    #     counters.append(counter)
    #     states.append(final_state)
    # with open('counts.csv', mode='w') as f:
    #     f.write('counts,cell,value\n')
    #     for i in range(len(counters)):
    #         for cell in states[i].keys():
    #             f.write('{},{},{}\n'.format(str(counters[i]), 'c' + cell, str(states[i][cell])))
