class Board:
    def __init__(self, grid=None):
        if grid is None:
            self.grid = {'00': 0, '01': 0, '02': 0, '03': 0,
                         '10': 0, '11': 0, '12': 0, '13': 0,
                         '20': 0, '21': 0, '22': 0, '23': 0,
                         '30': 0, '31': 0, '32': 0, '33': 0}
            self.random_spawn()
        else:
            self.grid = grid
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

    def push(self, l):
        c = 0  # current index
        has_merged = False
        while c < 4:
            if self.grid[l[c]] == 0:
                nnz = c
                while nnz < 4 and self.grid[l[nnz]] == 0:
                    nnz += 1  # increment while there are 0's
                if nnz == 4:
                    break  # if nnz is 4 then the entire list after c has 0's => task is finished
                self.grid[l[c]] = self.grid[l[nnz]]
                self.grid[l[nnz]] = 0
                if c > 0 and not has_merged and self.grid[l[c - 1]] == self.grid[l[c]]:
                    self.grid[l[c - 1]] += self.grid[l[c - 1]]
                    self.grid[l[c]] = 0
                self.state_changed = True
                has_merged = False
            if c < 3 and self.grid[l[c]] == self.grid[l[c + 1]]:
                has_merged = True
                self.grid[l[c]] += self.grid[l[c]]
                self.grid[l[c + 1]] = 0
                self.state_changed = True
            c += 1

    def push_all(self, lists):
        for l in lists:
            self.push(l)

    def push_up(self):
        self.push_all([['30', '20', '10', '00'], ['31', '21', '11', '01'], ['32', '22', '12', '02'], ['33', '23', '13', '03']])

    def push_down(self):
        self.push_all([['00', '10', '20', '30'], ['01', '11', '21', '31'], ['02', '12', '22', '32'], ['03', '13', '23', '33']])

    def push_left(self):
        self.push_all([['03', '02', '01', '00'], ['13', '12', '11', '10'], ['23', '22', '21', '20'], ['33', '32', '31', '30']])

    def push_right(self):
        self.push_all([['00', '01', '02', '03'], ['10', '11', '12', '13'], ['20', '21', '22', '23'], ['30', '31', '32', '33']])

    def print_grid(self):
        print(f"{self.grid['00']} {self.grid['01']} {self.grid['02']} {self.grid['03']}")
        print(f"{self.grid['10']} {self.grid['11']} {self.grid['12']} {self.grid['13']}")
        print(f"{self.grid['20']} {self.grid['21']} {self.grid['22']} {self.grid['23']}")
        print(f"{self.grid['30']} {self.grid['31']} {self.grid['32']} {self.grid['33']}")

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
    return {'w': 1, 'a': 3, 's': 0, 'd': 2}.get(k)


def game(delay=0.25, iterations=None, action_func=random_input, show=True, grid=None):
    import time
    import os
    board = Board(grid)
    action_map = {0: board.push_up, 1: board.push_down, 2: board.push_left, 3: board.push_right}
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
        action_map.get(action)()
        if action_func == random_input:
            time.sleep(delay)
        if board.state_changed:
            board.random_spawn()
        if show:
            board.print_grid()
    return counter, board.grid


if __name__ == '__main__':
    import time
    # grid = {'00': 0, '01': 0, '02': 0, '03': 0,
    #         '10': 4, '11': 0, '12': 4, '13': 0,
    #         '20': 8, '21': 4, '22': 2, '23': 0,
    #         '30': 16, '31': 8, '32': 2, '33': 0}
    game(action_func=random_input, show=True, delay=0.25)
    # counters = []
    # states = []
    # start_time = time.time()
    # for i in range(5000):
    #     if not i % 1000:
    #         print(i)
    #     counter, final_state = game(iterations=500, delay=0.0, action_func=random_input, show=False)
    #     counters.append(counter)
    #     states.append(final_state)
    # print(time.time() - start_time)
    # with open('counts.csv', mode='w') as f:
    #     f.write('counts,cell,value\n')
    #     for i in range(len(counters)):
    #         for cell in states[i].keys():
    #             f.write('{},{},{}\n'.format(str(counters[i]), 'c' + cell, str(states[i][cell])))
