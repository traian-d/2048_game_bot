class Board:

    def __init__(self):
        self.grid = {'00': 0, '01': 0, '02': 0, '03': 0,
                     '10': 0, '11': 0, '12': 0, '13': 0,
                     '20': 0, '21': 0, '22': 0, '23': 0,
                     '30': 0, '31': 0, '32': 0, '33': 0}
        self.random_spawn()

    def random_spawn(self):
        import random
        zeros = [el for el in self.grid if not self.grid[el]]
        position = random.randint(0, len(zeros) - 1)
        self.grid[zeros[position]] = 2

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

        self.grid[c1] = collapsed_list[3]
        self.grid[c2] = collapsed_list[2]
        self.grid[c3] = collapsed_list[1]
        self.grid[c4] = collapsed_list[0]

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


if __name__ == '__main__':
    import os
    board = Board()
    board.print_grid()
    k = input()
    while k != 'c':
        os.system('clear')
        if k == 'w':
            board.push_up()
        elif k == 's':
            board.push_down()
        elif k == 'a':
            board.push_left()
        elif k == 'd':
            board.push_right()
        board.random_spawn()
        board.print_grid()
        k = input()
