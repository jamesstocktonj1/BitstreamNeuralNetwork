


def bv(n):
    return (1 << n)

def take_bit(value, n):
    return ((value & bv(n)) >> n)


class LFSR:

    def __init__(self, starting_state):
        self.reg = starting_state

    def shift(self):
        feedback = (self.reg & 0b00000001) ^ ((self.reg & 0b00000100) >> 2) ^ ((self.reg & 0b00100000) >> 5) ^ ((self.reg & 0b01000000) >> 6)
        self.reg = ((self.reg >> 1) | (feedback << 7)) & 0xFF
    
    def get(self):
        return self.reg

    def put(self):
        print("Value: {:08b}: \t {}".format(self.reg, self.reg))


class LFSR16:
    def __init__(self, starting_state):
        self.reg = starting_state

    def shift(self):
        feedback = take_bit(self.reg, 0) ^ take_bit(self.reg, 3) ^ take_bit(self.reg, 12) ^ take_bit(self.reg, 14) ^ take_bit(self.reg, 15)
        self.reg = ((self.reg >> 1) | (feedback << 15)) & 0xFFFF
    
    def get(self):
        return (self.reg << 8) & 0xff

    def put(self):
        print("Value: {:08b}: \t {}".format(self.reg, self.reg))