

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