from LFSR import LFSR
import matplotlib.pyplot as plt


def main():
    sr = LFSR(0b11010)
    sr.put()

    val = 184
    total = 0

    for i in range(255):
        sr.shift()
        sr.put()
        if sr.get() < val:
            total += 1

    print("Value: {:.04f}".format((val-1) / 255))
    print("BS Value: {:.04f}".format(total / 255))


if __name__ == "__main__":
    main()