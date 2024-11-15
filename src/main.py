from potline import PotLine

if __name__ == '__main__':
    potline = PotLine('config.yaml', 1, True, True)
    potline.run()