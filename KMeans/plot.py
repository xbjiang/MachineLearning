#!/usr/bin/python
#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt

if __name__ == '__main__':
    fp = open('result.txt', 'r')
    colors = ['ro', 'go', 'bo', 'mo', 'co']
    plt.figure()
    for line in fp:
        a = line.strip('\n').split()
        xi = float(a[0])
        yi = float(a[1])
        kind = int(a[2])
        plt.plot(xi, yi, colors[kind], markersize=5)
    fp.close()
    plt.savefig('result.png')

