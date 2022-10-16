import os
import numpy as np


ROOT = 'D:\\SBandDataAll\\SBandDataUnzip'

rainy_days = []
rainy_days.append(['Year', 'Month', 'Day'])
for dir_ in sorted(os.listdir(ROOT)):
    year, month, day = dir_[:4], dir_[4:6], dir_[6:8]
    rainy_days.append([year, month, day])

rainy_days = np.array(rainy_days)
np.savetxt('rainy_days.csv', rainy_days, fmt='%s', delimiter=',')



