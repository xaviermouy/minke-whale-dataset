from ecosound.core.measurement import Measurement
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('TkAgg')

in_file = r'D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc'

meas = Measurement()
meas.from_netcdf(in_file)

meas.filter('confidence>=2', inplace=True)

ax = meas.data.snr.hist(bins=30, facecolor='lightgray',edgecolor = "black")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("# minke whale calls")
plt.show()

print('done')