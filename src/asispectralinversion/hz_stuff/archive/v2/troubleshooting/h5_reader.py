import h5py

with h5py.File('/Users/clevenger/Desktop/dasc_test/20230216_100000.h5', 'r') as hdf:
    print("Q data:", hdf['Q'][:])
    print("E0 data:", hdf['E0'][:])
