## What is hdf5 file
HDF5 file is a structure that can store dataset and group.You can think that datasets is a file, and group is a folder which to store file and another group.Group's structure looks like a python dictionary and datasets looks like a python numpy data_group.

## Read
```
import h5py
f = h5py.File('test.hdf5','r')
```

We can use keys() to check the label content of the hdf5 file
```
f.keys()
```

We also can create a subgroup in the file
```
group = f.create_group("sub_group")
```

And use iteration or visit(), visititerms() to show all the file. visit() need a fucntion to be its parameter
```
for filename in f:
  print(filename)
```

```
def printFileName(name):
  print(name)

f.visit(printFileName)
```

## Reference
[Reference1](https://blog.csdn.net/yudf2010/article/details/50353292)
[Reference2](https://docs.h5py.org/en/stable/high/group.html)
