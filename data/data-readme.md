# Readme for sleep_dataset

Every folder contains the data of one patient. For every patient
there are segments each 32 seconds long saved as .npy files of shape
(segment_length * new_fs, ) (parameters from convert_data.py).

The naming convention for the folders is 'dataset-id-gender'

- dataset: Which dataset was the patient from. "mesa" or "shhs"
- id: Internal dataset id of the patient
- gender: Gender of the patient. "F" or "M"

```
sleep_dataset
└───mesa-000001-F
│   │   00.npy
│   │   01.npy
│   │   ...
│   │   19.npy
│   
└───shhs-200080-M
│   │   00.npy
│   │   01.npy
│   │   ...
│   │   19.npy
│ 
└───...
│ 
...
```