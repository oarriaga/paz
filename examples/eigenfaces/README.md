This example requires you to manually download the [FERPlus](https://github.com/microsoft/FERPlus) and [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) csv dataset files.

More explicitly, you must extract the files "fer2013new.csv" and "fer2013.csv" from FERPlus and FER respectively, and put them inside the directory the directory ~/.keras/paz/datasets/FERPlus (if this directory is not there you must create it).

After that you must be able to run:

``python eigenfaces.py``

Alternatively you can extract the "fer2013.csv" and "fer2013new.csv" file and located them in a directory ``my_data`` and explicitly pass the ``args.data_path`` to ``eigenfaces.py`` i.e.

``python eigenfaces.py --data_path my_data``

### **To create database follow the following directory structure:**

```

├── database
│   ├── images
│   │   ├── <person1_name>
│   │   ├── ├── image1.png
│   │   ├── ├── .
│   │   ├── ├── .
│   │   ├── <person2_name>
│   │   ├── ├── image1.png
│   │   ├── ├── .
│   │   ├── ├── .
│   │   ├── .
│   │   ├── .
│   ├── database.npy
├── experiments
│   ├── eigenfaces.npy
│   ├── eigenvalues.npy
│   ├── mean_face.npy
├── database.py
├── demo.py
├── eigenfaces.py
├── pipelines.py
├── processors.py

```