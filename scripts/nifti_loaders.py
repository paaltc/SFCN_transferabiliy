import nibabel as nib
import numpy as np

def load_nifti_image(file_path):
    """Load a NIfTI image and return its data as a numpy array."""
    img = nib.load(file_path)
    img = img.get_fdata()
    img = np.expand_dims(img, 0)
    return img

def load_nifti_wrapper(img, label):
    img = tf.py_function(load_nifti_image, [img], tf.float32)
    return img, label

def create_generator(df, img_column, target_column, batch_size=3):
    dataset = tf.data.Dataset.from_tensor_slices((df[img_column].values, df[target_column].values))
    dataset = dataset.map(load_nifti_wrapper)
    dataset = dataset.batch(batch_size)
    return dataset
