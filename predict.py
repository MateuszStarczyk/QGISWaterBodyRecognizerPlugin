import os
import tempfile

import numpy as np
from osgeo import (gdal)
from qgis._core import QgsMessageLog
from tifffile import tifffile

from .deeplabv3plus import Deeplabv3
from .progressBar import progressBar

MODEL_NDWI = 'model-ndwi.hdf5'
MODEL_ANDWI = 'model-andwi.hdf5'


def get_model(algorithm: str = 'deeplabv3plus', input_size: int = 114, num_classes: int = 1, channels: int = 4):
    if algorithm == 'deeplabv3plus':
        model, model_name = (
            Deeplabv3(weights=None, input_shape=(input_size, input_size, channels),
                      classes=num_classes), "deeplabv3plus")
    else:
        raise Exception('{} is an invalid algorithm'.format(algorithm))

    return model, model_name


def predict(self, inRaster, index, bands):
    gdal.UseExceptions()

    model, model_name = get_model()

    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              MODEL_NDWI if index == 'ndwi' else MODEL_ANDWI)
    if not os.path.isfile(model_path):
        s = 'Model file not found. Check plugin directory for file {}' \
            .format(MODEL_NDWI if index == 'ndwi' else MODEL_ANDWI)
        pushFeedback(s, 'gui')
        return s

    model.load_weights(model_path)

    # save_folder = os.path.join('images', os.path.splitext(weight_file)[0])

    outRaster = os.path.join(tempfile.mkdtemp(), 'temp.tif')

    predict_image(self, inRaster=inRaster, outRaster=outRaster, model=model, index=index, bands=bands)


def predict_image(self, inRaster, outRaster, model, bands, feedback="gui", index='ndwi', size=114, batch=15):
    """!@brief The function classify the whole raster image, using per block image analysis.
    The classifier is given in classifier and options in kwargs
        Input :
            inRaster : Filtered image name ('sample_filtered.tif',str)
            outRaster :Raster image name ('outputraster.tif',str)
            model : model file got from precedent step ('model', str)
            inMask : mask to
            confidenceMap :  map of confidence per pixel
            NODATA : Default set to 0 (int)
            SCALE : Default set to None
            classifier = Default 'GMM'
        Output :
            nothing but save a raster image and a confidence map if asked
    """
    prediction_cutoff = 0.5

    if feedback == 'gui':
        progress = progressBar(self, inMaxStep=100)
    try:
        # Open Raster and get additionnal information
        inRaster = inRaster.currentLayer()
        inRaster = inRaster.dataProvider().dataSourceUri()

        # get raster proj
        raster = gdal.Open(inRaster, gdal.GA_ReadOnly)
        if raster is None:
            # fix_print_with_import
            print('Impossible to open ' + inRaster)
            exit()

        # Get the geoinformation
        GeoTransform = raster.GetGeoTransform()
        Projection = raster.GetProjection()

        raw_data = tifffile.imread(inRaster).transpose([1, 2, 0])
        image_data = scale_image_percentile(raw_data)

        x, new_size, splits, img_width, img_height = get_patches(image=image_data, network_size=114, index=index,
                                                                 bands=bands)

        if feedback == 'gui':
            progress.addStep(25)
        cutoff_array = np.full((len(x), size, size, 1), fill_value=prediction_cutoff)

        y_result = model.predict(x, batch_size=batch, verbose=1)
        y_result = np.append(cutoff_array, y_result, axis=3)

        if feedback == 'gui':
            progress.addStep(25)
        out = np.zeros((new_size, new_size, 2))

        for row in range(splits):
            for col in range(splits):
                out[size * row:size * (row + 1), size * col:size * (col + 1), :] = y_result[
                                                                                   row * splits + col,
                                                                                   :, :, :]

        if feedback == 'gui':
            progress.addStep(25)
        result = np.argmax(np.squeeze(out), axis=-1).astype(np.ubyte)
        result = result[:img_width, :img_height]

        # Initialize the output
        if not os.path.exists(os.path.dirname(outRaster)):
            os.makedirs(os.path.dirname(outRaster))

        tifffile.imwrite(outRaster, result)
        dst_ds = gdal.Open(inRaster, gdal.GA_Update)
        dst_ds.SetGeoTransform(GeoTransform)
        dst_ds.SetProjection(Projection)

        # Clean/Close variables
        raster = None
        dst_ds = None

        if feedback == 'gui':
            progress.addStep(25)
        self.iface.addRasterLayer(outRaster)
    finally:
        if feedback == 'gui':
            progress.reset()


def pushFeedback(message, feedback=None):
    isNum = isinstance(message, (float, int))

    if feedback and feedback is not True:
        if feedback == 'gui':
            if not isNum:
                QgsMessageLog.logMessage(message=str(message))
        else:
            if isNum:
                feedback.setProgress(message)
            else:
                feedback.setProgressText(message)
    else:
        if not isNum:
            print(str(message))


def scale_image_percentile(matrix, lower_percentile=1, higher_percentile=99):
    """
    Remove outliers from data
    """
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    mins = np.percentile(matrix, lower_percentile, axis=0)
    maxs = np.percentile(matrix, higher_percentile, axis=0) - mins

    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def get_patches(image, network_size, bands, index='ndwi'):
    x_train_multi = image
    x_train = np.concatenate((np.atleast_3d(x_train_multi[..., bands[0]]),
                              np.atleast_3d(x_train_multi[..., bands[1]]),
                              np.atleast_3d(x_train_multi[..., bands[2]])), axis=2)
    x_index = np.empty(shape=(x_train_multi.shape[0], x_train_multi.shape[1], 1))
    if index == "ndwi":
        x_index = np.true_divide(np.subtract(x_train_multi[..., bands[1]],
                                             x_train_multi[..., bands[3]]),
                                 np.add(x_train_multi[..., bands[1]],
                                        x_train_multi[..., bands[3]]) + 1e-5)
    elif index == "andwi":
        x_rgb_sum = np.add(np.add(x_train_multi[..., 4], x_train_multi[..., 2]), x_train_multi[..., 1])
        x_index = np.true_divide(
            np.subtract(
                np.subtract(
                    np.subtract(
                        x_rgb_sum,
                        x_train_multi[..., bands[3]]
                    ),
                    x_train_multi[..., bands[4]]
                ),
                x_train_multi[..., bands[5]]
            ),
            np.add(
                np.add(
                    np.add(
                        x_rgb_sum,
                        x_train_multi[..., bands[3]]
                    ),
                    x_train_multi[..., bands[4]]
                ),
                x_train_multi[..., bands[5]]
            ) + 1e-5
        )
    x_train = np.concatenate((x_train, np.atleast_3d(x_index)), axis=2)

    image_width = x_train.shape[0]
    image_height = x_train.shape[1]

    # Integer ceil division
    splits = max(-(-image_width // network_size),
                 -(-image_height // network_size))

    new_size = splits * network_size

    x_train_pad = np.zeros((new_size, new_size, 4))
    x_train_pad[:x_train.shape[0], :x_train.shape[1], :] = x_train

    x = np.empty((splits * splits, network_size, network_size, 4))

    for col in range(splits):
        for row in range(splits):
            x_start = network_size * col
            y_start = network_size * row

            x[col * splits + row] = x_train_pad[x_start:x_start + network_size, y_start:y_start + network_size]

    x = np.array(x)

    return x, new_size, splits, image_width, image_height
