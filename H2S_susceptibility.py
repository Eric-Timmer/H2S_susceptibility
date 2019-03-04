import rasterio
from rasterio.sample import sample_gen
import os
import pandas as pd
import pickle

from osgeo import gdal
from pyspatialml import predict
import glob
import geopandas as gpd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


def load_data(shp_path, predictors_path, out_path, load_raw_data=True, pkl_file_loc="loaded_data2.pkl"):
    if load_raw_data is True:
        xy, y, z_midpoints = get_shp_file_data(shp_path, field="H2S", depth_field="Z_midpoint")
        X = pd.DataFrame(xy, columns=["X", "Y"])
        X = load_raster_training_data(predictors_path, X, xy)
        X = X.drop(columns=["X", "Y"])
        X.to_csv(out_path)
        pickle.dump([X, y, xy], open(pkl_file_loc, "wb"))
    else:
        X, y, xy = pickle.load(open(pkl_file_loc, "rb"))

    return X, y, xy


def load_raster_training_data(directory, X, xy):
    print("Loading .tif training dataset")
    files = glob.glob(directory + '\*.tif')
    df = pd.DataFrame(xy, columns=["X", "Y"])
    for i in files:
        with rasterio.open(i) as src:
            training_data = np.vstack([s for s in sample_gen(src, xy)])
            training_data = np.ma.masked_equal(training_data, src.nodatavals)
            if isinstance(training_data.mask, np.bool_):
                mask_arr = np.empty(training_data.shape, dtype='bool')
                mask_arr[:] = False
                training_data.mask = mask_arr

            basename = os.path.basename(i)
            df[basename] = training_data

    # merge the raster dataset with the .shp dataset
    X = pd.merge(left=X, right=df, on=["X", "Y"])
    pd.DataFrame.to_csv(X, "out.csv")
    return X


def load_shp_training_data(directory, dataframe, xy, z_midpoints, gas_path, lith_path, buffer=5000):
    files = glob.glob(directory + '\*.shp')
    print("Loading .shp training dataset")
    total_count = 0
    of_data = (len(files) - 1) * 3 * len(xy)
    for i in files:
        basename = os.path.basename(i)

        # defined which columns to extract data from .shp files.
        if basename == lith_path:

            extract = ["Li_mgl", "Li_bd1", "RATION_87Sr"]
        elif basename ==gas_path:
            extract = ["H2S", "C1_C2", "Midpoint_Z"]
        else:
            continue
        training_data = gpd.read_file(i, encoding="utf-8")
        shp_x = training_data.bounds["minx"]
        shp_y = training_data.bounds["miny"]

        count = 0
        for x, y in xy:
            in_buffer = list()
            for x_s, y_s in zip(shp_x, shp_y):
                distance = ((x - x_s)**2 + (y - y_s)**2)**.5
                if distance <= buffer:
                    in_buffer.append([x_s, y_s])

            # find the data within the buffer
            for j in extract:
                data = list()
                for x_s, y_s in in_buffer:

                    row = pd.DataFrame(training_data.cx[x_s, y_s])
                    index = list(row.keys())
                    # I have to do this silly work around due to a stupid encoding error
                    for idx, k in enumerate(index):
                        if k.encode("utf-8") == j.encode("utf-8"):
                            row_values = row.values[0]
                            data.append(row_values[idx])
                            break

                if len(data) == 0:
                    data = np.NaN
                else:
                    if j == "H2S":
                        data = max(data)  # if there is any H2S in the vicinity, pick 1
                    else:
                        data = np.average(data)
                        if j == "Midpoint_Z":
                            z_distance = z_midpoints[count]
                            data = z_distance - data
                total_count += 1
                percent_done = total_count / float(of_data) * 100
                print("\r%i %% of .shp data loaded, working on %s" % (int(percent_done), basename), end="", flush=True)
                dataframe.loc[count, basename+"_"+j] = data
            count += 1
    dataframe.dropna(axis=0, thresh=3, how="all", inplace=True)
    dataframe.dropna(axis=1, how="all", inplace=True)
    dataframe.to_csv("out_shp.csv")
    return dataframe


def get_shp_file_data(shp_file_location, field, depth_field):
    print("Loading .shp label dataset")
    shp_file = gpd.read_file(shp_file_location)
    xy = shp_file.bounds.iloc[:, 2:].as_matrix()
    y = shp_file[field].values
    z_midpoints = shp_file[depth_field].values

    return xy, y, z_midpoints


def workflow(predictor_path, X, y, xy):
    predictor_files = X.columns

    split_data = False
    if split_data == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=.25,
                                                            random_state=1,
                                                            stratify=None,
                                                            shuffle=True)
        # define the classifier with standardization of the input features in a pipeline

    # remove mean and standard deviation
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X = preprocessing.minmax_scale(X)

    # # convert back to dataframe for clearer results
    X = pd.DataFrame(X, columns=predictor_files)

    try:
        X.to_csv("standard_out.csv")
        print("Standardized data exported")
    except PermissionError:
        print("Standardized spreadsheet not exported.")


    logit_model = sm.Logit(y, X, missing="drop")
    print(logit_model)
    results = logit_model.fit_regularized()
    print(results.summary())


    lr = Pipeline([('classifier', LogisticRegressionCV(n_jobs=1))])
    lr.fit(X, y)

    # export data...
    predictors = "out.vrt"
    predictor_files = glob.glob(predictor_path + '\*.tif')

    temp = list()
    for i in predictor_files:
        if os.path.basename(i) not in to_drop:
            temp.append(i)

    outds = gdal.BuildVRT(
        destName=predictors,
        srcDSOrSrcDSTab=temp,
        separate=True,
        resolution='highest',
        resampleAlg='bilinear')
    outds.FlushCache()
    outfile = 'prediction.tif'
    predict(estimator=lr, raster=predictors, file_path=outfile, predict_type='prob', indexes=1)


def main():
    print("Loading data...")
    shp_path, predictors_path, out_path, predictor_path = "", "", "", ""
    X, y, xy = load_data(shp_path, predictors_path, out_path, load_raw_data=True)
    print("DATA LOADED")
    workflow(predictor_path, X, y, xy)


if __name__ == "__main__":
    main()


