import os
from sklearn.preprocessing import StandardScaler
from Project import build_features
from tensorflow.keras.models import load_model
import joblib
import numpy as np

if __name__ == '__main__':
    local_csv = "data/wig20_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"
    
    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    
    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)[-30:]
    
    print(features)
    
    features = features.drop(columns=['Date','y_t'])
    
    model = load_model(f"runs/{market_name}_run/model.h5", compile=False)
    scaler_X = joblib.load(f"runs/{market_name}_run/scaler_X")
    scaler_y = joblib.load(f"runs/{market_name}_run/scaler_y")
    
    X = scaler_X.transform(features).reshape(1, 30, features.shape[1])
    y_pred = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()[0]
    
    print(f"Predicted next close: {y_pred}")