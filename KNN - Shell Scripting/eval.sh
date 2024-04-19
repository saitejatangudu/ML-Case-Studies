#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "DATA_FILE is missing"
    exit 1
fi

DATA_FILE="$1"

# Load the embeddings and labels
echo "Loading data..."
PREDICTIONS=$(python -c "import numpy as np; import pandas as pd;
data = pd.DataFrame(np.load('$DATA_FILE',allow_pickle=True));import Myclass; model = Myclass.KNN();fit_data = pd.DataFrame(np.load('data.npy',allow_pickle=True));
model.fit(fit_data);predictions = model.predict(data);result_df =pd.DataFrame(predictions,index=[0]).T;result_df.columns=['score'];print(result_df);result_df.to_csv('predictions.csv', index=True);")


# Load the pre-trained KNN model
echo "Loading KNN model..."
echo "Making predictions..."
cat predictions.csv