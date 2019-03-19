# Stock-Market-Prediction

Program Language:

    Python                      3.6

Libraries:

    matplotlib
    pandas
    numpy
    jupyter notebook            1.0.0
    TensorFlow                  1.12.0
    scikit-learn                0.20.3

testData: data

How to install environment(in Linux)?

    1. Install Anaconda3:bash ~/Downloads Anaconda3-5.3.0-Linux-x86_64.sh

    2. Create virtual environment: conda create -n env_name python=3.6

    3. Start virtual environment: conda activate env_name

    4. Install libraries: conda install lib_name

How to run our code after installation?

    1. Start virtual environment

    2. Type jupyter notebook, then you will get into jupyter notebook page on a browser

    3. Train LSTM: Open codes/LSTM_train and run. You can choose whatever stocks listed in the code. After training, it will store model in file called "LSTM". So if you want to train from the begining. Remember to remove LSTM.

    4. Run LSTM: Open codes/LSTM_run and run. It will read file "LSTM" to get a trained model and run all the stocks. The outputs are the prediction accuracy of rise/decline trend of each stock.

    5. Train and Run SVM: Open codes/SVM and run. It will all the stocks and also generate the prediction accuracy of rise/decline trend of each stock.

That's all, thanks for reading

