# PM2.5 prediction using LSTM-based encoder-decoder model with attention
- 每部分文件作用描述：
    - data/：存放相关非代码文件，包括：
        - 空气质量数据（./data/beijing_20140101-20141231/...）
        - 预处理过后数据文件（df_train.pkl, df_val.pkl, df_test.pkl，由preprocess.py生成）
        - 最优模型文件（checkpoint_zfr.pt）
    - dataset.py：定义数据集类
    - engine.py：定义训练函数、测试一个数据集函数
    - main.ipynb：包含数据可视化以及模型的实验结果
    - model.py：实现了模型
    - preprocess.py：实现预处理，包括读取csv，数据清理与缺失值处理，并输出到pkl文件
    - reproduce.py：运行时首先判断data文件夹内是否有预处理好的数据，如无则开始预处理；其次判断data文件夹内是否存在checkpoint_zfr.pt，如无则开始训练模型，如有则直接加载。之后开始测试。
    - utils.py：包含一些辅助函数
- 运行环境：Win10 64-bit; Anaconda/Miniconda最新版（下载地址略）
    - conda create -n zfr python=3.7.7 pytorch=1.5.0 cudatoolkit=10.2 -c pytorch
    - conda activate zfr
    - conda install matplotlib seaborn pandas scikit-learn 
- 整理数据的命令：详见data文件夹描述，下载之后unzip并mv即可。
- 训练最优模型命令：
    - python reproduce.py
    - 详见reproduce.py文件描述
- 测试命令：同上
- 用提交的最优模型测试的命令：同上
    - 预期的结果：print出6h、6类的准确率
    

