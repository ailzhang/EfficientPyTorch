python -c 'print(hash("hello"))' # 跑多次结果是不一样的
PYTHONHASHSEED=0 python -c 'print(hash("hello"))' #跑多次结果是一样的
