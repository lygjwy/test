import ast

dir_list = [
    'adam-bce-mean',
    'adam-bce-sum',
    'adam-mse-mean',
    'adam-mse-sum',
    'sgd-bce-mean',
    'sgd-mse-mean'
]

for dir in dir_list:
    file = '-'.join(dir.split('-')[1:]) + '-200.txt'
    with open('./' + dir + '/' + file) as f:
        train_loss = ast.literal_eval(f.readline())
        test_loss = ast.literal_eval(f.readline())

    print(dir)
    print(train_loss[100])
    print(test_loss[100])