from sklearn import tree

#buoc 1: thu thap du lieu
#buoc 2: xu li du lieu
#buoc 3: xay dung model
#buoc 4: Du doan ket qua
#buoc 5: Danh gia xem model cos hieu qua khong
my_tree = tree.DecisionTreeClassifier()

features = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 4, 6],
    [5, 4, 4, 3],
    [1, 4, 4, 7],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 2, 7]
]

labels = [0, 1, 1, 0, 0, 0, 0, 1]

res = my_tree.fit(features, labels)

predict_res = res.predict([[1, 4, 3, 6]])
print(predict_res)