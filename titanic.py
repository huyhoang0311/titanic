import pandas as pd
from sklearn.linear_model import LogisticRegression

# Đọc dữ liệu
train_data = pd.read_csv("C:/Users/ADMIN/.vscode/py/titanic/train.csv")
test_data = pd.read_csv("C:/Users/ADMIN/.vscode/py/titanic/test.csv")

# Xử lý giá trị bị thiếu trong cả hai tập
train_data = train_data.dropna(subset=['Age', 'Embarked', 'Fare'])
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Embarked'] = test_data['Embarked'].fillna('S')
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Mã hóa biến dạng (Sex, Embarked)
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)



# Đảm bảo các cột của test_data đồng nhất với train_data
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0  # Thêm cột bị thiếu với giá trị mặc định
test_data = test_data[train_data.columns]  # Đảm bảo thứ tự cột giống nhau

# Xác định các cột đặc trưng và nhãn
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X_train = train_data[features]
y_train = train_data['Survived']

# Kiểm tra nhãn trong tập test
if 'Survived' in test_data.columns:
    y_test = test_data['Survived']
else:
    y_test = None

X_test = test_data[features]

# Huấn luyện mô hình
model = LogisticRegression(max_iter=1000)  # Tăng max_iter để đảm bảo hội tụ
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Kiểm tra kết quả và lưu dự đoán
# Kiểm tra nhãn trong tập test

submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': y_pred
    })
submission.to_csv("submission.csv", index=False)
print("Predictions saved to 'submission.csv'.")
