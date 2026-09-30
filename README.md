# Titanic — Machine Learning from Disaster

使用機器學習預測 Titanic 乘客是否生還，參加 Kaggle 的入門級二元分類競賽：

> Titanic - Machine Learning from Disaster

本專案的目標是建立一個完整、可重現、可解釋的機器學習流程，從資料探索、資料清理、特徵工程、模型訓練、驗證評估，到最後產生可提交至 Kaggle 的 `submission.csv`。

---

## 1. 專案目標

本專案需要根據 Titanic 乘客資料，預測每一位乘客是否在事故中生還。

模型的預測目標如下：

- `0`：乘客未生還
- `1`：乘客生還

這是一個二元分類問題。

### 主要問題

> 根據乘客的性別、年齡、艙等、票價、登船港口、家庭人數與其他特徵，預測乘客是否生還。

---

## 2. Kaggle 挑戰資訊

- 競賽名稱：Titanic - Machine Learning from Disaster
- 競賽類型：Binary Classification
- 評估指標：Accuracy
- 訓練資料筆數：891 筆
- 測試資料筆數：418 筆
- 主要資料來源：Kaggle Titanic Competition
- Kaggle 競賽頁面：

https://www.kaggle.com/competitions/titanic

### 評估方式

Kaggle 使用 Accuracy 評估模型：

```text
Accuracy = 正確預測的筆數 / 總預測筆數
```

模型會使用 `train.csv` 進行訓練，接著預測 `test.csv` 中乘客的 `Survived`。

由於 `test.csv` 不包含真實的 `Survived` 標籤，因此本地開發時必須使用訓練資料切分或交叉驗證來評估模型。

---

## 3. 目前資料結構

目前 Repository 預期包含以下資料：

```text
.
├── README.md
└── data
    ├── gender_submission.csv
    ├── test.csv
    └── train.csv
```

### 3.1 `data/train.csv`

訓練資料包含乘客特徵與實際生還結果。

主要欄位：

| 欄位 | 說明 |
|---|---|
| `PassengerId` | 乘客識別碼 |
| `Survived` | 目標欄位，`0` 表示未生還，`1` 表示生還 |
| `Pclass` | 票艙等級，`1`、`2`、`3` |
| `Name` | 乘客姓名 |
| `Sex` | 乘客性別 |
| `Age` | 乘客年齡 |
| `SibSp` | 同行的兄弟姊妹或配偶人數 |
| `Parch` | 同行的父母或子女人數 |
| `Ticket` | 票號 |
| `Fare` | 票價 |
| `Cabin` | 船艙編號 |
| `Embarked` | 登船港口 |

`train.csv` 包含模型訓練所需的目標欄位 `Survived`。

### 3.2 `data/test.csv`

測試資料包含與 `train.csv` 類似的乘客特徵，但不包含 `Survived`。

模型必須使用訓練完成後的結果，預測 `test.csv` 中每一筆資料的 `Survived`。

### 3.3 `data/gender_submission.csv`

這是 Kaggle 提供的範例提交檔。

它不是測試集的真實答案，而是使用一個簡單規則產生的基準結果：

```text
女性乘客預測為生還
男性乘客預測為未生還
```

此檔案可以用來確認 Kaggle 提交格式，但不應該被當成測試資料的真實標籤，也不應該拿來訓練模型。

---

## 4. 專案實作要求

請 Agent Session 完成一個完整且可執行的 Titanic 機器學習專案。

實作內容至少需要包含：

1. 載入資料
2. 確認資料格式與欄位
3. 探索性資料分析
4. 缺失值分析
5. 資料前處理
6. 特徵工程
7. 建立基準模型
8. 訓練多個分類模型
9. 使用一致的驗證方法比較模型
10. 選擇最佳模型
11. 產生 Kaggle 提交檔
12. 驗證提交檔格式
13. 保存模型評估結果
14. 撰寫使用說明
15. 確保程式可以從 Repository 根目錄執行

---

## 5. 建議專案結構

Agent 應該將專案整理成類似以下結構：

```text
.
├── README.md
├── data
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── notebooks
│   └── titanic_analysis.ipynb
├── models
│   └── .gitkeep
├── outputs
│   ├── .gitkeep
│   └── submission.csv
├── reports
│   ├── .gitkeep
│   └── model_comparison.csv
├── requirements.txt
└── .gitignore
```

如果 Agent 判斷使用單一 Python 程式比較適合，也可以採用以下較簡單的結構：

```text
.
├── README.md
├── data
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── titanic_model.py
├── requirements.txt
├── outputs
│   └── submission.csv
└── reports
    └── model_comparison.csv
```

無論採用哪一種結構，都必須確保：

- 程式可以正常執行
- 路徑不應該寫死成特定電腦路徑
- 可以從 Repository 根目錄執行
- 可以重複產生相同或接近的結果
- 不應該把測試集標籤或外部答案加入訓練資料

---

## 6. 資料探索與品質檢查

實作時必須先檢查以下項目：

### 6.1 資料形狀

確認：

- `train.csv` 是否包含 891 筆資料
- `test.csv` 是否包含 418 筆資料
- `train.csv` 是否包含 `Survived`
- `test.csv` 是否不包含 `Survived`
- 訓練集與測試集的共同欄位是否一致

### 6.2 欄位型別

檢查每個欄位的資料型別，尤其是：

- 數值欄位：`Age`、`SibSp`、`Parch`、`Fare`
- 類別欄位：`Sex`、`Embarked`
- 文字欄位：`Name`、`Ticket`、`Cabin`

### 6.3 缺失值

分析以下欄位的缺失情況：

- `Age`
- `Cabin`
- `Embarked`
- `Fare`

不得直接無聲地刪除大量資料。若需要刪除資料，必須在程式或報告中說明原因。

### 6.4 目標欄位分布

分析 `Survived` 的分布：

- 生還人數
- 未生還人數
- 生還比例
- 類別是否不平衡

### 6.5 基本統計

至少分析：

- 不同性別的生還率
- 不同艙等的生還率
- 年齡分布
- 票價分布
- 登船港口分布
- 家庭人數與生還率的關聯

---

## 7. 資料前處理要求

資料前處理必須使用可重複且不會造成資料洩漏的方法。

### 7.1 目標欄位

將：

```python
target = "Survived"
```

從訓練特徵中分離。

`Survived` 不可以出現在模型輸入特徵中。

### 7.2 `PassengerId`

`PassengerId` 主要是識別用途，預設不應直接作為模型特徵。

可以保留它用於最後產生提交檔，但不要讓模型依賴乘客 ID 的數值大小。

### 7.3 `Age`

`Age` 存在缺失值，需要進行補值。

建議方法：

- 使用中位數補值
- 或按照 `Pclass` 與 `Sex` 分組後補值
- 若採用更進階方法，必須避免使用驗證集資訊造成資料洩漏

### 7.4 `Embarked`

`Embarked` 是類別欄位。

建議：

- 使用眾數補值
- 再進行 One-Hot Encoding

### 7.5 `Fare`

`Fare` 如果有缺失值，建議使用中位數補值。

可視情況新增：

- `FarePerPerson`
- `FareBand`

但必須在報告中說明處理方式。

### 7.6 `Cabin`

`Cabin` 缺失值較多，不建議直接把完整 Cabin 字串當成高基數類別。

可以選擇：

- 建立 `HasCabin`，表示是否有船艙資料
- 擷取 Cabin 的甲板字母
- 對缺失值使用 `Unknown`
- 或在模型比較後決定是否排除原始 `Cabin`

### 7.7 `Sex`

`Sex` 必須轉換成模型可以使用的格式，例如：

- One-Hot Encoding
- 或使用二元映射

不得直接將文字欄位傳入不支援文字輸入的模型。

### 7.8 類別欄位

對以下欄位進行適當編碼：

- `Sex`
- `Embarked`
- `Title`
- `Deck`
- 其他新增的類別特徵

建議使用 `ColumnTransformer` 與 `Pipeline`，讓補值、編碼與模型訓練形成完整流程。

---

## 8. 建議特徵工程

請至少實作以下幾項特徵工程，並比較加入前後的效果。

### 8.1 `FamilySize`

```text
FamilySize = SibSp + Parch + 1
```

代表乘客包含自己在內的家庭人數。

### 8.2 `IsAlone`

```text
IsAlone = 1 if FamilySize == 1 else 0
```

表示乘客是否獨自旅行。

### 8.3 `Title`

從 `Name` 擷取稱謂，例如：

- Mr
- Miss
- Mrs
- Master
- Rare

應該將稀有稱謂適當合併，避免產生過多稀疏類別。

### 8.4 `Surname`

可以從姓名擷取姓氏，用於推測家庭關係。

但必須謹慎處理，避免造成高基數特徵或過度擬合。

### 8.5 `Mother`

可以根據年齡、性別、稱謂與 `Parch` 等欄位建立母親相關特徵。

此特徵屬於進階特徵，可選擇性實作。

### 8.6 `FarePerPerson`

如果有家庭人數資訊，可以嘗試：

```text
FarePerPerson = Fare / FamilySize
```

需要處理分母為零或缺失值的情況。

### 8.7 `Deck`

從 `Cabin` 擷取第一個字元作為甲板資訊。

缺失的 Cabin 可以標記為：

```text
Unknown
```

### 8.8 年齡區間

可以建立年齡區間，例如：

- Child
- Teenager
- Adult
- Senior

但不能同時無限制地保留太多高度相關特徵，應該透過驗證結果決定是否保留。

---

## 9. 基準模型

實作時必須先建立簡單的 Baseline，方便比較後續模型是否真的改善。

建議包含：

### 9.1 Gender Baseline

使用以下規則：

```text
女性預測為 1
男性預測為 0
```

此結果可以和 `gender_submission.csv` 的邏輯對照。

### 9.2 Logistic Regression

建立一個經過：

- 缺失值補值
- 類別欄位編碼
- 數值欄位處理

的 Logistic Regression 模型。

### 9.3 Decision Tree

建立一個基本 Decision Tree，並限制：

- `max_depth`
- `min_samples_leaf`
- `random_state`

避免樹模型過度擬合。

---

## 10. 建議比較的模型

至少比較三種以上模型。

建議模型包括：

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. HistGradientBoostingClassifier
6. Extra Trees Classifier

如果環境允許，也可以加入：

- XGBoost
- LightGBM
- CatBoost

但不應該為了使用複雜模型而引入不必要的依賴。

### 模型比較原則

所有模型應該：

- 使用相同的訓練資料
- 使用相同的驗證策略
- 使用一致的評估指標
- 記錄訓練時間與驗證結果
- 固定 `random_state`
- 避免使用測試集進行模型選擇

---

## 11. 驗證策略

由於 `test.csv` 沒有公開標籤，本地評估必須使用 `train.csv`。

### 11.1 基本驗證

建議使用：

```text
Train / Validation Split
```

例如：

- 80% 訓練資料
- 20% 驗證資料
- `stratify=y`
- 固定 `random_state`

### 11.2 交叉驗證

為了降低單次切分造成的偶然性，建議使用：

```text
StratifiedKFold
```

建議設定：

- `n_splits=5`
- `shuffle=True`
- 固定 `random_state`

至少記錄：

- 平均 Accuracy
- Accuracy 標準差
- 每個 fold 的 Accuracy

### 11.3 禁止資料洩漏

不得：

- 使用 `test.csv` 的答案訓練模型
- 使用 `gender_submission.csv` 當作測試集真實答案
- 在切分資料前使用整份資料計算會影響模型的統計值
- 讓驗證集資料參與補值器或編碼器的學習
- 使用 Kaggle 排行榜結果反覆針對驗證集過度調整

---

## 12. 模型評估內容

每個模型至少需要記錄：

| 欄位 | 說明 |
|---|---|
| `model` | 模型名稱 |
| `validation_accuracy` | 驗證集 Accuracy |
| `cv_mean_accuracy` | 交叉驗證平均 Accuracy |
| `cv_std_accuracy` | 交叉驗證標準差 |
| `training_time` | 訓練時間 |
| `feature_count` | 使用的特徵數量 |

結果應該輸出到：

```text
reports/model_comparison.csv
```

如果有視覺化，也可以輸出：

```text
reports/
├── survival_by_sex.png
├── survival_by_class.png
├── feature_importance.png
└── model_comparison.png
```

---

## 13. 最佳模型選擇

最佳模型不應該只根據一次 validation split 決定。

建議依照以下順序選擇：

1. 交叉驗證平均 Accuracy
2. 交叉驗證標準差
3. 模型複雜度
4. 是否容易重現
5. 是否容易解釋
6. Kaggle 提交結果

如果兩個模型分數接近，優先選擇：

- 較簡單
- 較穩定
- 較容易維護
- 較少外部依賴

的模型。

---

## 14. 產生 Kaggle 提交檔

模型完成訓練後，必須使用完整的 `train.csv` 重新訓練最佳模型，然後對 `test.csv` 進行預測。

輸出檔案：

```text
outputs/submission.csv
```

提交檔必須符合以下格式：

```csv
PassengerId,Survived
892,0
893,1
894,0
```

### 提交檔規則

`submission.csv` 必須：

- 只有兩個欄位
- 欄位名稱必須是 `PassengerId` 與 `Survived`
- 包含 418 筆測試資料
- 加上標題列後總共 419 行
- `PassengerId` 必須來自 `test.csv`
- `Survived` 只能是 `0` 或 `1`
- 不可以包含索引欄位
- 不可以包含多餘欄位
- 不可以有重複的 `PassengerId`
- 不可以有缺失值

### 程式驗證

產生提交檔後，程式必須自動檢查：

```python
assert list(submission.columns) == ["PassengerId", "Survived"]
assert len(submission) == len(test)
assert submission["PassengerId"].is_unique
assert submission["Survived"].isin([0, 1]).all()
assert submission["Survived"].notna().all()
```

---

## 15. 建議執行方式

### 15.1 建立虛擬環境

```bash
python -m venv .venv
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 15.2 安裝依賴

```bash
pip install -r requirements.txt
```

### 15.3 執行完整流程

如果採用模組化結構：

```bash
python -m src.train
python -m src.evaluate
python -m src.predict
```

如果採用單一程式：

```bash
python titanic_model.py
```

### 15.4 執行後應該產生

```text
outputs/submission.csv
reports/model_comparison.csv
```

如果有模型保存需求，可以產生：

```text
models/best_model.joblib
```

---

## 16. 建議 `requirements.txt`

至少包含：

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
jupyter
```

如果實際程式沒有使用某些套件，請不要加入不必要的依賴。

---

## 17. 程式品質要求

Agent 實作時必須遵守以下規範：

### 17.1 可重現性

- 所有模型設定 `random_state`
- 不使用絕對路徑
- 不依賴個人電腦上的特殊目錄
- 不依賴手動修改 CSV
- 程式可以從乾淨環境重新執行

### 17.2 程式碼風格

- 使用清楚的函式名稱
- 每個主要步驟拆分成函式
- 避免單一函式過長
- 加入必要的型別提示
- 加入適量 docstring
- 避免重複程式碼
- 使用 logging 或清楚的輸出訊息

### 17.3 錯誤處理

程式應該檢查：

- 資料檔是否存在
- 必要欄位是否存在
- 訓練集是否包含 `Survived`
- 測試集是否錯誤地包含 `Survived`
- 產生的提交檔是否符合格式
- 模型是否成功完成訓練

### 17.4 資料安全

不得將以下內容加入 Repository：

- Kaggle API Token
- 個人密碼
- 私密憑證
- 個人絕對路徑
- 不必要的大型模型檔
- 未經允許的外部資料

---

## 18. `.gitignore` 建議

請建立 `.gitignore`：

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.env
.DS_Store

models/*
!models/.gitkeep

outputs/*
!outputs/.gitkeep

reports/*
!reports/.gitkeep
```

如果希望將最終提交檔加入 Repository，可以額外保留：

```gitignore
!outputs/submission.csv
```

---

## 19. 預期輸出

完成後，Repository 至少應該包含：

```text
.
├── README.md
├── data
│   ├── gender_submission.csv
│   ├── test.csv
│   └── train.csv
├── requirements.txt
├── src/
├── outputs/
│   └── submission.csv
├── reports/
│   └── model_comparison.csv
└── .gitignore
```

---

## 20. 完成標準

Agent Session 必須在完成後確認以下項目：

### 資料處理

- [ ] 成功讀取 `data/train.csv`
- [ ] 成功讀取 `data/test.csv`
- [ ] 成功處理缺失值
- [ ] 沒有將 `Survived` 錯誤放入測試資料
- [ ] 沒有使用 `gender_submission.csv` 作為真實測試標籤

### 模型

- [ ] 至少完成三種模型比較
- [ ] 使用固定的驗證策略
- [ ] 使用 Accuracy 評估
- [ ] 有記錄模型比較結果
- [ ] 沒有資料洩漏
- [ ] 有選出最佳模型

### 提交檔

- [ ] 已產生 `outputs/submission.csv`
- [ ] 只有 `PassengerId` 與 `Survived`
- [ ] 包含 418 筆預測
- [ ] `Survived` 僅包含 0 和 1
- [ ] 沒有缺失值
- [ ] `PassengerId` 沒有重複
- [ ] 可以直接上傳 Kaggle

### 文件

- [ ] README 已說明安裝方式
- [ ] README 已說明執行方式
- [ ] README 已說明資料欄位
- [ ] README 已說明模型流程
- [ ] README 已說明輸出檔案
- [ ] README 已說明如何產生 Kaggle submission

---

## 21. 建議的 Agent Session 任務

請 Agent Session 依照以下順序執行：

### Phase 1：檢查現有 Repository

1. 查看目前 Repository 的完整檔案結構。
2. 確認 `data/train.csv`、`data/test.csv`、`data/gender_submission.csv` 是否存在。
3. 檢查 CSV 欄位、資料筆數與缺失值。
4. 確認目前的 `README.md` 是否需要保留或重寫。

### Phase 2：建立基礎專案

1. 建立 Python 專案結構。
2. 建立 `requirements.txt`。
3. 建立 `.gitignore`。
4. 建立資料載入與驗證模組。
5. 建立基本執行入口。

### Phase 3：資料分析與前處理

1. 完成資料品質檢查。
2. 完成缺失值處理。
3. 完成類別欄位編碼。
4. 完成數值欄位處理。
5. 實作至少三項特徵工程。
6. 確保所有前處理都被包裝在 Pipeline 中。

### Phase 4：模型訓練與比較

1. 建立 Gender Baseline。
2. 建立 Logistic Regression。
3. 建立 Decision Tree。
4. 建立 Random Forest。
5. 視情況加入 Gradient Boosting 或其他模型。
6. 使用 StratifiedKFold 評估模型。
7. 將模型結果輸出至 `reports/model_comparison.csv`。

### Phase 5：產生提交檔

1. 根據驗證結果選擇最佳模型。
2. 使用完整訓練資料重新訓練。
3. 預測 `data/test.csv`。
4. 產生 `outputs/submission.csv`。
5. 執行提交檔格式驗證。
6. 確認可以直接上傳 Kaggle。

### Phase 6：測試與文件

1. 從 Repository 根目錄測試完整執行流程。
2. 修正所有錯誤與警告。
3. 確認沒有絕對路徑。
4. 更新 README 的實際執行指令。
5. 記錄最終模型與驗證結果。
6. 回報修改過的檔案與執行結果。

---

## 22. Agent 完成後的回報格式

完成後請 Agent 使用以下格式回報：

```text
## Implementation Summary

### Added Files
- 列出新增檔案

### Modified Files
- 列出修改檔案

### Data Validation
- train.csv 筆數：
- test.csv 筆數：
- 缺失值處理：
- 目標欄位：

### Models Evaluated
- 模型名稱與驗證結果

### Best Model
- 模型名稱：
- Cross-validation Accuracy：
- Standard deviation：

### Generated Files
- outputs/submission.csv
- reports/model_comparison.csv

### Validation
- submission 欄位是否正確：
- submission 筆數是否正確：
- 是否只有 0/1：
- 是否有缺失值：
- 是否成功執行完整流程：

### Notes
- 仍存在的限制
- 後續可改進方向
```

---

## 23. 後續可改進方向

完成基本版本後，可以進一步嘗試：

- 更精細的 `Title` 合併策略
- 家庭群組特徵
- Ticket 群組大小
- 同票號乘客分析
- 家庭成員生還率特徵
- 模型超參數搜尋
- RandomizedSearchCV
- GridSearchCV
- Ensemble Voting
- Stacking
- 特徵重要性分析
- SHAP 解釋
- 不同隨機種子下的穩定性測試
- Kaggle 提交結果與本地驗證結果的比較

但所有進階方法都必須避免資料洩漏，並保留一個簡單、可重現、容易理解的基準版本。

---

## 24. 專案限制

本專案的目標是學習完整的機器學習流程，不是保證達到最高 Kaggle 排名。

需要注意：

- Titanic 資料集規模較小
- 模型分數可能因資料切分而變動
- Kaggle Public Leaderboard 分數不一定代表模型泛化能力
- 反覆根據 Kaggle 分數調整模型可能造成過度擬合
- `gender_submission.csv` 只是基準提交範例
- `test.csv` 的真實標籤不可用於本地訓練

---

## 25. License and Data Usage

本專案使用 Kaggle Titanic Competition 提供的資料。

資料使用與競賽規則請以 Kaggle 官方頁面為準：

https://www.kaggle.com/competitions/titanic

本 Repository 中的程式碼僅供學習、研究與競賽實作使用。
