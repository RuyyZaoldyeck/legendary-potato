# legendary-potato
Machine Learning / Deep Learning utilities

# Handling imbalanced sets
```
# plotting the imbalance in terms of counts
df.target.value_counts().plot(kind="bar", title="titulo");

# print proportion of a binary class target column
def percentage(part, whole):
  return 100 * float(part) / float(whole)
  
no_claim, claim = df.target.value_counts()
print(f'No claim {no_claim}')
print(f'Claim {claim}')
print(f'Claim proportion {round(percentage(claim, claim + no_claim), 2)}%')

# print percent of missing rows
row_count = df.shape[0]

for c in df.columns:
  m_count = df[df[c] == -1][c].count()
  if m_count > 0:    
    print(f'{c} - {m_count} ({round(percentage(m_count, row_count), 3)}%) rows missing')
```
# Imputing data
```
# imputing numerical data / imputing categorical data
from sklearn.impute import SimpleImputer

mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
cat_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')

for c in num_columns:
  df[c] = mean_imp.fit_transform(df[[c]]).ravel()
  
for c in cat_columns:
  df[c] = cat_imp.fit_transform(df[[c]]).ravel()
```
# One hot encoding
```
# Pandas one hot encoding for categorical columns
df = pd.get_dummies(df, columns=cat_columns)
```
# Evaluation
```
# plotting the confusion matrix
from sklearn.metrics import confusion_matrix

def plot_cm(labels, predictions, p=0.5):

  tick_labels = ['No claim', 'Claim']

  cm = confusion_matrix(labels, predictions > p)
  ax = sns.heatmap(cm, annot=True, fmt="d")
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  ax.set_xticklabels(tick_labels)
  ax.set_yticklabels(tick_labels)
  
# Printing more metrics
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def build_model(train_data, metrics=["accuracy"]):
  model = keras.Sequential([
    keras.layers.Dense(
      units=36, 
      activation='relu',
      input_shape=(train_data.shape[-1],)
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=1, activation='sigmoid'),
  ])

  model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=metrics
  )

  return model

from sklearn.metrics import f1_score

def print_metrics(model, evaluation_results, predictions):
  for name, value in zip(model.metrics_names, evaluation_results):
    print(name, ': ', value)
  print(f'f1 score: {f1_score(y_test, np.round(y_pred.flatten()))}')
  print()

# Plotting ROC curve

from sklearn.metrics import roc_curve

def plot_roc(labels, predictions):
  fp, tp, _ = roc_curve(labels, predictions)

  plt.plot(fp, tp, label='ROC', linewidth=3)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.plot(
      [0, 1], [0, 1], 
      linestyle='--', 
      linewidth=2, 
      color='r',
      label='Chance', 
      alpha=.8
  )
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  plt.legend(loc="lower right")

```

# Resampling
```
# Oversampling minority class (in this case claim is the minority class)
from sklearn.utils import resample

claim_upsampled = resample(claim,
                          replace=True, 
                          n_samples=len(no_claim),
                          random_state=RANDOM_SEED)
                       
upsampled = pd.concat([no_claim, claim_upsampled])

# Undersampling majority class ( same as oversampling )

# Generating synthetic data using imblearn
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=RANDOM_SEED, ratio=1.0);
X_train, y_train = sm.fit_sample(X_train, y_train)

```
