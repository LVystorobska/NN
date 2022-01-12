import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import plotly.graph_objs as go
from plotly.offline import iplot
import torch.optim as optim
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 8)})

def plot_predictions(df_result):
    data = []
    
    value = go.Scatter(
        x=df_result.index,
        y=df_result.TempAvgF,
        mode='lines',
        name='values',
        marker=dict(),
        text=df_result.index,
        line=dict(color='rgba(0,0,0, 0.3)'),
    )
    data.append(value)

    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode='lines',
        line={'dash': 'dot'},
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)
    
    layout = dict(
        title='Predictions vs Actual Values for the average Temp',
        xaxis=dict(title='Time', ticklen=5, zeroline=False),
        yaxis=dict(title='Temperature', ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)

def plot_dataset(df, title):
    data = []
    
    value = go.Scatter(
        x=df.index,
        y=df.TempAvgF,
        mode='lines',
        name='values',
        marker=dict(),
        text=df.index,
        line=dict(color='rgba(0,0,0, 0.3)'),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title='Date', ticklen=5, zeroline=False),
        yaxis=dict(title='Value', ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)




device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv('austin_weather.csv')
df = df.set_index(['Date'])

df = df.drop(['PrecipitationSumInches','Events', 'TempHighF', 'TempLowF'], axis=1)
# df = df.drop(['PrecipitationSumInches','Events', 'TempHighF', 'TempLowF','DewPointHighF','DewPointAvgF','DewPointLowF','HumidityHighPercent','HumidityAvgPercent','HumidityLowPercent','SeaLevelPressureHighInches','SeaLevelPressureAvgInches','SeaLevelPressureLowInches','VisibilityHighMiles','VisibilityAvgMiles','VisibilityLowMiles','WindHighMPH','WindAvgMPH','WindGustMPH'], axis=1)
# df['TempHighF'] = df['TempHighF'].astype(float)
# df['TempLowF'] = df['TempLowF'].astype(float)
df['TempAvgF'] = df['TempAvgF'].astype(int)
df['DewPointHighF'] = df['DewPointHighF'].astype(float)
df['DewPointAvgF'] = df['DewPointAvgF'].astype(float)
df['DewPointLowF'] = df['DewPointLowF'].astype(float)
df['HumidityHighPercent'] = df['HumidityHighPercent'].astype(float)
df['HumidityAvgPercent'] = df['HumidityAvgPercent'].astype(float)
df['HumidityLowPercent'] = df['HumidityLowPercent'].astype(float)
df['VisibilityHighMiles'] = df['VisibilityHighMiles'].astype(float)
df['SeaLevelPressureLowInches'] = df['SeaLevelPressureLowInches'].astype(float)
df['VisibilityAvgMiles'] = df['VisibilityAvgMiles'].astype(float)
df['VisibilityLowMiles'] = df['VisibilityLowMiles'].astype(float)
df['WindHighMPH'] = df['WindHighMPH'].astype(float)
df['WindAvgMPH'] = df['WindAvgMPH'].astype(float)
df['WindGustMPH'] = df['WindGustMPH'].astype(float)

df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()

def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f'lag{n}'] = df_n['TempAvgF'].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

input_dim = 120
df_timelags = generate_time_lags(df, input_dim)
df_timelags


data_by_features = (df.assign(day = df.index.day)
                    .assign(month = df.index.month)
                    .assign(day_of_week = df.index.dayofweek)
                    .assign(week_of_year = df.index.week)
                )

def onehot_concat(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col)
    
    return pd.concat([df, dummies], axis=1).drop(columns=cols)

data_by_features = onehot_concat(data_by_features, ['month','day','day_of_week','week_of_year'])

data_by_features = df_timelags
# print(data_by_features.head())


def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(data_by_features, 'TempAvgF', 0.2)


def get_scaler(scaler):
    scalers = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'maxabs': MaxAbsScaler,
        'robust': RobustScaler,
    }
    return scalers.get(scaler.lower())()

scaler = get_scaler('robust')
X_train_arr = scaler.fit_transform(X_train)
X_val_arr = scaler.transform(X_val)
X_test_arr = scaler.transform(X_test)

y_train_arr = scaler.fit_transform(y_train)
y_val_arr = scaler.transform(y_val)
y_test_arr = scaler.transform(y_test)


from torch.utils.data import DataLoader, Dataset

batch_size = 64
dropout = 0.2
weight_decay = 1e-6

train_features = torch.Tensor(X_train_arr)
train_targets = torch.Tensor(y_train_arr)
val_features = torch.Tensor(X_val_arr)
val_targets = torch.Tensor(y_val_arr)
test_features = torch.Tensor(X_test_arr)
test_targets = torch.Tensor(y_test_arr)

# dataset to implement slising window
class CustomDataset(Dataset):
    def __init__(self, data, window):
        self.data_x = data[0]
        self.data_y = data[1]
        self.window = window
        # self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__()
 
    def __getitem__(self, index):
        x = self.data_x[index:index+self.window]
        y = self.data_y[index+self.window]
        return x, y
 
    def __len__(self):
        return len(self.data_y) -  self.window 
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())


train = (train_features, train_targets)
val = (val_features, val_targets)
test = (test_features, test_targets)

# window size
seq_length = 5

train_loader = DataLoader( CustomDataset(train, seq_length),batch_size=batch_size)
val_loader = DataLoader(CustomDataset(val, seq_length), batch_size=batch_size)
test_loader = DataLoader(CustomDataset(test, seq_length), batch_size=batch_size)
test_loader_one = DataLoader(CustomDataset(test, seq_length), batch_size=1)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, nonlinearity='relu', batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, h0 = self.rnn(x, h0.detach())
        # (batch_size, seq_length, hidden_size) => fully conn
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def set_model(model, model_params):
    models = {'rnn': RNNModel}
    return models.get(model.lower())(**model_params)


class Optimizer:
    def __init__(self, model, loss_fn, optimizer):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y):

        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_loader, val_loader, has_window=True, batch_size=64, n_epochs=50, n_features=1):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []

            for x_batch, y_batch in train_loader:
                # print(x_batch.shape)
                if has_window:
                    batch_size, _, _ = x_batch.shape
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    if has_window:
                        batch_size, _, _ = x_val.shape
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(f'[{epoch}/{n_epochs}] Train loss balue: {training_loss:.4f}\t Test loss value: {validation_loss:.4f}')

    def evaluate(self, test_loader, batch_size=1, n_features=1):

        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_loss_data(self):
        plt.plot(self.train_losses, 'g', label='Training loss')
        plt.plot(self.val_losses, 'purple', label='Validation loss')
        plt.legend()
        plt.title('Losses')
        plt.show()
        plt.close()



# print(len(X_train.columns))
input_dim = len(X_train.columns)
output_dim = 1
hidden_dim = 64
layer_dim = 1
batch_size = 64
n_epochs = 200
learning_rate = 1e-2


model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = set_model('rnn', model_params)

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# adaptive moment estimation
# better adamW

opt = Optimizer(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, has_window=True, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={'TempAvgF': vals, 'prediction': preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [['TempAvgF', 'prediction']])
    return df_result

def calculate_metrics(df):
    result_metrics = {'mae' : mean_absolute_error(df.TempAvgF, df.prediction),
                      'r2' : r2_score(df.TempAvgF, df.prediction)}
    
    print('Mean Absolute Error:       ', result_metrics['mae'])
    print('R^2 Score:                 ', result_metrics['r2'])
    return result_metrics

df_result = format_predictions(predictions, values, X_test, scaler)
result_metrics = calculate_metrics(df_result)

opt.plot_loss_data()
plot_predictions(df_result)