import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from prince import MCA

class low_frequency_to_others:
    def __init__(self, threshold=10, name='others', verbose=True):
        self.min_count = threshold
        self.category_map = defaultdict(lambda: name)
        self.name = name
        self.verbose = verbose

    def fit(self, X):
        freq = X.value_counts()
        self.category_map.update(freq[freq >= self.min_count].index.to_series().to_dict())

    def transform(self, X):
        transformed_X = X.map(self.category_map).fillna(self.name)
        num_changed = (transformed_X == self.name).sum()
        if num_changed > 0:
            if self.verbose:
                print(f"Columns:({X.name}) '{self.name}'로 {num_changed}개 변환")
        else:
            if self.verbose:
                print(f"Columns:({X.name}) 변환 X")
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class cat_to_embedding:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.embeddings = {}

    def fit(self, X):
        columns = X.columns
        for col in columns:
            unique_categories = X[col].nunique()
            embedding = nn.Embedding(num_embeddings=unique_categories, embedding_dim=self.embedding_dim)
            self.embeddings[col] = embedding

    def transform(self, X):
        X_transformed = X.copy()
        columns = X.columns
        for col in columns:
            # 카테고리 값을 임베딩 인덱스로 변환
            category_to_index = {category: idx for idx, category in enumerate(X[col].unique())}
            X_transformed[col] = X[col].map(category_to_index)

            # 임베딩 적용
            indices = torch.tensor(X_transformed[col].values)
            embedded_values = self.embeddings[col](indices).detach().numpy()

            # 임베딩된 값을 원래 데이터프레임에 추가
            embedding_columns = [f"{col}_embedding_{i}" for i in range(self.embedding_dim)]
            embedding_df = pd.DataFrame(embedded_values, columns=embedding_columns)
            X_transformed = pd.concat([X_transformed.drop(columns=[col]), embedding_df], axis=1)
        
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



class AE(nn.Module):
    def __init__(self, hidden_dims, learning_rate=0.001, epochs=100):
        super(AE, self).__init__()
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embedding_layers = {}
        self.input_dim = None
        self.encoder = None
        self.decoder = None
        self.embedding_dim_sum = 0

    def build_model(self):
        # 인코더 네트워크 구성
        encoder_layers = []
        current_dim = self.input_dim + self.embedding_dim_sum
        for h_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.LeakyReLU(negative_slope=0.01))
            current_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 디코더 네트워크 구성 (인코더의 반대 순서로)
        decoder_layers = []
        for h_dim in reversed(self.hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.LeakyReLU(negative_slope=0.01))
            current_dim = h_dim
            
        decoder_layers.append(nn.Linear(current_dim, self.input_dim + self.embedding_dim_sum))
        # decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def train_model(self, X_train, X_val=None, is_early_stop=True):
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        patience = 10

        best_model_state = None

        for epoch in range(self.epochs):
            self.encoder.train()
            self.decoder.train()
            optimizer.zero_grad()

            # 순전파
            encoded = self.encoder(X_train)
            reconstructed = self.decoder(encoded)

            # 역전파
            loss = criterion(reconstructed, X_train)
            loss.backward(retain_graph=True)  # retain_graph 생략, 기본값 False
            optimizer.step()

            if X_val is not None:
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    val_encoded = self.encoder(X_val)
                    val_reconstructed = self.decoder(val_encoded)
                    val_loss = criterion(val_reconstructed, X_val)

                if is_early_stop:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        best_epoch = epoch
                        best_model_state = {
                            'encoder': self.encoder.state_dict(),
                            'decoder': self.decoder.state_dict()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f'Early stopping at epoch {epoch+1}')
                            break
            if epoch % 10 == 0 :
                print(f'epoch:{epoch} train loss: {loss}, val loss: {val_loss}')
                
        if X_val is None:  # val이 없으면 마지막 상태를 사용
            best_loss = loss.item()
            best_epoch = self.epochs

        if best_model_state is not None:
            self.encoder.load_state_dict(best_model_state['encoder'])
            self.decoder.load_state_dict(best_model_state['decoder'])

        if X_val is not None:
            return best_loss, best_epoch
        else:
            return loss, len(self.epochs)

    def cat_to_embedding(self, X, cat_cols):
        embeddings = []
        for col in cat_cols:
            num_categories = X[col].nunique()
            embedding_dim = min(50, (num_categories + 1) // 2)
            self.embedding_dim_sum += embedding_dim

            embedding_layer = nn.Embedding(num_categories, embedding_dim)
            self.embedding_layers[col] = embedding_layer

            X[col] = X[col].astype('category').cat.codes
            embedded_tensor = embedding_layer(torch.tensor(X[col].values, dtype=torch.long))
            embeddings.append(embedded_tensor)

        continuous_tensor = torch.tensor(X.drop(columns=cat_cols).values, dtype=torch.float32)
        X_combined = torch.cat(embeddings + [continuous_tensor], dim=1)
        return X_combined

    def fit(self, X, val=None, cat_cols=None, seed=42, early_stop=True):
        if cat_cols:
            continuous_columns = [col for col in X.columns if col not in cat_cols]
            self.input_dim = len(continuous_columns)
            X_train = self.cat_to_embedding(X, cat_cols)
        else:
            self.input_dim = X.shape[1]
            X_train = torch.tensor(X.values, dtype=torch.float32)

        if self.encoder is None or self.decoder is None:
            self.build_model()

        if val is not None:
            if cat_cols:
                X_val = self.cat_to_embedding(val, cat_cols)
            else:
                X_val = torch.tensor(val.values, dtype=torch.float32)

            loss, best_epoch = self.train_model(X_train, X_val, is_early_stop=early_stop)
        else:
            loss, best_epoch = self.train_model(X_train, is_early_stop=False)

        print(f'best_epoch: {best_epoch}, validation_loss: {loss}')

    def transform(self, X, cat_cols=None):
        if cat_cols:
            X_tensor = self.cat_to_embedding(X, cat_cols)
        else:
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        
        # if isinstance(X, pd.DataFrame):
        #     X_tensor = torch.tensor(X.values, dtype=torch.float32)
        # else:
        #     X_tensor = X
            
        with torch.no_grad():
            encoded = self.encoder(X_tensor)
        return pd.DataFrame(encoded.numpy(), columns=[f"encoded_{i}" for i in range(self.hidden_dims[-1])])

    def fit_transform(self, X, val=None, cat_cols=None, seed=42, early_stop=True):
        self.fit(X, val=val, cat_cols=cat_cols, seed=seed, early_stop=early_stop)
        return self.transform(X, cat_cols=cat_cols)



def cal_mca_components(X, threshold=0.8):
    """
    최적의 MCA 컴포넌트 수를 계산
    """
    
    # MCA 모델 학습 (모든 컴포넌트를 계산)
    mca = MCA(n_components=X.shape[1])
    mca = mca.fit(X)
    
    # 각 컴포넌트의 설명된 이질성 비율 (inertia) 계산
    explained_inertia = mca.eigenvalues_
    
    # 누적 이질성 비율 계산
    cumulative_inertia = np.cumsum(explained_inertia)
    
    # 임계값을 초과하는 최소 컴포넌트 수 계산
    n_components = np.argmax(cumulative_inertia >= threshold) + 1
    
    return n_components, explained_inertia







