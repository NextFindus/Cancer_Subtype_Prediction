import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd


class TissueGatedTransformer(nn.Module):

    def __init__(self, num_features=20318, num_tissues=5, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Tissue embedding and attention gate
        self.tissue_embedding = nn.Embedding(num_tissues, 16)
        self.attention_gate = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_features),
            nn.Sigmoid()
        )

        # Feature projection
        self.feature_projection = nn.Linear(num_features, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, d_model))

        # Regression head
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.SELU(),
            nn.Linear(32, 1)
        )

    def forward(self, methylation, tissue_id):
        """
        methylation: [batch_size, num_features]
        tissue_id: [batch_size] (long tensor with tissue indices)
        """
        # Tissue-specific gating
        tissue_emb = self.tissue_embedding(tissue_id)  # [batch_size, 16]
        attn_weights = self.attention_gate(tissue_emb)  # [batch_size, num_features]
        gated_features = methylation * attn_weights  # [batch_size, num_features]

        projected = self.feature_projection(gated_features)  # [batch_size, d_model]

        transformer_input = projected + self.positional_encoding  # [batch_size, d_model]
        transformer_input = transformer_input.unsqueeze(1)  # [batch_size, 1, d_model]

        # Transformer processing
        transformer_output = self.transformer(transformer_input)  # [batch_size, 1, d_model]

        # Regression prediction
        pooled = transformer_output.squeeze(1)  # [batch_size, d_model]
        age_pred = self.regression_head(pooled)  # [batch_size, 1]

        return age_pred.squeeze(-1)


def train(model, train_loader, val_loader, num_epochs=100, lr=0.001, model_path='best_model.pth'):
    """
    Train the model and save the best version based on validation loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        for methylation, tissue_id, ages in train_loader:
            methylation = methylation.to(device)
            tissue_id = tissue_id.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()
            outputs = model(methylation, tissue_id)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * methylation.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for methylation, tissue_id, ages in val_loader:
                methylation = methylation.to(device)
                tissue_id = tissue_id.to(device)
                ages = ages.to(device)

                outputs = model(methylation, tissue_id)
                loss = criterion(outputs, ages)
                val_loss += loss.item() * methylation.size(0)

        # Calculate epoch metrics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved at {model_path} with val loss: {val_loss:.4f}')

    return model


def save_model(model, path='tissue_transformer.pth'):
    """Save model state dictionary"""
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def load_model(model_class, path, **model_args):
    """Load model state dictionary and initialize model"""
    model = model_class(**model_args)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(model, test_data, tissue_ids, output_csv='predictions.csv'):
    """
    Make predictions and save results to CSV
    Args:
        test_data: Tensor of shape [num_samples, num_features]
        tissue_ids: Tensor of shape [num_samples] with tissue indices
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create DataLoader for batching
    dataset = TensorDataset(test_data, tissue_ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for methylation, tissue_id in loader:
            methylation = methylation.to(device)
            tissue_id = tissue_id.to(device)
            batch_preds = model(methylation, tissue_id)
            predictions.append(batch_preds.cpu())

    # Combine results and save
    predictions = torch.cat(predictions).numpy()
    results_df = pd.DataFrame({
        'sample_id': range(len(predictions)),
        'predicted_age': predictions
    })
    results_df.to_csv(output_csv, index=False)
    print(f'Saved predictions to {output_csv}')
    return predictions

if __name__ == "__main__":

    num_samples = 100
    num_features = 20318
    num_tissues = 5

    # Create sample data
    methylation_data = torch.rand(num_samples, num_features)
    print("methylation data ",methylation_data)
    tissue_ids = torch.randint(0, num_tissues, (num_samples,))
    print("tissue ids ",tissue_ids)
    ages = torch.rand(num_samples) * 80
    print("ages ",ages)

    # Split into train/val/test (80/10/10)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    train_data = methylation_data[:train_size]
    train_tissues = tissue_ids[:train_size]
    train_ages = ages[:train_size]

    val_data = methylation_data[train_size:train_size + val_size]
    val_tissues = tissue_ids[train_size:train_size + val_size]
    val_ages = ages[train_size:train_size + val_size]

    test_data = methylation_data[train_size + val_size:]
    test_tissues = tissue_ids[train_size + val_size:]

    train_dataset = TensorDataset(train_data, train_tissues, train_ages)
    val_dataset = TensorDataset(val_data, val_tissues, val_ages)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = TissueGatedTransformer(
        num_features=num_features,
        num_tissues=num_tissues,
        d_model=64,
        nhead=4,
        num_layers=3
    )

    # Train model
    trained_model = train(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        lr=0.001,
        model_path='best_model.pth'
    )

    save_model(trained_model, 'final_model.pth')


    # Make prediction
    loaded_model = load_model(
        TissueGatedTransformer,
        'best_model.pth',
        num_features=num_features,
        num_tissues=num_tissues,
        d_model=64,
        nhead=4,
        num_layers=3
    )
    loaded_model.eval()

    predictions = predict(
        loaded_model,
        test_data,
        test_tissues,
        output_csv='test_predictions.csv'
    )
    print("Predictions:", predictions[:5])
