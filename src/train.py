import torch.optim as optim
from torch import nn
import torch
from src.model import NRMSModel
from src.data_loader import data_load
from src.evaluate import evaluate

def train(NAME, BATCH_SIZE = 64, LEARNING_RATE = 1e-4, EPOCHS = 5, NEG_SAMPLE_K = 4, MAX_HISTORY = 50,
          MAX_TITLE_LEN = 30, NUM_HEADS = 16, HEAD_DIM = 16):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, embedding_matrix = data_load(BATCH_SIZE, NEG_SAMPLE_K, MAX_HISTORY,
          MAX_TITLE_LEN)

    # Initialize
    model = NRMSModel(embedding_matrix, NUM_HEADS, HEAD_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            history = batch['history'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].to(device)
            hist_mask = batch['hist_mask'].to(device)


            optimizer.zero_grad()
            scores = model(history, candidates, hist_mask)


            # labels[:,0] is always the positive (index 0)
            target = torch.zeros(scores.size(0),
                    dtype=torch.long).to(device)
            loss = criterion(scores, target)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')


        # Save checkpoint
        torch.save(model.state_dict(),
                f'models/nrms_epoch{epoch+1}{NAME}.pt')
    
    evaluate(model=model, val_loader=val_loader, device=device)
