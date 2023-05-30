import optuna
import wandb
import torch.nn as nn
import torch 
import numpy as np
import config
from models.conTextTransformer import ConTextTransformer
from data.dataloader import Dataloader
import torch.nn.functional as F

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    # This part of the code will use the log_softmax function to compute the log of the softmax function
    # and then use the nll_loss function to compute the negative log likelihood loss.
    # This is done because the log_softmax function is numerically more stable than the softmax function.
    
    for i, (data_img, data_txt, txt_mask, target) in enumerate(data_loader):
        data_img = data_img.to(config.device)
        data_txt = data_txt.to(config.device)
        txt_mask = txt_mask.to(config.device)
        target = target.to(config.device)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        output = model(data_img, data_txt, txt_mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Print loss every 100 batches
        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data_img)) + '/' + '{:5}'.format(total_samples) +
                ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
    return i

    
def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for data_img, data_txt, txt_mask, target in data_loader:
            
            data_img = data_img.to(config.device)
            data_txt = data_txt.to(config.device)
            txt_mask = txt_mask.to(config.device)
            target = target.to(config.device)
            
            criterion = nn.CrossEntropyLoss()
            output = model(data_img, data_txt, txt_mask)
            loss = criterion(output, target)
            _, pred = torch.max(output, dim=1)

            total_loss += loss
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nTRAIN - Average train loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + ' {:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

    return (correct_samples / total_samples).to('cpu').numpy(), avg_loss


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(config.device), labels.to(config.device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(acc, example_ct, epoch, loss):
    # Where the magic happens
    wandb.log({"epoch": epoch, "train_accuracy": acc, "train_loss": loss}, step=example_ct)
    print(f"TRAIN - Accuracy after {str(example_ct).zfill(5)} examples: {acc:.3f}\n")

def train(model, train_loader, criterion, optimizer, scheduler, epochs, run_name):
    
    # Keep track of loss and accuracy
    train_loss_history, test_loss_history = [], []
    acc_history = []
    best_acc = 0
    best_loss = np.inf
    example_ct = 0  # number of examples seen
    
    # Train the model that has the best hyperparameters
    for epoch in range(1, epochs + 1):
        
        print('\nEpoch:', epoch)
        iter_in_epoch = train_epoch(model, optimizer, train_loader, train_loss_history)
        acc, loss = evaluate(model, train_loader, test_loss_history)
        
        acc_history.append(acc)
        if acc>best_acc: 
            best_acc = acc
        
        example_ct += len(train_loader.dataset)
        
        # Save the model with the best accuracy
        if loss<best_loss: 
            acc_best_epoch_loss = acc
            best_epoch_loss = loss
            
        scheduler.step()

        train_log(acc, example_ct, epoch, loss)


    return best_epoch_loss, acc_best_epoch_loss
                
    

###############################

def objective(trial):
    # Sample hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    mlp_dim = trial.suggest_int("mlp_dim", 256, 1024, log=True)
    depth = trial.suggest_int("depth", 1, 3)
    heads = trial.suggest_int("heads", 2, 6)
    epochs = 5

    # Starting WandrB run.
    config = {"trial_lr": lr,
              "trial_mlp_dim": mlp_dim,
              "trial_depth": depth,
              "trail_heads":heads,
              "epochs":epochs,
              "dataset":"base",
              "architecture": "Context Transformer",
              "pretrained_model":config.pretrained_model,
              "text_model": config.text_model, 
              }
    
    run = wandb.init(project="bussiness_uab_optuna",
                         name=f"trial_",
                         group="sampling",
                         config=config,
                         reinit=True)


    # Train the model and return the validation accuracy
    train_loader, test_loader, eval_loader = Dataloader().get_loaders()
    model = ConTextTransformer(
        image_size=config.image_size,
        num_classes=config.num_classes,
        channels=config.channels,
        dim=config.dim,
        depth=depth,
        heads=heads, 
        mlp_dim=mlp_dim
    ).to(config.device)

    epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=config.gamma)
    criterion = nn.CrossEntropyLoss()
    best_epoch_loss, acc_best_epoch_loss = train(model, train_loader, criterion, optimizer, scheduler, epochs)

    # WandB logging.
    with run:
        run.log({"best_epoch_loss": best_epoch_loss, "acc_best_epoch_loss":acc_best_epoch_loss}, step=trial.number)

    return best_epoch_loss

def hyperparameter_tuning():

    # Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    # Create the summary run.
    summary = wandb.init(project="bussiness_uab_optuna",
                         name="summary",
                         job_type="logging")

    # Getting the study trials.
    trials = study.trials

    # WandB summary.
    for step, trial in enumerate(trials):
        # Logging the loss.
        summary.log({"loss": trial.value}, step=step)

        # Logging the parameters.
        for k, v in trial.params.items():
            summary.log({k: v}, step=step)



    # Get best trial
    best_trial = study.best_trial
    best_lr = best_trial.params["lr"]
    best_mlp_dim = best_trial.params["mlp_dim"]
    best_depth = best_trial.params["depth"]
    heads = best_trial.params["heads"]

    print("\n\n\n")
    print("---------------------------------Best trial:------------------------------------------------")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
    print("--------------------------------------------------------------------------------------------")
    print("\n\n\n")




if __name__ == "__main__":
    hyperparameter_tuning()