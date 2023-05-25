import optuna
import torch.nn as nn
from train import * 
import config
from models.conTextTransformer import ConTextTransformer
from data.dataloader import *

def objective(trial):
    # Sample hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    mlp_dim = trial.suggest_int("mlp_dim", 32, 512, log=True)
    depth = trial.suggest_int("depth", 1, 3)
    heads = trial.suggest_int("heads", 3, 5)

    # Starting WandrB run.
    config = {"trial_lr": lr,
              "trial_mlp_dim": mlp_dim,
              "trial_depth": depth,
              "trail_heads":heads
              }
    
    run = wandb.init(project="bussiness_uab_optuna",
                         name=f"trial_",
                         group="sampling",
                         config=config,
                         reinit=True)


    # Train the model and return the validation accuracy
    train_loader, test_loader = Dataloader().get_loaders()
    model = ConTextTransformer(
        image_size=config.image_size,
        num_classes=config.num_classes,
        channels=config.channels,
        dim=config.dim,
        depth=depth,
        heads=heads, 
        mlp_dim=mlp_dim
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=config.gamma)
    criterion = nn.CrossEntropyLoss()
    train(model, train_loader, criterion, optimizer, scheduler)
    accuracy, avg_loss = evaluate(model, test_loader, [])

    # WandB logging.
    with run:
        run.log({"accuracy": accuracy, "loss":avg_loss}, step=trial.number)

    return avg_loss

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
    
def main():
    hyperparameter_tuning()