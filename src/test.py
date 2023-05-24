import wandb
import torch
import config
from wandb import AlertLevel
from datetime import timedelta
import torch.nn.functional as F
import torch.nn as nn
import os


def test(model, test_loader, save=False, run_name=None):
    
    # Run the model on some test examples
    with torch.no_grad():
        
        correct, total = 0, 0
        for data_img, data_txt, txt_mask, target in test_loader:
            data_img = data_img.to(config.device)
            data_txt = data_txt.to(config.device)
            txt_mask = txt_mask.to(config.device)
            target = target.to(config.device)
            criterion = nn.CrossEntropyLoss()
            output = model(data_img, data_txt, txt_mask)
            loss = criterion(output, target)
            _, predicted = torch.max(output, dim=1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print(f"\nTEST - Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}\n")
        
        acc = correct / total
        threshold = 0.7
        
        if acc < threshold:
            wandb.alert(
                title='Low accuracy',
                text=f'Accuracy {acc} is below the acceptable threshold {threshold}',
                level=AlertLevel.WARN,
                wait_duration=timedelta(minutes=10)
            )
        
        wandb.log({"test_accuracy": acc})
        
    with open(os.path.join("./results", run_name, "config.txt"), "w") as f:
        f.write("Test - Accuracy: {}\n".format(acc))


    if save:
        print(len(data_img))
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model,  # model being run
                          data_img,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")
        print("Model saved")
