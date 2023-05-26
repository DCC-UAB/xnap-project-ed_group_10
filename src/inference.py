import torch
import config
from utils.utils import context_inference
import numpy as np
from models.conTextTransformer import ConTextTransformer


def inference_test():
    
    model = ConTextTransformer(
        image_size=config.image_size,
        num_classes=config.num_classes,
        channels=config.channels,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads, 
        mlp_dim=config.mlp_dim
    )
    
    model.load_state_dict(torch.load('./src/models/all_best_params.pth'))
    model.to(config.device)
    model.eval()

    class_labels = {
        1: "Bakery", 
        2: "Barber", 
        3: "Bistro", 
        4: "Bookstore", 
        5: "Cafe", 
        6: "ComputerStore", 
        7: "CountryStore", 
        8: "Diner", 
        9: "DiscounHouse", 
        10: "Dry Cleaner", 
        11: "Funeral", 
        12: "Hotspot", 
        13: "MassageCenter", 
        14: "MedicalCenter", 
        15: "PackingStore", 
        16: "PawnShop", 
        17: "PetShop", 
        18: "Pharmacy", 
        19: "Pizzeria", 
        20: "RepairShop", 
        21: "Restaurant", 
        22: "School", 
        23: "SteakHouse", 
        24: "Tavern", 
        25: "TeaHouse", 
        26: "Theatre", 
        27: "Tobacco", 
        28: "Motel"
        }

    # !wget -q https://gailsbread.co.uk/wp-content/uploads/2017/11/Summertown-1080x675.jpg
    
    img = './data/Summertown-1080x675.jpg'

    import matplotlib.pyplot as plt
    plt.imshow(plt.imread(img))

    OCR_tokens = [] # Let's imagine our OCR model does not recognize any text

    probs = context_inference(model, img, OCR_tokens)
    class_id = np.argmax(probs)
    print('Prediction without text: {} ({})'.format(class_labels[class_id+1], probs[0,class_id]))

    OCR_tokens = [
        'GAIL', 
        'ARTISAN', 
        'BAKERY'
        ] # Simulate a perfect OCR output

    probs = context_inference(model, img, OCR_tokens)
    class_id = np.argmax(probs)
    print('Prediction with text:\t {} ({})'.format(class_labels[class_id+1], probs[0,class_id]))

    # Prediction without text: Diner (0.3832743763923645)
    # Prediction with text:	 Bakery (0.9980818033218384)


if __name__ == '__main__':
    inference_test()