from utils.dataset_detection import EgoHandDataset
from torchvision import transforms
from tqdm.auto import tqdm

IMAGE_DIR = 'EgoHand_dataset/_LABELLED_SAMPLES'
ANNO_PATH = 'EgoHand_dataset/metadata.mat'

dataset = EgoHandDataset(IMAGE_DIR, ANNO_PATH, transform=transforms.Compose([
    transforms.ToTensor(),
]))

# prog_bar = tqdm(dataset, total=len(dataset))
# for i, data in enumerate(prog_bar):
#     prog_bar.set_description(desc=f"IDX: {i}")

print(len(dataset))
dataset.visualize(4799)