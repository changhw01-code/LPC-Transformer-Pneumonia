# LPC-Transformer-Pneumonia
Official implementation of LPC-Transformer for multi-class lung X-ray &amp; CT pneumonia classification
1.Project File Structure
# Project Root
├─ dataloader.py
├─ lpc_transformer.py
├─ train.py
└─ test.py

# Auto-generated folders (created automatically when running code, no manual creation required)
├─ dataset/
│  ├─ train/  # Store 11 categories of training images (each category as an independent subfolder)
│  └─ test/   # Store 11 categories of test images (each category as an independent subfolder)
├─ output/     # Output files: confusion matrix, PCA visualization, t-SNE visualization, classification report CSV
└─ best_model.pth  # Saved optimal model weight generated after training


2.Environment & Dependencies Installation
Option 1: Install via requirements.txt
pip install -r requirements.txt

Option 2: One-click pip installation command
pip install torch torchvision timm pytorch-optimizer pillow numpy pandas matplotlib seaborn scikit-learn tqdm

3. Dataset Preparation
Dataset Name: CL-COVIDset
Official Download Link:
https://www.kaggle.com/datasets/wumengqiu01/chiong-continual-learning-of-covid19
Dataset Placement Rules
Download the complete dataset from the above Kaggle link;
Extract the dataset and split into train and test folders;
Put the two folders into the project path ./dataset/train and ./dataset/test respectively;
Each folder contains 11 independent subfolders, corresponding to the 11 pneumonia classification categories defined in test.py;
The code uses PyTorch standard ImageFolder data loading mode, no path modification is required.
Category List (11 classes)
0: CT Normal
1: CT Omicron and Delta Variant
2: CT Other Pneumonia
3: CT Wildtype SARS-CoV-2
4: X-Ray Bacterial Pneumonia
5: X-Ray MERS
6: X-Ray Normal
7: X-Ray Omicron and Delta Variant
8: X-Ray Other Viral Pneumonia
9: X-Ray SARS
10: X-Ray Wildtype SARS-CoV-2

4. Run Steps
1）Download CL-COVIDset dataset from Kaggle and place files following the rules above;
2）Install all required dependencies;
3）Train the model and auto-run full evaluation:
python train.py
After training completes, the script will automatically generate all quantitative metrics and visualization charts under ./output/.
4） Separate test (only execute when best_model.pth weight file exists):

5. Tested Reproduction Environment
Python: 3.9 / 3.10
PyTorch: 2.1.0, CUDA 11.8
GPU: NVIDIA RTX 3090 / 4090 (minimum 6GB VRAM for training)
Operating System: Windows 11 / Ubuntu 20.04 / Ubuntu 22.04

6. Output Description
The ./output folder will store all experimental results after running:
Swin_Report.csv: Complete classification report (Precision, Recall, F1-score per class)
Swin_ConfusionMatrix.jpg: Confusion matrix heatmap
Swin_PCA.jpg: Feature distribution PCA visualization
Swin_TSNE.jpg: Feature distribution t-SNE visualization



